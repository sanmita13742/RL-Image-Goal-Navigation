"""
src/data/dinov3_encoder.py
============================================================
Phase 3 + 4: Frozen DINOv3 encoding with batch inference.

CONFIRMED live shapes (2026-08-12):
  model(x).shape                : (1, 384)          ← classification head
  forward_features(x).shape     : (1, 1377, 384)    ← all tokens
  Special tokens                : 5 (1 CLS + 4 register)
  Patch tokens                  : 1372 (= 28 × 49)
  Patch grid                    : (1, 28, 49, 384)
  Center crop (CROP_H=14, W=25) : (1, 14, 25, 384)
  Embedding dim D               : 384
  SSD (blank image)             : ~0.0079

PAPER SPECIFIES:
  - DINOv3 ViT-S/16
  - Input: 448 × 784 (H × W)
  - Output: 28 × 49 patch embeddings
  - SSD threshold: 0.02
  - Model must remain frozen (eval, no_grad)

PAPER DOES NOT SPECIFY:
  - Image normalization (implementation uses ToTensor() only -> [0,1])
  - Whether φ(o) is full grid or pooled (implementation: x_norm_clstoken -> [384])
  - SSD crop dimensions (implementation: 14×25 from encoder_v2.py)

This module:
  1. Loads frozen DINOv3 (eval, no_grad)
  2. Encodes images in batches
  3. Computes per-frame:
       - patch_grid: [28, 49, 384]   (for SSD)
       - phi:        [384]            (x_norm_clstoken, for state/reward)
       - ssd:        float            (scalar, for goal filtering)
       - valid_goal: bool             (ssd > threshold)
  4. Saves to data/processed/dinov3/

Output files:
  phi_vectors.npy    — float32 array [N, 384]  (~108 MB for N=72000)
  ssd_scores.npy     — float32 array [N]       (~288 KB for N=72000)
  valid_goals.npy    — bool    array [N]
  encoder_meta.json  — shapes, model name, config
"""

import sys
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
import timm
from pathlib import Path
from PIL import Image
from torchvision import transforms


VENV_PYTHON = str(Path(__file__).parent.parent.parent / "vision-dinov3")


class FrozenDINOv3:
    """
    Frozen DINOv3 encoder. Computes patch grids, pooled phi vectors, and SSD.

    PAPER SPECIFIES:
      - Model: DINOv3 ViT-S/16
      - Input: 448 × 784 RGB
      - Frozen (eval + no_grad)
      - Output: 28 × 49 patch embeddings

    PAPER DOES NOT SPECIFY:
      - Image normalization -> IMPLEMENTATION CHOICE: ToTensor() only
      - SSD crop -> IMPLEMENTATION CHOICE: 14×25 center crop (from encoder_v2.py)
      - φ(o) form -> IMPLEMENTATION CHOICE: x_norm_clstoken -> [384]
    """

    # Architecture constants (confirmed by live probe)
    PATCH_H       = 28
    PATCH_W       = 49
    N_SPECIAL     = 5        # 1 CLS + 4 register tokens
    EMBED_DIM     = 384
    CROP_H        = 14       # center crop for SSD (encoder_v2.py hardcoded)
    CROP_W        = 25
    SSD_THRESHOLD = 0.02     # PAPER SPECIFIES

    def __init__(
        self,
        model_name:    str = "vit_small_patch16_dinov3",
        image_h:       int = 448,
        image_w:       int = 784,
        ssd_threshold: float = 0.02,
        device:        str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.ssd_threshold = ssd_threshold
        self.image_h = image_h
        self.image_w = image_w

        print(f"  Loading {model_name} on {self.device}...")
        self.model = timm.create_model(model_name, pretrained=True).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
            
        if hasattr(self.model, "num_features"):
            self.EMBED_DIM = self.model.num_features
        elif hasattr(self.model, "embed_dim"):
            self.EMBED_DIM = self.model.embed_dim
        else:
            self.EMBED_DIM = 384
            
        print(f"  Model loaded. Parameters frozen. Dim={self.EMBED_DIM}")

        # PAPER DOES NOT SPECIFY normalization.
        # IMPLEMENTATION CHOICE: ToTensor() only, matching encoder_v2.py
        self.transform = transforms.Compose([
            transforms.Resize((image_h, image_w)),
            transforms.ToTensor(),
            # No ImageNet normalization — matches encoder_v2.py
        ])

    def _load_image(self, path: str) -> torch.Tensor:
        """Load and preprocess a single image -> [1, 3, H, W]."""
        img = Image.open(path).convert("RGB")
        return self.transform(img).unsqueeze(0)

    def _forward_batch(self, batch: torch.Tensor):
        """
        batch: [B, 3, H, W]
        Returns:
          patch_grid: [B, 28, 49, 384]
          phi:        [B, 384]   L2-normalized x_norm_clstoken
          ssd:        [B]        scalar SSD per image
        """
        batch = batch.to(self.device)
        with torch.no_grad():
            feats = self.model.forward_features(batch)   # [B, 1377, 384]

        # Remove 5 special tokens -> patch tokens [B, 1372, 384]
        patches = feats[:, self.N_SPECIAL:]
        B, N, D = patches.shape
        assert N == self.PATCH_H * self.PATCH_W, \
            f"Expected {self.PATCH_H * self.PATCH_W} patches, got {N}"

        # Reshape to spatial grid [B, 28, 49, 384]
        grid = patches.reshape(B, self.PATCH_H, self.PATCH_W, D)

        # ── φ(o): DINOv3 native normalized CLS token ────────────────────────────
        # PAPER SPECIFIES: "final hidden-layer patch embeddings / normalized visual representation"
        # IMPLEMENTATION CHOICE: Use native normalized CLS token.
        phi = feats[:, 0]                                      # CLS token [B, D]
        phi = F.normalize(phi, p=2, dim=-1)                    # L2 normalize for cosine similarity

        # ── SSD: center crop → spatial std ────────────────────────────────────
        sh = (self.PATCH_H - self.CROP_H) // 2   # = 7
        sw = (self.PATCH_W - self.CROP_W) // 2   # = 12
        crop    = grid[:, sh:sh+self.CROP_H, sw:sw+self.CROP_W, :]  # [B,14,25,384]
        tokens  = crop.reshape(B, -1, D)                              # [B,350,384]
        tokens  = F.normalize(tokens, p=2, dim=-1)
        ssd     = tokens.std(dim=1).mean(dim=-1)                      # [B]

        return (
            grid.cpu().half() if grid is not None else None,
            phi.cpu(),
            ssd.cpu(),
        )

    def encode_paths(
        self,
        image_paths:  list,
        batch_size:   int = 32,
        return_grids: bool = False,
        return_pooled:bool = True,
    ):
        """
        Encode a list of image paths.

        Returns:
          phi_vectors  : np.ndarray [N, 384]  float32  (if return_pooled=True)
          ssd_scores   : np.ndarray [N]       float32
          valid_goals  : np.ndarray [N]       bool
          grids        : np.ndarray [N,28,49,384] float16  (if return_grids=True)
        """
        N = len(image_paths)
        phi_vectors = np.zeros((N, self.EMBED_DIM), dtype=np.float32) if return_pooled else None
        ssd_scores  = np.zeros(N, dtype=np.float32)
        grids       = None
        if return_grids:
            grids = np.zeros((N, self.PATCH_H, self.PATCH_W, self.EMBED_DIM),
                             dtype=np.float16)

        t_start = time.time()
        n_done  = 0

        for i in range(0, N, batch_size):
            batch_paths = image_paths[i : i + batch_size]

            # Load batch
            tensors = []
            for p in batch_paths:
                tensors.append(self._load_image(p))
            batch = torch.cat(tensors, dim=0)   # [B, 3, H, W]

            grid_b, phi_b, ssd_b = self._forward_batch(batch)
            B = len(batch_paths)

            if return_pooled:
                phi_vectors[i:i+B] = phi_b.numpy()
            ssd_scores[i:i+B]  = ssd_b.numpy()
            if return_grids:
                grids[i:i+B] = grid_b.numpy()

            n_done += B
            if n_done % max(batch_size * 10, 100) == 0 or n_done == N:
                elapsed  = time.time() - t_start
                fps      = n_done / elapsed
                eta      = (N - n_done) / fps if fps > 0 else 0
                print(f"    [{n_done:6d}/{N:6d}]  "
                      f"{fps:.1f} img/s  ETA {eta:.0f}s")

        elapsed = time.time() - t_start
        valid_goals = ssd_scores > self.ssd_threshold

        print(f"\n  Encoding complete:")
        print(f"    Frames        : {N:,}")
        print(f"    Device        : {self.device}")
        print(f"    Batch size    : {batch_size}")
        print(f"    Time          : {elapsed:.1f}s")
        print(f"    Throughput    : {N/elapsed:.1f} img/s")
        print(f"    SSD min/max   : {ssd_scores.min():.4f} / {ssd_scores.max():.4f}")
        print(f"    Valid goals   : {valid_goals.sum():,} / {N:,} "
              f"({100*valid_goals.mean():.1f}%)")

        return phi_vectors, ssd_scores, valid_goals, grids


def save_embeddings(
    out_dir:     Path,
    phi_vectors: np.ndarray,
    ssd_scores:  np.ndarray,
    valid_goals: np.ndarray,
    frame_index: "pd.DataFrame",
    cfg:         dict,
    grids:       np.ndarray = None,
):
    """Save encoded embeddings and metadata."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if phi_vectors is not None:
        np.save(out_dir / "phi_vectors.npy", phi_vectors)
    np.save(out_dir / "ssd_scores.npy",  ssd_scores)
    np.save(out_dir / "valid_goals.npy", valid_goals)

    if grids is not None:
        np.save(out_dir / "patch_grids.npy", grids)

    # Save global_step -> index mapping for fast lookup
    frame_index[["global_step", "segment_id", "segment_step", "rgb_path"]].to_parquet(
        out_dir / "embedding_index.parquet"
    )

    meta = {
        **cfg,
        "n_frames":          int(len(ssd_scores)),
        "ssd_shape":         list(ssd_scores.shape),
        "n_valid_goals":     int(valid_goals.sum()),
        "pct_valid_goals":   float(100 * valid_goals.mean()),
        "ssd_min":           float(ssd_scores.min()),
        "ssd_max":           float(ssd_scores.max()),
        "ssd_mean":          float(ssd_scores.mean()),
        "ssd_median":        float(np.median(ssd_scores)),
        "ssd_std":           float(ssd_scores.std()),
        "store_full_grids":  grids is not None,
        "feature_key":       "x_norm_clstoken",
    }
    with open(out_dir / "encoder_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Saved to {out_dir}:")
    if phi_vectors is not None:
        print(f"    phi_vectors.npy      {phi_vectors.nbytes / 1e6:.1f} MB")
    print(f"    ssd_scores.npy       {ssd_scores.nbytes / 1e3:.0f} KB")
    print(f"    valid_goals.npy      {valid_goals.nbytes / 1e3:.0f} KB")
    if grids is not None:
        print(f"    patch_grids.npy      {grids.nbytes / 1e6:.1f} MB")
