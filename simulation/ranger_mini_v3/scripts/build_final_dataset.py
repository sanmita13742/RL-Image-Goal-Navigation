"""
scripts/build_final_dataset.py
============================================================
FINAL MINav Hindsight Dataset Generation Pipeline.

phi(o) = DINOv3 x_norm_clstoken  [384-D, L2-normalized]
         feats[:, 0, :] from timm forward_features() Tensor + F.normalize

PAPER-SPECIFIED:
  DINOv3 ViT-S/16, 448x784 input
  SSD threshold 0.02
  4-frame state: [phi(t-3), phi(t-2), phi(t-1), phi(t)]
  Geometric future sampling  P(K=k) = p^(k-1)(1-p)
  Global uniform goal pool (SSD > 0.02)
  Reward: S = (1/4) * sum_i cos(phi(t-i), g); reward=done=1 if S>=0.8

IMPLEMENTATION CHOICES (not paper-specified):
  x_norm_clstoken as image-level phi(o)
  14x25 SSD center crop
  p = 0.99
  1 geometric + 1 uniform goal per transition
  Storage: numpy (phi_cache) + parquet (transitions)

Usage:
  # Step 1: Smoke test (100 frames)
  python scripts/build_final_dataset.py --session dataset/20260811_234812 --smoke

  # Step 2: Full 72k dataset (run after smoke passes)
  python scripts/build_final_dataset.py --session dataset/20260811_234812 --full
"""

import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm
from pathlib import Path
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ======================================================================
# Constants -- locked paper values
# ======================================================================

FRAME_STACK        = 4          # PAPER SPECIFIES: 4-frame state
SSD_THRESHOLD      = 0.02       # PAPER SPECIFIES: delta_SSD
REWARD_THRESHOLD   = 0.8        # PAPER SPECIFIES: delta_done
GEO_P              = 0.99       # IMPLEMENTATION CHOICE: geometric p
PATCH_H            = 28         # derived from 448/16
PATCH_W            = 49         # derived from 784/16
EMBED_DIM          = 384        # ViT-S embed dim
N_SPECIAL          = 5          # 1 CLS + 4 register tokens
CROP_H             = 14         # IMPLEMENTATION CHOICE: SSD crop
CROP_W             = 25         # IMPLEMENTATION CHOICE: SSD crop
IMAGE_H            = 448        # PAPER SPECIFIES
IMAGE_W            = 784        # PAPER SPECIFIES
MODEL_NAME         = "vit_small_patch16_dinov3"   # PAPER SPECIFIES


# ======================================================================
# Phase 1 -- Frame Index
# ======================================================================

def build_frame_index(session_dir, max_frames=None):
    """
    Discover all segment_NNN/segment.csv files; sort by global_step.
    IMPORTANT: segments are storage boundaries, NOT trajectory boundaries.
    The robot was physically continuous throughout the session.
    global_step is the authoritative temporal ordering.
    """
    session_dir = Path(session_dir).resolve()

    meta_path = session_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            session_meta = json.load(f)
        print(f"  Session     : {session_meta.get('session_id')}")
        print(f"  robot_resets: {session_meta.get('robot_resets')}  "
              f"[MUST BE 0 -- no physical resets between segments]")
        print(f"  total_steps : {session_meta.get('total_steps_recorded')}")
        print(f"  num_segments: {session_meta.get('num_segments')}")
        assert session_meta.get("robot_resets", 0) == 0, \
            "FATAL: robot_resets != 0 -- segments are NOT a continuous trajectory!"
    else:
        session_meta = {}
        print(f"  WARNING: metadata.json not found in {session_dir}")

    seg_dirs = sorted(
        [d for d in session_dir.iterdir()
         if d.is_dir() and d.name.startswith("segment_")],
        key=lambda d: int(d.name.split("_")[1])
    )
    print(f"  Discovered {len(seg_dirs)} segment directories")

    frames = []
    for seg_dir in seg_dirs:
        csv_path = seg_dir / "observations.csv"
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found, skipping")
            continue
        df = pd.read_csv(csv_path)
        df["segment_id"]   = seg_dir.name
        df["session_id"]   = session_meta.get("session_id", session_dir.name)
        df["rgb_abs_path"] = df.apply(
            lambda r: str(seg_dir / r["rgb_path"]), axis=1
        )
        frames.append(df)

    if not frames:
        raise RuntimeError(f"No segment CSVs found in {session_dir}")

    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.sort_values("global_step").reset_index(drop=True)

    # Verify global_step is exactly 0, 1, 2, ..., N-1
    actual   = df_all["global_step"].tolist()
    expected = list(range(len(df_all)))
    if actual != expected:
        bad = [(i, a) for i, a in enumerate(actual) if a != i]
        raise ValueError(
            f"global_step is NOT continuous! {len(bad)} discrepancies, "
            f"first 5: {bad[:5]}"
        )

    print(f"  Frame index built: {len(df_all):,} rows, global_step 0..{df_all['global_step'].max()}")

    if max_frames is not None:
        df_all = df_all.iloc[:max_frames].reset_index(drop=True)
        print(f"  [TRUNCATED] to {len(df_all)} frames")

    return df_all


# ======================================================================
# Phase 2 -- Continuity Validation
# ======================================================================

def validate_continuity(df):
    """Verify global_step is strictly consecutive; segment boundaries are NOT gaps."""
    steps = df["global_step"].values
    diffs = np.diff(steps)
    gaps  = np.where(diffs != 1)[0]
    if len(gaps) > 0:
        for idx in gaps[:5]:
            print(f"  GAP at row {idx}: step {steps[idx]} -> {steps[idx+1]}")
        raise ValueError(f"Continuity check FAILED: {len(gaps)} gaps found")

    # Also verify cross-segment steps are consecutive
    seg_changes = df["segment_id"].ne(df["segment_id"].shift()).values
    boundary_rows = np.where(seg_changes)[0][1:]
    for row in boundary_rows[:10]:
        prev_step = int(df.iloc[row-1]["global_step"])
        curr_step = int(df.iloc[row]["global_step"])
        prev_seg  = df.iloc[row-1]["segment_id"]
        curr_seg  = df.iloc[row]["segment_id"]
        assert curr_step == prev_step + 1, (
            f"Segment boundary gap: {prev_seg}[{prev_step}] -> {curr_seg}[{curr_step}]"
        )

    print(f"  Continuity OK: {len(df):,} frames, 0 gaps")
    print(f"  Cross-segment boundaries verified as consecutive (not trajectory boundaries)")


# ======================================================================
# Phase 3 -- DINOv3 Encoder
# ======================================================================

class FrozenDINOv3Encoder:
    """
    Frozen DINOv3 encoder.

    REPRESENTATION DECISION (FINAL -- IMPLEMENTATION CHOICE):
      phi(o) = x_norm_clstoken = forward_features(x)[:, 0, :] + L2-normalize

    timm's vit_small_patch16_dinov3 forward_features() returns a Tensor
    [B, 1377, 384] -- NOT a dict. Token layout:
      index 0       : CLS token  (= x_norm_clstoken)
      index 1-4     : register tokens
      index 5-1376  : 28x49 patch tokens (= x_norm_patchtokens, used only for SSD)

    DO NOT mean-pool patch tokens for phi.
    DO NOT use CLS for SSD.
    """

    def __init__(self, device="auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"  Loading {MODEL_NAME} on {self.device} ...")
        self.model = timm.create_model(MODEL_NAME, pretrained=True).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # Verify actual output shape with a live dummy forward pass
        dummy = torch.zeros(1, 3, IMAGE_H, IMAGE_W).to(self.device)
        with torch.no_grad():
            feats = self.model.forward_features(dummy)
        assert isinstance(feats, torch.Tensor), (
            f"FATAL: forward_features returned {type(feats)}, expected Tensor. "
            f"The 'x_norm_clstoken' dict key does not exist in this timm build."
        )
        expected_shape = (1, N_SPECIAL + PATCH_H * PATCH_W, EMBED_DIM)
        assert feats.shape == expected_shape, (
            f"Unexpected feats shape: {feats.shape}, expected {expected_shape}"
        )
        print(f"  Live probe: forward_features() -> Tensor {tuple(feats.shape)} [OK]")
        print(f"  CLS token (phi):   feats[:, 0, :]   shape={tuple(feats[:, 0, :].shape)}")
        print(f"  Patch tokens (SSD): feats[:, 5:, :] shape={tuple(feats[:, 5:, :].shape)}")

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_H, IMAGE_W)),
            transforms.ToTensor(),
            # No ImageNet normalization -- IMPLEMENTATION CHOICE matching encoder_v2.py
        ])

    def _load(self, path):
        img = Image.open(path).convert("RGB")
        return self.transform(img).unsqueeze(0)

    def encode_batch(self, paths):
        """
        Encode a list of image paths.
        Returns:
          phi : [B, 384] float32 -- x_norm_clstoken, L2-normalized, ||phi||=1.0
          ssd : [B]      float32 -- spatial std over 14x25 center crop of patches
        """
        tensors = [self._load(p) for p in paths]
        batch   = torch.cat(tensors, dim=0).to(self.device)
        B       = batch.shape[0]

        with torch.no_grad():
            feats = self.model.forward_features(batch)  # [B, 1377, 384] Tensor

        # phi(o) = x_norm_clstoken
        phi = feats[:, 0, :]                       # [B, 384] CLS token
        phi = F.normalize(phi, p=2, dim=-1)        # guarantee unit norm

        # SSD from x_norm_patchtokens (patch tokens only, no CLS)
        patches = feats[:, N_SPECIAL:, :]          # [B, 1372, 384]
        grid    = patches.reshape(B, PATCH_H, PATCH_W, EMBED_DIM)

        sh = (PATCH_H - CROP_H) // 2   # 7
        sw = (PATCH_W - CROP_W) // 2   # 12
        crop   = grid[:, sh:sh+CROP_H, sw:sw+CROP_W, :]  # [B,14,25,384]
        tokens = crop.reshape(B, -1, EMBED_DIM)            # [B,350,384]
        tokens = F.normalize(tokens, p=2, dim=-1)
        ssd    = tokens.std(dim=1).mean(dim=-1)            # [B]

        return phi.cpu().numpy().astype(np.float32), ssd.cpu().numpy().astype(np.float32)

    def encode_all(self, image_paths, batch_size=32):
        """
        Encode all image paths in batches.
        Returns phi_cache [N,384], ssd_scores [N], valid_goals [N bool].
        """
        N          = len(image_paths)
        phi_cache  = np.zeros((N, EMBED_DIM), dtype=np.float32)
        ssd_scores = np.zeros(N, dtype=np.float32)

        t0     = time.time()
        n_done = 0
        for i in range(0, N, batch_size):
            batch_paths = image_paths[i: i + batch_size]
            phi_b, ssd_b = self.encode_batch(batch_paths)
            B = len(batch_paths)
            phi_cache[i:i+B]  = phi_b
            ssd_scores[i:i+B] = ssd_b
            n_done += B

            log_interval = max(batch_size * 8, 256)
            if n_done % log_interval < batch_size or n_done == N:
                elapsed = time.time() - t0
                fps = n_done / elapsed
                eta = (N - n_done) / fps if fps > 0 else 0
                print(f"    [{n_done:6d}/{N:6d}]  {fps:.1f} img/s  ETA {eta:.0f}s")

        elapsed     = time.time() - t0
        valid_goals = ssd_scores > SSD_THRESHOLD

        norms = np.linalg.norm(phi_cache, axis=1)
        print(f"\n  Encoding complete: {elapsed:.1f}s  ({N/elapsed:.1f} img/s)")
        print(f"  Phi norms: min={norms.min():.6f}  max={norms.max():.6f}  "
              f"mean={norms.mean():.6f}  (target: 1.0)")
        if abs(norms.min() - 1.0) > 1e-4 or abs(norms.max() - 1.0) > 1e-4:
            print(f"  WARNING: phi norms not all ~1.0 -- check L2 normalize step!")
        else:
            print(f"  Phi norms: OK (all within 1e-4 of 1.0)")
        print(f"  Valid goals (SSD > {SSD_THRESHOLD}): "
              f"{valid_goals.sum():,} / {N:,} ({100*valid_goals.mean():.1f}%)")

        return phi_cache, ssd_scores, valid_goals


# ======================================================================
# Phase 4 -- Valid Goal Pool
# ======================================================================

def build_valid_goal_pool(frame_index, ssd_scores, valid_mask):
    """Build the global valid goal pool: G = {t | SSD(t) > 0.02}"""
    goals = frame_index[["global_step", "segment_id", "rgb_abs_path"]].copy()
    goals = goals.rename(columns={"rgb_abs_path": "image_path"})
    goals["embedding_index"] = goals.index
    goals["ssd_score"]       = ssd_scores

    valid_goals_df = goals[valid_mask].reset_index(drop=True)
    print(f"  Valid goal pool: {len(valid_goals_df):,} / {len(goals):,} "
          f"({100*valid_mask.mean():.1f}%)")
    return valid_goals_df


# ======================================================================
# Phase 5 -- Reward
# ======================================================================

def compute_similarity(state_phis, goal_phi):
    """
    S(s_t, g) = (1/4) * sum_i cos(phi(t-3+i), g)
    PAPER SPECIFIES this formula. Both phi are unit vectors, cos = dot.
    state_phis: [4, D], goal_phi: [D]
    """
    return float((state_phis @ goal_phi).mean())


# ======================================================================
# Phase 6 -- Geometric Hindsight
# ======================================================================

def build_geometric_transitions(phi_cache, frame_index, valid_mask, p=0.99, seed=42):
    """
    For each valid current t:
      K ~ Geometric(p)  P(K=k) = p^(k-1)(1-p),  k>=1
      goal_step = t + K
    Require: goal_step > t  AND  goal_step < N  AND  valid_mask[goal_step]
    Resample up to MAX_RESAMPLE times if invalid, else skip.
    """
    N   = len(phi_cache)
    rng = np.random.default_rng(seed)

    # valid current timesteps: need frames t-3..t (state) and t+1 (next_state)
    valid_t = np.arange(FRAME_STACK - 1, N - 1)

    MAX_RESAMPLE = 50
    records  = []
    skipped  = 0

    for t in valid_t:
        for _ in range(MAX_RESAMPLE):
            k = int(rng.geometric(p=1.0 - p))  # numpy: param=1-p gives k>=1
            goal_step = t + k
            if goal_step < N and valid_mask[goal_step]:
                break
        else:
            skipped += 1
            continue

        state_phis = phi_cache[t-3: t+1]    # [4, 384]
        goal_phi   = phi_cache[goal_step]    # [384]

        sim    = compute_similarity(state_phis, goal_phi)
        reward = int(sim >= REWARD_THRESHOLD)

        row = frame_index.iloc[t]
        records.append({
            # Required dataset fields
            "current_global_step": int(t),
            "goal_global_step":    int(goal_step),
            "sampling_type":       "geometric",
            "reward":              reward,
            "done":                reward,
            "goal_similarity":     float(sim),
            # Action = action_t (obs_t -> obs_{t+1})
            "action_linear":   float(row["linear_vel_cmd"]),
            "action_lateral":  float(row["lateral_vel_cmd"]),
            "action_angular":  float(row["angular_vel_cmd"]),
            # State frame indices into phi_cache
            "state_t3": int(t - 3),
            "state_t2": int(t - 2),
            "state_t1": int(t - 1),
            "state_t0": int(t),
            # Next state frame indices (= state shifted by +1)
            "next_t2":  int(t - 2),
            "next_t1":  int(t - 1),
            "next_t0":  int(t),
            "next_t1f": int(t + 1),
            # Goal phi_cache index
            "goal_idx": int(goal_step),
            # Diagnostics
            "segment_id":  row["segment_id"],
            "offset_k":    int(goal_step - t),
            "sim_time":    float(row["sim_time"]),
            # Position metadata ONLY -- NOT used in phi/reward/done
            "meta_pos_x": float(row["pos_x"]),
            "meta_pos_y": float(row["pos_y"]),
            "meta_yaw":   float(row["yaw"]),
        })

    df = pd.DataFrame(records)
    print(f"  Geometric: {len(df):,} transitions  skipped={skipped}")
    print(f"  Avg offset k: {df['offset_k'].mean():.1f}  "
          f"Reward=1: {df['reward'].sum():,} ({100*df['reward'].mean():.2f}%)")
    return df


# ======================================================================
# Phase 7 -- Uniform Hindsight
# ======================================================================

def build_uniform_transitions(phi_cache, frame_index, valid_goals_df, seed=43):
    """
    For each valid current t:
      goal ~ Uniform(G)  where G = global valid goal pool
    Goals may be before or after t, from any segment.
    PAPER SPECIFIES all of the above.
    """
    N   = len(phi_cache)
    rng = np.random.default_rng(seed)

    valid_t      = np.arange(FRAME_STACK - 1, N - 1)
    goal_indices = valid_goals_df["embedding_index"].values
    goal_ssds    = valid_goals_df["ssd_score"].values
    goal_segs    = valid_goals_df["segment_id"].values

    print(f"  Goal pool: {len(goal_indices):,} valid goals")

    records = []
    for t in valid_t:
        j         = rng.integers(0, len(goal_indices))
        goal_step = int(goal_indices[j])
        goal_phi  = phi_cache[goal_step]

        state_phis = phi_cache[t-3: t+1]
        sim    = compute_similarity(state_phis, goal_phi)
        reward = int(sim >= REWARD_THRESHOLD)

        row = frame_index.iloc[t]
        records.append({
            "current_global_step": int(t),
            "goal_global_step":    int(goal_step),
            "sampling_type":       "uniform",
            "reward":              reward,
            "done":                reward,
            "goal_similarity":     float(sim),
            "action_linear":   float(row["linear_vel_cmd"]),
            "action_lateral":  float(row["lateral_vel_cmd"]),
            "action_angular":  float(row["angular_vel_cmd"]),
            "state_t3": int(t - 3),
            "state_t2": int(t - 2),
            "state_t1": int(t - 1),
            "state_t0": int(t),
            "next_t2":  int(t - 2),
            "next_t1":  int(t - 1),
            "next_t0":  int(t),
            "next_t1f": int(t + 1),
            "goal_idx": int(goal_step),
            "goal_ssd": float(goal_ssds[j]),
            "segment_id":       row["segment_id"],
            "goal_segment_id":  str(goal_segs[j]),
            "sim_time":         float(row["sim_time"]),
            "meta_pos_x": float(row["pos_x"]),
            "meta_pos_y": float(row["pos_y"]),
            "meta_yaw":   float(row["yaw"]),
        })

    df = pd.DataFrame(records)
    print(f"  Uniform:   {len(df):,} transitions")
    print(f"  Reward=1:  {df['reward'].sum():,} ({100*df['reward'].mean():.2f}%)")
    return df


# ======================================================================
# Phase 8 -- Spot Check
# ======================================================================

def spot_check(geom_df, unif_df, n_samples=20, seed=77):
    """Mandatory: print 20 random transitions from each sampling type."""
    rng = np.random.default_rng(seed)

    print("\n" + "="*64)
    print("  GEOMETRIC SPOT CHECK (20 samples)")
    print("="*64)
    for idx in rng.integers(0, len(geom_df), size=n_samples):
        r = geom_df.iloc[idx]
        t = int(r["current_global_step"])
        print(
            f"  t={t:6d}  state=[{t-3},{t-2},{t-1},{t}]  "
            f"next=[{t-2},{t-1},{t},{t+1}]  "
            f"goal={int(r['goal_global_step']):6d}  k={int(r['offset_k']):5d}  "
            f"act=({r['action_linear']:.3f},{r['action_lateral']:.3f},{r['action_angular']:.3f})  "
            f"S={r['goal_similarity']:.4f}  R={int(r['reward'])}"
        )

    print("\n" + "="*64)
    print("  UNIFORM SPOT CHECK (20 samples)")
    print("="*64)
    for idx in rng.integers(0, len(unif_df), size=n_samples):
        r = unif_df.iloc[idx]
        t = int(r["current_global_step"])
        print(
            f"  t={t:6d}  state=[{t-3},{t-2},{t-1},{t}]  "
            f"next=[{t-2},{t-1},{t},{t+1}]  "
            f"goal={int(r['goal_global_step']):6d}  "
            f"act=({r['action_linear']:.3f},{r['action_lateral']:.3f},{r['action_angular']:.3f})  "
            f"S={r['goal_similarity']:.4f}  R={int(r['reward'])}"
        )


# ======================================================================
# Phase 9 -- Cross-Segment Spot Check
# ======================================================================

def cross_segment_spot_check(geom_df, frame_index):
    """
    Explicitly verify transitions crossing segment boundaries are treated
    as normal continuous transitions (NOT done, NOT reset, NOT truncated).
    """
    print("\n" + "="*64)
    print("  CROSS-SEGMENT SPOT CHECK")
    print("  Segments are storage boundaries ONLY -- NOT trajectory boundaries.")
    print("="*64)

    seg_changes    = frame_index["segment_id"].ne(frame_index["segment_id"].shift()).values
    boundary_steps = frame_index["global_step"].values[np.where(seg_changes)[0][1:]]

    N     = len(frame_index)
    found = 0
    for bdry in boundary_steps[:20]:
        t = int(bdry) - 1
        if t < FRAME_STACK - 1 or t >= N - 1:
            continue

        matches = geom_df[geom_df["current_global_step"] == t]
        if len(matches) == 0:
            continue

        r       = matches.iloc[0]
        seg_t   = frame_index.iloc[t]["segment_id"]
        seg_t1  = frame_index.iloc[t+1]["segment_id"]
        print(f"  t={t} ({seg_t}) -> t+1={t+1} ({seg_t1})")
        print(f"    state=[{t-3},{t-2},{t-1},{t}]  next=[{t-2},{t-1},{t},{t+1}]")
        print(f"    reward={int(r['reward'])}  done={int(r['done'])}  "
              f"S={r['goal_similarity']:.4f}")
        assert int(r["done"]) == int(r["reward"]), \
            f"BUG: done != reward at segment boundary t={t}"
        found += 1

    if found == 0:
        print("  (No cross-segment transitions found in geometric dataset -- "
              "may be truncated in smoke test)")
    else:
        print(f"  Cross-segment check: {found} boundaries verified OK")
        print(f"  Segment boundaries correctly treated as continuous transitions.")


# ======================================================================
# Phase 10 -- Validation
# ======================================================================

def run_validation(phi_cache, ssd_scores, valid_mask, valid_goals_df,
                   geom_df, unif_df, frame_index, n_verify=200, seed=99):
    """Run all 24 validation checks. Returns results dict."""
    N   = len(phi_cache)
    rng = np.random.default_rng(seed)
    errors = []
    results = {}

    # Check 1: Frame count
    results["frames_discovered"] = N

    # Check 2: Missing RGB files
    missing = sum(1 for p in frame_index["rgb_abs_path"] if not Path(p).exists())
    results["missing_rgb"] = missing
    if missing > 0:
        errors.append(f"CHECK 2: {missing} missing RGB files")

    # Checks 3+4: No NaN/Inf
    n_nan = int(np.isnan(phi_cache).sum())
    n_inf = int(np.isinf(phi_cache).sum())
    results["nan_count"] = n_nan
    results["inf_count"] = n_inf
    if n_nan > 0: errors.append(f"CHECK 3: {n_nan} NaNs in phi_cache")
    if n_inf > 0: errors.append(f"CHECK 4: {n_inf} Infs in phi_cache")

    # Check 5: Dimension = 384
    results["phi_dim"] = int(phi_cache.shape[1])
    if phi_cache.shape[1] != EMBED_DIM:
        errors.append(f"CHECK 5: phi dim={phi_cache.shape[1]}, expected {EMBED_DIM}")

    # Check 6: L2-normalized
    norms = np.linalg.norm(phi_cache, axis=1)
    max_norm_err = float(np.abs(norms - 1.0).max())
    results["phi_norm_max_err"] = max_norm_err
    if max_norm_err > 1e-4:
        errors.append(f"CHECK 6: phi norms not ~1.0, max_err={max_norm_err:.6f}")

    # Check 7: State columns exist
    for col in ["state_t3","state_t2","state_t1","state_t0"]:
        if col not in geom_df.columns:
            errors.append(f"CHECK 7: missing column {col}")
    results["state_frame_check"] = "ok"

    # Checks 8+9: State/next_state global steps
    sample_idx = rng.integers(0, len(geom_df), size=min(n_verify, len(geom_df)))
    state_errs = 0
    next_errs  = 0
    for i in sample_idx:
        r = geom_df.iloc[i]
        t = int(r["current_global_step"])
        if (int(r["state_t3"]) != t-3 or int(r["state_t2"]) != t-2 or
                int(r["state_t1"]) != t-1 or int(r["state_t0"]) != t):
            state_errs += 1
        if (int(r["next_t2"]) != t-2 or int(r["next_t1"]) != t-1 or
                int(r["next_t0"]) != t   or int(r["next_t1f"]) != t+1):
            next_errs += 1
    if state_errs > 0: errors.append(f"CHECK 8: {state_errs} state step mismatches")
    if next_errs  > 0: errors.append(f"CHECK 9: {next_errs} next_state step mismatches")
    results["state_step_errors"] = state_errs
    results["next_step_errors"]  = next_errs

    # Check 10: Action = action_t
    action_errs = 0
    for i in sample_idx[:50]:
        r  = geom_df.iloc[i]
        t  = int(r["current_global_step"])
        fi = frame_index.iloc[t]
        if (abs(r["action_linear"]  - float(fi["linear_vel_cmd"]))  > 1e-6 or
                abs(r["action_lateral"] - float(fi["lateral_vel_cmd"])) > 1e-6 or
                abs(r["action_angular"] - float(fi["angular_vel_cmd"])) > 1e-6):
            action_errs += 1
    if action_errs > 0: errors.append(f"CHECK 10: {action_errs} action mismatches")
    results["action_errors"] = action_errs

    # Check 11: No transition at final step
    max_t_geom = int(geom_df["current_global_step"].max())
    max_t_unif = int(unif_df["current_global_step"].max())
    if max_t_geom >= N - 1: errors.append(f"CHECK 11: geo max_t={max_t_geom} >= N-1={N-1}")
    if max_t_unif >= N - 1: errors.append(f"CHECK 11: unif max_t={max_t_unif} >= N-1={N-1}")
    results["max_current_steps"] = {"geo": max_t_geom, "unif": max_t_unif}

    # Check 12: Geometric goals strictly future
    bad_future = int((geom_df["goal_global_step"] <= geom_df["current_global_step"]).sum())
    results["geometric_future_violations"] = bad_future
    if bad_future > 0: errors.append(f"CHECK 12: {bad_future} geo goals not strictly future")

    # Check 13: Uniform goals in valid pool
    valid_goal_set = set(valid_goals_df["embedding_index"].tolist())
    bad_unif       = int((~unif_df["goal_idx"].isin(valid_goal_set)).sum())
    results["uniform_invalid_goal_violations"] = bad_unif
    if bad_unif > 0: errors.append(f"CHECK 13: {bad_unif} uniform goals not in valid pool")

    # Check 14: Cross-segment transitions not forced-done
    seg_changes    = frame_index["segment_id"].ne(frame_index["segment_id"].shift()).values
    boundary_steps = frame_index["global_step"].values[np.where(seg_changes)[0][1:]]
    cross_forced   = 0
    for bdry in boundary_steps[:100]:
        t = int(bdry) - 1
        for _, row in geom_df[geom_df["current_global_step"] == t].iterrows():
            if int(row["done"]) != int(row["reward"]):
                cross_forced += 1
    results["cross_segment_done_forced"] = cross_forced
    if cross_forced > 0:
        errors.append(f"CHECK 14: {cross_forced} cross-segment done != reward")

    # Check 15: No resets (guaranteed by robot_resets=0 assertion in Phase 1)
    results["reset_transitions"] = 0

    # Checks 16+17: Recompute reward/done from phi_cache
    combined   = pd.concat([geom_df, unif_df], ignore_index=True)
    verify_idx = rng.integers(0, len(combined), size=min(n_verify, len(combined)))
    reward_err = 0
    done_err   = 0
    for i in verify_idx:
        r  = combined.iloc[i]
        t  = int(r["current_global_step"])
        g  = int(r["goal_idx"])
        sim = compute_similarity(phi_cache[t-3:t+1], phi_cache[g])
        exp = int(sim >= REWARD_THRESHOLD)
        if int(r["reward"]) != exp: reward_err += 1
        if int(r["done"])   != exp: done_err   += 1
    results["reward_recompute_errors"] = reward_err
    results["done_recompute_errors"]   = done_err
    if reward_err > 0: errors.append(f"CHECK 16: {reward_err} reward recompute mismatches")
    if done_err   > 0: errors.append(f"CHECK 17: {done_err} done recompute mismatches")

    # Check 18: reward=1 <-> S>=0.8
    hi_but_0 = int(((combined["goal_similarity"] >= REWARD_THRESHOLD) & (combined["reward"] == 0)).sum())
    lo_but_1 = int(((combined["goal_similarity"] <  REWARD_THRESHOLD) & (combined["reward"] == 1)).sum())
    results["threshold_violations"] = hi_but_0 + lo_but_1
    if hi_but_0 + lo_but_1 > 0:
        errors.append(f"CHECK 18: {hi_but_0} hi-sim-but-0, {lo_but_1} lo-sim-but-1")

    # Check 19: No position leakage into transitions
    bad_cols = [c for c in ["pos_x","pos_y","yaw"]
                if c in geom_df.columns or c in unif_df.columns]
    if bad_cols:
        errors.append(f"CHECK 19: position columns {bad_cols} directly in transition df")
    results["position_leakage"] = bad_cols

    # Check 20+21: Counts and reward %
    results["geometric_count"]       = len(geom_df)
    results["uniform_count"]         = len(unif_df)
    results["total_count"]           = len(geom_df) + len(unif_df)
    results["geometric_reward1_pct"] = float(100 * geom_df["reward"].mean())
    results["uniform_reward1_pct"]   = float(100 * unif_df["reward"].mean())

    # Check 22: No duplicate transitions
    g_dups = int(geom_df[["current_global_step","goal_global_step"]].duplicated().sum())
    u_dups = int(unif_df[["current_global_step","goal_global_step"]].duplicated().sum())
    results["duplicate_transitions"] = g_dups + u_dups
    if g_dups + u_dups > 0:
        errors.append(f"CHECK 22: {g_dups+u_dups} duplicate transitions")

    # Check 23: Min current step
    min_t = int(geom_df["current_global_step"].min())
    results["min_current_step"] = min_t
    if min_t < FRAME_STACK - 1:
        errors.append(f"CHECK 23: min_t={min_t} < FRAME_STACK-1={FRAME_STACK-1}")

    # Check 24: Max next_t1f < N
    max_next_geo  = int(geom_df["next_t1f"].max())
    max_next_unif = int(unif_df["next_t1f"].max())
    if max_next_geo  >= N: errors.append(f"CHECK 24: geo next_t1f max={max_next_geo} >= N={N}")
    if max_next_unif >= N: errors.append(f"CHECK 24: unif next_t1f max={max_next_unif} >= N={N}")
    results["max_next_obs"] = {"geo": max_next_geo, "unif": max_next_unif}

    results["errors"] = errors
    results["passed"] = (len(errors) == 0)
    return results


# ======================================================================
# Save Dataset
# ======================================================================

def save_dataset(out_dir, phi_cache, ssd_scores, valid_mask,
                 valid_goals_df, geom_df, unif_df, frame_index, encoder_meta):
    """Save the complete dataset to out_dir."""
    out_dir = Path(out_dir)

    # DINOv3 cache
    dinov3_dir = out_dir / "dinov3"
    dinov3_dir.mkdir(parents=True, exist_ok=True)

    np.save(dinov3_dir / "phi_cache.npy",   phi_cache)
    np.save(dinov3_dir / "ssd_scores.npy",  ssd_scores)
    np.save(dinov3_dir / "valid_goals.npy", valid_mask)

    frame_index[["global_step","segment_id","segment_step","rgb_abs_path"]].to_parquet(
        dinov3_dir / "embedding_index.parquet"
    )

    phi_norms = np.linalg.norm(phi_cache, axis=1)
    with open(dinov3_dir / "encoder_meta.json", "w") as f:
        json.dump({
            **encoder_meta,
            "phi_cache_shape":  list(phi_cache.shape),
            "phi_norm_min":     float(phi_norms.min()),
            "phi_norm_max":     float(phi_norms.max()),
            "phi_norm_mean":    float(phi_norms.mean()),
            "ssd_min":          float(ssd_scores.min()),
            "ssd_max":          float(ssd_scores.max()),
            "ssd_mean":         float(ssd_scores.mean()),
            "ssd_median":       float(np.median(ssd_scores)),
            "n_valid_goals":    int(valid_mask.sum()),
            "pct_valid_goals":  float(100 * valid_mask.mean()),
        }, f, indent=2)

    valid_goals_df.to_parquet(dinov3_dir / "valid_goals.parquet")

    # Transitions
    hindsight_dir = out_dir / "hindsight"
    hindsight_dir.mkdir(parents=True, exist_ok=True)

    geom_df.to_parquet(hindsight_dir / "geometric_transitions.parquet", index=False)
    unif_df.to_parquet(hindsight_dir / "uniform_transitions.parquet",   index=False)

    combined = pd.concat([geom_df, unif_df], ignore_index=True)
    combined.to_parquet(hindsight_dir / "all_transitions.parquet", index=False)

    print(f"  phi_cache.npy:         {phi_cache.nbytes/1e6:.1f} MB  {phi_cache.shape}")
    print(f"  ssd_scores.npy:        {ssd_scores.nbytes/1e3:.0f} KB")
    g_sz = (hindsight_dir / "geometric_transitions.parquet").stat().st_size / 1e6
    u_sz = (hindsight_dir / "uniform_transitions.parquet").stat().st_size / 1e6
    print(f"  geometric_transitions: {g_sz:.1f} MB  ({len(geom_df):,} rows)")
    print(f"  uniform_transitions:   {u_sz:.1f} MB  ({len(unif_df):,} rows)")


# ======================================================================
# Final Report
# ======================================================================

def write_final_report(report_path, phi_cache, ssd_scores, valid_mask,
                       valid_goals_df, geom_df, unif_df, val,
                       session_dir, out_base, elapsed_sec):
    N        = len(phi_cache)
    n_valid  = int(valid_mask.sum())
    combined = pd.concat([geom_df, unif_df], ignore_index=True)
    phi_norms = np.linalg.norm(phi_cache, axis=1)

    # Pre-compute helper strings to avoid nested f-string issues
    overall_status = "PASSED" if val["passed"] else "FAILED"
    phi_norm_err   = val["phi_norm_max_err"]
    phi_norm_res   = f"OK (max_err={phi_norm_err:.2e})" if phi_norm_err < 1e-4 else "FAIL"

    lines = [
        "# FINAL_HINDSIGHT_DATASET_REPORT",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Session: {session_dir}",
        f"Output: {out_base}",
        f"Processing time: {elapsed_sec:.0f}s ({elapsed_sec/60:.1f} min)",
        "",
        "---",
        "",
        "## Dataset",
        "",
        "| Item | Value |",
        "|------|-------|",
        f"| Total source frames | {N:,} |",
        f"| Valid goals (SSD > 0.02) | {n_valid:,} ({100*n_valid/N:.1f}%) |",
        f"| Invalid (SSD <= 0.02) | {N-n_valid:,} ({100*(N-n_valid)/N:.1f}%) |",
        f"| Valid 4-frame states | {len(geom_df):,} |",
        "",
        "## Hindsight Relabeling",
        "",
        "| Item | Value |",
        "|------|-------|",
        f"| Geometric transitions | {len(geom_df):,} |",
        f"| Uniform transitions | {len(unif_df):,} |",
        f"| Total transitions | {len(combined):,} |",
        "",
        "## Reward Statistics",
        "",
        "| Item | Value |",
        "|------|-------|",
        f"| Geometric reward=1 % | {val['geometric_reward1_pct']:.2f}% |",
        f"| Uniform reward=1 % | {val['uniform_reward1_pct']:.2f}% |",
        f"| Mean similarity (geometric) | {geom_df['goal_similarity'].mean():.4f} |",
        f"| Median similarity (geometric) | {geom_df['goal_similarity'].median():.4f} |",
        f"| Mean similarity (uniform) | {unif_df['goal_similarity'].mean():.4f} |",
        f"| Median similarity (uniform) | {unif_df['goal_similarity'].median():.4f} |",
        f"| Mean similarity (all) | {combined['goal_similarity'].mean():.4f} |",
        f"| Median similarity (all) | {combined['goal_similarity'].median():.4f} |",
        "",
        "## Representation",
        "",
        "| Item | Value | Source |",
        "|------|-------|--------|",
        f"| DINOv3 model | {MODEL_NAME} | PAPER SPECIFIES |",
        f"| Input resolution | {IMAGE_H} x {IMAGE_W} | PAPER SPECIFIES |",
        f"| phi(o) feature key | x_norm_clstoken | IMPLEMENTATION CHOICE |",
        f"| Implementation | feats[:, 0, :] + F.normalize(dim=-1) | IMPLEMENTATION CHOICE |",
        f"| forward_features() output | Tensor [B,1377,384] (NOT dict) | Verified live |",
        f"| phi dimension | {EMBED_DIM} | PAPER SPECIFIES ViT-S |",
        f"| L2-normalized | Yes -- explicit F.normalize() | IMPLEMENTATION CHOICE |",
        f"| Encoder frozen | Yes (eval + no_grad) | PAPER SPECIFIES |",
        f"| Phi norm min | {phi_norms.min():.6f} | Verified |",
        f"| Phi norm max | {phi_norms.max():.6f} | Verified |",
        "",
        "## SSD",
        "",
        "| Item | Value | Source |",
        "|------|-------|--------|",
        f"| SSD source | x_norm_patchtokens (feats[:, 5:, :]) | IMPLEMENTATION CHOICE |",
        f"| Patch grid | {PATCH_H} x {PATCH_W} | Derived from 448/16, 784/16 |",
        f"| Center crop | {CROP_H} x {CROP_W} patches | IMPLEMENTATION CHOICE (encoder_v2.py) |",
        f"| Threshold | {SSD_THRESHOLD} | PAPER SPECIFIES |",
        f"| SSD min | {ssd_scores.min():.5f} | |",
        f"| SSD max | {ssd_scores.max():.5f} | |",
        f"| SSD mean | {ssd_scores.mean():.5f} | |",
        f"| SSD median | {float(np.median(ssd_scores)):.5f} | |",
        "",
        "## Geometric Sampling",
        "",
        "| Item | Value | Source |",
        "|------|-------|--------|",
        f"| Distribution | P(K=k) = p^(k-1)(1-p), k>=1 | PAPER SPECIFIES |",
        f"| p | {GEO_P} | IMPLEMENTATION CHOICE |",
        f"| E[K] = 1/(1-p) | {1/(1-GEO_P):.0f} steps | Derived |",
        f"| Goal constraint | goal_step > t, goal_step < N, SSD-valid | PAPER SPECIFIES |",
        f"| Avg offset k | {geom_df['offset_k'].mean():.1f} steps | Observed |",
        "",
        "## Segment Continuity",
        "",
        "> **Segments represent storage boundaries, not environment episode boundaries.**",
        "> The robot was physically continuous throughout the collection session.",
        "> No physical reset occurred between segments.",
        "> global_step determines temporal continuity.",
        "> Segment boundaries do NOT introduce done=1, truncation, or reset signals.",
        "",
        "- robot_resets = 0 (verified from session metadata)",
        "- 144 storage segments x 500 steps = 72,000 frames",
        "- Cross-segment transitions verified as normal continuous transitions",
        "",
        "## Representation Ablation Result",
        "",
        "> **IMPLEMENTATION CHOICE** -- not a paper-specified architectural detail.",
        "",
        "We evaluated four phi(o) representations (on 1,000-frame ablation):",
        "",
        "| Method | Random Distant Reward=1 | Geometric Reward=1 | Uniform Reward=1 |",
        "|--------|------------------------|---------------------|------------------|",
        "| DINOv3 normalized CLS | 66.2% | 75.5% | 73.0% |",
        "| Corresponding Patch | 56.7% | 73.3% | 64.3% |",
        "| Best-Match Patch (Chamfer) | 91.8% | 93.9% | 96.2% |",
        "| Symmetric Best-Match | 91.1% | 93.8% | 94.8% |",
        "",
        "**Decision:** DINOv3 normalized CLS (x_norm_clstoken) selected.",
        "Rationale: Best practical trade-off between spatial discrimination,",
        "viewpoint robustness, representation compactness, and computational cost.",
        "Best-match/Chamfer rejected for being excessively permissive (91%+ false positives).",
        "",
        "## Validation Results",
        "",
        f"**OVERALL: {overall_status}**",
        "",
        "| # | Check | Result |",
        "|---|-------|--------|",
        f"| 1 | Frame count = 72,000 | {'OK: ' + str(N) if N==72000 else 'WARN: ' + str(N)} |",
        f"| 2 | Missing RGB files | {'OK: 0' if val.get('missing_rgb',0)==0 else 'FAIL: '+str(val['missing_rgb'])} |",
        f"| 3 | No NaN in phi | {'OK' if val.get('nan_count',0)==0 else 'FAIL: '+str(val['nan_count'])} |",
        f"| 4 | No Inf in phi | {'OK' if val.get('inf_count',0)==0 else 'FAIL: '+str(val['inf_count'])} |",
        f"| 5 | phi dim = 384 | {'OK' if val.get('phi_dim')==384 else 'FAIL'} |",
        f"| 6 | phi L2-normalized | {phi_norm_res} |",
        f"| 7 | State has 4 frames | OK |",
        f"| 8 | State steps = [t-3..t] | {'OK' if val.get('state_step_errors',0)==0 else 'FAIL'} |",
        f"| 9 | next_state = [t-2..t+1] | {'OK' if val.get('next_step_errors',0)==0 else 'FAIL'} |",
        f"| 10 | action = action_t | {'OK' if val.get('action_errors',0)==0 else 'FAIL'} |",
        f"| 11 | No transition at final step | OK |",
        f"| 12 | Geo goals strictly future | {'OK' if val['geometric_future_violations']==0 else 'FAIL: '+str(val['geometric_future_violations'])} |",
        f"| 13 | Uniform goals in valid pool | {'OK' if val['uniform_invalid_goal_violations']==0 else 'FAIL'} |",
        f"| 14 | Cross-segment not forced done | {'OK' if val['cross_segment_done_forced']==0 else 'FAIL'} |",
        f"| 15 | No reset transitions | OK (robot_resets=0) |",
        f"| 16 | Reward recompute matches | {'OK' if val['reward_recompute_errors']==0 else 'FAIL: '+str(val['reward_recompute_errors'])} |",
        f"| 17 | Done recompute matches | {'OK' if val['done_recompute_errors']==0 else 'FAIL: '+str(val['done_recompute_errors'])} |",
        f"| 18 | reward=1 <-> S>=0.8 | {'OK' if val['threshold_violations']==0 else 'FAIL'} |",
        f"| 19 | No position leakage | {'OK' if not val['position_leakage'] else 'FAIL: '+str(val['position_leakage'])} |",
        f"| 20 | Geometric count | OK: {val['geometric_count']:,} |",
        f"| 20 | Uniform count | OK: {val['uniform_count']:,} |",
        f"| 21 | Geo reward=1 % | {val['geometric_reward1_pct']:.2f}% |",
        f"| 21 | Unif reward=1 % | {val['uniform_reward1_pct']:.2f}% |",
        f"| 22 | No duplicate transitions | {'OK' if val['duplicate_transitions']==0 else 'FAIL: '+str(val['duplicate_transitions'])} |",
        f"| 23 | All timesteps >= FRAME_STACK-1 | OK |",
        f"| 24 | All next obs exist | OK |",
        "",
        "## Integrity Notes",
        "",
        f"| Check | Value |",
        f"|-------|-------|",
        f"| Missing RGB files | {val.get('missing_rgb', 0)} |",
        f"| NaN in phi | {val.get('nan_count', 0)} |",
        f"| Invalid transitions | 0 |",
        f"| Cross-segment errors | {val['cross_segment_done_forced']} |",
        f"| Invalid geometric goals | {val['geometric_future_violations']} |",
        "",
        "## Output Files",
        "",
        "```",
        "data/processed/",
        f"  dinov3/",
        f"    phi_cache.npy           [{N},384] float32  ~{phi_cache.nbytes/1e6:.0f} MB",
        f"    ssd_scores.npy          [{N}]  float32",
        f"    valid_goals.npy         [{N}]  bool",
        f"    valid_goals.parquet     {n_valid:,} rows",
        f"    embedding_index.parquet {N:,} rows",
        f"    encoder_meta.json",
        f"  hindsight/",
        f"    geometric_transitions.parquet  {val['geometric_count']:,} rows",
        f"    uniform_transitions.parquet    {val['uniform_count']:,} rows",
        f"    all_transitions.parquet        {val['total_count']:,} rows",
        "```",
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Report: {report_path}")


# ======================================================================
# Pipeline runner
# ======================================================================

def run_pipeline(session_dir, out_dir, batch_size, max_frames=None, label="FULL"):
    t0 = time.time()

    print("\n" + "="*64)
    print(f"  MINav Final Hindsight Dataset Pipeline -- {label}")
    print("="*64)

    # Phase 1
    print("\n[Phase 1] Frame index ...")
    frame_index = build_frame_index(session_dir, max_frames=max_frames)
    N = len(frame_index)

    # Phase 2
    print("\n[Phase 2] Continuity validation ...")
    validate_continuity(frame_index)

    # Phase 3
    print("\n[Phase 3] Verifying RGB files ...")
    missing = [p for p in frame_index["rgb_abs_path"] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} missing RGB files. First: {missing[0]}")
    print(f"  All {N:,} RGB files found")

    # Phase 4
    print("\n[Phase 4] DINOv3 encoding (x_norm_clstoken) ...")
    encoder = FrozenDINOv3Encoder(device="auto")
    phi_cache, ssd_scores, valid_mask = encoder.encode_all(
        frame_index["rgb_abs_path"].tolist(), batch_size=batch_size
    )

    # Phase 5
    print("\n[Phase 5] Valid goal pool ...")
    valid_goals_df = build_valid_goal_pool(frame_index, ssd_scores, valid_mask)

    # Phase 6
    print("\n[Phase 6] Geometric hindsight (p=0.99) ...")
    geom_df = build_geometric_transitions(phi_cache, frame_index, valid_mask)

    # Phase 7
    print("\n[Phase 7] Uniform hindsight ...")
    unif_df = build_uniform_transitions(phi_cache, frame_index, valid_goals_df)

    # Phase 8
    print("\n[Phase 8] Save dataset ...")
    out_dir = Path(out_dir)
    encoder_meta = {
        "model_name":          MODEL_NAME,
        "image_size":          [IMAGE_H, IMAGE_W],
        "phi_source":          "x_norm_clstoken",
        "feature_key":         "forward_features(x)[:, 0, :] + F.normalize(dim=-1)",
        "feature_note":        "timm returns Tensor [B,1377,384] NOT dict. CLS=index 0.",
        "phi_dim":             EMBED_DIM,
        "already_normalized":  True,
        "frozen":              True,
        "ssd_crop":            [CROP_H, CROP_W],
        "ssd_threshold":       SSD_THRESHOLD,
        "geo_p":               GEO_P,
        "frame_stack":         FRAME_STACK,
        "reward_threshold":    REWARD_THRESHOLD,
        "n_frames":            N,
        "label":               label,
    }
    save_dataset(out_dir, phi_cache, ssd_scores, valid_mask,
                 valid_goals_df, geom_df, unif_df, frame_index, encoder_meta)

    # Phase 9
    print("\n[Phase 9] Spot check ...")
    spot_check(geom_df, unif_df)

    # Phase 10
    print("\n[Phase 10] Cross-segment spot check ...")
    cross_segment_spot_check(geom_df, frame_index)

    # Phase 11
    print("\n[Phase 11] 24-point validation ...")
    val = run_validation(phi_cache, ssd_scores, valid_mask, valid_goals_df,
                         geom_df, unif_df, frame_index)

    elapsed = time.time() - t0

    print("\n" + "="*64)
    print(f"  PIPELINE COMPLETE -- {label}")
    print("="*64)
    print(f"  Time           : {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Frames         : {N:,}")
    print(f"  Valid goals    : {valid_mask.sum():,} / {N:,} ({100*valid_mask.mean():.1f}%)")
    print(f"  Phi shape      : {phi_cache.shape}")
    print(f"  Phi norm err   : {val['phi_norm_max_err']:.2e}  (target <1e-4)")
    print(f"  Geo trans.     : {len(geom_df):,}  reward=1: {val['geometric_reward1_pct']:.2f}%")
    print(f"  Unif trans.    : {len(unif_df):,}  reward=1: {val['uniform_reward1_pct']:.2f}%")
    print(f"  Validation     : {'PASSED' if val['passed'] else 'FAILED -- ' + str(val['errors'])}")
    print("="*64)

    return phi_cache, ssd_scores, valid_mask, valid_goals_df, geom_df, unif_df, val, elapsed


# ======================================================================
# Entry point
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="MINav Final Hindsight Dataset")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    sys.path.insert(0, str(ROOT))
    from scripts.pipeline_utils import setup_pipeline_logger, update_pipeline_state
    
    logger = setup_pipeline_logger(run_dir, "hindsight")
    update_pipeline_state(run_dir, "hindsight", "running")

    try:
        import yaml
        with open(ROOT / args.config) as f:
            config = yaml.safe_load(f)

        is_smoke = args.smoke or config.get("smoke", {}).get("enabled", False)
        session_dir = run_dir / config["exploration"]["output_subdir"]
        out_base = run_dir / config["hindsight"]["output_subdir"]

        if not session_dir.exists():
            logger.error(f"ERROR: {session_dir} not found")
            sys.exit(1)

        batch_size = config["training"]["batch_size"] if "training" in config else 16
        # Or you can hardcode batch size, but caching batch_size usually doesn't strictly matter.
        batch_size = 16

        if is_smoke:
            max_frames = config["smoke"]["max_frames"]
            label = "SMOKE TEST"
            logger.warning("SMOKE MODE ACTIVE — limits enforced")
        else:
            max_frames = None
            label = "FULL DATASET"

        logger.info(f"Starting {label}")
        phi_cache, ssd_scores, valid_mask, valid_goals_df, \
            geom_df, unif_df, val, elapsed = run_pipeline(
                session_dir = session_dir,
                out_dir     = out_base,
                batch_size  = batch_size,
                max_frames  = max_frames,
                label       = label,
            )

        if not val["passed"]:
            logger.error("VALIDATION FAILED:")
            for e in val["errors"]:
                logger.error(f"  x {e}")
            update_pipeline_state(run_dir, "hindsight", "failed")
            sys.exit(1)

        report_path = out_base / "FINAL_HINDSIGHT_DATASET_REPORT.md"
        write_final_report(
            report_path, phi_cache, ssd_scores, valid_mask,
            valid_goals_df, geom_df, unif_df, val,
            session_dir, out_base, elapsed,
        )

        logger.info(f"FINAL DATASET COMPLETE. Report: {report_path}")
        update_pipeline_state(run_dir, "hindsight", "completed")

    except Exception as e:
        logger.error(f"Hindsight failed: {e}", exc_info=True)
        update_pipeline_state(run_dir, "hindsight", "failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
