"""
scripts/encode_dinov3.py
============================================================
Phase 17 small test + Phase 4 full encode script.

Runs the complete pipeline end-to-end:
  1. Build frame index from session
  2. Validate continuity
  3. Encode with DINOv3 (frozen)
  4. Build valid goal set
  5. Generate geometric transitions
  6. Generate uniform transitions

Usage:
  # Small test (1000 frames):
  python scripts/encode_dinov3.py --session dataset/20260811_234812 --test

  # Full dataset:
  python scripts/encode_dinov3.py --session dataset/20260811_234812

The script stops after the small test and reports results.
Run with --full to process the entire dataset.
"""

import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

# Add src/ and root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.index_segments  import build_frame_index
from src.data.dinov3_encoder  import FrozenDINOv3, save_embeddings
from src.data.goal_set        import build_goal_set
from src.data.hindsight_geometric import build_geometric_dataset
from src.data.hindsight_uniform   import build_uniform_dataset


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_pipeline(
    session_dir:   Path,
    out_base:      Path,
    cfg:           dict,
    max_frames:    int = None,
    batch_size:    int = 16,
):
    t0 = time.time()
    is_test = max_frames is not None
    mode    = f"SMALL TEST ({max_frames} frames)" if is_test else "FULL DATASET"

    print("=" * 64)
    print(f"  MINav Hindsight Pipeline  —  {mode}")
    print("=" * 64)

    # ── Phase 1: Frame index ──────────────────────────────────────────────────
    print("\n[Phase 1] Building frame index...")
    idx_dir = out_base / "frame_index"
    df_all  = build_frame_index(session_dir, idx_dir)

    if max_frames:
        df_all = df_all.iloc[:max_frames].reset_index(drop=True)
        print(f"  [TEST] Truncated to {len(df_all)} frames")

    # ── Phase 2: Continuity validation ───────────────────────────────────────
    print("\n[Phase 2] Validating continuity...")
    from scripts.validate_continuity import validate
    ok = validate(df_all)
    if not ok:
        print("FATAL: Continuity validation failed. Aborting.")
        sys.exit(1)

    # ── Phase 3+4: DINOv3 encoding ────────────────────────────────────────────
    print("\n[Phase 3+4] Encoding with frozen DINOv3...")
    cache_cfg = cfg.get("cache", {})
    return_grids = cache_cfg.get("store_full_grid", False)
    return_pooled = cache_cfg.get("store_pooled", True)
    
    enc_cfg = cfg.get("dinov3", {})
    encoder = FrozenDINOv3(
        model_name    = enc_cfg.get("model_name", "vit_small_patch16_dinov3"),
        image_h       = enc_cfg.get("image_height", 448),
        image_w       = enc_cfg.get("image_width", 784),
        ssd_threshold = cfg.get("ssd", {}).get("threshold", 0.02),
        device        = enc_cfg.get("device", "auto"),
    )

    image_paths = df_all["rgb_abs_path"].tolist()
    phi_vectors, ssd_scores, valid_goals, grids = encoder.encode_paths(
        image_paths, batch_size=batch_size, return_grids=return_grids, return_pooled=return_pooled
    )

    enc_dir = out_base / "dinov3"
    enc_meta = {
        "model_name":      enc_cfg.get("model_name"),
        "image_size":      [enc_cfg.get("image_height", 448), enc_cfg.get("image_width", 784)],
        "patch_grid":      [28, 49],
        "embed_dim":       384,
        "n_special_tokens": 5,
        "ssd_crop":        [14, 25],
        "ssd_threshold":   0.02,
        "normalize_imagenet": False,
        "phi_description": "L2-norm mean-pooled patch tokens [384]",
        "paper_specifies_normalization": False,   # explicit audit trail
        "paper_specifies_pooling":       False,
        "paper_specifies_p_value":       False,
        "is_test_run":    is_test,
        "n_frames":       len(df_all),
    }
    save_embeddings(enc_dir, phi_vectors, ssd_scores, valid_goals, df_all, enc_meta, grids=grids)

    # ── Phase 5: Valid goal set ───────────────────────────────────────────────
    print("\n[Phase 5] Building valid goal set...")
    goals_dir = out_base / "valid_goals"
    valid_goals_df = build_goal_set(df_all, ssd_scores, valid_goals, goals_dir)

    # ── Phase 8: Geometric hindsight ─────────────────────────────────────────
    print("\n[Phase 8] Geometric future sampling...")
    geom_cfg  = cfg.get("geometric", {})
    p_value   = geom_cfg.get("p", 0.99)
    num_goals = cfg.get("test", {}).get("num_goals_per_transition", 1) if is_test \
                else cfg.get("hindsight", {}).get("num_goals_per_transition", 1)

    geom_dir  = out_base / "hindsight_geometric"
    geom_df   = build_geometric_dataset(
        phi_vectors  = phi_vectors,
        frame_index  = df_all,
        p            = p_value,
        num_goals    = num_goals,
        out_dir      = geom_dir,
        max_frames   = None,    # already truncated above
    )

    # ── Phase 9: Uniform hindsight ────────────────────────────────────────────
    print("\n[Phase 9] Global uniform goal sampling...")
    unif_dir = out_base / "hindsight_uniform"
    unif_df  = build_uniform_dataset(
        phi_vectors    = phi_vectors,
        frame_index    = df_all,
        valid_goals_df = valid_goals_df,
        ssd_scores     = ssd_scores,
        num_goals      = num_goals,
        out_dir        = unif_dir,
        max_frames     = None,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 64)
    print(f"  PIPELINE COMPLETE  ({mode})")
    print("=" * 64)
    print(f"  Frames processed     : {len(df_all):,}")
    print(f"  DINOv3 output shape  : [{len(df_all)}, 384]  (phi vectors)")
    print(f"  Patch grid           : 28 × 49, D=384")
    print(f"  SSD threshold        : 0.02  [PAPER SPECIFIES]")
    print(f"  Valid goals          : {valid_goals.sum():,} / {len(df_all):,}  ({100*valid_goals.mean():.1f}%)")
    print(f"  Geometric p value    : {p_value}  [NOT paper-specified]")
    print(f"  Geometric transitions: {len(geom_df):,}")
    print(f"  Uniform transitions  : {len(unif_df):,}")
    print(f"  Reward=1 (geom)      : {geom_df['reward'].sum():,}  ({100*geom_df['reward'].mean():.1f}%)")
    print(f"  Reward=1 (unif)      : {unif_df['reward'].sum():,}  ({100*unif_df['reward'].mean():.1f}%)")
    print(f"  Total time           : {elapsed:.1f}s")
    print("=" * 64)

    return phi_vectors, ssd_scores, valid_goals, valid_goals_df, geom_df, unif_df


def main():
    parser = argparse.ArgumentParser(description="MINav DINOv3 Encoding + Hindsight Relabeling")
    parser.add_argument("--session",     required=True,   help="Session directory")
    parser.add_argument("--config",      default="configs/minav_hindsight.yaml")
    parser.add_argument("--out",         default="data/processed")
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--test",        action="store_true",
                        help="Run small test (1000 frames) before full dataset")
    parser.add_argument("--full",        action="store_true",
                        help="Process the full dataset (only after test passes)")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    session_dir = Path(args.session)
    out_base    = Path(args.out)

    if args.test or not args.full:
        # Phase 17: small test first
        test_frames = cfg.get("test", {}).get("max_frames", 1000)
        print(f"\nRunning small end-to-end test with {test_frames} frames.")
        print("Run with --full to process the entire dataset after test passes.\n")

        test_out = out_base / "test_run"
        run_pipeline(
            session_dir  = session_dir,
            out_base     = test_out,
            cfg          = cfg,
            max_frames   = test_frames,
            batch_size   = args.batch_size,
        )
        print("\n[TEST PASSED] Small end-to-end test complete.")
        print("Review results in data/processed/test_run/")
        print("Then run with --full to process the entire dataset.")

    if args.full:
        print("\nRunning FULL dataset encoding...")
        run_pipeline(
            session_dir = session_dir,
            out_base    = out_base,
            cfg         = cfg,
            max_frames  = None,
            batch_size  = args.batch_size,
        )


if __name__ == "__main__":
    main()
