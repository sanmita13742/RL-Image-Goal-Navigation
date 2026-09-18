"""
src/data/goal_set.py
============================================================
Phase 5: Build global valid goal set from SSD-filtered embeddings.

PAPER SPECIFIES:
  G = {phi(o) | o in D and sigma_spa(o) > delta_SSD}
  delta_SSD = 0.02

Each valid goal record contains:
  global_step     — temporal index
  segment_id      — for provenance
  image_path      — absolute path to RGB image
  embedding_index — index into phi_vectors.npy
  ssd_score       — SSD value

Output: data/processed/valid_goals/valid_goals.parquet
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


def build_goal_set(
    frame_index:  pd.DataFrame,
    ssd_scores:   np.ndarray,
    valid_mask:   np.ndarray,
    out_dir:      Path,
) -> pd.DataFrame:
    """
    Build and save the valid goal set.

    Args:
        frame_index : full frame index DataFrame (N rows, sorted by global_step)
        ssd_scores  : [N] float32 array of SSD scores
        valid_mask  : [N] bool array  (ssd_scores > threshold)
        out_dir     : where to save valid_goals.parquet

    Returns:
        DataFrame with valid goal records
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    N = len(frame_index)
    assert len(ssd_scores) == N
    assert len(valid_mask) == N

    # Build goal DataFrame
    goals = frame_index[["global_step", "segment_id", "rgb_abs_path"]].copy()
    goals = goals.rename(columns={"rgb_abs_path": "image_path"})
    goals["embedding_index"] = goals.index   # index into phi_vectors.npy
    goals["ssd_score"]       = ssd_scores

    # Filter to valid only
    valid_goals = goals[valid_mask].reset_index(drop=True)

    # Save
    out_path = out_dir / "valid_goals.parquet"
    valid_goals.to_parquet(out_path, index=False)

    # Also save all-frame SSD for analysis
    all_ssd = goals[["global_step", "segment_id", "ssd_score"]].copy()
    all_ssd["valid"] = valid_mask.tolist()
    all_ssd.to_parquet(out_dir / "all_ssd_scores.parquet", index=False)

    # Summary
    meta = {
        "total_observations":  N,
        "valid_goals":         int(valid_mask.sum()),
        "invalid_observations":N - int(valid_mask.sum()),
        "pct_valid":           float(100 * valid_mask.mean()),
        "ssd_min":             float(ssd_scores.min()),
        "ssd_max":             float(ssd_scores.max()),
        "ssd_mean":            float(ssd_scores.mean()),
        "ssd_median":          float(np.median(ssd_scores)),
        "ssd_std":             float(ssd_scores.std()),
        "ssd_threshold":       0.02,  # PAPER SPECIFIES
    }
    with open(out_dir / "goal_set_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nValid goal set:")
    print(f"  Total observations : {N:,}")
    print(f"  Valid goals        : {meta['valid_goals']:,}  ({meta['pct_valid']:.1f}%)")
    print(f"  Invalid            : {meta['invalid_observations']:,}")
    print(f"  SSD range          : [{meta['ssd_min']:.4f}, {meta['ssd_max']:.4f}]")
    print(f"  SSD mean ± std     : {meta['ssd_mean']:.4f} ± {meta['ssd_std']:.4f}")
    print(f"  Saved -> {out_path}")

    return valid_goals
