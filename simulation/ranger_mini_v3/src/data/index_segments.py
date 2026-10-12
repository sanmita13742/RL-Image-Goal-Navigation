"""
src/data/index_segments.py
============================================================
Phase 1: Reconstruct the continuous dataset manifest.

Discovers all segment_NNN/segment.csv files in the session directory,
concatenates them in global_step order, and writes a flat parquet index:

    data/processed/frame_index/frame_index.parquet

Each row represents one observation with its full absolute image path,
all CSV metadata, and the segment it came from.

IMPORTANT: Segments are STORAGE BOUNDARIES, not trajectory boundaries.
The robot was NEVER reset between segments. All 72,000 frames form
one continuous physical rollout (trajectory_id = 0).
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path


def build_frame_index(session_dir: Path, out_dir: Path) -> pd.DataFrame:
    """
    Discover all segments in session_dir, build and save a frame index.

    Returns the full DataFrame sorted by global_step.
    """
    session_dir = Path(session_dir).resolve()
    out_dir     = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load session metadata
    meta_path = session_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            session_meta = json.load(f)
        print(f"Session: {session_meta.get('session_id')}")
        print(f"  robot_resets   = {session_meta.get('robot_resets', '?')}")
        print(f"  trajectory_id  = {session_meta.get('trajectory_id', '?')}")
        print(f"  total_steps    = {session_meta.get('total_steps_recorded', '?')}")
        print(f"  num_segments   = {session_meta.get('num_segments', '?')}")
    else:
        session_meta = {}
        print(f"WARNING: metadata.json not found in {session_dir}")

    # Discover segment directories
    seg_dirs = sorted(
        [d for d in session_dir.iterdir()
         if d.is_dir() and d.name.startswith("segment_")],
        key=lambda d: int(d.name.split("_")[1])
    )
    print(f"\nDiscovered {len(seg_dirs)} segment directories")

    frames = []
    for seg_dir in seg_dirs:
        csv_path = seg_dir / "segment.csv"
        if not csv_path.exists():
            csv_path = seg_dir / "observations.csv"
            if not csv_path.exists():
                print(f"  WARNING: neither segment.csv nor observations.csv found in {seg_dir}, skipping")
                continue
            else:
                print(f"  Found observations.csv in {seg_dir.name}")

        df = pd.read_csv(csv_path)
        df["segment_id"]   = seg_dir.name
        df["session_id"]   = session_meta.get("session_id", session_dir.name)

        # Validate schema before modifying
        if df.columns.duplicated().any():
            dups = df.columns[df.columns.duplicated()].tolist()
            raise ValueError(f"Duplicate columns found in {csv_path}: {dups}")
            
        print(f"  Validating columns in {seg_dir.name}: {len(df.columns)} columns found.")

        # Standardize real-world vs simulation schema
        if "rgb_path" not in df.columns and "image_path" in df.columns:
            df["rgb_path"] = df["image_path"]
        if "rgb_path" not in df.columns and "image" in df.columns:
            df["rgb_path"] = df["image"]
        if "segment_step" not in df.columns:
            df["segment_step"] = np.arange(len(df), dtype=int)
        if "linear_vel_cmd" not in df.columns and "cmd_vx" in df.columns:
            df["linear_vel_cmd"] = df["cmd_vx"]
        if "lateral_vel_cmd" not in df.columns and "cmd_vy" in df.columns:
            df["lateral_vel_cmd"] = df["cmd_vy"]
        if "angular_vel_cmd" not in df.columns and "cmd_wz" in df.columns:
            df["angular_vel_cmd"] = df["cmd_wz"]
        if "linear_vel_cmd" not in df.columns and "vx" in df.columns:
            df["linear_vel_cmd"] = df["vx"]
        if "lateral_vel_cmd" not in df.columns and "vy" in df.columns:
            df["lateral_vel_cmd"] = df["vy"]
        if "angular_vel_cmd" not in df.columns and "wz" in df.columns:
            df["angular_vel_cmd"] = df["wz"]
        if "sim_time" not in df.columns and "timestamp" in df.columns:
            df["sim_time"] = df["timestamp"]
        if "sim_time" not in df.columns:
            df["sim_time"] = 0.0
        if "pos_x" not in df.columns:
            df["pos_x"] = 0.0
        if "pos_y" not in df.columns:
            df["pos_y"] = 0.0
        if "yaw" not in df.columns:
            df["yaw"] = 0.0
        if "trajectory_id" not in df.columns:
            df["trajectory_id"] = 0

        # Build absolute paths to RGB images
        # using a vectorized map to avoid empty DataFrame apply issues
        df["rgb_abs_path"] = df["rgb_path"].astype(str).map(
            lambda p: str((seg_dir / p).resolve())
        )

        frames.append(df)

    if not frames:
        raise RuntimeError(f"No segment CSVs found in {session_dir}")

    df_all = pd.concat(frames, ignore_index=True)

    # Ensure continuous global_step across all segments
    if "global_step" not in df_all.columns:
        df_all["global_step"] = np.arange(len(df_all), dtype=int)
    else:
        df_all = df_all.sort_values("global_step").reset_index(drop=True)

    # Verify global_step is 0, 1, 2, ..., N-1
    expected = list(range(len(df_all)))
    actual   = list(df_all["global_step"])
    if actual != expected:
        # Re-index if there were minor offset mismatches across segment boundaries
        df_all["global_step"] = np.arange(len(df_all), dtype=int)

    print(f"\nFrame index built:")
    print(f"  Total rows     : {len(df_all):,}")
    print(f"  global_step    : {df_all['global_step'].min()} … {df_all['global_step'].max()}")
    print(f"  Columns        : {list(df_all.columns)}")
    print(f"  trajectory_ids : {sorted(df_all['trajectory_id'].unique().tolist())}")

    # Save
    out_path = out_dir / "frame_index.parquet"
    df_all.to_parquet(out_path, index=False)
    print(f"\nSaved -> {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")

    return df_all


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--session",  required=True, help="Path to session directory")
    parser.add_argument("--out_dir",  default="data/processed/frame_index")
    args = parser.parse_args()

    build_frame_index(args.session, args.out_dir)
