#!/usr/bin/env python3
"""
realworld/scripts/build_dataset.py — Build dataset from real-world exploration data.
=====================================================================================
Thin wrapper that calls the EXISTING shared modules from the MuJoCo pipeline:
  1. index_segments → frame_index.parquet
  2. FrozenDINOv3.encode_paths() → phi_vectors.npy, ssd_scores.npy
  3. build_goal_set() → valid_goals.parquet
  4. build_geometric_dataset() → geometric_transitions.parquet
  5. build_uniform_dataset() → uniform_transitions.parquet

The data format from DataRecorder is identical to the MuJoCo pipeline,
so all downstream code works unchanged.
"""

import sys
import argparse
import yaml
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Import shared MuJoCo-pipeline modules (they have no MuJoCo dependency)
MUJOCO_ROOT = ROOT / "simulation" / "ranger_mini_v3"
sys.path.insert(0, str(MUJOCO_ROOT))

from src.data.index_segments import build_frame_index
from src.data.dinov3_encoder import FrozenDINOv3, save_embeddings
from src.data.goal_set import build_goal_set
from src.data.hindsight_geometric import build_geometric_dataset
from src.data.hindsight_uniform import build_uniform_dataset


def main():
    parser = argparse.ArgumentParser(description="Build dataset from real-world exploration")
    parser.add_argument("--config", type=str, default="realworld/configs/realworld_pipeline.yaml")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Run directory containing exploration/ subdirectory")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--smoke", action="store_true", help="Use reduced frame count")
    args = parser.parse_args()

    config_path = ROOT / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    run_dir = Path(args.run_dir).resolve()
    explore_subdir = config["exploration"]["output_subdir"]
    session_dir = run_dir / explore_subdir
    out_dir = run_dir / config.get("hindsight", {}).get("output_subdir", "processed")

    if not session_dir.exists():
        print(f"ERROR: Exploration data not found at {session_dir}")
        sys.exit(1)

    is_smoke = args.smoke or config.get("smoke", {}).get("enabled", False)

    # ── Phase 1: Build frame index ──
    print("\n[1/5] Building frame index...")
    frame_index_dir = out_dir / "frame_index"
    frame_index = build_frame_index(session_dir, frame_index_dir)

    # ── Phase 2: Encode with DINOv3 ──
    print("\n[2/5] Encoding with DINOv3...")
    h_cfg = config.get("hindsight", {})
    device = args.device

    encoder = FrozenDINOv3(
        model_name=h_cfg.get("dino_model", "vit_small_patch16_dinov3"),
        image_h=h_cfg.get("input_resolution", [448, 784])[0],
        image_w=h_cfg.get("input_resolution", [448, 784])[1],
        ssd_threshold=h_cfg.get("ssd_threshold", 0.02),
        device=device,
    )

    image_paths = frame_index["rgb_abs_path"].tolist()
    max_frames = config.get("smoke", {}).get("max_frames", None) if is_smoke else None
    if max_frames:
        image_paths = image_paths[:max_frames]
        frame_index = frame_index.iloc[:max_frames].reset_index(drop=True)

    batch_size = 32
    phi_vectors, ssd_scores, valid_goals_mask, _ = encoder.encode_paths(
        image_paths, batch_size=batch_size, return_grids=False, return_pooled=True,
    )

    dinov3_dir = out_dir / "dinov3"
    cfg_meta = {
        "source": "realworld",
        "model_name": h_cfg.get("dino_model", "vit_small_patch16_dinov3"),
        "ssd_threshold": h_cfg.get("ssd_threshold", 0.02),
    }
    save_embeddings(dinov3_dir, phi_vectors, ssd_scores, valid_goals_mask, frame_index, cfg_meta)

    # ── Phase 3: Build goal set ──
    print("\n[3/5] Building valid goal set...")
    goal_dir = out_dir / "valid_goals"
    valid_goals_df = build_goal_set(frame_index, ssd_scores, valid_goals_mask, goal_dir)

    # ── Phase 4: Geometric hindsight ──
    print("\n[4/5] Building geometric hindsight dataset...")
    hindsight_dir = out_dir / "hindsight"
    num_goals = config.get("smoke", {}).get("num_goals_per_transition", 1) if is_smoke else 1
    build_geometric_dataset(
        phi_vectors=phi_vectors,
        frame_index=frame_index,
        p=h_cfg.get("geometric_p", 0.99),
        num_goals=num_goals,
        out_dir=hindsight_dir,
        max_frames=max_frames,
        seed=config["run"]["seed"],
    )

    # ── Phase 5: Uniform hindsight ──
    print("\n[5/5] Building uniform hindsight dataset...")
    build_uniform_dataset(
        phi_vectors=phi_vectors,
        frame_index=frame_index,
        valid_goals_df=valid_goals_df,
        ssd_scores=ssd_scores,
        num_goals=num_goals,
        out_dir=hindsight_dir,
        max_frames=max_frames,
        seed=config["run"]["seed"],
    )

    print("\n✓ Dataset build complete!")
    print(f"  Output: {out_dir}")


if __name__ == "__main__":
    main()
