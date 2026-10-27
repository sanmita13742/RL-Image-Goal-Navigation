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
import numpy as np
import pandas as pd
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
from realworld.src.synthetic_augmentation import (
    generate_temporal_reversal_transitions,
    add_action_noise,
    mine_curriculum_uniform_transitions,
)


def main():
    parser = argparse.ArgumentParser(description="Build dataset from real-world exploration")
    parser.add_argument("--config", type=str, default="realworld/configs/realworld_pipeline.yaml")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Run directory containing exploration/ subdirectory")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--smoke", action="store_true", help="Use reduced frame count")
    parser.add_argument("--num-goals", type=int, default=None,
                        help="Number of hindsight goals per transition (overrides config)")
    parser.add_argument("--augment-reversal", action="store_true",
                        help="Generate synthetic temporal reversal transitions (negative vx)")
    parser.add_argument("--augment-noise", action="store_true",
                        help="Add action noise perturbations for regularization")
    parser.add_argument("--augment-curriculum", action="store_true",
                        help="Mine balanced near-goal and hard-negative curriculum transitions")
    parser.add_argument("--force-reencode", action="store_true",
                        help="Force re-encoding even if cached DINOv3 embeddings exist")
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

    image_paths = frame_index["rgb_abs_path"].tolist()
    max_frames = config.get("smoke", {}).get("max_frames", None) if is_smoke else None
    if max_frames:
        image_paths = image_paths[:max_frames]
        frame_index = frame_index.iloc[:max_frames].reset_index(drop=True)

    dinov3_dir = out_dir / "dinov3"
    phi_cache_file = dinov3_dir / "phi_vectors.npy"
    ssd_file = dinov3_dir / "ssd_scores.npy"
    valid_file = dinov3_dir / "valid_goals.npy"

    phi_vectors = None
    if not args.force_reencode and phi_cache_file.exists() and ssd_file.exists() and valid_file.exists():
        try:
            cached_phi = np.load(phi_cache_file)
            cached_ssd = np.load(ssd_file)
            cached_valid = np.load(valid_file)
            if len(cached_phi) == len(image_paths) and len(cached_ssd) == len(image_paths):
                print(f"Found existing DINOv3 cache ({len(cached_phi):,} frames) at {dinov3_dir}.")
                print("Skipping re-encoding and reusing cached embeddings.")
                phi_vectors = cached_phi
                ssd_scores = cached_ssd
                valid_goals_mask = cached_valid
            else:
                print(f"Cached shape {cached_phi.shape} mismatch with image count {len(image_paths)}. Re-encoding...")
        except Exception as e:
            print(f"Failed to load cached embeddings: {e}. Re-encoding...")

    if phi_vectors is None:
        encoder = FrozenDINOv3(
            model_name=h_cfg.get("dino_model", "vit_small_patch16_dinov3"),
            image_h=h_cfg.get("input_resolution", [448, 784])[0],
            image_w=h_cfg.get("input_resolution", [448, 784])[1],
            ssd_threshold=h_cfg.get("ssd_threshold", 0.02),
            device=device,
        )

        batch_size = 32
        phi_vectors, ssd_scores, valid_goals_mask, _ = encoder.encode_paths(
            image_paths, batch_size=batch_size, return_grids=False, return_pooled=True,
        )

    cfg_meta = {
        "source": "realworld",
        "model_name": h_cfg.get("dino_model", "vit_small_patch16_dinov3"),
        "ssd_threshold": h_cfg.get("ssd_threshold", 0.02),
    }
    save_embeddings(dinov3_dir, phi_vectors, ssd_scores, valid_goals_mask, frame_index, cfg_meta)

    # Optional Strategy 4: Aggregate 1D LiDAR range scans if recorded
    seg_dirs = sorted(
        [d for d in session_dir.iterdir() if d.is_dir() and d.name.startswith("segment_")],
        key=lambda d: int(d.name.split("_")[1])
    )
    lidar_scans_list = []
    for sd in seg_dirs:
        lpath = sd / "lidar_scans.npy"
        if lpath.exists():
            lidar_scans_list.append(np.load(lpath))

    if lidar_scans_list:
        lidar_vectors = np.concatenate(lidar_scans_list, axis=0)
        if max_frames:
            lidar_vectors = lidar_vectors[:max_frames]
        lidar_dir = out_dir / "lidar"
        lidar_dir.mkdir(parents=True, exist_ok=True)
        np.save(lidar_dir / "lidar_vectors.npy", lidar_vectors)
        print(f"  ✓ Aggregated {len(lidar_vectors):,} 1D LiDAR scans -> {lidar_dir / 'lidar_vectors.npy'}")

    # ── Phase 3: Build goal set ──
    print("\n[3/5] Building valid goal set...")
    goal_dir = out_dir / "valid_goals"
    valid_goals_df = build_goal_set(frame_index, ssd_scores, valid_goals_mask, goal_dir)

    # ── Phase 4: Geometric hindsight ──
    print("\n[4/5] Building geometric hindsight dataset...")
    hindsight_dir = out_dir / "hindsight"
    
    # Determine num_goals
    if is_smoke:
        num_goals = config.get("smoke", {}).get("num_goals_per_transition", 1)
    elif args.num_goals is not None:
        num_goals = args.num_goals
    else:
        num_goals = h_cfg.get("num_goals", 10)

    p_geom = h_cfg.get("geometric_p", [0.70, 0.90, 0.99])
    p_weights = h_cfg.get("geometric_p_weights", [4, 3, 3] if isinstance(p_geom, list) and len(p_geom) == 3 else None)

    df_geom = build_geometric_dataset(
        phi_vectors=phi_vectors,
        frame_index=frame_index,
        p=p_geom,
        num_goals=num_goals,
        out_dir=hindsight_dir,
        max_frames=max_frames,
        seed=config["run"]["seed"],
        p_weights=p_weights,
    )

    # Synthetic Temporal Reversal Augmentation
    if args.augment_reversal or h_cfg.get("augment_reversal", False):
        print("\nApplying synthetic temporal reversal augmentation...")
        df_rev = generate_temporal_reversal_transitions(
            phi_vectors=phi_vectors,
            frame_index=frame_index,
            p_values=p_geom if isinstance(p_geom, list) else [0.70, 0.90, 0.99],
            p_weights=p_weights if isinstance(p_geom, list) else None,
            num_goals=num_goals,
            seed=config["run"]["seed"],
        )
        if len(df_rev) > 0:
            df_geom = pd.concat([df_geom, df_rev], ignore_index=True)
            print(f"Total geometric transitions after reversal: {len(df_geom):,}")

    # Action Noise Augmentation
    if args.augment_noise or h_cfg.get("augment_noise", False):
        print("\nApplying action noise regularization...")
        df_geom_noisy = add_action_noise(df_geom, seed=config["run"]["seed"])
        df_geom = pd.concat([df_geom, df_geom_noisy], ignore_index=True)
        print(f"Total geometric transitions after action noise: {len(df_geom):,}")

    # Save final geometric dataset
    df_geom.to_parquet(hindsight_dir / "geometric_transitions.parquet", index=False)

    # ── Phase 5: Uniform hindsight ──
    print("\n[5/5] Building uniform hindsight dataset...")
    if args.augment_curriculum or h_cfg.get("augment_curriculum", False):
        print("Using goal proximity curriculum mining for uniform transitions...")
        df_unif = mine_curriculum_uniform_transitions(
            phi_vectors=phi_vectors,
            frame_index=frame_index,
            valid_goals_df=valid_goals_df,
            seed=config["run"]["seed"],
        )
    else:
        df_unif = build_uniform_dataset(
            phi_vectors=phi_vectors,
            frame_index=frame_index,
            valid_goals_df=valid_goals_df,
            ssd_scores=ssd_scores,
            num_goals=num_goals,
            out_dir=hindsight_dir,
            max_frames=max_frames,
            seed=config["run"]["seed"],
        )

    if args.augment_noise or h_cfg.get("augment_noise", False):
        df_unif_noisy = add_action_noise(df_unif, seed=config["run"]["seed"] + 1)
        df_unif = pd.concat([df_unif, df_unif_noisy], ignore_index=True)
        print(f"Total uniform transitions after action noise: {len(df_unif):,}")

    df_unif.to_parquet(hindsight_dir / "uniform_transitions.parquet", index=False)

    print("\n[SUCCESS] Dataset build complete!")
    print(f"  Output: {out_dir}")


if __name__ == "__main__":
    main()
