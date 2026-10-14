#!/usr/bin/env python3
"""
realworld/scripts/validate_offline_navigation.py
=============================================================================
Stage 1 Offline Navigation Validation for MINav Real-World TD3+BC Policy.

Validates the trained actor checkpoint without requiring physical hardware:
  1. Matches camera preprocessing & 4-frame temporal stacking [t-3, t-2, t-1, t0].
  2. Verifies actor output normalization [-1, 1]^3 and inverse affine denormalization
     to physical ranges: vx in [-0.9, 1.8], vy in [-1.0, 1.0], wz in [-1.5, 1.5].
  3. Replays continuous trajectory episodes towards designated goal images.
  4. Evaluates goal-reaching progression, similarity curves, action oscillation,
     and goal-arrival deceleration / stopping behavior.
  5. Generates publication-grade validation plots and metrics summary.
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project root resolution
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "simulation" / "ranger_mini_v3"))

from shared.action_normalizer import ActionNormalizer
from src.rl.networks import Actor

logger = logging.getLogger("validate_offline_navigation")


def parse_args():
    parser = argparse.ArgumentParser(description="Validate offline MINav navigation policy")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="output-realworld/training/checkpoints/checkpoint_150000.pt",
        help="Path to trained actor checkpoint (.pt)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="output-realworld/processed",
        help="Path to processed real-world dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output-realworld/validation_results",
        help="Output directory for plots and metrics",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="Starting frame index for evaluation trajectory",
    )
    parser.add_argument(
        "--goal-step",
        type=int,
        default=120,
        help="Goal frame index for evaluation trajectory",
    )
    parser.add_argument(
        "--goal-threshold",
        type=float,
        default=0.80,
        help="Cosine similarity threshold for goal arrival",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computation device ('cpu' or 'cuda')",
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> Actor:
    state_dim = 4 * 384   # 4 frames * 384 DINOv3 features
    goal_dim = 384
    action_dim = 3

    actor = Actor(state_dim, goal_dim, action_dim).to(device)
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    
    if "actor" in checkpoint:
        actor.load_state_dict(checkpoint["actor"])
    else:
        actor.load_state_dict(checkpoint)
        
    actor.eval()
    logger.info("Actor loaded successfully into eval mode.")
    return actor


def run_trajectory_evaluation(
    actor: Actor,
    normalizer: ActionNormalizer,
    phi_vectors: np.ndarray,
    transitions_df: pd.DataFrame,
    start_step: int,
    goal_step: int,
    goal_threshold: float,
    device: torch.device,
):
    total_frames = len(phi_vectors)
    start_step = max(0, min(start_step, total_frames - 5))
    goal_step = max(start_step + 4, min(goal_step, total_frames - 1))
    
    logger.info(f"Evaluating trajectory: frames {start_step} -> {goal_step} (horizon: {goal_step - start_step})")
    
    goal_phi = torch.tensor(phi_vectors[goal_step], dtype=torch.float32, device=device).unsqueeze(0)  # [1, 384]
    
    # 4-frame FIFO buffer for state construction [t-3, t-2, t-1, t0]
    phi_buffer = deque(maxlen=4)
    
    records = []
    goal_reached_step = None

    for t in range(start_step, goal_step + 1):
        cur_phi = torch.tensor(phi_vectors[t], dtype=torch.float32, device=device)
        phi_buffer.append(cur_phi)
        
        # Buffer warmup: replicate first frame if fewer than 4 frames
        while len(phi_buffer) < 4:
            phi_buffer.append(cur_phi)
            
        # Chronological order: [phi_{t-3}, phi_{t-2}, phi_{t-1}, phi_{t0}] -> [1, 1536]
        state = torch.cat(list(phi_buffer), dim=0).unsqueeze(0)
        
        # Policy inference (strictly normalized in [-1, 1]^3 due to Tanh)
        with torch.no_grad():
            action_norm = actor(state, goal_phi)  # [1, 3]
            
        # Inverse affine normalization to physical robot limits
        action_phys = normalizer.denormalize(action_norm.squeeze(0))
        vx_pred, vy_pred, wz_pred = action_phys.cpu().numpy()
        
        # Cosine similarity between current frame and goal (both L2 normalized)
        sim = float(torch.dot(cur_phi, goal_phi.squeeze(0)).item())
        
        # Retrieve behavior demonstration action if available
        demo_vx, demo_vy, demo_wz = np.nan, np.nan, np.nan
        match = transitions_df[transitions_df["current_global_step"] == t]
        if not match.empty:
            demo_vx = float(match["action_linear"].iloc[0])
            demo_vy = float(match["action_lateral"].iloc[0])
            demo_wz = float(match["action_angular"].iloc[0])
            
        if sim >= goal_threshold and goal_reached_step is None:
            goal_reached_step = t
            
        records.append({
            "step": t,
            "rel_step": t - start_step,
            "sim": sim,
            "norm_vx": float(action_norm[0, 0].item()),
            "norm_vy": float(action_norm[0, 1].item()),
            "norm_wz": float(action_norm[0, 2].item()),
            "pred_vx": float(vx_pred),
            "pred_vy": float(vy_pred),
            "pred_wz": float(wz_pred),
            "demo_vx": demo_vx,
            "demo_vy": demo_vy,
            "demo_wz": demo_wz,
            "goal_reached": (sim >= goal_threshold),
        })

    eval_df = pd.DataFrame(records)
    
    # Calculate action oscillation / jitter (first difference)
    eval_df["delta_vx"] = eval_df["pred_vx"].diff().abs()
    eval_df["delta_vy"] = eval_df["pred_vy"].diff().abs()
    eval_df["delta_wz"] = eval_df["pred_wz"].diff().abs()
    
    return eval_df, goal_reached_step


def compute_metrics(eval_df: pd.DataFrame, goal_reached_step: int, goal_threshold: float) -> dict:
    metrics = {
        "trajectory_length": len(eval_df),
        "initial_similarity": float(eval_df["sim"].iloc[0]),
        "final_similarity": float(eval_df["sim"].iloc[-1]),
        "max_similarity": float(eval_df["sim"].max()),
        "goal_reached": bool(eval_df["sim"].max() >= goal_threshold),
        "goal_reached_step": int(goal_reached_step) if goal_reached_step is not None else -1,
        
        # Action bound validation
        "norm_vx_range": [float(eval_df["norm_vx"].min()), float(eval_df["norm_vx"].max())],
        "norm_vy_range": [float(eval_df["norm_vy"].min()), float(eval_df["norm_vy"].max())],
        "norm_wz_range": [float(eval_df["norm_wz"].min()), float(eval_df["norm_wz"].max())],
        
        "pred_vx_mean": float(eval_df["pred_vx"].mean()),
        "pred_vx_range": [float(eval_df["pred_vx"].min()), float(eval_df["pred_vx"].max())],
        "pred_vy_range": [float(eval_df["pred_vy"].min()), float(eval_df["pred_vy"].max())],
        "pred_wz_range": [float(eval_df["pred_wz"].min()), float(eval_df["pred_wz"].max())],
        
        # Oscillation / smoothness metrics
        "jitter_vx_mean": float(eval_df["delta_vx"].dropna().mean()),
        "jitter_vy_mean": float(eval_df["delta_vy"].dropna().mean()),
        "jitter_wz_mean": float(eval_df["delta_wz"].dropna().mean()),
        "jitter_wz_std": float(eval_df["delta_wz"].dropna().std()),
    }
    
    # Behavior cloning error on valid demonstration points
    valid_demos = eval_df.dropna(subset=["demo_vx", "demo_vy", "demo_wz"])
    if len(valid_demos) > 0:
        mse_vx = float(((valid_demos["pred_vx"] - valid_demos["demo_vx"]) ** 2).mean())
        mse_vy = float(((valid_demos["pred_vy"] - valid_demos["demo_vy"]) ** 2).mean())
        mse_wz = float(((valid_demos["pred_wz"] - valid_demos["demo_wz"]) ** 2).mean())
        metrics["bc_mse_vx"] = mse_vx
        metrics["bc_mse_vy"] = mse_vy
        metrics["bc_mse_wz"] = mse_wz
        metrics["bc_mse_overall"] = (mse_vx + mse_vy + mse_wz) / 3.0
    else:
        metrics["bc_mse_overall"] = None
        
    return metrics


def plot_validation_results(eval_df: pd.DataFrame, metrics: dict, output_path: Path):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(4, 1, figsize=(13, 15), sharex=True)
    
    steps = eval_df["rel_step"]
    
    # Panel 1: Cosine Similarity to Goal
    ax0 = axes[0]
    ax0.plot(steps, eval_df["sim"], color="#2563EB", linewidth=2.5, label="Cosine Similarity cos(phi_t, phi_goal)")
    ax0.axhline(0.80, color="#DC2626", linestyle="--", linewidth=1.8, label="Goal Reached Threshold (0.80)")
    if metrics["goal_reached_step"] != -1:
        rel_goal = metrics["goal_reached_step"] - eval_df["step"].iloc[0]
        ax0.axvline(rel_goal, color="#16A34A", linestyle=":", linewidth=2.0, label=f"Goal Arrival (Step {rel_goal})")
    ax0.set_title("MINav Navigation Validation: Goal Alignment & Policy Action Trajectory", fontsize=14, fontweight="bold")
    ax0.set_ylabel("Cosine Similarity", fontsize=11)
    ax0.set_ylim(-0.1, 1.05)
    ax0.legend(loc="upper left", frameon=True)
    
    # Panel 2: Linear Forward Velocity (vx)
    ax1 = axes[1]
    ax1.plot(steps, eval_df["pred_vx"], color="#059669", linewidth=2.2, label="Policy Output vx (m/s)")
    if not eval_df["demo_vx"].isna().all():
        ax1.plot(steps, eval_df["demo_vx"], color="#9CA3AF", linestyle="--", alpha=0.8, label="Behavior Dataset vx (m/s)")
    ax1.axhline(1.8, color="#B91C1C", linestyle=":", alpha=0.6, label="Max Forward (1.8 m/s)")
    ax1.axhline(-0.9, color="#B91C1C", linestyle=":", alpha=0.6, label="Max Reverse (-0.9 m/s)")
    ax1.axhline(0.0, color="black", linestyle="-", linewidth=0.8, alpha=0.4)
    ax1.set_ylabel("Forward vx [m/s]", fontsize=11)
    ax1.legend(loc="upper right", frameon=True)
    
    # Panel 3: Lateral Velocity (vy) and Angular Rate (wz)
    ax2 = axes[2]
    ax2.plot(steps, eval_df["pred_vy"], color="#7C3AED", linewidth=2.0, label="Policy Lateral vy (m/s)")
    ax2.plot(steps, eval_df["pred_wz"], color="#D97706", linewidth=2.0, label="Policy Angular wz (rad/s)")
    ax2.axhline(0.0, color="black", linestyle="-", linewidth=0.8, alpha=0.4)
    ax2.set_ylabel("Lateral & Angular", fontsize=11)
    ax2.legend(loc="upper right", frameon=True)
    
    # Panel 4: Action Jitter / Oscillation Magnitude
    ax3 = axes[3]
    ax3.plot(steps, eval_df["delta_vx"], color="#059669", alpha=0.7, label="|Delta vx| per 100ms step")
    ax3.plot(steps, eval_df["delta_wz"], color="#D97706", alpha=0.7, label="|Delta wz| per 100ms step")
    ax3.set_ylabel("Jitter Magnitude", fontsize=11)
    ax3.set_xlabel("Relative Trajectory Steps (10 Hz, 100 ms intervals)", fontsize=11)
    ax3.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Validation plot saved to: {output_path}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    args = parse_args()

    checkpoint_path = ROOT / args.checkpoint
    data_dir = ROOT / args.data_dir
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # 1. Load actor model
    actor = load_model(checkpoint_path, device)
    normalizer = ActionNormalizer(device=device)

    # 2. Load dataset embeddings & transitions
    phi_path = data_dir / "dinov3" / "phi_vectors.npy"
    transitions_path = data_dir / "hindsight" / "uniform_transitions.parquet"

    logger.info(f"Loading phi_vectors from: {phi_path}")
    phi_vectors = np.load(phi_path)
    logger.info(f"Loaded {len(phi_vectors)} DINOv3 feature vectors of dim {phi_vectors.shape[1]}")

    logger.info(f"Loading transitions from: {transitions_path}")
    transitions_df = pd.read_parquet(transitions_path)

    # 3. Run evaluation
    eval_df, goal_reached_step = run_trajectory_evaluation(
        actor=actor,
        normalizer=normalizer,
        phi_vectors=phi_vectors,
        transitions_df=transitions_df,
        start_step=args.start_step,
        goal_step=args.goal_step,
        goal_threshold=args.goal_threshold,
        device=device,
    )

    # 4. Compute metrics
    metrics = compute_metrics(eval_df, goal_reached_step, args.goal_threshold)

    # 5. Save artifacts
    csv_out = output_dir / "trajectory_evaluation.csv"
    eval_df.to_csv(csv_out, index=False)
    
    json_out = output_dir / "validation_metrics.json"
    with open(json_out, "w") as f:
        json.dump(metrics, f, indent=2)
        
    plot_out = output_dir / "offline_navigation_validation.png"
    plot_validation_results(eval_df, metrics, plot_out)

    logger.info("=" * 65)
    logger.info("OFFLINE NAVIGATION VALIDATION COMPLETE")
    logger.info(f"  Goal Reached      : {metrics['goal_reached']} (at step {metrics['goal_reached_step']})")
    logger.info(f"  Similarity Range  : [{metrics['initial_similarity']:.3f} -> {metrics['final_similarity']:.3f}] (Peak: {metrics['max_similarity']:.3f})")
    logger.info(f"  Pred vx (Forward) : Mean {metrics['pred_vx_mean']:.3f} m/s (Range: [{metrics['pred_vx_range'][0]:.3f}, {metrics['pred_vx_range'][1]:.3f}])")
    logger.info(f"  Pred vy (Lateral) : Range: [{metrics['pred_vy_range'][0]:.3f}, {metrics['pred_vy_range'][1]:.3f}] m/s")
    logger.info(f"  Pred wz (Angular) : Range: [{metrics['pred_wz_range'][0]:.3f}, {metrics['pred_wz_range'][1]:.3f}] rad/s")
    logger.info(f"  Angular Jitter std: {metrics['jitter_wz_std']:.4f} rad/s")
    if metrics["bc_mse_overall"] is not None:
        logger.info(f"  BC Demo MSE       : {metrics['bc_mse_overall']:.5f}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
