#!/usr/bin/env python3
"""
realworld/scripts/evaluate_deployment_metrics.py — Aggregate and Generate Navigation Metrics.
===========================================================================================
Calculates and formats the publication-ready metrics table from deployment/evaluation trials:
  - Success Rate (SR % ↑): Mean ± Std
  - Navigation Time (Time s ↓): Mean ± Std
  - Success weighted by Time Length (STL % ↑): Mean ± Std
across Simple, Standard, and Complex environment splits.

Usage:
  # Generate summary table from existing logged deployment episodes:
  python realworld/scripts/evaluate_deployment_metrics.py --log-dir results/deployment_metrics

  # Generate benchmark report with simulated/synthetic test trials:
  python realworld/scripts/evaluate_deployment_metrics.py --generate-mock --num-episodes 15
"""

import sys
import argparse
import logging
from pathlib import Path

# Path setup
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from realworld.src.deployment_metrics_logger import DeploymentMetricsLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("evaluate_deployment_metrics")


def generate_mock_trials(logger_inst: DeploymentMetricsLogger, num_per_split: int = 10):
    """Populates realistic deployment evaluation trials across Simple, Standard, Complex."""
    import random
    import numpy as np

    logger.info(f"Generating {num_per_split} demonstration evaluation trials per split...")

    configs = {
        "simple": {"sr_target": 0.95, "time_range": (8.0, 14.0), "opt_time": 10.0, "max_s": 300},
        "standard": {"sr_target": 0.85, "time_range": (16.0, 26.0), "opt_time": 20.0, "max_s": 500},
        "complex": {"sr_target": 0.70, "time_range": (25.0, 38.0), "opt_time": 30.0, "max_s": 800},
    }

    for split, cfg in configs.items():
        for i in range(num_per_split):
            ep_id = f"trial_{split}_{i+1:03d}"
            logger_inst.start_episode(
                split=split,
                goal_image=f"maps/{split}/goals/goal_{i+1:02d}.png",
                max_steps=cfg["max_s"],
                control_freq=10.0,
                optimal_time_sec=cfg["opt_time"],
                goal_threshold=0.80,
                episode_id=ep_id,
            )

            is_success = (random.random() < cfg["sr_target"])
            if is_success:
                actual_time = random.uniform(*cfg["time_range"])
                steps = int(actual_time * 10)
                final_sim = random.uniform(0.81, 0.92)
                reason = "goal_reached"
            else:
                actual_time = cfg["max_s"] / 10.0
                steps = cfg["max_s"]
                final_sim = random.uniform(0.40, 0.68)
                reason = "timeout"

            # Simulate step telemetry
            for s in range(0, steps, 5):
                sim_prog = (s / steps) * final_sim
                logger_inst.log_step(
                    step=s,
                    similarity=sim_prog,
                    cmd_vx=0.45,
                    cmd_vy=0.0,
                    cmd_wz=0.1,
                    odom_x=s * 0.04,
                    odom_y=0.0,
                    safety_blocked=False,
                )

            # Back-calculate start time for accurate duration recording
            logger_inst.active_episode["start_time"] = logger_inst.active_episode["start_time"] - actual_time

            logger_inst.end_episode(
                success=is_success,
                termination_reason=reason,
                final_similarity=final_sim,
            )


def main():
    parser = argparse.ArgumentParser(description="Calculate and format navigation deployment metrics")
    parser.add_argument("--log-dir", type=str, default="results/deployment_metrics",
                        help="Directory containing deployment_episodes.csv")
    parser.add_argument("--generate-mock", action="store_true",
                        help="Generate synthetic benchmark trial dataset to verify metric pipelines")
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="Number of episodes per split if generating mock trials (default: 10)")
    parser.add_argument("--method-name", type=str, default="MINav (TD3+BC)",
                        help="Name of evaluated method")
    parser.add_argument("--backbone", type=str, default="DINOv3 (ViT-S/16)",
                        help="Name of visual representation backbone")
    args = parser.parse_args()

    metrics_logger = DeploymentMetricsLogger(
        log_dir=args.log_dir,
        method_name=args.method_name,
        visual_backbone=args.backbone,
    )

    if args.generate_mock:
        generate_mock_trials(metrics_logger, num_per_split=args.num_episodes)

    summary = metrics_logger.generate_summary_report()

    if not summary:
        logger.warning(f"No deployment records found in {args.log_dir}/deployment_episodes.csv")
        return

    # Print pretty report
    tex_file = Path(args.log_dir) / "deployment_metrics_table.tex"
    md_file = Path(args.log_dir) / "deployment_metrics_table.md"

    print("\n" + "=" * 80)
    print("MINAV DEPLOYMENT METRICS BENCHMARK REPORT")
    print(f"Method: {summary['method']} | Backbone: {summary['visual_backbone']} | Policy Params: {summary['policy_params']:,}")
    print("=" * 80)
    print(f"{'Split':<12} | {'Trials':<8} | {'SR (%)':<16} | {'Time (s)':<16} | {'STL (%)':<16} | {'Path (m)':<10}")
    print("-" * 80)

    for split in ["simple", "standard", "complex"]:
        data = summary["splits"].get(split, {})
        sr_str = f"{data.get('sr_pct', 0.0):.1f} ± {data.get('sr_std', 0.0):.1f}"
        time_str = f"{data.get('time_mean_s', 0.0):.1f} ± {data.get('time_std_s', 0.0):.1f}"
        stl_str = f"{data.get('stl_mean_pct', 0.0):.1f} ± {data.get('stl_std_pct', 0.0):.1f}"
        path_str = f"{data.get('path_mean_m', 0.0):.2f}"
        print(f"{split.capitalize():<12} | {data.get('n_episodes', 0):<8} | {sr_str:<16} | {time_str:<16} | {stl_str:<16} | {path_str:<10}")

    print("=" * 80)
    print(f"\nArtifacts Saved:")
    print(f"  • LaTeX Table : {tex_file}")
    print(f"  • Markdown    : {md_file}")
    print(f"  • Summary JSON: {Path(args.log_dir) / 'deployment_metrics_summary.json'}")
    print(f"  • Master CSV  : {Path(args.log_dir) / 'deployment_episodes.csv'}\n")


if __name__ == "__main__":
    main()
