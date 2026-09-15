#!/usr/bin/env python3
"""
realworld/scripts/train.py — Train TD3+BC on real-world data.
=============================================================
Thin wrapper that calls the EXISTING training modules from the MuJoCo pipeline.
The dataset format from build_dataset.py is identical to MuJoCo, so the
trainer works unchanged.
"""

import sys
import argparse
import yaml
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

MUJOCO_ROOT = ROOT / "simulation" / "ranger_mini_v3"
sys.path.insert(0, str(MUJOCO_ROOT))

from src.rl.utils import set_seed
from src.rl.dataset import MinavHindsightDataset
from src.rl.td3_bc import TD3_BC
from src.rl.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train TD3+BC on real-world data")
    parser.add_argument("--config", type=str, default="realworld/configs/realworld_pipeline.yaml")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config_path = ROOT / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    run_dir = Path(args.run_dir).resolve()
    t_cfg = config.get("training", {})

    # Resolve device
    import torch
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    set_seed(config["run"]["seed"])

    # Dataset path
    dataset_path = run_dir / config.get("hindsight", {}).get("output_subdir", "processed")

    is_smoke = args.smoke or config.get("smoke", {}).get("enabled", False)
    total_steps = config.get("smoke", {}).get("training_steps", 500) if is_smoke else t_cfg.get("total_gradient_steps", 1000000)

    print(f"\nTraining TD3+BC on real-world data")
    print(f"  Dataset : {dataset_path}")
    print(f"  Device  : {device}")
    print(f"  Steps   : {total_steps}")
    print(f"  Smoke   : {is_smoke}")

    # Load dataset
    dataset = MinavHindsightDataset(
        dataset_path=dataset_path,
        feature_loading=t_cfg.get("feature_loading", "ram"),
        device=device,
    )

    # Build model
    # state_dim = 4 frames × 384 dim = 1536
    # goal_dim = 384
    # action_dim = 3
    agent = TD3_BC(
        state_dim=4 * 384,
        action_dim=3,
        goal_dim=384,
        device=device,
        lr_actor=t_cfg.get("lr_actor", 3e-4),
        lr_critic=t_cfg.get("lr_critic", 3e-4),
        gamma=t_cfg.get("gamma", 0.99),
        tau=t_cfg.get("tau", 0.005),
        policy_delay=t_cfg.get("policy_delay", 2),
        target_noise=t_cfg.get("target_noise", 0.2),
        target_noise_clip=t_cfg.get("target_noise_clip", 0.5),
        lambda_bc=t_cfg.get("lambda_bc", 0.001),
    )

    # Train
    output_dir = run_dir / t_cfg.get("output_subdir", "training")
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        agent=agent,
        dataset=dataset,
        output_dir=output_dir,
        total_steps=total_steps,
        batch_size=t_cfg.get("batch_size", 256),
        checkpoint_frequency=t_cfg.get("checkpoint_every", 10000),
    )

    trainer.train()
    print(f"\n✓ Training complete! Checkpoints saved to {output_dir}")


if __name__ == "__main__":
    main()
