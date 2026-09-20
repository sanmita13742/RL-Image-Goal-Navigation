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

    # Build the exact config dict that Trainer expects, mapped from pipeline config
    train_cfg = t_cfg.copy()
    train_cfg["seed"] = config["run"]["seed"]
    train_cfg["dataset_path"] = str(dataset_path)
    train_cfg["output_dir"] = str(run_dir / t_cfg.get("output_subdir", "training"))
    train_cfg["feature_loading"] = "ram" # Or mmap, hardcode ram for speed
    train_cfg["fqe_frequency"] = train_cfg.get("checkpoint_every", 100000)
    train_cfg["checkpoint_frequency"] = train_cfg.get("checkpoint_every", 100000)
    train_cfg["learning_rate_actor"] = train_cfg["lr_actor"]
    train_cfg["learning_rate_critic"] = train_cfg["lr_critic"]

    if is_smoke:
        train_cfg["total_gradient_steps"] = total_steps
        train_cfg["checkpoint_frequency"] = max(1, train_cfg["total_gradient_steps"] // 2)
        train_cfg["fqe_frequency"] = train_cfg["checkpoint_frequency"]

    trainer = Trainer(train_cfg, device)

    trainer.train()
    print(f"\n✓ Training complete! Checkpoints saved to {train_cfg['output_dir']}")


if __name__ == "__main__":
    main()
