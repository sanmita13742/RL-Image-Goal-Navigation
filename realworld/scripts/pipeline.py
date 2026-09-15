#!/usr/bin/env python3
"""
realworld/scripts/pipeline.py — Real-world pipeline orchestrator.
==================================================================
Runs the full pipeline: pre-flight → exploration → dataset → training.

Pre-flight checks are MANDATORY and cannot be skipped.

Usage:
  # Full pipeline (dry-run exploration by default):
  python realworld/scripts/pipeline.py --config realworld/configs/realworld_pipeline.yaml

  # Full pipeline with armed exploration:
  python realworld/scripts/pipeline.py --config realworld/configs/realworld_pipeline.yaml --arm

  # Smoke test:
  python realworld/scripts/pipeline.py --config realworld/configs/realworld_pipeline.yaml --smoke
"""

import sys
import argparse
import subprocess
import datetime
import yaml
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    parser = argparse.ArgumentParser(description="Real-world MINav pipeline")
    parser.add_argument("--config", default="realworld/configs/realworld_pipeline.yaml")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--arm", action="store_true", help="ARM robot for real motion")
    parser.add_argument("--smoke", action="store_true", help="Smoke test mode")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip offline pre-flight tests (NOT recommended)")
    args = parser.parse_args()

    # Determine run directory
    run_id = args.run_id or f"realworld_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = ROOT / "realworld" / "runs" / run_id

    if run_dir.exists():
        print(f"ERROR: Run directory {run_dir} already exists.")
        sys.exit(1)

    run_dir.mkdir(parents=True)
    print(f"\n{'='*60}")
    print(f"REAL-WORLD MINAV PIPELINE")
    print(f"{'='*60}")
    print(f"Run ID  : {run_id}")
    print(f"Mode    : {'ARMED' if args.arm else 'DRY-RUN'}")
    print(f"Smoke   : {args.smoke}")
    print(f"Device  : {args.device}")
    print(f"Output  : {run_dir}")
    print(f"{'='*60}\n")

    # Copy config
    config_path = ROOT / args.config
    shutil.copy(config_path, run_dir / "config_used.yaml")

    # ── Phase 0: Pre-flight checks ──
    if not args.skip_preflight:
        print("\n[0/3] PRE-FLIGHT CHECKS (offline)")
        preflight_cmd = [
            sys.executable, str(ROOT / "realworld" / "tests" / "smoke_test.py"),
            "--phase", "offline",
        ]
        result = subprocess.run(preflight_cmd)
        if result.returncode != 0:
            print("\n[FAIL] PRE-FLIGHT CHECKS FAILED. Fix issues before running pipeline.")
            sys.exit(1)
        print("[PASS] Pre-flight checks passed.\n")
    else:
        print("!! Skipping pre-flight checks (--skip-preflight)\n")

    # ── Phase 1: Exploration ──
    print("\n[1/3] DATA EXPLORATION")
    explore_cmd = [
        sys.executable, str(ROOT / "realworld" / "scripts" / "run_exploration.py"),
        "--config", args.config,
        "--run-dir", str(run_dir),
    ]
    if args.arm:
        explore_cmd.append("--arm")
    if args.smoke:
        explore_cmd.append("--smoke")

    result = subprocess.run(explore_cmd)
    if result.returncode != 0:
        print("[FAIL] Exploration failed.")
        sys.exit(1)

    # ── Phase 2: Dataset processing ──
    print("\n[2/3] DATASET PROCESSING (DINOv3 + Hindsight)")
    dataset_cmd = [
        sys.executable, str(ROOT / "realworld" / "scripts" / "build_dataset.py"),
        "--config", args.config,
        "--run-dir", str(run_dir),
        "--device", args.device,
    ]
    if args.smoke:
        dataset_cmd.append("--smoke")

    result = subprocess.run(dataset_cmd)
    if result.returncode != 0:
        print("[FAIL] Dataset processing failed.")
        sys.exit(1)

    # ── Phase 3: Training ──
    print("\n[3/3] TD3+BC TRAINING")
    train_cmd = [
        sys.executable, str(ROOT / "realworld" / "scripts" / "train.py"),
        "--config", args.config,
        "--run-dir", str(run_dir),
        "--device", args.device,
    ]
    if args.smoke:
        train_cmd.append("--smoke")

    result = subprocess.run(train_cmd)
    if result.returncode != 0:
        print("[FAIL] Training failed.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"[SUCCESS] PIPELINE COMPLETE")
    print(f"  Run ID  : {run_id}")
    print(f"  Output  : {run_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
