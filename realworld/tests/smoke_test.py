#!/usr/bin/env python3
"""
realworld/tests/smoke_test.py — Mandatory pre-flight smoke test.
================================================================
Runs a complete suite of validation checks to ensure the hardware
and software are ready before the robot is armed.

Phases:
  offline: No hardware/ROS needed. Tests math, data format, imports.
  online: Requires ROS 2. Tests topic connectivity, rates, and dry-run execution.
  all: Runs both phases.

Usage:
  python realworld/tests/smoke_test.py --phase offline
  python realworld/tests/smoke_test.py --phase all
"""

import sys
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def run_pytest(test_file: str, name: str) -> bool:
    """Run a pytest file and report result."""
    print(f"\n>> Running: {name}")
    print("-" * 40)
    
    cmd = [sys.executable, "-m", "pytest", str(ROOT / "realworld" / "tests" / test_file), "-v"]
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"\n[PASS]: {name}")
        return True
    else:
        print(f"\n[FAIL]: {name}")
        return False


def run_offline_tests() -> bool:
    """Run tests that don't require hardware or ROS."""
    print("\n" + "="*50)
    print("PHASE 1: OFFLINE CHECKS")
    print("="*50)
    
    tests = [
        ("test_pink_noise_offline.py", "Pink Noise Generation"),
        ("test_action_normalization.py", "Action Normalization"),
        ("test_data_recorder.py", "Data Recorder Format"),
        ("test_safety_monitor.py", "Safety Monitor Logic"),
        ("test_shared_modules.py", "Shared Modules (DINOv3, Hindsight)"),
        ("test_cmd_vel_dryrun.py", "Command Velocity Dry-Run Logic"),
    ]
    
    all_passed = True
    for file, name in tests:
        if not run_pytest(file, name):
            all_passed = False
            
    return all_passed


def run_online_tests() -> bool:
    """Run tests that require a running ROS 2 environment."""
    import os
    os.environ["MINAV_ROS2_AVAILABLE"] = "1"
    
    print("\n" + "="*50)
    print("PHASE 2: ROS 2 CONNECTIVITY")
    print("="*50)
    
    tests = [
        ("test_ros2_connectivity.py", "Topic Existence"),
        ("test_camera_feed.py", "RealSense Camera Feed"),
        ("test_odom_sanity.py", "Odometry Sanity"),
    ]
    
    all_passed = True
    for file, name in tests:
        if not run_pytest(file, name):
            all_passed = False
            
    # Phase 3: Dry-run exploration loop
    print("\n" + "="*50)
    print("PHASE 3: DRY-RUN EXPLORATION (30s)")
    print("="*50)
    
    cmd = [
        sys.executable, 
        str(ROOT / "realworld" / "scripts" / "run_exploration.py"),
        "--smoke"
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n[PASS]: Dry-Run Exploration")
    else:
        print("\n[FAIL]: Dry-Run Exploration")
        all_passed = False
            
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Real-world MINav pre-flight smoke test")
    parser.add_argument("--phase", choices=["offline", "online", "all"], default="all",
                        help="Which tests to run (default: all)")
    args = parser.parse_args()
    
    success = True
    
    if args.phase in ["offline", "all"]:
        if not run_offline_tests():
            success = False
            
    if args.phase in ["online", "all"]:
        if not run_online_tests():
            success = False
            
    print("\n" + "="*50)
    if success:
        print("ALL PRE-FLIGHT CHECKS PASSED")
        print("The robot is safe to arm for real-world exploration.")
        sys.exit(0)
    else:
        print("[FAILED] SMOKE TEST FAILED")
        print("Please fix the failing tests before attempting to run the robot.")
        sys.exit(1)


if __name__ == "__main__":
    main()
