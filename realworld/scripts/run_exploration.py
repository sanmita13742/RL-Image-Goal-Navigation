#!/usr/bin/env python3
"""
realworld/scripts/run_exploration.py — Real-world exploration entry point.
==========================================================================
Usage:
  # Dry-run (DEFAULT — robot will NOT move):
  python realworld/scripts/run_exploration.py --config realworld/configs/realworld_pipeline.yaml

  # Armed (robot WILL move — requires all pre-flight checks):
  python realworld/scripts/run_exploration.py --config realworld/configs/realworld_pipeline.yaml --arm

  # Smoke test (30 seconds, dry-run):
  python realworld/scripts/run_exploration.py --config realworld/configs/realworld_pipeline.yaml --smoke

IMPORTANT: The robot does NOT start moving automatically.
  - Default mode is DRY-RUN: commands are logged but not published.
  - To enable motion, pass --arm explicitly.
  - Pre-flight checks (smoke_test.py) should pass before arming.
"""

import sys
import argparse
import logging
import datetime
import yaml
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import rclpy
from rclpy.executors import MultiThreadedExecutor

from realworld.src.ros2_robot import ROS2RangerMiniV3
from realworld.src.ros2_camera import ROS2Camera
from realworld.src.ros2_lidar import ROS2LiDAR
from realworld.src.safety_monitor import SafetyMonitor
from realworld.src.data_recorder import DataRecorder
from realworld.src.exploration_runner import ExplorationRunner


def setup_logging(run_dir: Path) -> None:
    """Configure logging to both stdout and file."""
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(run_dir / "exploration.log", mode="a", encoding="utf-8"),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Real-world MINav exploration")
    parser.add_argument("--config", type=str, default="realworld/configs/realworld_pipeline.yaml",
                        help="Path to pipeline config YAML")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Output directory for this run (auto-generated if not set)")
    parser.add_argument("--arm", action="store_true",
                        help="ARM the robot for real motion (default: dry-run)")
    parser.add_argument("--smoke", action="store_true",
                        help="Run a 30-second smoke test (dry-run)")
    args = parser.parse_args()

    # Load config
    config_path = ROOT / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Determine run directory
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        run_id = config["run"].get("id") or f"realworld_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = ROOT / "realworld" / "runs" / run_id

    setup_logging(run_dir)
    logger = logging.getLogger("run_exploration")

    # Determine mode
    dry_run = not args.arm
    if args.smoke:
        dry_run = True
        duration_minutes = config.get("smoke", {}).get("exploration_minutes", 0.5)
        logger.warning("SMOKE MODE — 30s dry-run exploration")
    else:
        duration_minutes = config["exploration"]["duration_minutes"]

    logger.info("=" * 60)
    logger.info("REAL-WORLD EXPLORATION")
    logger.info("=" * 60)
    logger.info(f"  Config    : {config_path}")
    logger.info(f"  Run dir   : {run_dir}")
    logger.info(f"  Duration  : {duration_minutes} minutes")
    logger.info(f"  Mode      : {'ARMED' if not dry_run else 'DRY-RUN'}")
    logger.info("=" * 60)

    if not dry_run:
        logger.warning("⚠️  ROBOT IS ARMED — it WILL move!")
        logger.warning("    Press Ctrl+C to emergency stop at any time.")

    # Initialize ROS 2
    rclpy.init()

    try:
        # Create ROS 2 nodes
        robot_cfg = config.get("robot", {})
        robot = ROS2RangerMiniV3(
            cmd_vel_topic=robot_cfg.get("cmd_vel_topic", "/cmd_vel"),
            odom_topic=robot_cfg.get("odom_topic", "/odom"),
            dry_run=dry_run,
            max_linear_vel=robot_cfg.get("max_linear_vel", 1.8),
            min_linear_vel=robot_cfg.get("min_linear_vel", -0.9),
            max_lateral_vel=robot_cfg.get("max_lateral_vel", 1.0),
            max_angular_vel=robot_cfg.get("max_angular_vel", 1.5),
        )

        cam_cfg = config.get("camera", {})
        camera = ROS2Camera(
            topic=cam_cfg.get("topic", "/camera/color/image_raw"),
            encoding=cam_cfg.get("encoding", "rgb8"),
            expected_width=cam_cfg.get("width", 640),
            expected_height=cam_cfg.get("height", 480),
        )

        lidar_cfg = config.get("lidar", {})
        lidar = ROS2LiDAR(
            topic=lidar_cfg.get("topic", "/scanner/cloud"),
            projection_width=lidar_cfg.get("projection_width", 640),
            projection_height=lidar_cfg.get("projection_height", 60),
            max_range=lidar_cfg.get("max_range", 10.0),
            min_range=lidar_cfg.get("min_range", 0.1),
            height_min=lidar_cfg.get("height_min", -0.3),
            height_max=lidar_cfg.get("height_max", 1.0),
            fov_deg=lidar_cfg.get("fov_deg", 360.0),
        )

        safety_cfg = config.get("safety", {})
        safety = SafetyMonitor(
            robot=robot,
            camera=camera,
            lidar=lidar,
            camera_timeout_s=safety_cfg.get("camera_timeout_s", 2.0),
            odom_timeout_s=safety_cfg.get("odom_timeout_s", 2.0),
            lidar_timeout_s=safety_cfg.get("lidar_timeout_s", 3.0),
        )

        explore_cfg = config.get("exploration", {})
        session_dir = run_dir / explore_cfg.get("output_subdir", "exploration")
        recorder = DataRecorder(
            session_dir=session_dir,
            segment_size=explore_cfg.get("segment_size", 1000),
        )

        runner = ExplorationRunner(
            robot=robot,
            camera=camera,
            lidar=lidar,
            safety=safety,
            recorder=recorder,
            control_freq_hz=explore_cfg.get("control_freq_hz", 10.0),
            duration_minutes=duration_minutes,
            beta=explore_cfg.get("pink_noise_beta", 1),
        )

        # Run exploration
        success = runner.run()

        if success:
            logger.info("Exploration finished successfully.")
        else:
            logger.error("Exploration failed or was aborted.")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
