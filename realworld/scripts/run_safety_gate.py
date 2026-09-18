#!/usr/bin/env python3
"""
realworld/scripts/run_safety_gate.py — Real-world LiDAR safety gate entry point.
================================================================================
Usage:
  python realworld/scripts/run_safety_gate.py --config realworld/configs/realworld_pipeline.yaml

This script launches the LidarSafetyGate ROS 2 node, which subscribes to 
the raw point cloud and the MINav cmd_vel, and publishes safe commands.
"""

import sys
import argparse
import yaml
import logging
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import rclpy
from rclpy.parameter import Parameter
from realworld.src.lidar_safety_gate import LidarSafetyGate

def main():
    parser = argparse.ArgumentParser(description="Real-world MINav Safety Gate")
    parser.add_argument("--config", type=str, default="realworld/configs/realworld_pipeline.yaml",
                        help="Path to pipeline config YAML")
    args = parser.parse_args()

    config_path = ROOT / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    safety_gate_cfg = config.get("safety_gate", {})
    
    # Setup simple logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    logger = logging.getLogger("run_safety_gate")

    logger.info("=" * 60)
    logger.info("LIDAR SAFETY GATE")
    logger.info("=" * 60)
    logger.info(f"  Config    : {config_path}")
    logger.info("=" * 60)

    rclpy.init()

    # Construct parameter overrides
    param_overrides = []
    for key, value in safety_gate_cfg.items():
        if isinstance(value, float):
            param_overrides.append(Parameter(key, Parameter.Type.DOUBLE, value))
        elif isinstance(value, int):
            param_overrides.append(Parameter(key, Parameter.Type.INTEGER, value))
        elif isinstance(value, str):
            param_overrides.append(Parameter(key, Parameter.Type.STRING, value))
        elif isinstance(value, bool):
            param_overrides.append(Parameter(key, Parameter.Type.BOOL, value))

    try:
        node = LidarSafetyGate(parameter_overrides=param_overrides)
        rclpy.spin(node)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    finally:
        if 'node' in locals():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
