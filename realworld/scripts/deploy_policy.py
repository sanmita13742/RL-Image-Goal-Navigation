#!/usr/bin/env python3
"""
realworld/scripts/deploy_policy.py — Deploy trained policy on real robot.
=========================================================================
Loads a trained TD3+BC actor checkpoint and runs closed-loop image-goal
navigation on the real Ranger Mini V3.

Pipeline per step:
  1. Get RGB frame from RealSense.
  2. Encode with DINOv3 → phi(o) [384-D].
  3. Construct 4-frame state [1536-D].
  4. Actor forward pass → normalized action [-1, 1].
  5. Denormalize → physical action → /cmd_vel (or dry-run).

IMPORTANT: Starts in dry-run mode by default. Pass --arm to enable motion.
"""

import sys
import argparse
import logging
import time
from pathlib import Path
from collections import deque

import numpy as np
import torch

# Project root
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

MUJOCO_ROOT = ROOT / "simulation" / "ranger_mini_v3"
sys.path.insert(0, str(MUJOCO_ROOT))

import rclpy

from realworld.src.ros2_robot import ROS2RangerMiniV3
from realworld.src.ros2_camera import ROS2Camera
from realworld.src.ros2_lidar import ROS2LiDAR
from realworld.src.safety_monitor import SafetyMonitor
from shared.drive_command import DriveCommand
from shared.action_normalizer import ActionNormalizer
from src.rl.networks import Actor
from src.data.dinov3_encoder import FrozenDINOv3

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Deploy trained policy")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to actor checkpoint")
    parser.add_argument("--goal-image", type=str, required=True, help="Path to goal image")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--arm", action="store_true", help="ARM robot for real motion")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max inference steps")
    parser.add_argument("--control-freq", type=float, default=10.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load DINOv3 encoder
    logger.info("Loading DINOv3 encoder...")
    encoder = FrozenDINOv3(device=str(device))

    # Encode goal image
    logger.info(f"Encoding goal image: {args.goal_image}")
    goal_phi, _, _, _ = encoder.encode_paths([args.goal_image])
    goal_phi_tensor = torch.tensor(goal_phi[0], dtype=torch.float32, device=device)
    logger.info(f"Goal phi shape: {goal_phi_tensor.shape}")

    # Load actor
    logger.info(f"Loading actor from: {args.checkpoint}")
    state_dim = 4 * 384  # 4-frame state
    goal_dim = 384
    action_dim = 3

    actor = Actor(state_dim, goal_dim, action_dim).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    normalizer = ActionNormalizer(device=str(device))

    # Initialize ROS 2
    rclpy.init()
    dry_run = not args.arm

    try:
        robot = ROS2RangerMiniV3(dry_run=dry_run)
        camera = ROS2Camera()
        lidar = ROS2LiDAR()
        safety = SafetyMonitor(robot, camera, lidar)

        # Frame buffer for 4-frame state
        phi_buffer = deque(maxlen=4)
        control_period = 1.0 / args.control_freq

        logger.info("Waiting for sensors...")
        time.sleep(3)

        if not dry_run:
            if not safety.arm():
                logger.error("Failed to arm. Aborting.")
                return

        logger.info(f"Running policy ({'ARMED' if not dry_run else 'DRY-RUN'})...")

        for step in range(args.max_steps):
            loop_start = time.time()

            # Spin ROS
            rclpy.spin_once(robot, timeout_sec=0.01)
            rclpy.spin_once(camera, timeout_sec=0.01)
            rclpy.spin_once(lidar, timeout_sec=0.01)

            # Safety check
            if not dry_run and not safety.check():
                logger.error("Safety violation! Stopping.")
                break

            # Get frame
            rgb = camera.get_frame()
            if rgb is None:
                time.sleep(0.01)
                continue

            # Encode frame with DINOv3
            from PIL import Image as PILImage
            import tempfile, os
            # Save to temp file for encoder (encoder expects file paths)
            tmp_path = Path(tempfile.gettempdir()) / "_minav_deploy_frame.png"
            PILImage.fromarray(rgb).save(tmp_path)
            phi, _, _, _ = encoder.encode_paths([str(tmp_path)])
            phi_tensor = torch.tensor(phi[0], dtype=torch.float32, device=device)

            phi_buffer.append(phi_tensor)

            # Need 4 frames for state
            if len(phi_buffer) < 4:
                logger.info(f"Buffering frames: {len(phi_buffer)}/4")
                time.sleep(control_period)
                continue

            # Construct 4-frame state [1536]
            state = torch.cat(list(phi_buffer), dim=0).unsqueeze(0)  # [1, 1536]

            # Actor inference
            with torch.no_grad():
                action_norm = actor(state, goal_phi_tensor.unsqueeze(0))  # [1, 3]

            # Denormalize
            action_phys = normalizer.denormalize(action_norm.squeeze(0))
            vx, vy, omega = action_phys.cpu().numpy()

            cmd = DriveCommand(v_linear=float(vx), v_lateral=float(vy), v_angular=float(omega))
            robot.send_command(cmd)

            # Check if goal reached
            sim = float((state.squeeze(0).reshape(4, 384) @ goal_phi_tensor).mean())
            if sim >= 0.8:
                logger.info(f"🎯 GOAL REACHED! Similarity: {sim:.3f}")
                break

            if step % 10 == 0:
                logger.info(f"Step {step}: sim={sim:.3f} vx={vx:.2f} vy={vy:.2f} ω={omega:.2f}")

            # Rate control
            elapsed = time.time() - loop_start
            if elapsed < control_period:
                time.sleep(control_period - elapsed)

    except KeyboardInterrupt:
        logger.warning("Interrupted")
    finally:
        robot.emergency_stop()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
