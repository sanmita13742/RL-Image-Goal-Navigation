#!/usr/bin/env python3
"""
realworld/scripts/deploy_policy.py — Deploy trained MINav policy on Ranger Mini V3.
==================================================================================
Loads a trained TD3+BC actor checkpoint and runs closed-loop image-goal
navigation on the real Ranger Mini V3 robot.

Strict Architecture & Safety Requirements:
  1. Camera Input Matching:
     - RealSense RGB (640x480) -> DINOv3 input (448x784).
     - ToTensor() conversion, no ImageNet normalization.
     - Frozen DINOv3 ViT-Small/16 feature extraction -> CLS token index 0.
     - L2 normalization -> 384-D embedding per frame.
     - 4-frame FIFO temporal state construction [t-3, t-2, t-1, t0] -> [1, 1536].
  2. Action Mapping & Inverse Normalization:
     - Policy outputs normalized actions [-1, 1]^3 via nn.Tanh().
     - MUST apply inverse affine transformation (ActionNormalizer.denormalize):
         vx in [-0.9, 1.8] m/s, vy in [-1.0, 1.0] m/s, wz in [-1.5, 1.5] rad/s.
     - Maps vx -> linear.x, vy -> linear.y, wz -> angular.z.
     - Never publish raw actor outputs directly to /cmd_vel.
  3. Independent Collision Protection & Command Routing:
     - Policy commands publish to /minav/cmd_vel by default.
     - The independent LidarSafetyGate node intercepts /minav/cmd_vel,
       evaluates the 10 Hz /rslidar_points swept footprint, and arbitrates /cmd_vel.
     - Built-in SafetyMonitor watchdog halts robot if any stream drops.
  4. Progressive Deployment & Stopping:
     - Starts in DRY-RUN mode by default. Pass --arm to enable physical drive.
     - Configurable --speed-scale (default 0.3) for safe supervised trials.
     - Decelerates near goal (sim >= 0.70) and halts completely at sim >= 0.80.
"""

import sys
import argparse
import logging
import time
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage

# Project root resolution
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

logger = logging.getLogger("deploy_policy")


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy trained MINav navigation policy")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to actor checkpoint (.pt)")
    parser.add_argument("--goal-image", type=str, required=True, help="Path to goal image (RGB)")
    parser.add_argument("--cmd-topic", type=str, default="/minav/cmd_vel",
                        help="Topic to publish policy commands (default: /minav/cmd_vel routed through safety gate)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cpu', 'cuda', 'auto')")
    parser.add_argument("--arm", action="store_true", help="ARM robot for real physical motion (default: DRY-RUN)")
    parser.add_argument("--speed-scale", type=float, default=0.30,
                        help="Speed scaling factor (0.0 to 1.0) for supervised trials (default: 0.30)")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum control steps")
    parser.add_argument("--control-freq", type=float, default=10.0, help="Control loop frequency in Hz")
    parser.add_argument("--goal-threshold", type=float, default=0.80, help="Cosine similarity threshold for goal arrival")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")

    # 1. Resolve compute device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Compute device: {device}")

    # 2. Load DINOv3 visual encoder
    logger.info("Initializing frozen DINOv3 visual encoder (vit_small_patch16_dinov3)...")
    encoder = FrozenDINOv3(device=str(device))

    # 3. Encode goal image
    goal_path = Path(args.goal_image)
    if not goal_path.exists():
        logger.error(f"Goal image not found: {goal_path}")
        return
    logger.info(f"Encoding goal image: {goal_path}")
    goal_rgb = np.array(PILImage.open(goal_path).convert("RGB"))
    goal_phi = encoder.encode_frame(goal_rgb)  # [384] tensor on device
    goal_phi_batch = goal_phi.unsqueeze(0)     # [1, 384]
    logger.info(f"Goal embedding ready. Shape: {goal_phi.shape}, L2-norm: {torch.norm(goal_phi).item():.3f}")

    # 4. Load Actor network
    state_dim = 4 * 384   # 4 consecutive 384-D frame embeddings = 1536
    goal_dim = 384
    action_dim = 3

    actor = Actor(state_dim, goal_dim, action_dim).to(device)
    logger.info(f"Loading actor weights from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "actor" in ckpt:
        actor.load_state_dict(ckpt["actor"])
    else:
        actor.load_state_dict(ckpt)
    actor.eval()
    logger.info("Actor network initialized in eval mode.")

    # 5. Initialize Action Normalizer
    normalizer = ActionNormalizer(device=str(device))
    speed_scale = max(0.05, min(1.0, args.speed_scale))
    logger.info(f"Speed scaling set to: {speed_scale * 100:.0f}% (Forward max: {1.8 * speed_scale:.2f} m/s)")

    # 6. Initialize ROS 2
    rclpy.init()
    dry_run = not args.arm
    control_period = 1.0 / args.control_freq

    try:
        # Connect to robot, camera, LiDAR
        robot = ROS2RangerMiniV3(cmd_vel_topic=args.cmd_topic, dry_run=dry_run)
        camera = ROS2Camera()
        lidar = ROS2LiDAR()
        safety = SafetyMonitor(robot, camera, lidar)

        # 4-frame FIFO buffer for state construction [t-3, t-2, t-1, t0]
        phi_buffer = deque(maxlen=4)

        logger.info("Awaiting sensor initialization (3 seconds)...")
        for _ in range(30):
            rclpy.spin_once(robot, timeout_sec=0.1)
            rclpy.spin_once(camera, timeout_sec=0.1)
            rclpy.spin_once(lidar, timeout_sec=0.1)

        mode_str = "ARMED (Physical Motion Enabled)" if not dry_run else "DRY-RUN (Safe Simulation / Wheels Raised)"
        logger.info("=" * 65)
        logger.info(f"MINAV POLICY DEPLOYMENT: {mode_str}")
        logger.info(f"  Command Topic : {args.cmd_topic}")
        logger.info(f"  Control Rate  : {args.control_freq:.1f} Hz (Period: {control_period * 1000:.1f} ms)")
        logger.info(f"  Speed Scale   : {speed_scale:.2f}")
        logger.info("=" * 65)

        if not dry_run:
            if not safety.arm():
                logger.error("SAFETY ARMING FAILED! One or more sensor streams are missing or stale.")
                return
            logger.info("Safety monitor ARMED. Commencing closed-loop navigation...")

        step = 0
        while step < args.max_steps and rclpy.ok():
            loop_start = time.time()

            # Spin ROS nodes
            rclpy.spin_once(robot, timeout_sec=0.005)
            rclpy.spin_once(camera, timeout_sec=0.005)
            rclpy.spin_once(lidar, timeout_sec=0.005)

            # Safety Watchdog Check
            if not dry_run and not safety.check():
                logger.error("SAFETY WATCHDOG VIOLATION! Halting robot immediately.")
                robot.emergency_stop()
                break

            # Acquire camera frame
            rgb = camera.get_frame()
            if rgb is None:
                time.sleep(0.005)
                continue

            # In-memory DINOv3 feature encoding (Zero Disk I/O)
            phi_tensor = encoder.encode_frame(rgb)  # [384] tensor on device
            phi_buffer.append(phi_tensor)

            # Warm-up: Wait for 4 frames to fill FIFO buffer
            if len(phi_buffer) < 4:
                logger.info(f"Buffering initial temporal state: {len(phi_buffer)}/4 frames...")
                time.sleep(control_period)
                continue

            # Construct chronological 4-frame state: [phi_{t-3}, phi_{t-2}, phi_{t-1}, phi_{t0}] -> [1, 1536]
            state = torch.cat(list(phi_buffer), dim=0).unsqueeze(0)

            # Policy forward pass -> normalized action in [-1, 1]^3
            with torch.no_grad():
                action_norm = actor(state, goal_phi_batch)  # [1, 3]

            # Inverse affine denormalization to physical velocities
            action_phys = normalizer.denormalize(action_norm.squeeze(0))
            vx, vy, omega = action_phys.cpu().numpy()

            # Cosine similarity to goal (both unit L2 normalized)
            sim = float(torch.dot(phi_tensor, goal_phi).item())

            # Goal arrival and stopping behavior
            if sim >= args.goal_threshold:
                logger.info(f"🎯 GOAL REACHED! Final Cosine Similarity: {sim:.3f} >= {args.goal_threshold:.2f}")
                robot.send_command(DriveCommand(0.0, 0.0, 0.0))
                break

            # Smooth deceleration when entering goal approach zone [0.70, goal_threshold]
            decel_factor = 1.0
            if sim >= 0.70:
                # Linearly ramp down forward speed as similarity increases
                decel_factor = max(0.15, (args.goal_threshold - sim) / (args.goal_threshold - 0.70))

            # Apply safe scaling to physical velocities
            vx_cmd = float(vx) * speed_scale * decel_factor
            vy_cmd = float(vy) * speed_scale
            omega_cmd = float(omega) * speed_scale

            # Send command to robot (/minav/cmd_vel)
            cmd = DriveCommand(v_linear=vx_cmd, v_lateral=vy_cmd, v_angular=omega_cmd)
            robot.send_command(cmd)

            if step % 10 == 0:
                logger.info(
                    f"Step {step:4d} | sim: {sim:.3f} | "
                    f"cmd: [vx={vx_cmd:+.2f}, vy={vy_cmd:+.2f}, ω={omega_cmd:+.2f}] | "
                    f"raw_norm: [{action_norm[0,0]:+.2f}, {action_norm[0,1]:+.2f}, {action_norm[0,2]:+.2f}]"
                )

            step += 1

            # Precision 10 Hz rate control
            elapsed = time.time() - loop_start
            if elapsed < control_period:
                time.sleep(control_period - elapsed)

    except KeyboardInterrupt:
        logger.warning("Deployment interrupted by operator.")
    finally:
        logger.info("Shutting down deployment node. Sending emergency stop...")
        robot.emergency_stop()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
