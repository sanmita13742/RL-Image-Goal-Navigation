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
import threading

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
# Project root resolution
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

MUJOCO_ROOT = ROOT / "simulation" / "ranger_mini_v3"
sys.path.insert(0, str(MUJOCO_ROOT))

try:
    import rclpy
    from std_msgs.msg import Bool
    from realworld.src.ros2_robot import ROS2RangerMiniV3
    from realworld.src.ros2_camera import ROS2Camera
    from realworld.src.ros2_lidar import ROS2LiDAR
    from realworld.src.safety_monitor import SafetyMonitor
except ImportError:
    rclpy = None
    Bool = None
    ROS2RangerMiniV3 = None
    ROS2Camera = None
    ROS2LiDAR = None
    SafetyMonitor = None

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
                        help="Linear speed scaling factor (0.0 to 1.0) for supervised trials (default: 0.30)")
    parser.add_argument("--angular-scale", type=float, default=None,
                        help="Angular speed scaling factor (0.0 to 1.0, default: matches --speed-scale, capped at 2x --speed-scale).")
    parser.add_argument("--stream", action="store_true", default=False,
                        help="Enable continuous background 10 Hz command streaming (default: False, synchronous per inference step)")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum control steps")
    parser.add_argument("--control-freq", type=float, default=10.0, help="Control loop frequency in Hz")
    parser.add_argument("--goal-threshold", type=float, default=0.70,
                        help="Cosine similarity threshold for goal arrival (default: 0.70)")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")

    # 1. Resolve compute device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested via --device cuda, but CUDA is not available or Torch was compiled without CUDA. Falling back to CPU.")
        device = torch.device("cpu")
    elif args.device == "auto":
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
    angular_scale = max(0.05, min(1.0, args.angular_scale if args.angular_scale is not None else speed_scale))
    max_angular = 2.0 * speed_scale
    if angular_scale > max_angular:
        logger.warning(
            f"Requested angular_scale ({angular_scale:.2f}) is > 2x speed_scale ({speed_scale:.2f}). "
            f"Capping angular_scale to {max_angular:.2f} to prevent severe rotational distortion."
        )
        angular_scale = max_angular
    logger.info(
        f"Speed scaling: Linear={speed_scale * 100:.0f}% (Forward max: {1.8 * speed_scale:.2f} m/s) | "
        f"Angular={angular_scale * 100:.0f}% (Turn max: {1.5 * angular_scale:.2f} rad/s) | "
        f"Mode={'Continuous 10Hz Streamer' if args.stream else 'Synchronous (Per-inference direct)'}"
    )

    # 6. Initialize ROS 2
    if rclpy is None:
        logger.error("ROS 2 (rclpy/std_msgs) is not available. Ensure your ROS 2 environment is sourced.")
        return
    rclpy.init()
    dry_run = not args.arm
    control_period = 1.0 / args.control_freq

    # Command streaming state (used if --stream is enabled)
    current_cmd = DriveCommand(0.0, 0.0, 0.0)
    cmd_lock = threading.Lock()
    last_cmd_time = time.time()
    stream_active = True

    try:
        # Connect to robot, camera, LiDAR
        robot = ROS2RangerMiniV3(cmd_vel_topic=args.cmd_topic, dry_run=dry_run)
        camera = ROS2Camera()
        lidar = ROS2LiDAR()
        safety = SafetyMonitor(robot, camera, lidar)

        # Monitor Safety Gate block status
        is_safety_blocked = False
        def safety_gate_callback(msg: Bool):
            nonlocal is_safety_blocked
            is_safety_blocked = msg.data
        safety_sub = robot.create_subscription(Bool, "/safety_gate/blocked", safety_gate_callback, 10)

        # Launch background command streamer thread only if --stream requested
        if args.stream:
            def streamer_loop():
                stream_period = 1.0 / args.control_freq
                while stream_active and rclpy.ok():
                    now = time.time()
                    with cmd_lock:
                        if is_safety_blocked:
                            cmd_to_send = DriveCommand(0.0, 0.0, 0.0)
                        elif now - last_cmd_time > 0.4:
                            cmd_to_send = DriveCommand(0.0, 0.0, 0.0)
                        else:
                            cmd_to_send = current_cmd
                    if not dry_run:
                        robot.send_command(cmd_to_send)
                    time.sleep(stream_period)

            streamer_thread = threading.Thread(target=streamer_loop, daemon=True)
            streamer_thread.start()

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
        logger.info(f"  Speed Scale   : Linear={speed_scale:.2f}, Angular={angular_scale:.2f}")
        logger.info("=" * 65)

        if not dry_run:
            if not safety.arm():
                logger.error("SAFETY ARMING FAILED! One or more sensor streams are missing or stale.")
                return
            logger.info("Safety monitor ARMED. Commencing closed-loop navigation...")

        was_safety_blocked = False
        blocked_ticks = 0
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
                with cmd_lock:
                    current_cmd = DriveCommand(0.0, 0.0, 0.0)
                robot.emergency_stop()
                break


            # Acquire camera frame
            rgb, frame_time = camera.get_frame()
            if rgb is None:
                time.sleep(0.005)
                continue

            # In-memory DINOv3 feature encoding (Zero Disk I/O)
            phi_tensor = encoder.encode_frame(rgb)  # [384] tensor on device
            phi_buffer.append(phi_tensor)

            # Warm-up / Re-buffering: Wait for 4 frames to fill FIFO buffer
            if len(phi_buffer) < 4:
                logger.info(f"Buffering temporal state: {len(phi_buffer)}/4 frames...")
                stop_cmd = DriveCommand(0.0, 0.0, 0.0)
                if not args.stream:
                    robot.send_command(stop_cmd)
                else:
                    with cmd_lock:
                        current_cmd = stop_cmd
                        last_cmd_time = time.time()
                time.sleep(control_period)
                continue

            # Construct chronological 4-frame state: [phi_{t-3}, phi_{t-2}, phi_{t-1}, phi_{t0}] -> [1, 1536]
            state = torch.cat(list(phi_buffer), dim=0).unsqueeze(0)

            # Policy forward pass -> normalized action in [-1, 1]^3
            with torch.inference_mode():
                action_norm = actor(state, goal_phi_batch)  # [1, 3]

            # Inverse affine denormalization to physical velocities
            action_phys = normalizer.denormalize(action_norm.squeeze(0))
            vx, vy, omega = action_phys.cpu().numpy()

            # Cosine similarity to goal (both unit L2 normalized)
            sim = float(torch.dot(phi_tensor, goal_phi).item())

            # Goal arrival and stopping behavior
            if sim >= args.goal_threshold:
                logger.info(f"🎯 GOAL REACHED! Final Cosine Similarity: {sim:.3f} >= {args.goal_threshold:.2f}")
                with cmd_lock:
                    current_cmd = DriveCommand(0.0, 0.0, 0.0)
                robot.emergency_stop()
                break

            # Smooth deceleration when entering goal approach zone [goal_threshold - 0.10, goal_threshold]
            decel_factor = 1.0
            approach_zone_start = max(0.0, args.goal_threshold - 0.10)
            if sim >= approach_zone_start and args.goal_threshold > approach_zone_start:
                # Linearly ramp down forward speed as similarity increases
                decel_factor = max(0.15, (args.goal_threshold - sim) / (args.goal_threshold - approach_zone_start))

            # Apply safe scaling to physical velocities
            vx_cmd = float(vx) * speed_scale * decel_factor
            vy_cmd = float(vy) * speed_scale * decel_factor
            omega_cmd = float(omega) * angular_scale

            # If safety gate reported block, suppress forward and lateral translation
            if is_safety_blocked:
                blocked_ticks += 1
                vx_cmd = 0.0
                vy_cmd = 0.0

                # Escaped Steering Window:
                # Allow turning for up to 15 ticks (1.5s, ~25 deg) so policy can pivot away
                # from the obstacle toward the goal corridor, matching exploration behavior.
                max_pivot_ticks = 15
                if blocked_ticks <= max_pivot_ticks:
                    omega_cmd = float(omega) * angular_scale
                else:
                    # Timeout reached: halt rotation to prevent 144-degree spin away from goal
                    omega_cmd = 0.0
            else:
                blocked_ticks = 0

            # Send command to robot (/minav/cmd_vel)
            if not args.stream:
                cmd = DriveCommand(v_linear=vx_cmd, v_lateral=vy_cmd, v_angular=omega_cmd)
                robot.send_command(cmd)
            else:
                with cmd_lock:
                    current_cmd = DriveCommand(v_linear=vx_cmd, v_lateral=vy_cmd, v_angular=omega_cmd)
                    last_cmd_time = time.time()

            if not is_safety_blocked:
                if step % 10 == 0:
                    logger.info(
                        f"Step {step:4d} | sim: {sim:.3f} | "
                        f"cmd: [vx={vx_cmd:+.2f}, vy={vy_cmd:+.2f}, ω={omega_cmd:+.2f}] | "
                        f"raw_norm: [{action_norm[0,0]:+.2f}, {action_norm[0,1]:+.2f}, {action_norm[0,2]:+.2f}]"
                    )
                step += 1
            else:
                if blocked_ticks % 10 == 1:
                    status = f"pivoting ({blocked_ticks}/{max_pivot_ticks})" if blocked_ticks <= max_pivot_ticks else "held stationary"
                    logger.warning(
                        f"Step {step:4d} | ⚠️ SAFETY BLOCKED ({blocked_ticks} ticks, {status}) | "
                        f"sim: {sim:.3f} | cmd: [vx=0, vy=0, ω={omega_cmd:+.2f}]"
                    )

            # Precision control rate timing
            elapsed = time.time() - loop_start
            if elapsed < control_period:
                time.sleep(control_period - elapsed)

    except KeyboardInterrupt:
        logger.warning("Deployment interrupted by operator.")
    finally:
        logger.info("Shutting down deployment node. Sending emergency stop...")
        stream_active = False
        with cmd_lock:
            current_cmd = DriveCommand(0.0, 0.0, 0.0)
        robot.emergency_stop()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
