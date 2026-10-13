#!/usr/bin/env python3
"""
realworld/scripts/deploy_robot_gateway.py — Robot-Side CPU Control Gateway.
==========================================================================
Runs on the Ranger Mini V3 onboard CPU.
Receives AI actions from the Jetson via Ethernet LAN, validates freshness,
denormalizes actions to physical velocities, monitors safety watchdogs,
and routes commands to the LiDAR Safety Gate.

Pipeline:
  1. Subscribes to /minav/policy_action (from Jetson)
  2. Latency & Network Watchdog: Discards actions older than 400ms or halts if stream drops
  3. Denormalizes [-1, 1]^3 to physical velocities (ActionNormalizer)
  4. Applies speed scaling:
       - vx: --speed-scale (linear ceiling) + deceleration near goal
       - vy: --speed-scale
       - omega: --angular-scale (full rotational responsiveness)
  5. Continuous 10 Hz streaming to /minav/cmd_vel -> LidarSafetyGate -> /cmd_vel
"""

import sys
import argparse
import logging
import time
import threading
from pathlib import Path

import numpy as np

# Project root resolution
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, Twist
from std_msgs.msg import Bool

from realworld.src.ros2_robot import ROS2RangerMiniV3
from realworld.src.ros2_camera import ROS2Camera
from realworld.src.ros2_lidar import ROS2LiDAR
from realworld.src.safety_monitor import SafetyMonitor
from shared.drive_command import DriveCommand
from shared.action_normalizer import ActionNormalizer

logger = logging.getLogger("robot_gateway")


def parse_args():
    parser = argparse.ArgumentParser(description="Robot-Side Control Gateway for Jetson AI Policy")
    parser.add_argument("--action-topic", type=str, default="/minav/policy_action",
                        help="Topic from Jetson containing normalized actions (default: /minav/policy_action)")
    parser.add_argument("--cmd-topic", type=str, default="/minav/cmd_vel",
                        help="Topic to publish policy commands (default: /minav/cmd_vel routed through safety gate)")
    parser.add_argument("--arm", action="store_true", help="ARM robot for real physical motion (default: DRY-RUN)")
    parser.add_argument("--speed-scale", type=float, default=0.25,
                        help="Linear speed scaling factor (0.0 to 1.0) for supervised trials (default: 0.25)")
    parser.add_argument("--angular-scale", type=float, default=0.80,
                        help="Angular speed scaling factor (0.0 to 1.0, default: 0.80)")
    parser.add_argument("--control-freq", type=float, default=10.0, help="Command publishing frequency in Hz")
    parser.add_argument("--goal-threshold", type=float, default=0.80, help="Cosine similarity threshold for goal arrival")
    parser.add_argument("--max-action-age", type=float, default=0.40,
                        help="Maximum allowed action message age in seconds before watchdog triggers (default: 0.40s)")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")

    normalizer = ActionNormalizer(device="cpu")
    speed_scale = max(0.05, min(1.0, args.speed_scale))
    angular_scale = max(0.05, min(1.0, args.angular_scale))

    logger.info("=" * 65)
    logger.info("MINAV DISTRIBUTED ROBOT GATEWAY (CPU Side)")
    logger.info(f"  Speed Scaling : Linear={speed_scale * 100:.0f}% ({1.8 * speed_scale:.2f} m/s) | Angular={angular_scale * 100:.0f}% ({1.5 * angular_scale:.2f} rad/s)")
    logger.info(f"  Action Topic  : {args.action_topic}")
    logger.info(f"  Command Topic : {args.cmd_topic}")
    logger.info("=" * 65)

    rclpy.init()
    dry_run = not args.arm
    control_period = 1.0 / args.control_freq

    # Robot interfaces
    robot = ROS2RangerMiniV3(cmd_vel_topic=args.cmd_topic, dry_run=dry_run)
    camera = ROS2Camera()
    lidar = ROS2LiDAR()
    safety = SafetyMonitor(robot, camera, lidar)

    # State tracking
    current_cmd = DriveCommand(0.0, 0.0, 0.0)
    cmd_lock = threading.Lock()
    last_action_recv_time = 0.0
    latest_sim = 0.0
    latest_norm_action = np.zeros(3)
    is_safety_blocked = False

    # Subscriber: Jetson Policy Actions
    def action_callback(msg: TwistStamped):
        nonlocal last_action_recv_time, latest_sim, latest_norm_action, current_cmd
        now = time.time()
        last_action_recv_time = now

        norm_vx = float(msg.twist.linear.x)
        norm_vy = float(msg.twist.linear.y)
        norm_omega = float(msg.twist.angular.z)
        sim = float(msg.twist.linear.z)  # cosine similarity embedded in linear.z

        latest_sim = sim
        latest_norm_action = np.array([norm_vx, norm_vy, norm_omega])

        # Goal arrival check
        if sim >= args.goal_threshold:
            logger.info(f"🎯 GOAL REACHED! Final Cosine Similarity: {sim:.3f} >= {args.goal_threshold:.2f}")
            with cmd_lock:
                current_cmd = DriveCommand(0.0, 0.0, 0.0)
            return

        # Denormalize to physical velocities
        phys_action = normalizer.denormalize(latest_norm_action)
        vx, vy, omega = [float(x) for x in phys_action]

        # Deceleration ramp near goal
        decel_factor = 1.0
        if sim >= 0.70:
            decel_factor = max(0.15, (args.goal_threshold - sim) / (args.goal_threshold - 0.70))

        vx_cmd = vx * speed_scale * decel_factor
        vy_cmd = vy * speed_scale
        omega_cmd = omega * angular_scale

        with cmd_lock:
            current_cmd = DriveCommand(v_linear=vx_cmd, v_lateral=vy_cmd, v_angular=omega_cmd)

    action_sub = robot.create_subscription(TwistStamped, args.action_topic, action_callback, 10)

    # Subscriber: LiDAR Safety Gate Block Status
    def safety_blocked_callback(msg: Bool):
        nonlocal is_safety_blocked
        is_safety_blocked = msg.data
    safety_sub = robot.create_subscription(Bool, "/safety_gate/blocked", safety_blocked_callback, 10)

    # Continuous 10 Hz Command Streamer
    stream_active = True
    def streamer_loop():
        while stream_active and rclpy.ok():
            now = time.time()
            time_since_action = now - last_action_recv_time

            with cmd_lock:
                # Watchdog: If no action from Jetson in > max_action_age seconds, zero output
                if last_action_recv_time == 0.0 or time_since_action > args.max_action_age:
                    cmd_to_send = DriveCommand(0.0, 0.0, 0.0)
                else:
                    cmd_to_send = current_cmd

            if not dry_run:
                robot.send_command(cmd_to_send)
            time.sleep(control_period)

    streamer_thread = threading.Thread(target=streamer_loop, daemon=True)
    streamer_thread.start()

    logger.info("Awaiting sensor feeds and Jetson policy stream...")
    for _ in range(30):
        rclpy.spin_once(robot, timeout_sec=0.1)
        rclpy.spin_once(camera, timeout_sec=0.1)
        rclpy.spin_once(lidar, timeout_sec=0.1)

    if not dry_run:
        if not safety.arm():
            logger.error("SAFETY ARMING FAILED! Ensure camera, odom, and LiDAR streams are live.")
            stream_active = False
            return
        logger.info("Safety monitor ARMED. Commencing navigation gateway...")

    step = 0
    try:
        while rclpy.ok():
            rclpy.spin_once(robot, timeout_sec=0.01)
            rclpy.spin_once(camera, timeout_sec=0.01)
            rclpy.spin_once(lidar, timeout_sec=0.01)

            # Safety Watchdog
            if not dry_run and not safety.check():
                logger.error("SAFETY WATCHDOG VIOLATION! Halting robot.")
                robot.emergency_stop()
                break

            now = time.time()
            action_age = now - last_action_recv_time

            if latest_sim >= args.goal_threshold and last_action_recv_time > 0:
                logger.info("Arrived at goal! Gateway complete.")
                break

            if step % 10 == 0:
                with cmd_lock:
                    cmd = current_cmd
                stale_tag = " [⚠️ STALE JETSON STREAM]" if (last_action_recv_time > 0 and action_age > args.max_action_age) else ""
                blocked_tag = " [🛑 SAFETY BLOCKED]" if is_safety_blocked else ""
                logger.info(
                    f"Step {step:4d} | Age: {action_age * 1000:3.0f}ms | sim: {latest_sim:.3f} | "
                    f"cmd: [vx={cmd.v_linear:+.2f}, vy={cmd.v_lateral:+.2f}, ω={cmd.v_angular:+.2f}] | "
                    f"norm: [{latest_norm_action[0]:+.2f}, {latest_norm_action[1]:+.2f}, {latest_norm_action[2]:+.2f}]"
                    f"{stale_tag}{blocked_tag}"
                )

            step += 1
            time.sleep(control_period)

    except KeyboardInterrupt:
        logger.warning("Gateway interrupted by operator.")
    finally:
        logger.info("Shutting down gateway. Sending emergency stop...")
        stream_active = False
        with cmd_lock:
            current_cmd = DriveCommand(0.0, 0.0, 0.0)
        robot.emergency_stop()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
