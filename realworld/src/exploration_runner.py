"""
realworld/src/exploration_runner.py — Real-world pink-noise exploration loop.
=============================================================================
Orchestrates the 10 Hz control loop on the real Ranger Mini V3:
  1. Get RGB frame from RealSense camera.
  2. Get LiDAR depth image for collision avoidance.
  3. Get (x, y, yaw) from odometry.
  4. Compute pink-noise action via PrimitiveExplorationPolicy.
  5. Send command via ROS 2 /cmd_vel (or dry-run).
  6. Record step to MuJoCo-compatible dataset.

The runner operates in three modes:
  - DRY-RUN (default): Everything runs except /cmd_vel is not published.
  - ARMED: Full exploration with robot motion.
  - SMOKE: Short 30-second dry-run with full validation.
"""

from __future__ import annotations

import logging
import time
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from shared.drive_command import DriveCommand
from shared.exploration_base import PrimitiveExplorationPolicy

logger = logging.getLogger(__name__)


class ExplorationRunner:
    """Real-world exploration loop orchestrator.

    Parameters
    ----------
    robot : ROS2RangerMiniV3
        Robot interface for commands and odometry.
    camera : ROS2Camera
        Camera interface for RGB frames.
    lidar : ROS2LiDAR
        LiDAR interface for depth images.
    safety : SafetyMonitor
        Safety watchdog.
    recorder : DataRecorder
        Data recorder for saving exploration data.
    control_freq_hz : float
        Control loop frequency. Default: 10.0 (MINav paper).
    duration_minutes : float
        Exploration duration in minutes. Default: 120 (MINav paper).
    beta : int
        Pink noise spectral exponent. Default: 1 (MINav paper).
    """

    def __init__(
        self,
        robot,
        camera,
        lidar,
        safety,
        recorder,
        control_freq_hz: float = 10.0,
        duration_minutes: float = 120.0,
        beta: int = 1,
    ):
        self._robot = robot
        self._camera = camera
        self._lidar = lidar
        self._safety = safety
        self._recorder = recorder

        self._control_freq = control_freq_hz
        self._control_period = 1.0 / control_freq_hz
        self._duration_minutes = duration_minutes
        self._total_steps = int(duration_minutes * 60 * control_freq_hz)

        # Initialize exploration policy (uses shared module, no MuJoCo)
        self._policy = PrimitiveExplorationPolicy(control_freq_hz, beta=beta)

        # Fallback depth image (all far away = no obstacles)
        self._fallback_depth = np.full((60, 640), 10.0, dtype=np.float32)
        self._max_range = 10.0
        # Minimum near-range pixels per image third to count as a real obstacle
        self._min_obstacle_pixels = 50

    def _filter_sparse_depth(self, depth: np.ndarray) -> np.ndarray:
        """Filter sparse near-range noise from the LiDAR depth image.

        With only ~500 points in a 38400-pixel image, isolated near-range
        pixels are noise (ground reflections, chassis returns). Only keep
        near-range values in each image third if enough pixels support them.

        The collision avoidance triggers at < 0.6m, so we count pixels
        below 1.0m (with margin) and require at least min_obstacle_pixels
        to confirm a real obstacle.
        """
        filtered = depth.copy()
        h, w = filtered.shape
        # Count pixels in the collision-relevant range (< 1.0m)
        obstacle_threshold = 1.0

        for start_col, end_col in [(0, w // 3), (w // 3, 2 * w // 3), (2 * w // 3, w)]:
            region = filtered[:, start_col:end_col]
            near_count = (region < obstacle_threshold).sum()
            if near_count < self._min_obstacle_pixels:
                # Not enough evidence of a real obstacle — suppress noise
                region[region < obstacle_threshold] = self._max_range

        return filtered

    def run(self) -> bool:
        """Execute the exploration loop.

        Returns True on successful completion, False on failure/abort.
        """
        logger.info("=" * 60)
        logger.info("EXPLORATION RUNNER")
        logger.info("=" * 60)
        logger.info(f"  Duration     : {self._duration_minutes} minutes")
        logger.info(f"  Total steps  : {self._total_steps}")
        logger.info(f"  Control freq : {self._control_freq} Hz")
        logger.info(f"  Dry-run      : {self._robot.dry_run}")
        logger.info(f"  Armed        : {not self._robot.dry_run}")
        logger.info("=" * 60)

        # Open the data recorder
        self._recorder.open()

        # Wait for all sensors to be available (up to 10 seconds)
        if not self._wait_for_sensors(timeout=10.0):
            logger.error("Sensor initialization timed out. Aborting.")
            self._recorder.close()
            return False

        # If not in dry-run mode, arm the safety monitor
        if not self._robot.dry_run:
            if not self._safety.arm():
                logger.error("Failed to arm robot. Aborting.")
                self._recorder.close()
                return False

        start_time = time.time()
        step_count = 0

        try:
            for step in range(self._total_steps):
                loop_start = time.time()

                # 1. Safety check
                if not self._robot.dry_run and not self._safety.check():
                    logger.error(f"Safety violation at step {step}. Aborting.")
                    break

                # 2. Get camera frame
                rgb = self._camera.get_frame()
                if rgb is None:
                    logger.warning(f"Step {step}: no camera frame, skipping")
                    time.sleep(self._control_period)
                    continue

                # 3. Get LiDAR depth image
                depth = self._lidar.get_depth_image()
                if depth is None:
                    depth = self._fallback_depth  # No obstacles if LiDAR not ready
                else:
                    # Density filter: with a sparse 500-point LiDAR, isolated
                    # near-range pixels are noise. Only keep near-range readings
                    # in each image third if enough pixels support them.
                    depth = self._filter_sparse_depth(depth)

                # 4. Get odometry
                pose = self._robot.get_pose()

                # 5. Compute action
                cmd, prim = self._policy.get_action(
                    depth, pose.x, pose.y, pose.yaw
                )

                # 6. Send command
                self._robot.send_command(cmd)

                # 7. Record data
                self._recorder.record_step(
                    rgb_frame=rgb,
                    cmd=cmd,
                    pos_x=pose.x,
                    pos_y=pose.y,
                    yaw=pose.yaw,
                    wall_time=time.time(),
                )

                step_count += 1

                # Log progress
                if step_count % 100 == 0:
                    elapsed = time.time() - start_time
                    pct = 100.0 * step_count / self._total_steps
                    logger.info(
                        f"[{step_count:06d}/{self._total_steps}] "
                        f"{pct:.1f}% | {prim.name} | "
                        f"vx={cmd.v_linear:+.2f} vy={cmd.v_lateral:+.2f} "
                        f"ω={cmd.v_angular:+.2f} | "
                        f"pos=({pose.x:.2f}, {pose.y:.2f}) | "
                        f"elapsed={elapsed:.0f}s"
                    )

                # 8. Rate control — sleep for the remainder of the control period
                loop_elapsed = time.time() - loop_start
                sleep_time = self._control_period - loop_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif step_count % 500 == 0 and loop_elapsed > self._control_period * 1.5:
                    logger.warning(
                        f"Control loop overrun: {loop_elapsed*1000:.1f}ms "
                        f"(target: {self._control_period*1000:.1f}ms)"
                    )

        except KeyboardInterrupt:
            logger.warning("Exploration interrupted by user (Ctrl+C)")

        finally:
            # Always stop the robot and close the recorder
            self._robot.emergency_stop()
            self._safety.disarm(reason="exploration ended")

            total_elapsed = time.time() - start_time
            self._recorder.close(
                duration_minutes=self._duration_minutes,
                control_freq_hz=self._control_freq,
                total_steps_planned=self._total_steps,
            )

            logger.info("=" * 60)
            logger.info("EXPLORATION COMPLETE")
            logger.info(f"  Steps recorded : {step_count}")
            logger.info(f"  Elapsed        : {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)")
            logger.info(f"  E-stops        : {self._safety.estop_count}")
            logger.info("=" * 60)

        return step_count > 0

    def _wait_for_sensors(self, timeout: float = 10.0) -> bool:
        """Wait for camera, odom, and LiDAR to start publishing.

        Returns True if all sensors are live within timeout.
        """
        logger.info("Waiting for sensors...")
        start = time.time()

        while time.time() - start < timeout:
            # Background executor handles spinning; just poll cached values
            time.sleep(0.1)

            cam_ok = self._camera.has_frame
            odom_ok = self._robot.odom_received
            lidar_ok = self._lidar.has_scan

            if cam_ok and odom_ok and lidar_ok:
                logger.info("✓ All sensors live:")
                logger.info(f"  Camera : {self._camera.frame_count} frames")
                logger.info(f"  Odom   : received")
                logger.info(f"  LiDAR  : {self._lidar.scan_count} scans")
                return True

            status = []
            status.append(f"cam={'✓' if cam_ok else '✗'}")
            status.append(f"odom={'✓' if odom_ok else '✗'}")
            status.append(f"lidar={'✓' if lidar_ok else '✗'}")
            logger.debug(f"  Sensors: {', '.join(status)}")

        logger.error(
            f"Sensor timeout after {timeout}s: "
            f"cam={self._camera.has_frame}, "
            f"odom={self._robot.odom_received}, "
            f"lidar={self._lidar.has_scan}"
        )
        return False
