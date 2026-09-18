"""
realworld/src/exploration_runner.py — MINav-faithful pink-noise exploration loop.
==================================================================================
Orchestrates the real-world exploration on the Ranger Mini V3:
  1. Get RGB frame from RealSense camera.
  2. Sample smoothed pink-uniform-noise action (2 Hz + LERP).
  3. Apply independent LiDAR safety gate (block-only, no trajectory shaping).
  4. Send command via ROS 2 /cmd_vel (or dry-run).
  5. Record step: commanded action, executed action, odom, safety flag.

MINav exploration policy (arXiv:2603.26441):
  a_t = a_min + (a_max - a_min) · Φ(x_t / σ)

The exploration policy is PURE correlated noise — no depth input, no visit
grid, no primitives, no collision logic. LiDAR is an independent safety layer
that can only BLOCK unsafe commands (zero them out), never reshape the trajectory.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from shared.drive_command import DriveCommand
from shared.pink_uniform_policy import PinkUniformNoisePolicy, PolicyConfig

logger = logging.getLogger(__name__)


class ExplorationRunner:
    """Real-world MINav exploration loop orchestrator.

    Parameters
    ----------
    robot : ROS2RangerMiniV3
        Robot interface for commands and odometry.
    camera : ROS2Camera
        Camera interface for RGB frames.
    lidar : ROS2LiDAR
        LiDAR interface for depth images (safety gate only).
    safety : SafetyMonitor
        Sensor liveness watchdog.
    recorder : DataRecorder
        Data recorder for saving exploration data.
    policy_config : PolicyConfig
        Configuration for the PinkUniformNoisePolicy.
    duration_minutes : float
        Exploration duration in minutes. Default: 120 (MINav paper).
    safety_gate_min_depth : float
        Minimum obstacle distance (metres) before the safety gate
        blocks the command. Default: 0.5.
    safety_gate_min_pixels : int
        Minimum near-range pixels per sector to confirm a real
        obstacle (filters sparse LiDAR noise). Default: 75.
    """

    def __init__(
        self,
        robot,
        camera,
        lidar,
        safety,
        recorder,
        policy_config: PolicyConfig,
        duration_minutes: float = 120.0,
        safety_gate_min_depth: float = 0.5,
        safety_gate_min_pixels: int = 75,
    ):
        self._robot = robot
        self._camera = camera
        self._lidar = lidar
        self._safety = safety
        self._recorder = recorder

        self._control_freq = policy_config.control_freq_hz
        self._control_period = 1.0 / policy_config.control_freq_hz
        self._duration_minutes = duration_minutes
        self._total_steps = int(duration_minutes * 60 * policy_config.control_freq_hz)

        # MINav-faithful exploration policy (pure pink uniform noise)
        self._policy = PinkUniformNoisePolicy(policy_config)

        # Independent LiDAR safety gate parameters
        self._gate_min_depth = safety_gate_min_depth
        self._gate_min_pixels = safety_gate_min_pixels
        self._max_range = 10.0

        # Fallback depth image (all far away = no obstacles)
        self._fallback_depth = np.full((60, 640), self._max_range, dtype=np.float32)

        # Safety gate statistics
        self._gate_block_count = 0

    def _safety_gate(self, cmd: DriveCommand, depth: np.ndarray) -> tuple[DriveCommand, bool]:
        """Independent LiDAR safety gate — directional blocking.

        Checks sectors for near-range obstacles and clamps specific
        directional velocities (forward, left, right) to prevent collisions,
        while strictly preserving rotational velocity to allow escapes.

        Parameters
        ----------
        cmd : DriveCommand
            The commanded action from the exploration policy.
        depth : np.ndarray
            LiDAR-projected depth image (H, W) float32.

        Returns
        -------
        tuple[DriveCommand, bool]
            (executed_cmd, was_blocked)
        """
        h, w = depth.shape
        obstacle_threshold = self._gate_min_depth

        # Sectors mapped from atan2 in ros2_lidar.py
        right_sector = depth[:, 0 : w // 3]                 # -180 to -60 deg
        front_sector = depth[:, w // 3 : 2 * w // 3]        # -60 to +60 deg
        left_sector  = depth[:, 2 * w // 3 : w]             # +60 to +180 deg
        
        safe_vx = cmd.v_linear
        safe_vy = cmd.v_lateral
        safe_wz = cmd.v_angular
        blocked = False

        # 1. Front obstacle -> block forward motion
        if (front_sector < obstacle_threshold).sum() >= self._gate_min_pixels:
            if safe_vx > 0:
                safe_vx = 0.0
                blocked = True

        # 2. Right obstacle -> block rightward crab-walk (vy < 0)
        if (right_sector < obstacle_threshold).sum() >= self._gate_min_pixels:
            if safe_vy < 0:
                safe_vy = 0.0
                blocked = True
                
        # 3. Left obstacle -> block leftward crab-walk (vy > 0)
        if (left_sector < obstacle_threshold).sum() >= self._gate_min_pixels:
            if safe_vy > 0:
                safe_vy = 0.0
                blocked = True

        if blocked:
            self._gate_block_count += 1
            
        return DriveCommand(safe_vx, safe_vy, safe_wz), blocked

    def run(self) -> bool:
        """Execute the exploration loop.

        Returns True on successful completion, False on failure/abort.
        """
        logger.info("=" * 60)
        logger.info("EXPLORATION RUNNER — MINav Pink Uniform Noise")
        logger.info("=" * 60)
        logger.info(f"  Duration     : {self._duration_minutes} minutes")
        logger.info(f"  Total steps  : {self._total_steps}")
        logger.info(f"  Control freq : {self._control_freq} Hz")
        logger.info(f"  Policy freq  : {self._policy.config.policy_freq_hz} Hz")
        logger.info(f"  Smoothing α  : {self._policy.config.smoothing_alpha}")
        logger.info(f"  vx range     : {self._policy.config.vx_range}")
        logger.info(f"  vy range     : {self._policy.config.vy_range}")
        logger.info(f"  ωz range     : {self._policy.config.wz_range}")
        logger.info(f"  Safety gate  : {self._gate_min_depth}m / {self._gate_min_pixels}px")
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
        dts = []
        last_loop_start = start_time

        try:
            for step in range(self._total_steps):
                loop_start = time.time()
                if step > 0:
                    dts.append(loop_start - last_loop_start)
                last_loop_start = loop_start

                # 1. Safety watchdog check (sensor liveness)
                if not self._robot.dry_run and not self._safety.check():
                    logger.error(f"Safety violation at step {step}. Aborting.")
                    break

                # 2. Get camera frame
                rgb = self._camera.get_frame()
                if rgb is None:
                    logger.warning(f"Step {step}: no camera frame, skipping")
                    time.sleep(self._control_period)
                    continue

                # 3. Get odometry (for logging only — NOT fed to policy)
                pose = self._robot.get_pose()

                # 4. Sample action from pink-uniform-noise policy
                #    (internally handles 2 Hz sampling + LERP smoothing)
                cmd = self._policy.get_action()

                # 5. Independent LiDAR safety gate (block-only)
                depth = self._lidar.get_depth_image()
                if depth is None:
                    depth = self._fallback_depth
                executed_cmd, safety_blocked = self._safety_gate(cmd, depth)

                # 6. Send the (possibly blocked) command to the robot
                self._robot.send_command(executed_cmd)

                # 7. Record data: commanded, executed, odom, safety flag
                self._recorder.record_step(
                    rgb_frame=rgb,
                    cmd=cmd,
                    pos_x=pose.x,
                    pos_y=pose.y,
                    yaw=pose.yaw,
                    wall_time=time.time(),
                    executed_cmd=executed_cmd,
                    safety_blocked=safety_blocked,
                )

                step_count += 1

                # Log progress
                if step_count % 100 == 0:
                    elapsed = time.time() - start_time
                    pct = 100.0 * step_count / self._total_steps
                    gate_pct = (
                        100.0 * self._gate_block_count / step_count
                        if step_count > 0 else 0.0
                    )
                    logger.info(
                        f"[{step_count:06d}/{self._total_steps}] "
                        f"{pct:.1f}% | "
                        f"vx={cmd.v_linear:+.2f} vy={cmd.v_lateral:+.2f} "
                        f"ω={cmd.v_angular:+.2f} | "
                        f"pos=({pose.x:.2f}, {pose.y:.2f}) | "
                        f"gate={gate_pct:.1f}% | "
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

            # Calculate timing stats
            if dts:
                mean_dt = np.mean(dts)
                std_dt = np.std(dts)
                min_dt = np.min(dts)
                max_dt = np.max(dts)
                actual_publish_rate = 1.0 / mean_dt if mean_dt > 0 else 0.0
            else:
                mean_dt = std_dt = min_dt = max_dt = actual_publish_rate = 0.0

            logger.info("=" * 60)
            logger.info("EXPLORATION COMPLETE")
            logger.info(f"  Steps recorded : {step_count}")
            logger.info(f"  Elapsed        : {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)")
            logger.info(f"  Safety blocks  : {self._gate_block_count}")
            logger.info(f"  E-stops        : {self._safety.estop_count}")
            logger.info("  --- Timing ---")
            logger.info(f"  Target Rate    : {self._control_freq:.1f} Hz")
            logger.info(f"  Actual Rate    : {actual_publish_rate:.1f} Hz")
            logger.info(f"  Mean dt        : {mean_dt*1000:.1f} ms")
            logger.info(f"  Std dt         : {std_dt*1000:.1f} ms")
            logger.info(f"  Min dt         : {min_dt*1000:.1f} ms")
            logger.info(f"  Max dt         : {max_dt*1000:.1f} ms")
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
