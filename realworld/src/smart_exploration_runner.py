"""
realworld/src/smart_exploration_runner.py — Enhanced LiDAR-Aware Exploration.
==============================================================================
Orchestrates intelligent exploration on the Ranger Mini V3 by combining:
  1. Underlying PinkUniformNoisePolicy (unmodified 1/f noise generators)
  2. LidarAwarePolicy (action shaping, speed zones, panoramic rotation, coverage nudge)
  3. Independent LiDAR safety gate (defense-in-depth hardware protection)
  4. 72-bin 1D LiDAR range scan vector extraction (Strategy 4 observation state)

Provides a 100% drop-in alternative to ExplorationRunner.
Switchable via CLI flag (--mode smart) or config (exploration.mode: smart).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from shared.drive_command import DriveCommand
from shared.pink_uniform_policy import PolicyConfig
from shared.lidar_aware_policy import LidarAwarePolicy, SmartPolicyConfig
from shared.lidar_scan_extractor import extract_scan_from_depth_image
from realworld.src.exploration_runner import ExplorationRunner

logger = logging.getLogger(__name__)


class SmartExplorationRunner(ExplorationRunner):
    """Enhanced exploration loop orchestrator with LiDAR awareness.

    Parameters
    ----------
    robot : ROS2RangerMiniV3
        Robot interface for commands and odometry.
    camera : ROS2Camera
        Camera interface for RGB frames.
    lidar : ROS2LiDAR
        LiDAR interface for depth images.
    safety : SafetyMonitor
        Sensor liveness watchdog.
    recorder : DataRecorder
        Data recorder for saving exploration data.
    policy_config : PolicyConfig
        Configuration for the underlying PinkUniformNoisePolicy.
    smart_config : SmartPolicyConfig, optional
        Configuration for intelligent exploration strategies.
    record_lidar_scan : bool
        Whether to extract and record 72-bin 1D LiDAR range vectors (Strategy 4).
    duration_minutes : float
        Exploration duration in minutes. Default: 120.
    safety_gate_min_depth : float
        Minimum obstacle distance (metres) before safety gate blocks. Default: 0.55.
    safety_gate_min_pixels : int
        Minimum near-range pixels to confirm obstacle. Default: 75.
    """

    def __init__(
        self,
        robot,
        camera,
        lidar,
        safety,
        recorder,
        policy_config: PolicyConfig,
        smart_config: Optional[SmartPolicyConfig] = None,
        record_lidar_scan: bool = True,
        duration_minutes: float = 120.0,
        safety_gate_min_depth: float = 0.55,
        safety_gate_min_pixels: int = 75,
    ):
        super().__init__(
            robot=robot,
            camera=camera,
            lidar=lidar,
            safety=safety,
            recorder=recorder,
            policy_config=policy_config,
            duration_minutes=duration_minutes,
            safety_gate_min_depth=safety_gate_min_depth,
            safety_gate_min_pixels=safety_gate_min_pixels,
        )

        self._smart_config = smart_config or SmartPolicyConfig()
        # Wrap the base PinkUniformNoisePolicy in LidarAwarePolicy
        self._smart_policy = LidarAwarePolicy(
            base_policy=self._policy,
            config=self._smart_config,
        )
        self._record_lidar = record_lidar_scan
        self._shaped_action_count = 0
        self._panoramic_step_count = 0

    @property
    def smart_policy(self) -> LidarAwarePolicy:
        """The active smart policy instance."""
        return self._smart_policy

    def run(self) -> bool:
        """Execute the smart exploration loop.

        Returns True on successful completion, False on failure/abort.
        """
        logger.info("=" * 60)
        logger.info("SMART EXPLORATION RUNNER — Enhanced Ranger Mini V3")
        logger.info("=" * 60)
        logger.info(f"  Duration          : {self._duration_minutes} minutes")
        logger.info(f"  Total steps       : {self._total_steps}")
        logger.info(f"  Control freq      : {self._control_freq} Hz")
        logger.info(f"  Policy freq       : {self._policy.config.policy_freq_hz} Hz")
        logger.info(f"  Action shaping    : {self._smart_config.action_shaping_enabled}")
        logger.info(f"  Adaptive speed    : {self._smart_config.adaptive_speed_enabled}")
        logger.info(f"  Panoramic spin    : {self._smart_config.panoramic_rotations_enabled} (every {self._smart_config.panoramic_interval_s}s)")
        logger.info(f"  Coverage nudge    : {self._smart_config.coverage_nudge_enabled}")
        logger.info(f"  Record 1D LiDAR   : {self._record_lidar} (72 bins)")
        logger.info(f"  Backup safety gate: {self._gate_min_depth}m / {self._gate_min_pixels}px")
        logger.info(f"  Dry-run           : {self._robot.dry_run}")
        logger.info(f"  Armed             : {not self._robot.dry_run}")
        logger.info("=" * 60)

        # Open the data recorder
        self._recorder.open()

        # Wait for all sensors to be available
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

        last_record_wall_time = 0.0
        last_record_cam_time = None

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
                rgb, cam_time = self._camera.get_frame()

                # 3. Get odometry
                pose = self._robot.get_pose()

                # 4. Get LiDAR depth image
                depth = self._lidar.get_depth_image()
                if depth is None:
                    depth = self._fallback_depth

                # 5. Update smart policy state with current observations
                self._smart_policy.update_lidar_state(depth)
                self._smart_policy.update_pose(pose.x, pose.y, pose.yaw)

                # 6. Sample smart action (pre-emptively shapes away from obstacles,
                #    adapts speed to clearance, injects panoramic rotation, adds coverage nudge)
                cmd = self._smart_policy.get_action()
                diag = self._smart_policy.get_diagnostics()

                if diag.get("shaped", False):
                    self._shaped_action_count += 1
                if diag.get("panoramic_active", False):
                    self._panoramic_step_count += 1

                # 7. Independent LiDAR safety gate (defense-in-depth backup)
                executed_cmd, safety_blocked, gate_diag = self._safety_gate(cmd, depth)

                # 8. Send the command to the robot
                self._robot.send_command(executed_cmd)

                # 9. Extract 72-bin 1D LiDAR range scan if enabled (Strategy 4)
                lidar_scan = None
                if self._record_lidar:
                    lidar_scan = extract_scan_from_depth_image(
                        depth,
                        num_bins=72,
                        max_range=self._max_range,
                    )

                # 10. Record step (10 Hz max sampling rate)
                if rgb is not None and cam_time != last_record_cam_time:
                    if time.time() - last_record_wall_time >= 0.1:
                        target = self._smart_policy.target_action
                        self._recorder.record_step(
                            rgb_frame=rgb,
                            cmd=cmd,
                            pos_x=pose.x,
                            pos_y=pose.y,
                            pos_z=pose.z,
                            yaw=pose.yaw,
                            wall_time=cam_time,
                            executed_cmd=executed_cmd,
                            safety_blocked=safety_blocked,
                            target_cmd=target,
                            obstacle_distance=gate_diag["obstacle_distance"],
                            relevant_point_count=gate_diag["relevant_point_count"],
                            cloud_age=gate_diag["cloud_age"],
                            lidar_scan=lidar_scan,
                        )
                        step_count += 1
                        last_record_cam_time = cam_time
                        last_record_wall_time = time.time()

                # 11. Diagnostic logging every 100 control ticks (~5s)
                if step > 0 and step % 100 == 0:
                    elapsed = time.time() - start_time
                    pct = 100.0 * step / self._total_steps
                    gate_pct = (
                        100.0 * self._gate_block_count / step
                        if step > 0 else 0.0
                    )
                    shaped_pct = (
                        100.0 * self._shaped_action_count / step
                        if step > 0 else 0.0
                    )

                    recent_dts = dts[-100:]
                    loop_rate = len(recent_dts) / sum(recent_dts) if sum(recent_dts) > 0 else 0.0
                    last_cam = self._camera.last_frame_time
                    cam_age = (time.time() - last_cam) if last_cam else 0.0

                    grid = self._smart_policy.visit_grid
                    cells_visited = int(np.count_nonzero(grid))

                    logger.info(
                        f"[{step:06d}/{self._total_steps}] "
                        f"{pct:.1f}% | "
                        f"vx={cmd.v_linear:+.2f} vy={cmd.v_lateral:+.2f} "
                        f"ω={cmd.v_angular:+.2f} | "
                        f"scale={diag.get('speed_scale', 1.0):.2f} "
                        f"d_min={diag.get('front_min', 10.0):.2f}m | "
                        f"cells={cells_visited} pano={'YES' if diag.get('panoramic_active') else 'NO'} | "
                        f"shaped={shaped_pct:.1f}% gate={gate_pct:.1f}% | "
                        f"loop={loop_rate:.1f}Hz rec={step_count} | "
                        f"elapsed={elapsed:.0f}s"
                    )

                # 12. Rate control — sleep for the remainder of the control period
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

            grid = self._smart_policy.visit_grid
            cells_visited = int(np.count_nonzero(grid))

            logger.info("=" * 60)
            logger.info("SMART EXPLORATION COMPLETE")
            logger.info(f"  Steps recorded      : {step_count}")
            logger.info(f"  Elapsed             : {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)")
            logger.info(f"  Unique cells visited: {cells_visited}")
            logger.info(f"  Shaped actions      : {self._shaped_action_count}")
            logger.info(f"  Panoramic steps     : {self._panoramic_step_count}")
            logger.info(f"  Safety gate blocks  : {self._gate_block_count}")
            logger.info(f"  E-stops             : {self._safety.estop_count}")
            logger.info("  --- Timing ---")
            logger.info(f"  Target Rate         : {self._control_freq:.1f} Hz")
            logger.info(f"  Actual Rate         : {actual_publish_rate:.1f} Hz")
            logger.info(f"  Mean dt             : {mean_dt*1000:.1f} ms")
            logger.info(f"  Std dt              : {std_dt*1000:.1f} ms")
            logger.info("=" * 60)

        return step_count > 0
