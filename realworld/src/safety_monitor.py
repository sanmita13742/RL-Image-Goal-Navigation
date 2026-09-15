"""
realworld/src/safety_monitor.py — Watchdog and safety supervisor.
=================================================================
Monitors camera, odom, and LiDAR liveness. If any sensor times out,
publishes a zero-velocity e-stop to /cmd_vel.

The robot can only move when explicitly armed. Arming requires all
sensor streams to be live.

Safety invariants:
  1. Robot starts DISARMED. No motion commands are published.
  2. arm() checks all sensor streams are live before transitioning.
  3. Watchdog timer runs at configurable Hz, checking sensor freshness.
  4. Any sensor timeout → immediate e-stop → DISARMED.
  5. Emergency stop always publishes, even when disarmed (defense in depth).
"""

from __future__ import annotations

import logging
import time
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class SafetyMonitor:
    """Watchdog and safety supervisor for real-world exploration.

    Parameters
    ----------
    robot : ROS2RangerMiniV3
        Robot interface (for emergency_stop).
    camera : ROS2Camera
        Camera interface (for liveness check).
    lidar : ROS2LiDAR
        LiDAR interface (for liveness check).
    camera_timeout_s : float
        Max time without a camera frame before e-stop. Default: 2.0.
    odom_timeout_s : float
        Max time without an odom message before e-stop. Default: 2.0.
    lidar_timeout_s : float
        Max time without a LiDAR scan before e-stop. Default: 3.0.
    """

    def __init__(
        self,
        robot,
        camera,
        lidar,
        camera_timeout_s: float = 2.0,
        odom_timeout_s: float = 2.0,
        lidar_timeout_s: float = 3.0,
    ):
        self._robot = robot
        self._camera = camera
        self._lidar = lidar

        self._camera_timeout = camera_timeout_s
        self._odom_timeout = odom_timeout_s
        self._lidar_timeout = lidar_timeout_s

        self._armed = False
        self._lock = threading.Lock()
        self._estop_count = 0

    @property
    def is_armed(self) -> bool:
        """Whether the robot is armed for motion."""
        with self._lock:
            return self._armed

    def arm(self) -> bool:
        """Attempt to arm the robot for motion.

        Checks all sensor streams are live. Returns True on success.
        """
        with self._lock:
            if self._armed:
                logger.warning("Already armed")
                return True

            # Check all sensors have received at least one message
            issues = []
            if not self._camera.has_frame:
                issues.append("Camera: no frames received")
            if not self._robot.odom_received:
                issues.append("Odom: no messages received")
            if not self._lidar.has_scan:
                issues.append("LiDAR: no scans received")

            # Check freshness
            now = time.time()
            cam_time = self._camera.last_frame_time
            if cam_time is not None and (now - cam_time) > self._camera_timeout:
                issues.append(f"Camera: stale ({now - cam_time:.1f}s ago)")

            odom_time = self._robot.last_odom_time
            if odom_time is not None and (now - odom_time) > self._odom_timeout:
                issues.append(f"Odom: stale ({now - odom_time:.1f}s ago)")

            lidar_time = self._lidar.last_scan_time
            if lidar_time is not None and (now - lidar_time) > self._lidar_timeout:
                issues.append(f"LiDAR: stale ({now - lidar_time:.1f}s ago)")

            if issues:
                for issue in issues:
                    logger.error(f"ARM FAILED: {issue}")
                return False

            self._armed = True
            logger.info("✓ Robot ARMED — all sensor streams live")
            return True

    def disarm(self, reason: str = "manual") -> None:
        """Disarm the robot and send e-stop."""
        with self._lock:
            was_armed = self._armed
            self._armed = False

        self._robot.emergency_stop()
        if was_armed:
            logger.warning(f"Robot DISARMED: {reason}")

    def check(self) -> bool:
        """Run one safety watchdog check.

        Returns True if all sensors are healthy, False if e-stop was triggered.
        Should be called at watchdog_hz (e.g., 5 Hz).
        """
        with self._lock:
            if not self._armed:
                return True  # Not armed, nothing to check

        now = time.time()
        issues = []

        # Camera freshness
        cam_time = self._camera.last_frame_time
        if cam_time is None or (now - cam_time) > self._camera_timeout:
            elapsed = "never" if cam_time is None else f"{now - cam_time:.1f}s"
            issues.append(f"Camera timeout (last: {elapsed})")

        # Odom freshness
        odom_time = self._robot.last_odom_time
        if odom_time is None or (now - odom_time) > self._odom_timeout:
            elapsed = "never" if odom_time is None else f"{now - odom_time:.1f}s"
            issues.append(f"Odom timeout (last: {elapsed})")

        # LiDAR freshness
        lidar_time = self._lidar.last_scan_time
        if lidar_time is None or (now - lidar_time) > self._lidar_timeout:
            elapsed = "never" if lidar_time is None else f"{now - lidar_time:.1f}s"
            issues.append(f"LiDAR timeout (last: {elapsed})")

        if issues:
            self._estop_count += 1
            for issue in issues:
                logger.error(f"SAFETY VIOLATION #{self._estop_count}: {issue}")
            self.disarm(reason="; ".join(issues))
            return False

        return True

    @property
    def estop_count(self) -> int:
        """Total number of emergency stops triggered."""
        return self._estop_count
