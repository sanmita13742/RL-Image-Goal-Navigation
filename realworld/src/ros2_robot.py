"""
realworld/src/ros2_robot.py — ROS 2 interface to the Ranger Mini V3.
====================================================================
Publishes geometry_msgs/Twist to /cmd_vel.
Subscribes to nav_msgs/Odometry on /odom for position + heading.

Key safety features:
  - Dry-run mode (default): logs commands but does NOT publish.
  - Hard velocity clamping to physical action bounds (AGENTS.md).
  - Thread-safe pose access.

Confirmed from rosbag:
  /cmd_vel  → geometry_msgs/msg/Twist
  /odom     → nav_msgs/msg/Odometry  (~50 Hz)
"""

from __future__ import annotations

import math
import logging
import threading
from dataclasses import dataclass
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from shared.drive_command import DriveCommand

logger = logging.getLogger(__name__)


@dataclass
class RobotPose:
    """Current robot pose from odometry."""
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    timestamp_sec: float = 0.0


def _quaternion_to_yaw(q) -> float:
    """Extract yaw from a geometry_msgs Quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class ROS2RangerMiniV3(Node):
    """ROS 2 driver for the Ranger Mini V3 robot.

    Parameters
    ----------
    cmd_vel_topic : str
        Topic to publish velocity commands (default: /cmd_vel).
    odom_topic : str
        Topic to subscribe for odometry (default: /odom).
    dry_run : bool
        If True (default), log commands but do NOT publish to /cmd_vel.
        The robot will not move.
    max_linear_vel : float
        Maximum forward velocity (m/s). Default: 1.8 (AGENTS.md).
    min_linear_vel : float
        Maximum reverse velocity (m/s, negative). Default: -0.9 (AGENTS.md).
    max_lateral_vel : float
        Maximum lateral velocity (m/s). Default: 1.0.
    max_angular_vel : float
        Maximum angular velocity (rad/s). Default: 1.5.
    """

    def __init__(
        self,
        cmd_vel_topic: str = "/cmd_vel",
        odom_topic: str = "/odom",
        dry_run: bool = True,
        max_linear_vel: float = 1.8,
        min_linear_vel: float = -0.9,
        max_lateral_vel: float = 1.0,
        max_angular_vel: float = 1.5,
        node_name: str = "minav_robot",
    ):
        super().__init__(node_name)

        self.dry_run = dry_run
        self.max_linear_vel = max_linear_vel
        self.min_linear_vel = min_linear_vel
        self.max_lateral_vel = max_lateral_vel
        self.max_angular_vel = max_angular_vel

        # Publisher
        self._cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self._cmd_count = 0

        from rclpy.qos import qos_profile_sensor_data
        self._odom_sub = self.create_subscription(
            Odometry, odom_topic, self._odom_callback, qos_profile_sensor_data
        )

        # Thread-safe pose
        self._pose = RobotPose()
        self._pose_lock = threading.Lock()
        self._odom_received = False
        self._last_odom_time: Optional[float] = None

        mode = "DRY-RUN (no motion)" if dry_run else "ARMED (live commands)"
        logger.info(f"ROS2RangerMiniV3 initialized: {mode}")
        logger.info(f"  cmd_vel: {cmd_vel_topic}  |  odom: {odom_topic}")
        logger.info(f"  Limits: vx=[{min_linear_vel}, {max_linear_vel}], "
                     f"vy=±{max_lateral_vel}, ω=±{max_angular_vel}")

    def _odom_callback(self, msg: Odometry) -> None:
        """Update internal pose from odometry message."""
        yaw = _quaternion_to_yaw(msg.pose.pose.orientation)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        with self._pose_lock:
            self._pose = RobotPose(
                x=msg.pose.pose.position.x,
                y=msg.pose.pose.position.y,
                yaw=yaw,
                timestamp_sec=stamp,
            )
            self._odom_received = True
            self._last_odom_time = stamp

    def get_pose(self) -> RobotPose:
        """Return the latest robot pose (thread-safe copy)."""
        with self._pose_lock:
            return RobotPose(
                x=self._pose.x,
                y=self._pose.y,
                yaw=self._pose.yaw,
                timestamp_sec=self._pose.timestamp_sec,
            )

    @property
    def odom_received(self) -> bool:
        """Whether at least one odom message has been received."""
        with self._pose_lock:
            return self._odom_received

    @property
    def last_odom_time(self) -> Optional[float]:
        """Timestamp of the last odom message (seconds)."""
        with self._pose_lock:
            return self._last_odom_time

    def _clamp(self, val: float, lo: float, hi: float) -> float:
        """Clamp value to [lo, hi]."""
        return max(lo, min(hi, val))

    def send_command(self, cmd: DriveCommand) -> Twist:
        """Send a velocity command to the robot.

        In dry-run mode, the Twist message is constructed and logged
        but NOT published. In armed mode, the message is published.

        Velocity values are hard-clamped to physical limits.

        Parameters
        ----------
        cmd : DriveCommand
            Desired velocity command.

        Returns
        -------
        Twist
            The constructed (and optionally published) message.
        """
        twist = Twist()
        twist.linear.x = self._clamp(cmd.v_linear, self.min_linear_vel, self.max_linear_vel)
        twist.linear.y = self._clamp(cmd.v_lateral, -self.max_lateral_vel, self.max_lateral_vel)
        twist.angular.z = self._clamp(cmd.v_angular, -self.max_angular_vel, self.max_angular_vel)

        if self.dry_run:
            if self._cmd_count % 50 == 0:  # Log every 50th command to avoid spam
                logger.debug(
                    f"[DRY-RUN] cmd_vel: vx={twist.linear.x:.3f} "
                    f"vy={twist.linear.y:.3f} ω={twist.angular.z:.3f}"
                )
        else:
            self._cmd_pub.publish(twist)

        self._cmd_count += 1
        return twist

    def emergency_stop(self) -> None:
        """Publish a zero-velocity command regardless of dry-run mode."""
        twist = Twist()
        # Always publish e-stop, even in dry-run
        self._cmd_pub.publish(twist)
        logger.warning("EMERGENCY STOP: zero velocity published")

    def destroy_node(self) -> None:
        """Send e-stop before shutting down."""
        self.emergency_stop()
        super().destroy_node()
