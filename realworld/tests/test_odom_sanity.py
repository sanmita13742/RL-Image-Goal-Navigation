"""
realworld/tests/test_odom_sanity.py — Odometry sanity tests.
==============================================================
Requires ROS 2 + ranger_base. Skipped unless MINAV_ROS2_AVAILABLE=1.
"""

import sys
import time
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


@pytest.mark.ros2
class TestOdomSanity:
    """Validate odometry data from the Ranger Mini V3."""

    @pytest.fixture(autouse=True)
    def setup_ros(self):
        import rclpy
        rclpy.init()
        yield
        rclpy.shutdown()

    def test_odom_rate(self):
        """Assert odom publishes at ≥20 Hz (expected ~50 Hz)."""
        import rclpy
        from realworld.src.ros2_robot import ROS2RangerMiniV3

        robot = ROS2RangerMiniV3(dry_run=True)
        try:
            odom_count = 0

            class Counter:
                def __init__(self):
                    self.count = 0

            counter = Counter()
            original_cb = robot._odom_callback

            def counting_cb(msg):
                counter.count += 1
                original_cb(msg)

            robot._odom_sub.callback = counting_cb

            start = time.time()
            while time.time() - start < 3.0:
                rclpy.spin_once(robot, timeout_sec=0.05)

            elapsed = time.time() - start
            rate = counter.count / elapsed if elapsed > 0 else 0

            assert rate >= 20, f"Odom rate too low: {rate:.1f} Hz (expected ≥20 Hz)"
        finally:
            robot.destroy_node()

    def test_odom_yaw_range(self):
        """Yaw should be in [-π, π]."""
        import rclpy
        from realworld.src.ros2_robot import ROS2RangerMiniV3

        robot = ROS2RangerMiniV3(dry_run=True)
        try:
            start = time.time()
            while time.time() - start < 2.0:
                rclpy.spin_once(robot, timeout_sec=0.1)
                if robot.odom_received:
                    break

            pose = robot.get_pose()
            assert -math.pi <= pose.yaw <= math.pi, (
                f"Yaw out of range: {pose.yaw:.4f}"
            )
        finally:
            robot.destroy_node()

    def test_odom_position_finite(self):
        """Position values should be finite (not NaN/Inf)."""
        import rclpy
        from realworld.src.ros2_robot import ROS2RangerMiniV3

        robot = ROS2RangerMiniV3(dry_run=True)
        try:
            start = time.time()
            while time.time() - start < 2.0:
                rclpy.spin_once(robot, timeout_sec=0.1)
                if robot.odom_received:
                    break

            pose = robot.get_pose()
            assert math.isfinite(pose.x), f"x is not finite: {pose.x}"
            assert math.isfinite(pose.y), f"y is not finite: {pose.y}"
            assert math.isfinite(pose.yaw), f"yaw is not finite: {pose.yaw}"
        finally:
            robot.destroy_node()
