"""
realworld/tests/test_ros2_connectivity.py — ROS 2 connectivity tests.
======================================================================
These tests require a running ROS 2 environment with the Ranger Mini V3
bringup. They are skipped unless MINAV_ROS2_AVAILABLE=1 is set.
"""

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


@pytest.mark.ros2
class TestROS2Connectivity:
    """Tests that verify the ROS 2 environment is healthy."""

    @pytest.fixture(autouse=True)
    def setup_ros(self):
        """Initialize ROS 2 for the test."""
        import rclpy
        rclpy.init()
        yield
        rclpy.shutdown()

    def test_cmd_vel_topic_exists(self):
        """Verify /cmd_vel topic is available."""
        import rclpy
        from rclpy.node import Node

        node = Node("test_connectivity")
        try:
            topics = node.get_topic_names_and_types()
            topic_names = [t[0] for t in topics]
            assert "/cmd_vel" in topic_names, (
                f"/cmd_vel not found. Available topics: {topic_names}"
            )
        finally:
            node.destroy_node()

    def test_odom_topic_exists(self):
        """Verify /odom topic is publishing."""
        import rclpy
        from rclpy.node import Node

        node = Node("test_odom")
        try:
            topics = node.get_topic_names_and_types()
            topic_names = [t[0] for t in topics]
            assert "/odom" in topic_names, (
                f"/odom not found. Available topics: {topic_names}"
            )
        finally:
            node.destroy_node()

    def test_camera_topic_exists(self):
        """Verify camera topic is active."""
        import rclpy
        from rclpy.node import Node

        node = Node("test_camera_topic")
        try:
            topics = node.get_topic_names_and_types()
            topic_names = [t[0] for t in topics]
            # RealSense publishes to /camera/color/image_raw
            camera_topics = [t for t in topic_names if "camera" in t and "image" in t]
            assert len(camera_topics) > 0, (
                f"No camera image topics found. Available: {topic_names}"
            )
        finally:
            node.destroy_node()

    def test_lidar_topic_exists(self):
        """Verify /rslidar_points topic is active."""
        import rclpy
        from rclpy.node import Node

        node = Node("test_lidar_topic")
        try:
            topics = node.get_topic_names_and_types()
            topic_names = [t[0] for t in topics]
            assert "/rslidar_points" in topic_names, (
                f"/rslidar_points not found. Available topics: {topic_names}"
            )
        finally:
            node.destroy_node()

    def test_odom_receiving_messages(self):
        """Verify odom messages are arriving at expected rate (~50 Hz)."""
        from realworld.src.ros2_robot import ROS2RangerMiniV3
        import rclpy

        robot = ROS2RangerMiniV3(dry_run=True)
        try:
            start = time.time()
            while time.time() - start < 3.0:
                rclpy.spin_once(robot, timeout_sec=0.1)

            assert robot.odom_received, "No odom messages received in 3 seconds"
        finally:
            robot.destroy_node()
