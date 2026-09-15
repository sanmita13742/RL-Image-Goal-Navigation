"""
realworld/tests/test_camera_feed.py — Camera feed validation tests.
====================================================================
Requires ROS 2 + RealSense camera. Skipped unless MINAV_ROS2_AVAILABLE=1.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


@pytest.mark.ros2
class TestCameraFeed:
    """Validate that the RealSense camera produces valid frames."""

    @pytest.fixture(autouse=True)
    def setup_ros(self):
        import rclpy
        rclpy.init()
        yield
        rclpy.shutdown()

    def test_camera_receives_frames(self):
        """Subscribe for 3s, assert ≥10 frames received."""
        import rclpy
        from realworld.src.ros2_camera import ROS2Camera

        camera = ROS2Camera()
        try:
            start = time.time()
            while time.time() - start < 3.0:
                rclpy.spin_once(camera, timeout_sec=0.1)

            assert camera.frame_count >= 10, (
                f"Expected ≥10 frames in 3s, got {camera.frame_count}"
            )
        finally:
            camera.destroy_node()

    def test_camera_frame_shape(self):
        """Assert frame shape matches expected (H, W, 3)."""
        import rclpy
        from realworld.src.ros2_camera import ROS2Camera

        camera = ROS2Camera()
        try:
            start = time.time()
            while time.time() - start < 3.0:
                rclpy.spin_once(camera, timeout_sec=0.1)
                if camera.has_frame:
                    break

            frame = camera.get_frame()
            assert frame is not None, "No frame received"
            assert len(frame.shape) == 3, f"Expected 3D array, got shape {frame.shape}"
            assert frame.shape[2] == 3, f"Expected 3 channels, got {frame.shape[2]}"
        finally:
            camera.destroy_node()

    def test_camera_frame_not_blank(self):
        """Assert frames are not all-zeros (lens cap check)."""
        import rclpy
        from realworld.src.ros2_camera import ROS2Camera

        camera = ROS2Camera()
        try:
            start = time.time()
            while time.time() - start < 3.0:
                rclpy.spin_once(camera, timeout_sec=0.1)
                if camera.has_frame:
                    break

            frame = camera.get_frame()
            assert frame is not None
            assert frame.max() > 0, "Frame is all zeros — check lens cap!"
            assert frame.std() > 1.0, "Frame has very low variance — possible blank image"
        finally:
            camera.destroy_node()

    def test_camera_frame_dtype(self):
        """Assert frame dtype is uint8."""
        import rclpy
        from realworld.src.ros2_camera import ROS2Camera

        camera = ROS2Camera()
        try:
            start = time.time()
            while time.time() - start < 3.0:
                rclpy.spin_once(camera, timeout_sec=0.1)
                if camera.has_frame:
                    break

            frame = camera.get_frame()
            assert frame is not None
            assert frame.dtype == np.uint8, f"Expected uint8, got {frame.dtype}"
        finally:
            camera.destroy_node()
