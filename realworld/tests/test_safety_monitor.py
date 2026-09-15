"""
realworld/tests/test_safety_monitor.py — Safety watchdog tests.
================================================================
Tests the SafetyMonitor's arm/disarm logic, sensor timeout detection,
and emergency stop behavior. Uses mock objects (no ROS 2 needed).
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from realworld.src.safety_monitor import SafetyMonitor


def _make_mock_robot(odom_received=True, last_odom_time=None):
    robot = MagicMock()
    type(robot).odom_received = PropertyMock(return_value=odom_received)
    type(robot).last_odom_time = PropertyMock(
        return_value=last_odom_time if last_odom_time is not None else time.time()
    )
    return robot


def _make_mock_camera(has_frame=True, last_frame_time=None):
    camera = MagicMock()
    type(camera).has_frame = PropertyMock(return_value=has_frame)
    type(camera).last_frame_time = PropertyMock(
        return_value=last_frame_time if last_frame_time is not None else time.time()
    )
    return camera


def _make_mock_lidar(has_scan=True, last_scan_time=None):
    lidar = MagicMock()
    type(lidar).has_scan = PropertyMock(return_value=has_scan)
    type(lidar).last_scan_time = PropertyMock(
        return_value=last_scan_time if last_scan_time is not None else time.time()
    )
    return lidar


class TestSafetyMonitor:

    def test_starts_disarmed(self):
        """Safety monitor should start in disarmed state."""
        robot = _make_mock_robot()
        camera = _make_mock_camera()
        lidar = _make_mock_lidar()
        safety = SafetyMonitor(robot, camera, lidar)

        assert not safety.is_armed

    def test_arm_with_all_sensors_live(self):
        """Should arm successfully when all sensors are reporting."""
        robot = _make_mock_robot()
        camera = _make_mock_camera()
        lidar = _make_mock_lidar()
        safety = SafetyMonitor(robot, camera, lidar)

        assert safety.arm() is True
        assert safety.is_armed

    def test_arm_fails_without_camera(self):
        """Should refuse to arm if camera has no frames."""
        robot = _make_mock_robot()
        camera = _make_mock_camera(has_frame=False)
        lidar = _make_mock_lidar()
        safety = SafetyMonitor(robot, camera, lidar)

        assert safety.arm() is False
        assert not safety.is_armed

    def test_arm_fails_without_odom(self):
        """Should refuse to arm if odom not received."""
        robot = _make_mock_robot(odom_received=False)
        camera = _make_mock_camera()
        lidar = _make_mock_lidar()
        safety = SafetyMonitor(robot, camera, lidar)

        assert safety.arm() is False
        assert not safety.is_armed

    def test_arm_fails_without_lidar(self):
        """Should refuse to arm if LiDAR has no scans."""
        robot = _make_mock_robot()
        camera = _make_mock_camera()
        lidar = _make_mock_lidar(has_scan=False)
        safety = SafetyMonitor(robot, camera, lidar)

        assert safety.arm() is False
        assert not safety.is_armed

    def test_disarm_sends_estop(self):
        """Disarming should send emergency stop to robot."""
        robot = _make_mock_robot()
        camera = _make_mock_camera()
        lidar = _make_mock_lidar()
        safety = SafetyMonitor(robot, camera, lidar)

        safety.arm()
        safety.disarm(reason="test")

        robot.emergency_stop.assert_called()
        assert not safety.is_armed

    def test_estop_on_camera_timeout(self):
        """Should e-stop if camera frame is stale."""
        robot = _make_mock_robot()
        camera = _make_mock_camera(last_frame_time=time.time())
        lidar = _make_mock_lidar()
        safety = SafetyMonitor(robot, camera, lidar, camera_timeout_s=2.0)

        assert safety.arm() is True
        assert safety.is_armed
        
        # Simulate time passing
        type(camera).last_frame_time = PropertyMock(return_value=time.time() - 10.0)

        result = safety.check()
        assert result is False  # Safety violation
        assert not safety.is_armed  # Should be disarmed
        robot.emergency_stop.assert_called()

    def test_estop_on_odom_timeout(self):
        """Should e-stop if odom is stale."""
        robot = _make_mock_robot(last_odom_time=time.time())
        camera = _make_mock_camera()
        lidar = _make_mock_lidar()
        safety = SafetyMonitor(robot, camera, lidar, odom_timeout_s=2.0)

        assert safety.arm() is True
        type(robot).last_odom_time = PropertyMock(return_value=time.time() - 10.0)
        
        result = safety.check()
        assert result is False
        assert not safety.is_armed

    def test_estop_on_lidar_timeout(self):
        """Should e-stop if LiDAR scan is stale."""
        robot = _make_mock_robot()
        camera = _make_mock_camera()
        lidar = _make_mock_lidar(last_scan_time=time.time())
        safety = SafetyMonitor(robot, camera, lidar, lidar_timeout_s=3.0)

        assert safety.arm() is True
        type(lidar).last_scan_time = PropertyMock(return_value=time.time() - 10.0)
        
        result = safety.check()
        assert result is False

    def test_check_passes_when_healthy(self):
        """Check should pass when all sensors are fresh."""
        robot = _make_mock_robot()
        camera = _make_mock_camera()
        lidar = _make_mock_lidar()
        safety = SafetyMonitor(robot, camera, lidar)

        safety.arm()
        assert safety.check() is True
        assert safety.is_armed

    def test_check_noop_when_disarmed(self):
        """Check should return True (no-op) when not armed."""
        robot = _make_mock_robot()
        camera = _make_mock_camera(has_frame=False)
        lidar = _make_mock_lidar()
        safety = SafetyMonitor(robot, camera, lidar)

        # Not armed, check should be fine
        assert safety.check() is True

    def test_estop_count_increments(self):
        """E-stop counter should increment on each safety violation."""
        robot = _make_mock_robot()
        camera = _make_mock_camera(last_frame_time=time.time())
        lidar = _make_mock_lidar()
        safety = SafetyMonitor(robot, camera, lidar, camera_timeout_s=2.0)

        assert safety.arm() is True
        type(camera).last_frame_time = PropertyMock(return_value=time.time() - 10.0)
        
        safety.check()  # Will trigger e-stop
        assert safety.estop_count == 1
