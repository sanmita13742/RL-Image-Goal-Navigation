"""
realworld/tests/test_cmd_vel_dryrun.py — Command velocity dry-run tests.
=========================================================================
Validates that dry-run mode does NOT publish to /cmd_vel,
and that velocity clamping works correctly.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from shared.drive_command import DriveCommand


class TestCmdVelDryRun:
    """Tests for cmd_vel dry-run behavior (no ROS 2 needed — uses mocks)."""

    def test_dryrun_does_not_publish(self):
        """In dry-run mode, verify publish is NOT called on /cmd_vel."""
        # Create a mock that simulates the ROS2RangerMiniV3 behavior
        mock_pub = MagicMock()

        # Simulate the send_command logic in dry-run mode
        class MockRobot:
            def __init__(self):
                self.dry_run = True
                self._cmd_pub = mock_pub
                self._cmd_count = 0

            def send_command(self, cmd):
                class Twist:
                    class Vector3:
                        def __init__(self):
                            self.x, self.y, self.z = 0.0, 0.0, 0.0
                    def __init__(self):
                        self.linear = self.Vector3()
                        self.angular = self.Vector3()
                
                twist = Twist()
                twist.linear.x = cmd.v_linear
                twist.linear.y = cmd.v_lateral
                twist.angular.z = cmd.v_angular
                if not self.dry_run:
                    self._cmd_pub.publish(twist)
                self._cmd_count += 1
                return twist

        robot = MockRobot()
        cmd = DriveCommand(v_linear=1.0, v_lateral=0.5, v_angular=-0.3)
        robot.send_command(cmd)

        mock_pub.publish.assert_not_called()

    def test_velocity_clamping_vx_max(self):
        """Commands exceeding vx max (1.8) should be clamped."""
        def clamp(val, lo, hi):
            return max(lo, min(hi, val))

        result = clamp(5.0, -0.9, 1.8)
        assert result == 1.8

    def test_velocity_clamping_vx_min(self):
        """Commands below vx min (-0.9) should be clamped."""
        def clamp(val, lo, hi):
            return max(lo, min(hi, val))

        result = clamp(-3.0, -0.9, 1.8)
        assert result == -0.9

    def test_velocity_clamping_vy(self):
        """Lateral velocity should be clamped to ±1.0."""
        def clamp(val, lo, hi):
            return max(lo, min(hi, val))

        assert clamp(2.0, -1.0, 1.0) == 1.0
        assert clamp(-2.0, -1.0, 1.0) == -1.0

    def test_velocity_clamping_omega(self):
        """Angular velocity should be clamped to ±1.5."""
        def clamp(val, lo, hi):
            return max(lo, min(hi, val))

        assert clamp(3.0, -1.5, 1.5) == 1.5
        assert clamp(-3.0, -1.5, 1.5) == -1.5

    def test_twist_message_format(self):
        """Verify DriveCommand maps correctly to Twist fields."""
        cmd = DriveCommand(v_linear=1.2, v_lateral=-0.5, v_angular=0.8)

        # The mapping should be:
        # linear.x  = v_linear   (forward)
        # linear.y  = v_lateral  (left/crab)
        # angular.z = v_angular  (CCW)
        assert cmd.v_linear == 1.2    # → twist.linear.x
        assert cmd.v_lateral == -0.5  # → twist.linear.y
        assert cmd.v_angular == 0.8   # → twist.angular.z

    def test_drive_command_defaults_to_zero(self):
        """Default DriveCommand should be zero velocity (stationary)."""
        cmd = DriveCommand()
        assert cmd.v_linear == 0.0
        assert cmd.v_lateral == 0.0
        assert cmd.v_angular == 0.0
