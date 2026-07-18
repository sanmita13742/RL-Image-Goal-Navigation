"""
scout_mujoco/robot.py  —  Scout Mini concrete robot implementation
==================================================================
Wraps the ScoutMini skid-steer differential drive logic
using the RobotBase abstraction layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing from the parent RL/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from robot_base import RobotBase, DriveCommand, RobotDimensions


class ScoutMiniRobot(RobotBase):
    """
    Scout Mini — 4-wheel skid-steer mobile robot.

    Actuator order in scout_mini.xml:
        ctrl[0] = act_fl  (front-left  velocity)
        ctrl[1] = act_rl  (rear-left   velocity)
        ctrl[2] = act_fr  (front-right velocity)
        ctrl[3] = act_rr  (rear-right  velocity)
    """

    WHEEL_RADIUS: float = 0.08      # metres  (from URDF)
    TRACK_WIDTH:  float = 0.4165    # metres  (left–right wheel-centre distance)
    WHEELBASE:    float = 0.0       # skid-steer — wheelbase not used for steering
    ROBOT_NAME:   str   = "Scout Mini"

    # ── Abstract implementation ───────────────────────────────────────────────

    def apply_command(self, cmd: DriveCommand) -> None:
        """Convert (v_linear, v_angular) to wheel rad/s via skid-steer."""
        self._require_loaded()
        v_left, v_right = self._skid_steer(cmd.v_linear, cmd.v_angular)
        self.data.ctrl[0] = v_left    # front-left
        self.data.ctrl[1] = v_left    # rear-left
        self.data.ctrl[2] = v_right   # front-right
        self.data.ctrl[3] = v_right   # rear-right

    def get_dimensions(self) -> RobotDimensions:
        return RobotDimensions(
            wheel_radius = self.WHEEL_RADIUS,
            track_width  = self.TRACK_WIDTH,
            wheelbase    = self.WHEELBASE,
            mass_total   = 60.0,
            description  = "Skid-steer, 4WD, no steering joints",
        )

    def joint_names(self) -> list[str]:
        self._require_loaded()
        import mujoco
        return [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]

    def actuator_names(self) -> list[str]:
        self._require_loaded()
        import mujoco
        return [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _skid_steer(self, v_linear: float, v_angular: float) -> tuple[float, float]:
        """Convert twist to (v_left, v_right) wheel speeds in rad/s."""
        v_left  = (v_linear - v_angular * self.TRACK_WIDTH / 2.0) / self.WHEEL_RADIUS
        v_right = (v_linear + v_angular * self.TRACK_WIDTH / 2.0) / self.WHEEL_RADIUS
        return v_left, v_right
