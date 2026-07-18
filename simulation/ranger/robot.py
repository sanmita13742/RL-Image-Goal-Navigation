"""
ranger_mujoco/robot.py  —  Ranger Mini concrete robot implementation
====================================================================
Implements the RobotBase interface for the AgileX Ranger Mini,
which uses 4-Wheel Drive + 4-Wheel Steering (Ackermann geometry).

Actuator layout in ranger.xml:
    ctrl[0]  act_fl_steer   – FL steering position (rad)
    ctrl[1]  act_fr_steer   – FR steering position (rad)
    ctrl[2]  act_rl_steer   – RL steering position (rad)
    ctrl[3]  act_rr_steer   – RR steering position (rad)
    ctrl[4]  act_fl_drive   – FL wheel velocity (rad/s)
    ctrl[5]  act_fr_drive   – FR wheel velocity (rad/s)
    ctrl[6]  act_rl_drive   – RL wheel velocity (rad/s)
    ctrl[7]  act_rr_drive   – RR wheel velocity (rad/s)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Allow importing from parent RL/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from robot_base import (
    RobotBase, DriveCommand, RobotDimensions,
    ackermann_angles, ackermann_wheel_speeds,
)


class RangerRobot(RobotBase):
    """
    AgileX Ranger Mini — 4-Wheel Drive + 4-Wheel Steering robot.

    Drive model: Ackermann geometry for smooth curved paths.
    For spin-in-place: uses a virtual small turning radius so all
    four wheels steer symmetrically ±35° and spin differentially.
    """

    # ── Physical constants (from URDF analysis) ───────────────────────────────
    WHEEL_RADIUS: float = 0.160     # metres  (estimated from mesh + joint geometry)
    TRACK_WIDTH:  float = 0.560     # metres  (2 × 0.280 m from URDF y-offsets)
    WHEELBASE:    float = 0.890     # metres  (2 × 0.445 m from URDF x-offsets)
    MAX_STEER:    float = math.radians(35)   # ≈ 0.6109 rad — matches ctrlrange
    ROBOT_NAME:   str   = "Ranger Mini"

    # Drive mode constants
    MODE_ACKERMANN = "ackermann"   # Front steer only (or 4WS)
    MODE_CRAB      = "crab"        # All wheels steer same angle (lateral motion)
    MODE_SPIN      = "spin"        # In-place spin

    def __init__(self) -> None:
        super().__init__()
        self.drive_mode: str = self.MODE_ACKERMANN

    # ── Abstract implementation ───────────────────────────────────────────────

    def apply_command(self, cmd: DriveCommand) -> None:
        """
        Convert a DriveCommand to 8 actuator signals (4 steer + 4 drive).
        Uses Ackermann geometry for steering angle computation.
        """
        self._require_loaded()

        fl_steer, fr_steer, rl_steer, rr_steer = ackermann_angles(
            v_angular    = cmd.v_angular,
            v_linear     = cmd.v_linear,
            wheelbase    = self.WHEELBASE,
            track_width  = self.TRACK_WIDTH,
            max_steer_rad = self.MAX_STEER,
        )

        fl_drive, fr_drive, rl_drive, rr_drive = ackermann_wheel_speeds(
            v_linear   = cmd.v_linear,
            v_angular  = cmd.v_angular,
            wheel_radius = self.WHEEL_RADIUS,
            wheelbase    = self.WHEELBASE,
            track_width  = self.TRACK_WIDTH,
        )

        # Steering (position actuators)
        self.data.ctrl[0] = fl_steer
        self.data.ctrl[1] = fr_steer
        self.data.ctrl[2] = rl_steer
        self.data.ctrl[3] = rr_steer

        # Drive (velocity actuators — rad/s)
        self.data.ctrl[4] = fl_drive
        self.data.ctrl[5] = fr_drive
        self.data.ctrl[6] = rl_drive
        self.data.ctrl[7] = rr_drive

    def apply_crab_command(self, v_lateral: float, angle_rad: float) -> None:
        """
        Crab-steer mode: all four wheels point the same direction,
        allowing the robot to move sideways.

        Parameters
        ----------
        v_lateral : lateral velocity in m/s (+ = left)
        angle_rad : steering angle in rad (clamped to MAX_STEER)
        """
        self._require_loaded()
        angle = max(-self.MAX_STEER, min(self.MAX_STEER, angle_rad))
        w = v_lateral / self.WHEEL_RADIUS

        self.data.ctrl[0] = angle   # fl_steer
        self.data.ctrl[1] = angle   # fr_steer
        self.data.ctrl[2] = angle   # rl_steer
        self.data.ctrl[3] = angle   # rr_steer
        self.data.ctrl[4] = w       # fl_drive
        self.data.ctrl[5] = w       # fr_drive
        self.data.ctrl[6] = w       # rl_drive
        self.data.ctrl[7] = w       # rr_drive

    def stop(self) -> None:
        """Immediately zero all actuators."""
        self._require_loaded()
        for i in range(self.model.nu):
            self.data.ctrl[i] = 0.0

    def get_dimensions(self) -> RobotDimensions:
        return RobotDimensions(
            wheel_radius = self.WHEEL_RADIUS,
            track_width  = self.TRACK_WIDTH,
            wheelbase    = self.WHEELBASE,
            mass_total   = 88.757 + 4 * (2.095 + 11.468),
            description  = (
                "4WD4WS Ackermann steering; "
                "4 steering joints + 4 drive joints; "
                "URDF mass total ≈ 143 kg"
            ),
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

    # ── Sensor helpers ────────────────────────────────────────────────────────

    def read_imu(self) -> dict[str, list[float]]:
        """
        Return current IMU readings.

        Returns
        -------
        dict with keys 'accel' [ax, ay, az] m/s²  and  'gyro' [gx, gy, gz] rad/s
        """
        self._require_loaded()
        import mujoco
        import numpy as np
        accel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")
        gyro_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        accel_adr = self.model.sensor_adr[accel_id]
        gyro_adr  = self.model.sensor_adr[gyro_id]
        return {
            "accel": self.data.sensordata[accel_adr : accel_adr + 3].tolist(),
            "gyro":  self.data.sensordata[gyro_adr  : gyro_adr  + 3].tolist(),
        }

    def read_wheel_velocities(self) -> dict[str, float]:
        """Return wheel velocities (rad/s) from joint-velocity sensors."""
        self._require_loaded()
        import mujoco
        result = {}
        for name in ("fl_wheel_vel", "fr_wheel_vel", "rl_wheel_vel", "rr_wheel_vel"):
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            adr = self.model.sensor_adr[sid]
            result[name] = float(self.data.sensordata[adr])
        return result

    def read_steering_angles(self) -> dict[str, float]:
        """Return current steering angles (rad) from joint-position sensors."""
        self._require_loaded()
        import mujoco
        result = {}
        for name in ("fl_steer_pos", "fr_steer_pos", "rl_steer_pos", "rr_steer_pos"):
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            adr = self.model.sensor_adr[sid]
            result[name] = float(self.data.sensordata[adr])
        return result
