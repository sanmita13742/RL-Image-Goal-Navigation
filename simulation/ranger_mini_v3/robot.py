"""
robot.py  —  Ranger Mini V3 MuJoCo Interface
============================================
Provides the RangerMiniV3Robot class which adapts the 4WD4WS
Ackermann-driven Ranger Mini V3 to the unified RobotBase API.
"""

from pathlib import Path
import mujoco

# Import the shared base class and utilities from parent directory
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from robot_base import RobotBase, DriveCommand, RobotDimensions, compute_4ws_ik
    # WHEEL_RADIUS: float = 0.10
    # TRACK_WIDTH:  float = 0.36
    # WHEELBASE:    float = 0.494
class RangerMiniV3Robot(RobotBase):
    ROBOT_NAME:   str   = "RangerMiniV3"

    def __init__(self) -> None:
        self.WHEEL_RADIUS: float = 0.0
        self.TRACK_WIDTH:  float = 0.0
        self.WHEELBASE:    float = 0.0
        super().__init__()
        # Load the XML relative to this file
        xml_path = Path(__file__).parent / "ranger_mini_v3.xml"
        self.load(xml_path)

    def _post_load(self) -> None:
        """Cache actuator IDs for faster lookup during control loop."""
        if not self.model: return
        self._id_fl_steer = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_fl_steer")
        self._id_fr_steer = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_fr_steer")
        self._id_rl_steer = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_rl_steer")
        self._id_rr_steer = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_rr_steer")

        self._id_fl_drive = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_fl_drive")
        self._id_fr_drive = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_fr_drive")
        self._id_rl_drive = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_rl_drive")
        self._id_rr_drive = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_rr_drive")

        # Extract geometry directly from the MJCF model to prevent duplicated constants
        susp_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "fl_suspension_link")
        pos = self.model.body_pos[susp_id]
        self.WHEELBASE = abs(pos[0]) * 2.0
        self.TRACK_WIDTH = abs(pos[1]) * 2.0

        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "fl_wheel_col")
        self.WHEEL_RADIUS = self.model.geom_size[geom_id][0]

    def apply_command(self, cmd: DriveCommand) -> None:
        """Apply a given (v_linear, v_lateral, v_angular) to the 8 actuators."""
        if self.data is None:
            return

        # 1. Read current steering angles
        current = self.read_steering_angles()
        current_angles = (
            current.get("fl_steer_pos", 0.0),
            current.get("fr_steer_pos", 0.0),
            current.get("rl_steer_pos", 0.0),
            current.get("rr_steer_pos", 0.0),
        )

        # 2. Compute optimal continuous IK solutions
        (fl_ang, fr_ang, rl_ang, rr_ang), (fl_w, fr_w, rl_w, rr_w) = compute_4ws_ik(
            cmd=cmd,
            dims=self.get_dimensions(),
            current_angles=current_angles
        )

        # 3. Apply to ctrl array
        self.data.ctrl[self._id_fl_steer] = fl_ang
        self.data.ctrl[self._id_fr_steer] = fr_ang
        self.data.ctrl[self._id_rl_steer] = rl_ang
        self.data.ctrl[self._id_rr_steer] = rr_ang

        self.data.ctrl[self._id_fl_drive] = fl_w
        self.data.ctrl[self._id_fr_drive] = fr_w
        self.data.ctrl[self._id_rl_drive] = rl_w
        self.data.ctrl[self._id_rr_drive] = rr_w

    def get_dimensions(self) -> RobotDimensions:
        return RobotDimensions(
            wheel_radius=self.WHEEL_RADIUS,
            track_width=self.TRACK_WIDTH,
            wheelbase=self.WHEELBASE,
            mass_total=111.0,  # 75 + 4*(8+1)
            description="Ranger Mini V3 (4WD4WS Omni-Directional)"
        )

    def joint_names(self) -> list[str]:
        return [
            "root",
            "fl_suspension_joint", "fl_steering_joint", "fl_wheel",
            "fr_suspension_joint", "fr_steering_joint", "fr_wheel",
            "rl_suspension_joint", "rl_steering_joint", "rl_wheel",
            "rr_suspension_joint", "rr_steering_joint", "rr_wheel",
        ]

    def actuator_names(self) -> list[str]:
        return [
            "act_fl_steer", "act_fr_steer", "act_rl_steer", "act_rr_steer",
            "act_fl_drive", "act_fr_drive", "act_rl_drive", "act_rr_drive",
        ]

    # ── Sensor helpers ────────────────────────────────────────────────────────

    def read_imu(self) -> dict[str, list[float]]:
        """Return current IMU readings."""
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
