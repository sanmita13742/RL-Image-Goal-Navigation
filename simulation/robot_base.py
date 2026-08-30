"""
robot_base.py  —  Abstract robot interface for MuJoCo UGV simulations
=====================================================================
Defines the RobotBase class that every concrete robot implementation
must subclass.  The goal is a single, interchangeable API so that
drive.py / check_sensors.py / RL training code need not know which
physical platform is being simulated.

Usage
-----
    from robot_base import RobotBase

    class MyRobot(RobotBase):
        ...

Concrete robots currently implemented
--------------------------------------
    scout_mujoco/robot.py   – ScoutMiniRobot  (skid-steer, 4-wheel-drive)
    ranger_mujoco/robot.py  – RangerRobot      (Ackermann 4WD4WS)
"""

from __future__ import annotations

import abc
import dataclasses
import math
from pathlib import Path
from typing import Optional

import mujoco


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class DriveCommand:
    """Unified motion command fed to apply_command()."""
    v_linear:  float = 0.0   # m/s  (+ forward)
    v_lateral: float = 0.0   # m/s  (+ left, crab walk)
    v_angular: float = 0.0   # rad/s (+ turn left, CCW)


@dataclasses.dataclass
class RobotDimensions:
    """Key physical dimensions for a ground robot."""
    wheel_radius:  float   # metres
    track_width:   float   # metres – lateral distance between left and right wheel centres
    wheelbase:     float   # metres – longitudinal distance between front and rear axles
    mass_total:    float   # kg (approximate)
    description:   str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class RobotBase(abc.ABC):
    """
    Abstract base class for all MuJoCo ground-robot implementations.

    Subclasses **must** override every abstract method/property.
    They should also set the class-level constants (WHEEL_RADIUS, …)
    so generic utilities can query dimensions without instantiating the class.
    """

    # ── Class-level constants (override in subclasses) ────────────────────────
    WHEEL_RADIUS: float = NotImplemented   # metres
    TRACK_WIDTH:  float = NotImplemented   # metres
    WHEELBASE:    float = NotImplemented   # metres  (0 for skid-steer where N/A)
    ROBOT_NAME:   str   = "AbstractRobot"

    def __init__(self) -> None:
        self.model: Optional[mujoco.MjModel] = None
        self.data:  Optional[mujoco.MjData]  = None
        self._xml_path: Optional[Path] = None

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self, xml_path: str | Path) -> None:
        """Load the MJCF model and create simulation data.

        Parameters
        ----------
        xml_path:
            Path to the .xml MJCF file (absolute or relative to cwd).
        """
        self._xml_path = Path(xml_path)
        self.model = mujoco.MjModel.from_xml_path(str(self._xml_path))
        self.data  = mujoco.MjData(self.model)
        self._post_load()

    def _post_load(self) -> None:
        """Override to do any per-robot post-load setup (e.g. cache joint ids)."""

    # ── Abstract interface ────────────────────────────────────────────────────

    @abc.abstractmethod
    def apply_command(self, cmd: DriveCommand) -> None:
        """Write actuator commands to self.data.ctrl for the given twist.

        Parameters
        ----------
        cmd:
            Linear and angular velocity setpoints.
        """

    @abc.abstractmethod
    def get_dimensions(self) -> RobotDimensions:
        """Return the key physical dimensions for this robot."""

    @abc.abstractmethod
    def joint_names(self) -> list[str]:
        """Return the ordered list of joint names in this model."""

    @abc.abstractmethod
    def actuator_names(self) -> list[str]:
        """Return the ordered list of actuator names in this model."""

    # ── Concrete helpers (available to all subclasses) ────────────────────────

    def sensor_names(self) -> list[str]:
        """Return list of sensor names (reads directly from model)."""
        self._require_loaded()
        return [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            for i in range(self.model.nsensor)
        ]

    def camera_names(self) -> list[str]:
        """Return list of camera names in this model."""
        self._require_loaded()
        return [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(self.model.ncam)
        ]

    def body_names(self) -> list[str]:
        """Return list of body names (excluding 'world')."""
        self._require_loaded()
        names = []
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and name != "world":
                names.append(name)
        return names

    def step(self) -> None:
        """Advance the simulation by one timestep."""
        self._require_loaded()
        mujoco.mj_step(self.model, self.data)

    def reset(self) -> None:
        """Reset simulation to initial state."""
        self._require_loaded()
        mujoco.mj_resetData(self.model, self.data)

    def print_summary(self) -> None:
        """Print a human-readable summary of the robot model."""
        self._require_loaded()
        dims = self.get_dimensions()
        print("=" * 60)
        print(f"  Robot : {self.ROBOT_NAME}")
        print(f"  XML   : {self._xml_path}")
        print("=" * 60)
        print(f"  Bodies    ({self.model.nbody - 1}): {', '.join(self.body_names())}")
        print(f"  Joints    ({self.model.njnt }): {', '.join(self.joint_names())}")
        print(f"  Actuators ({self.model.nu   }): {', '.join(self.actuator_names())}")
        print(f"  Sensors   ({self.model.nsensor}): {', '.join(self.sensor_names())}")
        print(f"  Cameras   ({self.model.ncam  }): {', '.join(self.camera_names())}")
        print()
        print(f"  Wheel radius : {dims.wheel_radius:.4f} m")
        print(f"  Track width  : {dims.track_width:.4f} m")
        print(f"  Wheelbase    : {dims.wheelbase:.4f} m")
        print(f"  Total mass   : {dims.mass_total:.2f} kg")
        if dims.description:
            print(f"  Notes        : {dims.description}")
        print("=" * 60)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _require_loaded(self) -> None:
        if self.model is None or self.data is None:
            raise RuntimeError(
                f"{self.__class__.__name__}: call load(xml_path) before using the robot."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Utility: Ackermann geometry
# ─────────────────────────────────────────────────────────────────────────────

def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def _get_continuous_wheel_ik(
    x: float, y: float, 
    v_linear: float, v_lateral: float, v_angular: float, 
    current_angle: float
) -> tuple[float, float]:
    """Compute the required steering angle and drive speed (m/s) for a wheel at (x,y).
    
    Selects between forward and reverse driving to minimize the change from current_angle.
    Returns (target_angle, target_speed) with sign convention for axis="0 0 -1" (Z-down steering).
    """
    vx = v_linear - v_angular * y
    vy = v_lateral + v_angular * x
    speed = math.hypot(vx, vy)
    
    if speed < 1e-6:
        # If not moving, keep the current angle and zero speed
        return current_angle, 0.0
        
    # The physical steering direction (relative to robot +X)
    target_fwd = math.atan2(vy, vx)
    target_rev = normalize_angle(target_fwd + math.pi)
    
    # Since Z-down steering convention negates the physical angle in MuJoCo:
    # Physical angle theta corresponds to joint angle -theta.
    # We un-negate the current joint angle to work in physical coordinates.
    phys_curr = -current_angle
    
    # Calculate angular distance to both solutions
    diff_fwd = normalize_angle(target_fwd - phys_curr)
    diff_rev = normalize_angle(target_rev - phys_curr)
    
    # Select the solution with the smallest rotation
    if abs(diff_fwd) <= abs(diff_rev):
        best_phys = phys_curr + diff_fwd
        best_speed = speed
    else:
        best_phys = phys_curr + diff_rev
        best_speed = -speed
        
    # Convert back to joint coordinates (Z-down negates physical angle)
    return -best_phys, best_speed

def compute_4ws_ik(
    cmd: DriveCommand,
    dims: RobotDimensions,
    current_angles: tuple[float, float, float, float]
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Compute per-wheel steering angles (rad) and drive speeds (rad/s) using continuous 4WS IK.
    
    Returns:
        angles: (fl_a, fr_a, rl_a, rr_a)
        speeds: (fl_s, fr_s, rl_s, rr_s)
    """
    half_l = dims.wheelbase / 2.0
    half_t = dims.track_width / 2.0
    
    curr_fl, curr_fr, curr_rl, curr_rr = current_angles
    
    fl_a, fl_v = _get_continuous_wheel_ik( half_l,  half_t, cmd.v_linear, cmd.v_lateral, cmd.v_angular, curr_fl)
    fr_a, fr_v = _get_continuous_wheel_ik( half_l, -half_t, cmd.v_linear, cmd.v_lateral, cmd.v_angular, curr_fr)
    rl_a, rl_v = _get_continuous_wheel_ik(-half_l,  half_t, cmd.v_linear, cmd.v_lateral, cmd.v_angular, curr_rl)
    rr_a, rr_v = _get_continuous_wheel_ik(-half_l, -half_t, cmd.v_linear, cmd.v_lateral, cmd.v_angular, curr_rr)
    
    speeds = (
        fl_v / dims.wheel_radius,
        fr_v / dims.wheel_radius,
        rl_v / dims.wheel_radius,
        rr_v / dims.wheel_radius,
    )
    
    return (fl_a, fr_a, rl_a, rr_a), speeds
