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

def ackermann_angles(
    v_angular: float,
    v_linear:  float,
    wheelbase: float,
    track_width: float,
    max_steer_rad: float = math.radians(35),
) -> tuple[float, float, float, float]:
    """
    Compute per-wheel steering angles and drive speeds for an Ackermann vehicle.

    For pure differential (v_angular != 0, v_linear == 0) we use a virtual
    turning-radius approximation so the robot can spin in place.

    Returns
    -------
    fl_angle, fr_angle, rl_angle, rr_angle  (radians, + = steer left)
    """
    if abs(v_angular) < 1e-6:
        # Straight line
        return 0.0, 0.0, 0.0, 0.0

    if abs(v_linear) < 1e-6:
        # In-place spin — use small virtual radius
        R = wheelbase * 0.5
    else:
        R = v_linear / v_angular   # signed turning radius

    # Ackermann inner / outer angles
    # Front wheels steer, rear wheels counter-steer for 4WS
    try:
        fl_angle = math.atan2(wheelbase, R - track_width / 2)
        fr_angle = math.atan2(wheelbase, R + track_width / 2)
        rl_angle = -math.atan2(wheelbase * 0.5, R - track_width / 2)
        rr_angle = -math.atan2(wheelbase * 0.5, R + track_width / 2)
    except ZeroDivisionError:
        fl_angle = fr_angle = rl_angle = rr_angle = 0.0

    # Clamp
    clamp = lambda a: max(-max_steer_rad, min(max_steer_rad, a))
    return clamp(fl_angle), clamp(fr_angle), clamp(rl_angle), clamp(rr_angle)


def ackermann_wheel_speeds(
    v_linear:   float,
    v_angular:  float,
    wheel_radius: float,
    wheelbase:    float,
    track_width:  float,
) -> tuple[float, float, float, float]:
    """
    Compute per-wheel drive speeds (rad/s) for an Ackermann vehicle.
    Front-left, Front-right, Rear-left, Rear-right order.
    """
    if abs(v_angular) < 1e-6:
        w = v_linear / wheel_radius
        return w, w, w, w

    if abs(v_linear) < 1e-6:
        R = wheelbase * 0.5
    else:
        R = v_linear / v_angular

    half_t = track_width / 2.0
    half_l = wheelbase   / 2.0

    def speed(rx: float, ry: float) -> float:
        r_wheel = math.hypot(rx, ry)
        v_wheel = v_angular * r_wheel
        return v_wheel / wheel_radius

    fl = speed(half_l,  R - half_t)
    fr = speed(half_l,  R + half_t)
    rl = speed(-half_l, R - half_t)
    rr = speed(-half_l, R + half_t)
    return fl, fr, rl, rr
