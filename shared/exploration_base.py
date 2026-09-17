"""
shared/exploration_base.py — Platform-independent exploration policies.
======================================================================
Extracted from simulation/ranger_mini_v3/exploration_policies.py.
Uses shared.drive_command.DriveCommand instead of robot_base.DriveCommand,
so there is NO MuJoCo dependency.

The collision-avoidance depth logic remains — the real-world pipeline will
pass LiDAR-projected depth images through the same interface.
"""

import abc
import math
import random
from collections import defaultdict
from enum import Enum

import numpy as np

from shared.drive_command import DriveCommand
from shared.pink_noise import UniformColoredNoise


class Primitive(Enum):
    ACKERMANN = 1
    SPIN = 2
    TRAVERSE = 3
    DIAGONAL = 4
    REVERSE = 5


# ============================================================
# Base Exploration Policy
# ============================================================

class BaseExploration(abc.ABC):
    """Abstract base class with stateful, adaptive holonomic collision recovery.

    The depth_img parameter in get_action() accepts any 2D numpy array
    representing obstacle proximity. In MuJoCo this comes from the
    depth renderer; on the real robot it comes from LiDAR projection.
    """

    def __init__(self, control_freq: float):
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq

        self.current_cmd = DriveCommand()
        self.timer = 0

        # State Machine
        self.recovery_state = "NORMAL"
        self.recovery_timer = 0
        self.recovery_cmd = DriveCommand()

        # Metrics & Tracking
        self.primitive_counts = {p: 0 for p in Primitive}
        self.primitive_duration_sum = {p: 0.0 for p in Primitive}
        self.primitive_transitions = {p: {p2: 0 for p2 in Primitive} for p in Primitive}
        self.current_primitive = Primitive.ACKERMANN
        self.active_primitive_timer = 0.0

        # Adaptive Meta-Scheduler & Behavioral Loop Detection
        self.visit_grid = defaultdict(int)
        self.grid_res = 1.0
        self.visited_history = []  # Track new cells over window
        self.window_size = int(control_freq * 30)  # 30 second window

        self.x, self.y, self.yaw = 0.0, 0.0, 0.0

    def _switch_primitive(self, new_prim: Primitive):
        self.primitive_transitions[self.current_primitive][new_prim] += 1
        self.primitive_counts[self.current_primitive] += 1
        self.primitive_duration_sum[self.current_primitive] += self.active_primitive_timer

        self.current_primitive = new_prim
        self.active_primitive_timer = 0.0

    def _update_visit_grid(self, x: float, y: float):
        cell = (int(math.floor(x / self.grid_res)), int(math.floor(y / self.grid_res)))
        if self.visit_grid[cell] == 0:
            self.visited_history.append(1)
        else:
            self.visited_history.append(0)

        self.visit_grid[cell] += 1

        if len(self.visited_history) > self.window_size:
            self.visited_history.pop(0)

    def get_action(self, depth_img: np.ndarray, x: float, y: float, yaw: float) -> tuple:
        """Compute the next exploration action.

        Parameters
        ----------
        depth_img : np.ndarray
            2D array of obstacle proximity values. Lower = closer obstacle.
            In MuJoCo: direct depth render. On real robot: LiDAR projection.
        x, y : float
            Robot position in metres.
        yaw : float
            Robot heading in radians.

        Returns
        -------
        (DriveCommand, Primitive)
        """
        self.x, self.y, self.yaw = x, y, yaw
        self._update_visit_grid(x, y)

        h, w = depth_img.shape
        left_third = depth_img[:, :w//3]
        center_third = depth_img[:, w//3:2*w//3]
        right_third = depth_img[:, 2*w//3:]

        min_c = np.min(center_third)
        min_l = np.min(left_third)
        min_r = np.min(right_third)
        min_depth = min(min_c, min_l, min_r)

        # DEBUG PRINT TO DIAGNOSE SPINNING
        if self.timer % 10 == 0:
            print(f"[DEBUG EXPLORATION] min_l: {min_l:.2f}, min_c: {min_c:.2f}, min_r: {min_r:.2f} | min_depth: {min_depth:.2f} | current_prim: {self.current_primitive.name} | state: {self.recovery_state}")

        # ─── Collision Recovery State Machine ───

        if self.recovery_timer > 0:
            self.recovery_timer -= 1
            self.active_primitive_timer += self.dt
            if self.recovery_timer == 0:
                self.recovery_state = "NORMAL"
            return self.recovery_cmd, self.current_primitive

        # 1. Reactive Escape (Obstacles)
        if min_depth < 0.80:
            if min_l > 1.0:
                self.recovery_state = "REACTIVE_ESCAPE_TRAVERSE"
                self.recovery_cmd = DriveCommand(v_linear=0.0, v_lateral=0.8, v_angular=0.0)
                self.recovery_timer = int(self.control_freq * 1.5)
                self._switch_primitive(Primitive.TRAVERSE)
            elif min_r > 1.0:
                self.recovery_state = "REACTIVE_ESCAPE_TRAVERSE"
                self.recovery_cmd = DriveCommand(v_linear=0.0, v_lateral=-0.8, v_angular=0.0)
                self.recovery_timer = int(self.control_freq * 1.5)
                self._switch_primitive(Primitive.TRAVERSE)
            elif min_depth < 0.70:
                self.recovery_state = "REACTIVE_ESCAPE_REVERSE"
                steer = -1.0 if min_l > min_r else 1.0
                self.recovery_cmd = DriveCommand(v_linear=-0.8, v_lateral=0.0, v_angular=steer)
                self.recovery_timer = int(self.control_freq * 1.5)
                self._switch_primitive(Primitive.REVERSE)
            else:
                self.recovery_state = "REACTIVE_ESCAPE_SPIN"
                spin = 1.5 if min_l > min_r else -1.5
                self.recovery_cmd = DriveCommand(v_linear=0.0, v_lateral=0.0, v_angular=spin)
                self.recovery_timer = int(self.control_freq * 1.0)
                self._switch_primitive(Primitive.SPIN)

            self.active_primitive_timer += self.dt
            return self.recovery_cmd, self.current_primitive

        # 2. Proactive Escape (Behavioral Loop Detection)
        efficiency = sum(self.visited_history) / len(self.visited_history) if len(self.visited_history) > 0 else 1.0

        if len(self.visited_history) == self.window_size and efficiency < 0.005:
            self.recovery_state = "PROACTIVE_ESCAPE"
            if random.random() < 0.5:
                lat = random.choice([-1.0, 1.0])
                self.recovery_cmd = DriveCommand(v_linear=0.0, v_lateral=lat, v_angular=0.0)
                self.recovery_timer = int(self.control_freq * 3.0)
                self._switch_primitive(Primitive.TRAVERSE)
            else:
                self.recovery_cmd = DriveCommand(v_linear=-0.8, v_lateral=0.0, v_angular=random.uniform(-1, 1))
                self.recovery_timer = int(self.control_freq * 3.0)
                self._switch_primitive(Primitive.REVERSE)

            self.visited_history = []
            self.active_primitive_timer += self.dt
            return self.recovery_cmd, self.current_primitive

        # ─── Normal Exploration ───

        if self.timer <= 0:
            new_cmd, new_prim, hold_time = self._sample_primitive(efficiency)

            if new_prim != self.current_primitive:
                self._switch_primitive(new_prim)

            self.current_cmd = new_cmd
            self.timer = int(self.control_freq * hold_time)

        else:
            self.current_cmd = self._modulate_primitive(self.current_cmd, self.current_primitive)

        self.timer -= 1
        self.active_primitive_timer += self.dt
        return self.current_cmd, self.current_primitive

    @abc.abstractmethod
    def _sample_primitive(self, efficiency: float) -> tuple:
        pass

    @abc.abstractmethod
    def _modulate_primitive(self, cmd: DriveCommand, prim: Primitive) -> DriveCommand:
        pass


# ============================================================
# Primitive Exploration Policy (Pink Noise)
# ============================================================

class PrimitiveExplorationPolicy(BaseExploration):
    """Concrete exploration policy using pink-noise-modulated motion primitives.

    This is the MINav paper's exploration strategy: 5 motion primitives
    (Ackermann, Spin, Traverse, Diagonal, Reverse) with pink-noise-sampled
    velocities and adaptive meta-scheduling.
    """

    def __init__(self, control_freq: float, beta: int = 1):
        super().__init__(control_freq)
        self.beta = beta

        self.speed_noise = UniformColoredNoise(beta, range_val=(0.3, 1.8))
        self.steer_noise = UniformColoredNoise(beta, range_val=(-1.0, 1.0))
        self.lat_noise   = UniformColoredNoise(beta, range_val=(-1.0, 1.0))

        self.base_probs = {
            Primitive.ACKERMANN: 0.50,
            Primitive.DIAGONAL:  0.20,
            Primitive.TRAVERSE:  0.10,
            Primitive.SPIN:      0.10,
            Primitive.REVERSE:   0.10,
        }

    def _predict_cell(self, prim: Primitive, dist: float = 1.5) -> tuple:
        """Roughly predict the future cell if this primitive is chosen."""
        px, py = self.x, self.y
        if prim == Primitive.ACKERMANN:
            px += math.cos(self.yaw) * dist
            py += math.sin(self.yaw) * dist
        elif prim == Primitive.REVERSE:
            px -= math.cos(self.yaw) * dist
            py -= math.sin(self.yaw) * dist
        elif prim == Primitive.TRAVERSE:
            px += math.cos(self.yaw - math.pi/2) * dist
            py += math.sin(self.yaw - math.pi/2) * dist
        elif prim == Primitive.DIAGONAL:
            px += math.cos(self.yaw - math.pi/4) * dist
            py += math.sin(self.yaw - math.pi/4) * dist

        return (int(math.floor(px / self.grid_res)), int(math.floor(py / self.grid_res)))

    def _sample_primitive(self, efficiency: float) -> tuple:
        # SIMPLIFIED MODE FOR REAL WORLD: 
        # Only use ACKERMANN (forward/turn) for normal exploration.
        # Ignore the complex visit grid and primitive switching.
        prim = Primitive.ACKERMANN
        hold_time = random.uniform(1.0, 3.0)
        return self._modulate_primitive(DriveCommand(), prim), prim, hold_time

    def _modulate_primitive(self, cmd: DriveCommand, prim: Primitive) -> DriveCommand:
        v = self.speed_noise.sample()
        w = self.steer_noise.sample()
        lat = self.lat_noise.sample()

        if prim == Primitive.ACKERMANN:
            return DriveCommand(v_linear=v, v_lateral=0.0, v_angular=w)
        elif prim == Primitive.SPIN:
            return DriveCommand(v_linear=0.0, v_lateral=0.0, v_angular=w * 1.5)
        elif prim == Primitive.TRAVERSE:
            return DriveCommand(v_linear=0.0, v_lateral=lat, v_angular=0.0)
        elif prim == Primitive.DIAGONAL:
            return DriveCommand(v_linear=v, v_lateral=lat, v_angular=0.0)
        elif prim == Primitive.REVERSE:
            return DriveCommand(v_linear=-v * 0.5, v_lateral=0.0, v_angular=w)

        return DriveCommand()
