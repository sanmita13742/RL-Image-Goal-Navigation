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
        self.x, self.y, self.yaw = x, y, yaw

        h, w = depth_img.shape
        left_third = depth_img[:, :w//3]
        center_third = depth_img[:, w//3:2*w//3]
        right_third = depth_img[:, 2*w//3:]

        min_c = np.min(center_third)
        min_l = np.min(left_third)
        min_r = np.min(right_third)
        min_depth = min(min_c, min_l, min_r)

        # DEBUG PRINT
        if self.timer % 10 == 0:
            print(f"[DEBUG] min_l: {min_l:.2f}, min_c: {min_c:.2f}, min_r: {min_r:.2f} | min_depth: {min_depth:.2f}")

        # ─── Simple Collision Avoidance ───
        if min_depth < 0.80:
        # ─── Collision Recovery State Machine ───
        if self.recovery_timer > 0:
            self.recovery_timer -= 1
            return self.recovery_cmd, Primitive.REVERSE

        if min_depth < 0.80:
            # We are close to an obstacle. Execute a human-like car escape maneuver.
            # Steer towards the side with more space while backing up.
            steer = -1.0 if min_l > min_r else 1.0
            
            # Shift into reverse and steer away
            self.recovery_cmd = DriveCommand(v_linear=-0.25, v_lateral=0.0, v_angular=steer)
            self.recovery_timer = int(self.control_freq * 2.0) # Hold reverse for 2 seconds to clear it
            
            return self.recovery_cmd, Primitive.REVERSE

        # ─── Normal Exploration (Pink Uniform Noise) ───
        if self.timer <= 0:
            self.current_cmd, _, hold_time = self._sample_primitive(1.0)
            self.timer = int(self.control_freq * hold_time)

        self.timer -= 1
        return self.current_cmd, Primitive.ACKERMANN

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
    """Concrete exploration policy using pink-noise-modulated motion."""

    def __init__(self, control_freq: float, beta: int = 1):
        super().__init__(control_freq)
        self.beta = beta
        # Safe indoor speeds
        self.speed_noise = UniformColoredNoise(beta, range_val=(0.15, 0.35))
        self.steer_noise = UniformColoredNoise(beta, range_val=(-0.8, 0.8))

    def _sample_primitive(self, efficiency: float) -> tuple:
        """Sample a new sweeping arc from the pink noise generator."""
        cmd = DriveCommand(
            v_linear=self.speed_noise.sample(),
            v_lateral=0.0,
            v_angular=self.steer_noise.sample()
        )
        # Hold this arc for 1 to 3 seconds to create sweeping curves
        hold_time = random.uniform(1.0, 3.0)
        return cmd, Primitive.ACKERMANN, hold_time

    def _modulate_primitive(self, cmd: DriveCommand, prim: Primitive) -> DriveCommand:
        return cmd
