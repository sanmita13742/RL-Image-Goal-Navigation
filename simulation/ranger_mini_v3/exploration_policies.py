"""
ranger_mujoco/exploration_policies.py
============================================================
Exploration framework based on Motion Primitives, Colored Noise,
Adaptive Scheduling, and Behavioral Loop Detection.
"""

import abc
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import scipy.stats
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent))
from robot_base import DriveCommand


class Primitive(Enum):
    ACKERMANN = 1
    SPIN = 2
    TRAVERSE = 3
    DIAGONAL = 4
    REVERSE = 5


class ColoredNoise:
    """Dependency-free Colored Noise generator (Beta = 0, 1, 2)."""
    def __init__(self, beta: int = 1, cols: int = 16):
        self.beta = beta
        self.cols = cols
        
        # For Pink (Beta=1) Voss-McCartney
        self.vm_array = np.random.randn(cols)
        self.vm_counter = 0
        
        # For Brown (Beta=2) Random Walk
        self.brown_state = 0.0
        self.brown_damping = 0.05 

    def sample(self) -> float:
        if self.beta == 0:
            return np.random.randn()
        elif self.beta == 1:
            c = (self.vm_counter ^ (self.vm_counter + 1)).bit_length() - 1
            if c >= self.cols:
                c = self.cols - 1
            self.vm_array[c] = np.random.randn()
            self.vm_counter += 1
            white = np.random.randn()
            return (np.sum(self.vm_array) + white) / math.sqrt(self.cols + 1)
        elif self.beta == 2:
            self.brown_state = (1.0 - self.brown_damping) * self.brown_state + np.random.randn()
            return self.brown_state / math.sqrt(1.0 / (2.0 * self.brown_damping))
        else:
            raise ValueError("Beta must be 0 (White), 1 (Pink), or 2 (Brown).")


class UniformColoredNoise:
    """Transforms Colored Noise into a Uniform marginal distribution."""
    def __init__(self, beta: int = 1, range_val: tuple = (-1.0, 1.0)):
        self.noise = ColoredNoise(beta=beta)
        self.range = range_val
        
    def sample(self) -> float:
        raw = self.noise.sample()
        u = scipy.stats.norm.cdf(raw)
        return self.range[0] + (self.range[1] - self.range[0]) * u


class BaseExploration(abc.ABC):
    """Abstract base class with stateful, adaptive holonomic collision recovery."""
    
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
        self.window_size = int(control_freq * 30) # 30 second window
        
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

    def get_action(self, depth_img: np.ndarray, x: float, y: float, yaw: float) -> tuple[DriveCommand, Primitive]:
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
        
        # ─── Collision Recovery State Machine ───
        
        if self.recovery_timer > 0:
            self.recovery_timer -= 1
            self.active_primitive_timer += self.dt
            if self.recovery_timer == 0:
                self.recovery_state = "NORMAL"
            return self.recovery_cmd, self.current_primitive
            
        # 1. Reactive Escape (Obstacles)
        if min_depth < 0.6:
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
            elif min_depth < 0.3:
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
        # If we have collected enough history and efficiency is near zero
        efficiency = sum(self.visited_history) / len(self.visited_history) if len(self.visited_history) > 0 else 1.0
        
        if len(self.visited_history) == self.window_size and efficiency < 0.005:
            # We are trapped in a behavioral loop (e.g. corridor sweeping, huge revisits)
            self.recovery_state = "PROACTIVE_ESCAPE"
            # Force crab walk away or reverse
            if random.random() < 0.5:
                lat = random.choice([-1.0, 1.0])
                self.recovery_cmd = DriveCommand(v_linear=0.0, v_lateral=lat, v_angular=0.0)
                self.recovery_timer = int(self.control_freq * 3.0) # Escape for longer
                self._switch_primitive(Primitive.TRAVERSE)
            else:
                self.recovery_cmd = DriveCommand(v_linear=-0.8, v_lateral=0.0, v_angular=random.uniform(-1, 1))
                self.recovery_timer = int(self.control_freq * 3.0)
                self._switch_primitive(Primitive.REVERSE)
                
            self.visited_history = [] # Reset history to avoid cascading proactive escapes
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
    def _sample_primitive(self, efficiency: float) -> tuple[DriveCommand, Primitive, float]:
        pass

    @abc.abstractmethod
    def _modulate_primitive(self, cmd: DriveCommand, prim: Primitive) -> DriveCommand:
        pass


class PrimitiveExplorationPolicy(BaseExploration):
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
        
    def _predict_cell(self, prim: Primitive, dist: float = 1.5) -> tuple[int, int]:
        """Roughly predict the future cell if this primitive is chosen."""
        px, py = self.x, self.y
        if prim == Primitive.ACKERMANN:
            px += math.cos(self.yaw) * dist
            py += math.sin(self.yaw) * dist
        elif prim == Primitive.REVERSE:
            px -= math.cos(self.yaw) * dist
            py -= math.sin(self.yaw) * dist
        elif prim == Primitive.TRAVERSE:
            # Assuming lateral right is -90 deg from yaw
            px += math.cos(self.yaw - math.pi/2) * dist
            py += math.sin(self.yaw - math.pi/2) * dist
        elif prim == Primitive.DIAGONAL:
            px += math.cos(self.yaw - math.pi/4) * dist
            py += math.sin(self.yaw - math.pi/4) * dist
        
        return (int(math.floor(px / self.grid_res)), int(math.floor(py / self.grid_res)))

    def _sample_primitive(self, efficiency: float) -> tuple[DriveCommand, Primitive, float]:
        probs = self.base_probs.copy()
        
        # 1. Adaptive Meta-Scheduler: Boost escape primitives if coverage stagnates
        if len(self.visited_history) == self.window_size and efficiency < 0.05:
            probs[Primitive.ACKERMANN] *= 0.5
            probs[Primitive.TRAVERSE]  *= 2.0
            probs[Primitive.DIAGONAL]  *= 1.5
            probs[Primitive.REVERSE]   *= 1.5
            
        # 2. Soft Revisitation Penalty: Repel from hotspots
        for p in Primitive:
            projected = self._predict_cell(p)
            visits = self.visit_grid[projected]
            if visits > 0:
                probs[p] *= math.exp(-0.1 * visits)
                
        # Normalize
        total_p = sum(probs.values())
        if total_p == 0:
            probs = self.base_probs.copy() # fallback
            total_p = sum(probs.values())
            
        keys = list(probs.keys())
        weights = [probs[k]/total_p for k in keys]
        
        prim = random.choices(keys, weights=weights, k=1)[0]
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
