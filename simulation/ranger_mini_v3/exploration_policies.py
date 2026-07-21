"""
ranger_mujoco/exploration_policies.py
============================================================
Defines an abstract base class for exploration policies, along
with concrete implementations including True Pink Noise and OU.

Policies implemented:
  - WhiteNoiseExploration   : Jittery, uniform baseline
  - LinearExploration       : Piecewise-constant holding
  - OUExploration           : Ornstein-Uhlenbeck temporally correlated process
  - TruePinkExploration     : Voss-McCartney 1/f Pink Noise generator
  - UniformPinkExploration  : True Pink mapped to Uniform distribution

Includes stateful collision avoidance to prevent corner loops.
"""

import abc
import math
import random
import sys
from pathlib import Path
import numpy as np
import scipy.stats

# Allow importing DriveCommand
sys.path.insert(0, str(Path(__file__).parent.parent))
from robot_base import DriveCommand


class VossMcCartneyPinkNoise:
    """Dependency-free 1/f Pink Noise generator using the Voss-McCartney algorithm."""
    def __init__(self, cols=16):
        self.cols = cols
        self.array = np.random.randn(cols)
        self.counter = 0

    def sample(self) -> float:
        # Determine which column to update based on trailing zeros of counter
        # c = number of trailing zeros
        c = (self.counter ^ (self.counter + 1)).bit_length() - 1
        if c >= self.cols:
            c = self.cols - 1
            
        self.array[c] = np.random.randn()
        self.counter += 1
        
        # White noise component for the 0th index is always updated
        white = np.random.randn()
        
        return (np.sum(self.array) + white) / math.sqrt(self.cols + 1)


class ExplorationPolicy(abc.ABC):
    """Abstract base class for all exploration policies."""
    
    def __init__(self, control_freq: float, hold_time: float):
        self.control_freq = control_freq
        self.hold_steps = max(1, int(control_freq * hold_time))
        self.timer = 0
        self.current_cmd = DriveCommand(v_linear=0.0, v_angular=0.0)
        
        # Stateful collision avoidance
        self._avoid_timer = 0
        self._avoid_cmd = DriveCommand()
        self._evasion_spin_dir = 0.0

    def get_action(self, min_depth: float) -> DriveCommand:
        """Multi-stage stateful collision avoidance."""
        
        # Stage 1: Active avoidance manoeuvre in progress
        if self._avoid_timer > 0:
            self._avoid_timer -= 1
            return self._avoid_cmd
        
        # Stage 2: CRITICAL DANGER (< 0.4m) — Hard reverse and spin
        if min_depth < 0.4:
            if self._evasion_spin_dir == 0.0:
                self._evasion_spin_dir = random.choice([-1.0, 1.0])
            self._avoid_cmd = DriveCommand(v_linear=-0.8, v_lateral=0.0, v_angular=self._evasion_spin_dir * 1.5)
            self._avoid_timer = int(self.control_freq * 0.8) # Commit to evasion for 0.8s
            self.timer = 0
            return self._avoid_cmd

        # Stage 3: CAUTION (< 1.2m) — Stateful turning away from wall
        if min_depth < 1.2:
            factor = (min_depth - 0.4) / 0.8 # 0.0 to 1.0
            v = self.current_cmd.v_linear * factor * 0.5
            
            # Commit to a turning direction so we don't snake along the wall
            if self._evasion_spin_dir == 0.0:
                self._evasion_spin_dir = 1.0 if self.current_cmd.v_angular > 0 else -1.0
                if abs(self.current_cmd.v_angular) < 0.2:
                    self._evasion_spin_dir = random.choice([-1.0, 1.0])
                    
            w = self._evasion_spin_dir * 1.0
            return DriveCommand(v_linear=max(v, 0.0), v_lateral=0.0, v_angular=w)
            
        # CLEAR ZONE: Reset evasion state
        self._evasion_spin_dir = 0.0
            
        # Stage 4: Normal exploration
        if self.timer <= 0:
            v, w = self._sample_noise()
            self.current_cmd = DriveCommand(v_linear=v, v_angular=w)
            self.timer = self.hold_steps
            
        self.timer -= 1
        return self.current_cmd

    @abc.abstractmethod
    def _sample_noise(self) -> tuple[float, float]:
        pass


class WhiteNoiseExploration(ExplorationPolicy):
    def __init__(self, control_freq: float, v_range=(-0.1, 1.5), w_range=(-1.0, 1.0)):
        super().__init__(control_freq, hold_time=0.0)
        self.v_range = v_range
        self.w_range = w_range

    def _sample_noise(self) -> tuple[float, float]:
        v = random.uniform(*self.v_range)
        w = random.uniform(*self.w_range)
        return v, w


class LinearExploration(ExplorationPolicy):
    def __init__(self, control_freq: float, hold_time: float = 1.0, 
                 v_range=(-0.1, 1.5), w_range=(-1.0, 1.0)):
        super().__init__(control_freq, hold_time=hold_time)
        self.v_range = v_range
        self.w_range = w_range

    def _sample_noise(self) -> tuple[float, float]:
        v = random.uniform(*self.v_range)
        w = random.uniform(*self.w_range)
        if random.random() < 0.2:
            w = 0.0
        elif random.random() < 0.2:
            v = 0.0
        return v, w


class OUExploration(ExplorationPolicy):
    """Temporally correlated Ornstein-Uhlenbeck process (Red/Brownian noise)."""
    def __init__(self, control_freq: float,
                 v_mean: float = 0.8, v_sigma: float = 0.6,
                 w_mean: float = 0.0, w_sigma: float = 0.5,
                 theta: float = 0.15,
                 v_range: tuple = (0.0, 2.0),
                 w_range: tuple = (-1.0, 1.0)):
        super().__init__(control_freq, hold_time=0.0)
        self.v_mean, self.v_sigma = v_mean, v_sigma
        self.w_mean, self.w_sigma = w_mean, w_sigma
        self.theta = theta
        self.v_range = v_range
        self.w_range = w_range
        self.dt = 1.0 / control_freq
        self._v_state = v_mean
        self._w_state = w_mean
    
    def _sample_noise(self) -> tuple[float, float]:
        sqrt_dt = math.sqrt(self.dt)
        self._v_state += self.theta * (self.v_mean - self._v_state) * self.dt + self.v_sigma * sqrt_dt * np.random.randn()
        self._w_state += self.theta * (self.w_mean - self._w_state) * self.dt + self.w_sigma * sqrt_dt * np.random.randn()
        
        v = max(self.v_range[0], min(self.v_range[1], self._v_state))
        w = max(self.w_range[0], min(self.w_range[1], self._w_state))
        return v, w


class TruePinkExploration(ExplorationPolicy):
    """1/f Pink Noise scaled directly to bounds."""
    def __init__(self, control_freq: float, 
                 v_range: tuple = (0.0, 1.5), 
                 w_range: tuple = (-1.0, 1.0)):
        super().__init__(control_freq, hold_time=0.0)
        self.v_range = v_range
        self.w_range = w_range
        self.v_gen = VossMcCartneyPinkNoise()
        self.w_gen = VossMcCartneyPinkNoise()
        
    def _sample_noise(self) -> tuple[float, float]:
        # Generate pink noise (approx std ~1.0, mean ~0)
        raw_v = self.v_gen.sample()
        raw_w = self.w_gen.sample()
        
        # Scale into ranges assuming ~99% falls within [-3, 3]
        v = self.v_range[0] + (self.v_range[1] - self.v_range[0]) * ((raw_v / 6.0) + 0.5)
        w = self.w_range[0] + (self.w_range[1] - self.w_range[0]) * ((raw_w / 6.0) + 0.5)
        
        v = max(self.v_range[0], min(self.v_range[1], v))
        w = max(self.w_range[0], min(self.w_range[1], w))
        return v, w


class UniformPinkExploration(ExplorationPolicy):
    """Pink Noise transformed via standard normal CDF to guarantee a Uniform marginal distribution."""
    def __init__(self, control_freq: float, 
                 v_range: tuple = (-0.2, 1.5), 
                 w_range: tuple = (-1.0, 1.0)):
        super().__init__(control_freq, hold_time=0.0)
        self.v_range = v_range
        self.w_range = w_range
        self.v_gen = VossMcCartneyPinkNoise()
        self.w_gen = VossMcCartneyPinkNoise()
        
    def _sample_noise(self) -> tuple[float, float]:
        # raw is ~N(0, 1) but with 1/f spectrum
        raw_v = self.v_gen.sample()
        raw_w = self.w_gen.sample()
        
        # Map through CDF to get perfectly uniform [0, 1] marginals while keeping 1/f dynamics
        u_v = scipy.stats.norm.cdf(raw_v)
        u_w = scipy.stats.norm.cdf(raw_w)
        
        # Scale to range
        v = self.v_range[0] + (self.v_range[1] - self.v_range[0]) * u_v
        w = self.w_range[0] + (self.w_range[1] - self.w_range[0]) * u_w
        
        return v, w
