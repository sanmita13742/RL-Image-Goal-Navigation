"""
ranger_mujoco/exploration_policies.py
============================================================
Defines an abstract base class for exploration policies, along
with concrete implementations (e.g., Linear, White Noise).

DRY Architecture:
The base `ExplorationPolicy` handles all the boilerplate state:
1. Checking LiDAR depth for collision avoidance.
2. Holding actions for a specific "hold time" period.
3. Managing the timer.

Subclasses only need to implement `_sample_noise()` to define 
their specific noise distributions.
"""

import abc
import random
import sys
from pathlib import Path

# Allow importing DriveCommand
sys.path.insert(0, str(Path(__file__).parent.parent))
from robot_base import DriveCommand


class ExplorationPolicy(abc.ABC):
    """Abstract base class for all exploration policies."""
    
    def __init__(self, control_freq: float, hold_time: float):
        """
        Args:
            control_freq: Simulation control frequency (Hz).
            hold_time: How long to hold each sampled action (in seconds).
                       Set to 0.0 to sample a new action every step.
        """
        self.control_freq = control_freq
        # Ensure we hold for at least 1 control step
        self.hold_steps = max(1, int(control_freq * hold_time))
        self.timer = 0
        self.current_cmd = DriveCommand(v_linear=0.0, v_angular=0.0)

    def get_action(self, min_depth: float) -> DriveCommand:
        """Returns a DriveCommand based on the policy and sensor inputs."""
        
        # 1. Collision Avoidance (takes priority over exploration)
        if min_depth < 0.6:
            v = -0.2
            w = self.current_cmd.v_angular
            if abs(w) < 0.5:
                w = random.choice([-1.2, 1.2]) # Turn sharply away
            
            self.current_cmd = DriveCommand(v_linear=v, v_angular=w)
            self.timer = int(self.control_freq * 0.4) # Force hold turn for 0.4s
            return self.current_cmd
            
        # 2. Sample New Action if timer expired
        if self.timer <= 0:
            v, w = self._sample_noise()
            self.current_cmd = DriveCommand(v_linear=v, v_angular=w)
            self.timer = self.hold_steps
            
        # 3. Maintain current action
        self.timer -= 1
        return self.current_cmd

    @abc.abstractmethod
    def _sample_noise(self) -> tuple[float, float]:
        """
        Subclasses implement this to define their noise generation strategy.
        Returns:
            (v_linear, v_angular) tuple.
        """
        pass


class WhiteNoiseExploration(ExplorationPolicy):
    """
    White Noise Exploration Policy:
    Samples a new random action uniformly at every single control step.
    This results in very jittery motion but covers the action space densely.
    """
    def __init__(self, control_freq: float, v_range=(-0.1, 1.5), w_range=(-1.0, 1.0)):
        # hold_time=0.0 means the base class will sample a new noise every step
        super().__init__(control_freq, hold_time=0.0)
        self.v_range = v_range
        self.w_range = w_range

    def _sample_noise(self) -> tuple[float, float]:
        v = random.uniform(*self.v_range)
        w = random.uniform(*self.w_range)
        return v, w


class LinearExploration(ExplorationPolicy):
    """
    Linear Exploration Policy (Piecewise Constant Uniform Noise):
    Samples a random action and holds it for a specific duration, 
    resulting in smoother, more linear piecewise trajectories.
    """
    def __init__(self, control_freq: float, hold_time: float = 1.0, 
                 v_range=(-0.1, 1.5), w_range=(-1.0, 1.0)):
        # We hold the noise for `hold_time` seconds to create linear paths
        super().__init__(control_freq, hold_time=hold_time)
        self.v_range = v_range
        self.w_range = w_range

    def _sample_noise(self) -> tuple[float, float]:
        v = random.uniform(*self.v_range)
        w = random.uniform(*self.w_range)
        
        # Heuristic: occasionally drive perfectly straight or spin in place
        # to ensure good variety in the dataset.
        if random.random() < 0.2:
            w = 0.0
        elif random.random() < 0.2:
            v = 0.0
            
        return v, w
