"""
shared/pink_uniform_policy.py — MINav-faithful Pink Uniform Noise exploration.
================================================================================
Implements the exploration policy from MINav (arXiv:2603.26441) exactly:

    a_t = a_min + (a_max - a_min) · Φ(x_t / σ)

where x_t is a Gaussian pink-noise sequence (β=1, FFT spectral shaping).

Key design points:
  - 3 independent UniformColoredNoise channels for [vx, vy, ωz].
  - 2 Hz policy rate: a new raw action is sampled every 0.5s.
  - Velocity smoothing via LERP at the control loop rate (10 Hz).
  - NO depth/LiDAR input — this is pure correlated noise.
  - NO visit grid, meta-scheduler, or collision avoidance.
  - Safety is handled by an external, independent gate.

The CDF transform (Φ) gives uniform marginal action coverage while
preserving the 1/f temporal autocorrelation structure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from shared.drive_command import DriveCommand
from shared.pink_noise import UniformColoredNoise

logger = logging.getLogger(__name__)


@dataclass
class PolicyConfig:
    """Configuration for the Pink Uniform Noise exploration policy.

    Attributes
    ----------
    beta : float
        Spectral exponent for pink noise. β=1 is MINav default.
    policy_freq_hz : float
        Rate at which new raw actions are sampled from the noise
        generators. MINav paper: 2 Hz.
    control_freq_hz : float
        Rate of the outer control loop (velocity smoothing).
        MINav paper: 20 Hz. Actions are LERP-interpolated between
        policy ticks at this rate.
    smoothing_alpha : float
        LERP blending factor per control tick. Implementation
        parameter (not from MINav).
        0.0 = hold previous, 1.0 = snap to target instantly.
        With 10 control ticks per policy tick (20 Hz / 2 Hz),
        α=0.2 gives smooth convergence over the window.
    vx_range : tuple[float, float]
        Feasible forward velocity range (m/s). MINav maps surge
        to [0, vx_max]; reverse is not used during exploration.
        Ranger-specific conservative limit.
    vy_range : tuple[float, float]
        Feasible lateral (sway) velocity range (m/s).
        Ranger-specific conservative limit.
    wz_range : tuple[float, float]
        Feasible angular (yaw) velocity range (rad/s).
        Ranger-specific conservative limit.
    seed : int or None
        Optional random seed for reproducibility.
    """
    beta: float = 1.0
    policy_freq_hz: float = 2.0
    control_freq_hz: float = 20.0
    smoothing_alpha: float = 0.2
    vx_range: tuple = (0.0, 0.6)
    vy_range: tuple = (-0.3, 0.3)
    wz_range: tuple = (-1.0, 1.0)
    seed: Optional[int] = None


class PinkUniformNoisePolicy:
    """MINav-faithful Pink Uniform Noise exploration policy.

    This class generates temporally correlated exploration actions
    following the MINav paper exactly. Each action dimension has its
    own independent pink-noise generator. New raw actions are sampled
    at ``policy_freq_hz`` (2 Hz), and intermediate control ticks use
    LERP interpolation for smooth velocity transitions.

    Usage
    -----
    Call ``get_action()`` at every control loop tick (e.g., 10 Hz).
    The policy internally tracks when to sample new noise values vs.
    when to interpolate.

    Parameters
    ----------
    config : PolicyConfig
        Policy configuration.
    """

    def __init__(self, config: PolicyConfig = None):
        if config is None:
            config = PolicyConfig()
        self._cfg = config

        # Ticks between policy samples
        # e.g., 10 Hz control / 2 Hz policy = 5 ticks per new sample
        self._ticks_per_sample = max(1, round(
            config.control_freq_hz / config.policy_freq_hz
        ))
        self._tick_counter = 0

        # LERP blending factor
        self._alpha = config.smoothing_alpha

        # 3 independent pink-noise generators (one per action dimension)
        # Each produces Uniform[a_min, a_max] via the Gaussian CDF transform
        seed = config.seed
        self._vx_noise = UniformColoredNoise(
            beta=config.beta, range_val=config.vx_range,
            seed=seed,
        )
        self._vy_noise = UniformColoredNoise(
            beta=config.beta, range_val=config.vy_range,
            seed=(seed + 1) if seed is not None else None,
        )
        self._wz_noise = UniformColoredNoise(
            beta=config.beta, range_val=config.wz_range,
            seed=(seed + 2) if seed is not None else None,
        )

        # Current target action (from latest policy sample)
        self._target = DriveCommand(0.0, 0.0, 0.0)
        # Current smoothed action (output to the robot)
        self._current = DriveCommand(0.0, 0.0, 0.0)

        # Sample the first target immediately
        self._sample_new_target()

        logger.info(
            f"PinkUniformNoisePolicy initialized: β={config.beta}, "
            f"policy={config.policy_freq_hz}Hz, control={config.control_freq_hz}Hz, "
            f"α={config.smoothing_alpha}, "
            f"vx={config.vx_range}, vy={config.vy_range}, ωz={config.wz_range}"
        )

    def _sample_new_target(self) -> None:
        """Sample a new raw action from the 3 pink-noise generators.

        Each generator internally advances its correlated sequence by
        one step, preserving the 1/f temporal structure.
        """
        self._target = DriveCommand(
            v_linear=self._vx_noise.sample(),
            v_lateral=self._vy_noise.sample(),
            v_angular=self._wz_noise.sample(),
        )

    def _lerp_action(self) -> None:
        """Interpolate the current (smoothed) action toward the target.

        current = current + α * (target - current)

        This is MINav's "20 Hz velocity smoothing" adapted to whatever
        control_freq_hz is configured (default 10 Hz).
        """
        a = self._alpha
        self._current = DriveCommand(
            v_linear=self._current.v_linear + a * (self._target.v_linear - self._current.v_linear),
            v_lateral=self._current.v_lateral + a * (self._target.v_lateral - self._current.v_lateral),
            v_angular=self._current.v_angular + a * (self._target.v_angular - self._current.v_angular),
        )

    def get_action(self) -> DriveCommand:
        """Return the next exploration action.

        Call this at every control loop tick. Internally:
        - Every ``_ticks_per_sample`` ticks, a new raw action is
          sampled from the pink-noise generators (2 Hz policy rate).
        - On every tick, the output is LERP-interpolated toward the
          latest target (velocity smoothing).

        Returns
        -------
        DriveCommand
            The smoothed velocity command [vx, vy, ωz].
        """
        # Check if it's time for a new policy sample
        if self._tick_counter >= self._ticks_per_sample:
            self._sample_new_target()
            self._tick_counter = 0

        self._tick_counter += 1

        # LERP toward the target
        self._lerp_action()

        return DriveCommand(
            v_linear=self._current.v_linear,
            v_lateral=self._current.v_lateral,
            v_angular=self._current.v_angular,
        )

    @property
    def target_action(self) -> DriveCommand:
        """The current raw (pre-smoothing) target action."""
        return self._target

    @property
    def config(self) -> PolicyConfig:
        """The policy configuration."""
        return self._cfg
