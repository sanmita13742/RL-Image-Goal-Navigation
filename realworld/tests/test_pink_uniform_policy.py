"""
realworld/tests/test_pink_uniform_policy.py — Tests for the MINav-faithful policy.
====================================================================================
These tests run OFFLINE (no ROS 2 needed). They validate that
PinkUniformNoisePolicy produces correct action distributions,
temporal correlation, and LERP smoothing.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from shared.pink_uniform_policy import PinkUniformNoisePolicy, PolicyConfig
from shared.drive_command import DriveCommand


class TestPinkUniformNoisePolicy:
    """Tests for the MINav Pink Uniform Noise exploration policy."""

    def _make_policy(self, **kwargs) -> PinkUniformNoisePolicy:
        defaults = dict(
            beta=1.0,
            policy_freq_hz=2.0,
            control_freq_hz=20.0,
            smoothing_alpha=0.2,
            vx_range=(0.0, 0.6),
            vy_range=(-0.3, 0.3),
            wz_range=(-1.0, 1.0),
            seed=42,
        )
        defaults.update(kwargs)
        return PinkUniformNoisePolicy(PolicyConfig(**defaults))

    def test_returns_drive_command(self):
        """get_action() should return a DriveCommand."""
        policy = self._make_policy()
        cmd = policy.get_action()
        assert isinstance(cmd, DriveCommand)

    def test_actions_in_range(self):
        """All actions should be within configured ranges."""
        policy = self._make_policy()
        for _ in range(2000):
            cmd = policy.get_action()
            assert 0.0 <= cmd.v_linear <= 0.6, f"vx={cmd.v_linear} out of range"
            assert -0.3 <= cmd.v_lateral <= 0.3, f"vy={cmd.v_lateral} out of range"
            assert -1.0 <= cmd.v_angular <= 1.0, f"wz={cmd.v_angular} out of range"

    def test_temporal_autocorrelation_vx(self):
        """vx should have positive lag-1 autocorrelation (pink noise smoothness)."""
        policy = self._make_policy()
        vx = np.array([policy.get_action().v_linear for _ in range(5000)])

        mean = vx.mean()
        n = len(vx)
        c0 = np.sum((vx - mean) ** 2) / n
        c1 = np.sum((vx[:-1] - mean) * (vx[1:] - mean)) / n
        rho = c1 / c0 if c0 > 0 else 0

        assert rho > 0.5, (
            f"vx should have high autocorrelation (pink noise + LERP), got ρ={rho:.3f}"
        )

    def test_temporal_autocorrelation_wz(self):
        """ωz should have positive lag-1 autocorrelation."""
        policy = self._make_policy()
        wz = np.array([policy.get_action().v_angular for _ in range(5000)])

        mean = wz.mean()
        n = len(wz)
        c0 = np.sum((wz - mean) ** 2) / n
        c1 = np.sum((wz[:-1] - mean) * (wz[1:] - mean)) / n
        rho = c1 / c0 if c0 > 0 else 0

        assert rho > 0.5, (
            f"ωz should have high autocorrelation (pink noise + LERP), got ρ={rho:.3f}"
        )

    def test_uniform_coverage_vx(self):
        """vx should cover its full range with roughly uniform distribution."""
        policy = self._make_policy()
        vx = np.array([policy.get_action().v_linear for _ in range(10000)])

        # Check quartiles
        lo, hi = 0.0, 0.6
        q_edges = np.linspace(lo, hi, 5)
        for i in range(4):
            frac = np.sum((vx >= q_edges[i]) & (vx < q_edges[i + 1])) / len(vx)
            assert frac > 0.10, (
                f"vx quartile {i} has only {frac:.2%} of samples — poor coverage"
            )

    def test_uniform_coverage_wz(self):
        """ωz should cover its full range with roughly uniform distribution."""
        policy = self._make_policy()
        wz = np.array([policy.get_action().v_angular for _ in range(10000)])

        lo, hi = -1.0, 1.0
        q_edges = np.linspace(lo, hi, 5)
        for i in range(4):
            frac = np.sum((wz >= q_edges[i]) & (wz < q_edges[i + 1])) / len(wz)
            assert frac > 0.10, (
                f"ωz quartile {i} has only {frac:.2%} of samples — poor coverage"
            )

    def test_lerp_smoothing_reduces_jitter(self):
        """With LERP smoothing, consecutive actions should be closer than raw samples."""
        # Compare smoothed policy vs snap-to-target (alpha=1.0)
        policy_smooth = self._make_policy(smoothing_alpha=0.2, seed=42)
        policy_snap = self._make_policy(smoothing_alpha=1.0, seed=42)

        n = 2000
        smooth_vx = np.array([policy_smooth.get_action().v_linear for _ in range(n)])
        snap_vx = np.array([policy_snap.get_action().v_linear for _ in range(n)])

        # Mean absolute step change should be smaller for smoothed
        smooth_jitter = np.mean(np.abs(np.diff(smooth_vx)))
        snap_jitter = np.mean(np.abs(np.diff(snap_vx)))

        assert smooth_jitter < snap_jitter, (
            f"LERP smoothing should reduce jitter: smooth={smooth_jitter:.4f} "
            f"vs snap={snap_jitter:.4f}"
        )

    def test_policy_ticks_per_sample(self):
        """With 20 Hz control and 2 Hz policy, should sample every 10 ticks."""
        policy = self._make_policy()
        assert policy._ticks_per_sample == 10

    def test_reproducibility_with_seed(self):
        """Same seed should produce identical sequences."""
        policy1 = self._make_policy(seed=123)
        policy2 = self._make_policy(seed=123)

        cmds1 = [(c.v_linear, c.v_lateral, c.v_angular)
                 for c in (policy1.get_action() for _ in range(100))]
        cmds2 = [(c.v_linear, c.v_lateral, c.v_angular)
                 for c in (policy2.get_action() for _ in range(100))]

        for i, (c1, c2) in enumerate(zip(cmds1, cmds2)):
            assert c1 == c2, f"Step {i}: {c1} != {c2}"

    def test_no_high_freq_oscillation(self):
        """ωz should not have excessive consecutive sign changes."""
        policy = self._make_policy()
        wz = np.array([policy.get_action().v_angular for _ in range(5000)])

        signs = np.sign(wz)
        signs[np.abs(wz) < 0.01] = 0
        nonzero = signs[signs != 0]

        if len(nonzero) < 2:
            return  # Not enough data

        flips = np.sum(nonzero[:-1] != nonzero[1:])
        flip_rate = flips / (len(nonzero) - 1)

        assert flip_rate < 0.30, (
            f"ωz sign-flip rate {flip_rate:.2%} exceeds 30% threshold — "
            "high-frequency oscillation detected"
        )
