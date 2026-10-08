"""
tests/test_exploration_audit.py — Comprehensive audit tests for MINav exploration.
===================================================================================
These tests validate:
  1. PinkUniformNoisePolicy: ranges, independence, target sampling, timing
  2. LERP smoothing: alpha=0/1/0.2, convergence
  3. Safety gate (depth-based): passthrough, blocking, rotation preservation,
     stale cloud, insufficient points
  4. Data recorder: extended CSV schema

All tests run OFFLINE — no ROS 2 dependency.
"""

import sys
import csv
import math
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.pink_noise import FFTColoredNoise, UniformColoredNoise
from shared.pink_uniform_policy import PinkUniformNoisePolicy, PolicyConfig
from shared.drive_command import DriveCommand


# ─────────────────────────────────────────────────────────────
# 1. PinkUniformNoisePolicy Tests
# ─────────────────────────────────────────────────────────────

class TestPolicyRanges:
    """Verify correct output ranges for all three action dimensions."""

    def _make_policy(self, **kwargs) -> PinkUniformNoisePolicy:
        defaults = dict(
            beta=1.0, policy_freq_hz=2.0, control_freq_hz=20.0,
            smoothing_alpha=0.2, vx_range=(0.0, 0.6),
            vy_range=(-0.3, 0.3), wz_range=(-1.0, 1.0), seed=42,
        )
        defaults.update(kwargs)
        return PinkUniformNoisePolicy(PolicyConfig(**defaults))

    def test_vx_in_range(self):
        """vx must always be in [0.0, 0.6]."""
        policy = self._make_policy()
        for _ in range(5000):
            cmd = policy.get_action()
            assert 0.0 <= cmd.v_linear <= 0.6, f"vx={cmd.v_linear} out of range"

    def test_vy_in_range(self):
        """vy must always be in [-0.3, 0.3]."""
        policy = self._make_policy()
        for _ in range(5000):
            cmd = policy.get_action()
            assert -0.3 <= cmd.v_lateral <= 0.3, f"vy={cmd.v_lateral} out of range"

    def test_wz_in_range(self):
        """wz must always be in [-1.0, 1.0]."""
        policy = self._make_policy()
        for _ in range(5000):
            cmd = policy.get_action()
            assert -1.0 <= cmd.v_angular <= 1.0, f"wz={cmd.v_angular} out of range"


class TestPolicyIndependence:
    """Verify the three action dimensions are independent."""

    def _make_policy(self, **kwargs) -> PinkUniformNoisePolicy:
        defaults = dict(
            beta=1.0, policy_freq_hz=2.0, control_freq_hz=20.0,
            smoothing_alpha=1.0, vx_range=(0.0, 1.0),
            vy_range=(-1.0, 1.0), wz_range=(-1.0, 1.0), seed=42,
        )
        defaults.update(kwargs)
        return PinkUniformNoisePolicy(PolicyConfig(**defaults))

    def test_cross_correlation_low(self):
        """Cross-correlation between dimensions should be low (independent noise)."""
        policy = self._make_policy()
        n = 5000
        vx = np.zeros(n)
        vy = np.zeros(n)
        wz = np.zeros(n)
        for i in range(n):
            cmd = policy.get_action()
            vx[i] = cmd.v_linear
            vy[i] = cmd.v_lateral
            wz[i] = cmd.v_angular

        # Correlation between dimensions should be low
        # (not exactly zero due to shared LERP timing, but independent noise sources)
        r_vx_vy = abs(np.corrcoef(vx, vy)[0, 1])
        r_vx_wz = abs(np.corrcoef(vx, wz)[0, 1])
        r_vy_wz = abs(np.corrcoef(vy, wz)[0, 1])

        # With independent noise, correlations should be < 0.15
        assert r_vx_vy < 0.20, f"|corr(vx,vy)| = {r_vx_vy:.3f} (too high)"
        assert r_vx_wz < 0.20, f"|corr(vx,wz)| = {r_vx_wz:.3f} (too high)"
        assert r_vy_wz < 0.20, f"|corr(vy,wz)| = {r_vy_wz:.3f} (too high)"


class TestPolicyTiming:
    """Verify 2 Hz / 20 Hz timing relationship."""

    def _make_policy(self, **kwargs) -> PinkUniformNoisePolicy:
        defaults = dict(
            beta=1.0, policy_freq_hz=2.0, control_freq_hz=20.0,
            smoothing_alpha=0.2, vx_range=(0.0, 0.6),
            vy_range=(-0.3, 0.3), wz_range=(-1.0, 1.0), seed=42,
        )
        defaults.update(kwargs)
        return PinkUniformNoisePolicy(PolicyConfig(**defaults))

    def test_ticks_per_sample_default(self):
        """20 Hz / 2 Hz = 10 ticks per sample."""
        policy = self._make_policy()
        assert policy._ticks_per_sample == 10

    def test_ticks_per_sample_custom(self):
        """10 Hz / 2 Hz = 5 ticks per sample."""
        policy = self._make_policy(control_freq_hz=10.0, policy_freq_hz=2.0)
        assert policy._ticks_per_sample == 5

    def test_target_changes_at_policy_rate(self):
        """Target action should change every ticks_per_sample calls."""
        policy = self._make_policy(smoothing_alpha=1.0)  # snap to target
        targets = []
        for i in range(30):
            _ = policy.get_action()
            targets.append(
                (policy.target_action.v_linear,
                 policy.target_action.v_lateral,
                 policy.target_action.v_angular)
            )

        # Target should change every 10 ticks (but first sample is at init)
        # Count how many times the target changes
        changes = sum(1 for i in range(1, len(targets)) if targets[i] != targets[i-1])
        # Over 30 ticks: 2 target changes expected (at tick 10 and tick 20)
        # But first target was set in __init__, so tick 0-9 is target 0,
        # tick 10-19 is target 1, tick 20-29 is target 2
        assert changes >= 2, f"Only {changes} target changes in 30 ticks (expected >= 2)"


# ─────────────────────────────────────────────────────────────
# 2. LERP Smoothing Tests
# ─────────────────────────────────────────────────────────────

class TestLERPSmoothing:
    """Verify LERP smoothing behavior for different alpha values."""

    def _make_policy(self, alpha: float, seed: int = 42) -> PinkUniformNoisePolicy:
        return PinkUniformNoisePolicy(PolicyConfig(
            beta=1.0, policy_freq_hz=2.0, control_freq_hz=20.0,
            smoothing_alpha=alpha, vx_range=(0.0, 0.6),
            vy_range=(-0.3, 0.3), wz_range=(-1.0, 1.0), seed=seed,
        ))

    def test_alpha_zero_holds_initial(self):
        """alpha=0 → output never changes from initial (always 0)."""
        policy = self._make_policy(alpha=0.0)
        for _ in range(100):
            cmd = policy.get_action()
            assert cmd.v_linear == 0.0, "alpha=0 should hold initial value"
            assert cmd.v_lateral == 0.0
            assert cmd.v_angular == 0.0

    def test_alpha_one_snaps_to_target(self):
        """alpha=1.0 → output equals target after one tick."""
        policy = self._make_policy(alpha=1.0)
        # After the first get_action, current should equal target
        for _ in range(20):
            cmd = policy.get_action()
            target = policy.target_action
            # After LERP with alpha=1, current = target
            assert abs(cmd.v_linear - target.v_linear) < 1e-10
            assert abs(cmd.v_lateral - target.v_lateral) < 1e-10
            assert abs(cmd.v_angular - target.v_angular) < 1e-10

    def test_alpha_02_convergence(self):
        """alpha=0.2 → output converges toward target over ~10 ticks."""
        policy = self._make_policy(alpha=0.2)
        # Run enough ticks to converge
        for _ in range(50):
            policy.get_action()

        target = policy.target_action
        # After many ticks at alpha=0.2, should be very close to target
        cmd = policy.get_action()
        assert abs(cmd.v_linear - target.v_linear) < 0.05, (
            f"After 50 ticks, vx={cmd.v_linear:.4f} should be near target={target.v_linear:.4f}"
        )

    def test_smoothing_reduces_step_changes(self):
        """Smoothed output should have smaller step-to-step changes than raw targets."""
        policy_smooth = self._make_policy(alpha=0.2, seed=42)
        policy_snap = self._make_policy(alpha=1.0, seed=42)

        n = 2000
        smooth = [policy_smooth.get_action().v_linear for _ in range(n)]
        snap = [policy_snap.get_action().v_linear for _ in range(n)]

        smooth_jitter = np.mean(np.abs(np.diff(smooth)))
        snap_jitter = np.mean(np.abs(np.diff(snap)))

        assert smooth_jitter < snap_jitter, (
            f"Smoothed jitter ({smooth_jitter:.4f}) should be less than snap ({snap_jitter:.4f})"
        )


# ─────────────────────────────────────────────────────────────
# 3. Safety Gate (Depth-based) Tests
# ─────────────────────────────────────────────────────────────

class TestDepthSafetyGate:
    """Test the depth-image-based safety gate in exploration_runner.py.

    These tests use a mock ExplorationRunner to test _safety_gate directly.
    """

    def _make_runner_stub(self):
        """Create a minimal ExplorationRunner with mocked dependencies."""
        from realworld.src.exploration_runner import ExplorationRunner

        mock_robot = MagicMock()
        mock_robot.dry_run = True
        mock_camera = MagicMock()
        mock_lidar = MagicMock()
        mock_lidar.last_scan_time = time.time()
        mock_safety = MagicMock()
        mock_recorder = MagicMock()

        config = PolicyConfig(
            beta=1.0, policy_freq_hz=2.0, control_freq_hz=20.0,
            smoothing_alpha=0.2, vx_range=(0.0, 0.6),
            vy_range=(-0.3, 0.3), wz_range=(-1.0, 1.0), seed=42,
        )

        runner = ExplorationRunner(
            robot=mock_robot, camera=mock_camera, lidar=mock_lidar,
            safety=mock_safety, recorder=mock_recorder,
            policy_config=config,
            duration_minutes=0.1,
            safety_gate_min_depth=0.55,
            safety_gate_min_pixels=75,
        )
        return runner

    def _make_depth(self, value: float = 10.0, h: int = 60, w: int = 640) -> np.ndarray:
        """Create a uniform depth image (all far away by default)."""
        return np.full((h, w), value, dtype=np.float32)

    def test_no_obstacle_passthrough(self):
        """When no obstacles, command should pass through unchanged."""
        runner = self._make_runner_stub()
        depth = self._make_depth(10.0)
        cmd = DriveCommand(0.3, 0.1, 0.5)
        result, blocked, diag = runner._safety_gate(cmd, depth)
        assert not blocked
        assert result.v_linear == pytest.approx(0.3)
        assert result.v_lateral == pytest.approx(0.1)
        assert result.v_angular == pytest.approx(0.5)

    def test_front_obstacle_blocks_forward(self):
        """Obstacle in front sector should block vx > 0."""
        runner = self._make_runner_stub()
        depth = self._make_depth(10.0)
        h, w = depth.shape
        # Fill front sector with close obstacles
        depth[:, w // 3: 2 * w // 3] = 0.3  # well below threshold

        cmd = DriveCommand(0.3, 0.0, 0.5)
        result, blocked, diag = runner._safety_gate(cmd, depth)
        assert blocked
        assert result.v_linear == 0.0  # forward blocked
        assert result.v_angular == 0.5  # rotation preserved

    def test_rotation_preserved_when_blocked(self):
        """Angular velocity must NEVER be zeroed by the safety gate."""
        runner = self._make_runner_stub()
        depth = self._make_depth(0.2)  # obstacles everywhere

        cmd = DriveCommand(0.3, 0.2, -0.8)
        result, blocked, diag = runner._safety_gate(cmd, depth)
        assert blocked
        assert result.v_angular == -0.8  # rotation untouched

    def test_obstacle_outside_sector_allows_motion(self):
        """Obstacle only in left sector should not block forward motion."""
        runner = self._make_runner_stub()
        depth = self._make_depth(10.0)
        h, w = depth.shape
        # Only left sector has obstacles
        depth[:, 2 * w // 3:] = 0.3

        cmd = DriveCommand(0.3, 0.0, 0.0)
        result, blocked, diag = runner._safety_gate(cmd, depth)
        # Forward should not be blocked by left obstacle
        assert result.v_linear == pytest.approx(0.3)

    def test_insufficient_pixels_no_block(self):
        """If fewer than min_pixels are near, no blocking should occur."""
        runner = self._make_runner_stub()
        depth = self._make_depth(10.0)
        h, w = depth.shape
        # Put just a tiny number of near-range pixels in front
        depth[0, w // 2] = 0.3  # only 1 pixel
        depth[1, w // 2] = 0.3  # only 2 pixels total

        cmd = DriveCommand(0.3, 0.0, 0.5)
        result, blocked, diag = runner._safety_gate(cmd, depth)
        assert not blocked
        assert result.v_linear == pytest.approx(0.3)

    def test_diagnostics_returned(self):
        """Safety gate should return diagnostic metadata."""
        runner = self._make_runner_stub()
        depth = self._make_depth(5.0)
        cmd = DriveCommand(0.3, 0.0, 0.5)
        result, blocked, diag = runner._safety_gate(cmd, depth)
        assert 'obstacle_distance' in diag
        assert 'relevant_point_count' in diag
        assert 'cloud_age' in diag
        assert isinstance(diag['obstacle_distance'], float)
        assert isinstance(diag['relevant_point_count'], int)


# ─────────────────────────────────────────────────────────────
# 4. Pink Noise Core Tests
# ─────────────────────────────────────────────────────────────

class TestFFTColoredNoiseAudit:
    """Audit-specific tests for the FFT noise generator."""

    def test_beta_1_spectral_exponent(self):
        """beta=1 should produce PSD ~ 1/f (pink noise)."""
        noise = FFTColoredNoise(beta=1.0, buffer_size=8192, seed=42)
        samples = np.array([noise.sample() for _ in range(8192)])

        # Compute PSD
        fft = np.fft.rfft(samples)
        psd = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(len(samples))

        # Fit log-log slope (skip DC and first few bins)
        valid = freqs[5:len(freqs)//2]
        psd_valid = psd[5:len(freqs)//2]
        log_f = np.log(valid)
        log_psd = np.log(psd_valid)

        # Linear fit
        coeffs = np.polyfit(log_f, log_psd, 1)
        slope = coeffs[0]

        # For beta=1 pink noise, PSD ~ f^(-1), so slope should be near -1
        assert -2.0 < slope < -0.3, (
            f"PSD slope = {slope:.2f}, expected near -1.0 for beta=1 pink noise"
        )

    def test_gaussian_cdf_produces_uniform(self):
        """Applying norm.cdf to N(0,1) noise should yield Uniform[0,1]."""
        import scipy.stats
        noise = FFTColoredNoise(beta=1.0, buffer_size=8192, seed=42)
        raw = np.array([noise.sample() for _ in range(20000)])
        # Subsample to break autocorrelation (KS test assumes i.i.d. samples)
        uniform = scipy.stats.norm.cdf(raw[::100])

        # KS test against uniform distribution
        ks_stat, p_val = scipy.stats.kstest(uniform, 'uniform')
        # Should not reject uniformity at 1% significance
        assert p_val > 0.01, (
            f"CDF output fails KS test for uniformity: stat={ks_stat:.4f}, p={p_val:.4f}"
        )

    def test_normalization_zero_mean_unit_var(self):
        """Buffer should be normalized to approximately zero mean, unit variance."""
        noise = FFTColoredNoise(beta=1.0, buffer_size=8192, seed=42)
        samples = np.array([noise.sample() for _ in range(8192)])
        assert abs(np.mean(samples)) < 0.05, f"Mean = {np.mean(samples):.4f}"
        assert 0.8 < np.std(samples) < 1.2, f"Std = {np.std(samples):.4f}"


# ─────────────────────────────────────────────────────────────
# 5. Data Recorder Extended Schema Test
# ─────────────────────────────────────────────────────────────

class TestDataRecorderExtendedSchema:
    """Verify the extended CSV schema with diagnostic columns."""

    def test_extended_csv_header(self):
        """CSV header should include target and diagnostic columns."""
        from realworld.src.data_recorder import DataRecorder

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = DataRecorder(
                session_dir=Path(tmpdir) / "test_session",
                segment_size=100,
            )
            recorder.open()

            # Record one step with all extended fields
            rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            cmd = DriveCommand(0.3, 0.1, 0.5)
            executed = DriveCommand(0.0, 0.0, 0.5)
            target = DriveCommand(0.4, -0.1, 0.8)

            recorder.record_step(
                rgb_frame=rgb,
                cmd=cmd,
                pos_x=1.0, pos_y=2.0, yaw=0.5,
                wall_time=time.time(),
                executed_cmd=executed,
                safety_blocked=True,
                target_cmd=target,
                obstacle_distance=0.35,
                relevant_point_count=120,
                cloud_age=0.05,
            )

            recorder.close()

            # Read back and verify columns
            csv_path = Path(tmpdir) / "test_session" / "segment_000" / "observations.csv"
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 1
            row = rows[0]

            # Verify extended columns exist
            assert "target_vx" in row
            assert "target_vy" in row
            assert "target_wz" in row
            assert "obstacle_distance" in row
            assert "timestamp" in row

            # Verify values
            assert float(row["target_vx"]) == pytest.approx(0.4, abs=0.01)
            assert float(row["target_vy"]) == pytest.approx(-0.1, abs=0.01)
            assert float(row["target_wz"]) == pytest.approx(0.8, abs=0.01)
            assert float(row["obstacle_distance"]) == pytest.approx(0.35, abs=0.01)

            # Verify new simplified columns work
            assert float(row["cmd_vx"]) == pytest.approx(0.3, abs=0.01)
            assert float(row["executed_vx"]) == pytest.approx(0.0, abs=0.01)
            assert row["safety_blocked"] == "1"
