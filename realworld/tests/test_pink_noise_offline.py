"""
realworld/tests/test_pink_noise_offline.py — Pink noise generation tests.
=========================================================================
These tests run OFFLINE (no ROS 2 needed). They validate the FFT pink
noise generator and uniform colored noise transform.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from shared.pink_noise import FFTColoredNoise, UniformColoredNoise


class TestFFTColoredNoise:
    """Tests for the raw FFT colored noise generator."""

    def test_mean_near_zero(self):
        """10k samples should have near-zero mean (normalized to N(0,1))."""
        noise = FFTColoredNoise(beta=1.0, seed=42)
        samples = [noise.sample() for _ in range(10000)]
        mean = np.mean(samples)
        assert abs(mean) < 0.1, f"Mean should be near zero, got {mean:.4f}"

    def test_unit_variance(self):
        """10k samples should have approximately unit variance."""
        noise = FFTColoredNoise(beta=1.0, seed=42)
        samples = [noise.sample() for _ in range(10000)]
        std = np.std(samples)
        assert 0.7 < std < 1.3, f"Std should be near 1.0, got {std:.4f}"

    def test_buffer_regeneration(self):
        """Sampling beyond buffer_size should seamlessly regenerate."""
        noise = FFTColoredNoise(beta=1.0, buffer_size=100, seed=42)
        # Sample 250 times (will trigger 2 regenerations)
        samples = [noise.sample() for _ in range(250)]
        assert len(samples) == 250
        # All should be finite
        assert all(np.isfinite(s) for s in samples)

    def test_pink_autocorrelation(self):
        """Pink noise (β=1) should have positive lag-1 autocorrelation (temporal smoothness)."""
        noise = FFTColoredNoise(beta=1.0, buffer_size=8192, seed=42)
        samples = np.array([noise.sample() for _ in range(5000)])

        # Compute lag-1 autocorrelation
        mean = samples.mean()
        n = len(samples)
        c0 = np.sum((samples - mean) ** 2) / n
        c1 = np.sum((samples[:-1] - mean) * (samples[1:] - mean)) / n
        rho = c1 / c0 if c0 > 0 else 0

        assert rho > 0.3, (
            f"Pink noise should have positive autocorrelation, got ρ={rho:.3f}. "
            "This means the noise is not temporally smooth."
        )

    def test_white_noise_low_autocorrelation(self):
        """White noise (β=0) should have near-zero autocorrelation."""
        noise = FFTColoredNoise(beta=0.0, buffer_size=8192, seed=42)
        samples = np.array([noise.sample() for _ in range(5000)])

        mean = samples.mean()
        n = len(samples)
        c0 = np.sum((samples - mean) ** 2) / n
        c1 = np.sum((samples[:-1] - mean) * (samples[1:] - mean)) / n
        rho = c1 / c0 if c0 > 0 else 0

        assert abs(rho) < 0.15, f"White noise should have ~0 autocorrelation, got ρ={rho:.3f}"

    def test_reproducibility_with_seed(self):
        """Same seed should produce identical sequences."""
        noise1 = FFTColoredNoise(beta=1.0, seed=123)
        noise2 = FFTColoredNoise(beta=1.0, seed=123)

        s1 = [noise1.sample() for _ in range(100)]
        s2 = [noise2.sample() for _ in range(100)]

        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different sequences."""
        noise1 = FFTColoredNoise(beta=1.0, seed=1)
        noise2 = FFTColoredNoise(beta=1.0, seed=2)

        s1 = [noise1.sample() for _ in range(100)]
        s2 = [noise2.sample() for _ in range(100)]

        assert s1 != s2


class TestUniformColoredNoise:
    """Tests for the Uniform[min, max] colored noise transform."""

    def test_samples_in_range(self):
        """All samples should be within the configured range."""
        noise = UniformColoredNoise(beta=1.0, range_val=(0.3, 1.8), seed=42)
        samples = [noise.sample() for _ in range(5000)]
        assert all(0.3 <= s <= 1.8 for s in samples), "Samples outside range"

    def test_samples_in_negative_range(self):
        """Test with negative range (steering)."""
        noise = UniformColoredNoise(beta=1.0, range_val=(-1.5, 1.5), seed=42)
        samples = [noise.sample() for _ in range(5000)]
        assert all(-1.5 <= s <= 1.5 for s in samples)

    def test_uniform_coverage(self):
        """Samples should cover the full range reasonably uniformly."""
        noise = UniformColoredNoise(beta=1.0, range_val=(0.0, 1.0), seed=42)
        samples = np.array([noise.sample() for _ in range(10000)])

        # Check that all quartiles are reasonably populated
        q1 = np.sum(samples < 0.25)
        q2 = np.sum((samples >= 0.25) & (samples < 0.5))
        q3 = np.sum((samples >= 0.5) & (samples < 0.75))
        q4 = np.sum(samples >= 0.75)

        # Each quartile should have at least 15% of samples
        total = len(samples)
        for i, q in enumerate([q1, q2, q3, q4], 1):
            frac = q / total
            assert frac > 0.15, f"Quartile {i} has only {frac:.2%} of samples"

    def test_temporal_smoothness_preserved(self):
        """Uniform transform should preserve pink noise temporal smoothness."""
        noise = UniformColoredNoise(beta=1.0, range_val=(0.0, 1.0), seed=42)
        samples = np.array([noise.sample() for _ in range(5000)])

        # Lag-1 autocorrelation
        mean = samples.mean()
        n = len(samples)
        c0 = np.sum((samples - mean) ** 2) / n
        c1 = np.sum((samples[:-1] - mean) * (samples[1:] - mean)) / n
        rho = c1 / c0 if c0 > 0 else 0

        assert rho > 0.3, (
            f"Uniform pink noise should still be temporally smooth, got ρ={rho:.3f}"
        )
