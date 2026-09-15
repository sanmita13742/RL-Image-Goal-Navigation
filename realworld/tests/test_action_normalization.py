"""
realworld/tests/test_action_normalization.py — Action normalization tests.
==========================================================================
Validates the asymmetric affine normalization critical to AGENTS.md:
  Action bounds: [-0.9, -1.0, -1.5] to [1.8, 1.0, 1.5]
  Normalized to [-1, 1].
"""

import sys
from pathlib import Path

import numpy as np
import torch
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from shared.action_normalizer import ActionNormalizer


class TestActionNormalizer:

    @pytest.fixture
    def normalizer(self):
        return ActionNormalizer(device="cpu")

    def test_normalize_min_maps_to_neg1(self, normalizer):
        """Physical minimum [-0.9, -1.0, -1.5] should map to [-1, -1, -1]."""
        a_min = torch.tensor([-0.9, -1.0, -1.5])
        result = normalizer.normalize(a_min)
        expected = torch.tensor([-1.0, -1.0, -1.0])
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_normalize_max_maps_to_pos1(self, normalizer):
        """Physical maximum [1.8, 1.0, 1.5] should map to [1, 1, 1]."""
        a_max = torch.tensor([1.8, 1.0, 1.5])
        result = normalizer.normalize(a_max)
        expected = torch.tensor([1.0, 1.0, 1.0])
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_denormalize_roundtrip(self, normalizer):
        """normalize(denormalize(x)) ≈ x for arbitrary x."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            a_norm = torch.tensor(rng.uniform(-1, 1, size=3).astype(np.float32))
            roundtrip = normalizer.normalize(normalizer.denormalize(a_norm))
            torch.testing.assert_close(roundtrip, a_norm, atol=1e-5, rtol=1e-5)

    def test_normalize_roundtrip(self, normalizer):
        """denormalize(normalize(x)) ≈ x for arbitrary physical x."""
        rng = np.random.default_rng(42)
        a_min = np.array([-0.9, -1.0, -1.5])
        a_max = np.array([1.8, 1.0, 1.5])
        for _ in range(50):
            a = torch.tensor((a_min + rng.random(3) * (a_max - a_min)).astype(np.float32))
            roundtrip = normalizer.denormalize(normalizer.normalize(a))
            torch.testing.assert_close(roundtrip, a, atol=1e-5, rtol=1e-5)

    def test_asymmetric_vx_center(self, normalizer):
        """Midpoint of [-0.9, 1.8] is 0.45, NOT 0.0. normalize(0.45) ≈ 0.0.

        This is the CRITICAL test from AGENTS.md: simple division distorts
        the asymmetric vx behavior. The affine normalization must map the
        physical midpoint (0.45 m/s) to normalized 0.0.
        """
        midpoint = torch.tensor([0.45, 0.0, 0.0])  # vx midpoint, vy=0, omega=0
        result = normalizer.normalize(midpoint)
        # vx should map to 0.0
        assert abs(result[0].item()) < 1e-5, (
            f"vx=0.45 should normalize to 0.0, got {result[0].item():.6f}. "
            "This indicates the asymmetric normalization is broken."
        )
        # vy=0 should map to 0.0 (symmetric range)
        assert abs(result[1].item()) < 1e-5
        # omega=0 should map to 0.0 (symmetric range)
        assert abs(result[2].item()) < 1e-5

    def test_zero_velocity_is_not_center(self, normalizer):
        """Physical zero [0, 0, 0] should NOT map to normalized [0, 0, 0] for vx.

        Because vx range is [-0.9, 1.8], zero is NOT the midpoint.
        normalize(0) for vx should be negative (closer to min than max).
        """
        zero = torch.tensor([0.0, 0.0, 0.0])
        result = normalizer.normalize(zero)
        # vx: (0 - (-0.9)) / (1.8 - (-0.9)) * 2 - 1 = 0.9/2.7 * 2 - 1 = -0.333...
        expected_vx = 2.0 * (0.0 - (-0.9)) / (1.8 - (-0.9)) - 1.0
        assert abs(result[0].item() - expected_vx) < 1e-5
        assert result[0].item() < 0, "vx=0 should normalize to negative (asymmetric range)"

    def test_numpy_input(self, normalizer):
        """Should accept numpy arrays and Python lists."""
        a_np = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        result = normalizer.normalize(a_np)
        assert isinstance(result, torch.Tensor)

        a_list = [0.5, 0.0, 0.0]
        result2 = normalizer.normalize(a_list)
        assert isinstance(result2, torch.Tensor)

    def test_batch_normalization(self, normalizer):
        """Should handle batched inputs [B, 3]."""
        batch = torch.tensor([
            [-0.9, -1.0, -1.5],
            [1.8, 1.0, 1.5],
            [0.45, 0.0, 0.0],
        ])
        result = normalizer.normalize(batch)
        assert result.shape == (3, 3)
        # Row 0: all -1
        torch.testing.assert_close(result[0], torch.tensor([-1.0, -1.0, -1.0]), atol=1e-5, rtol=1e-5)
        # Row 1: all +1
        torch.testing.assert_close(result[1], torch.tensor([1.0, 1.0, 1.0]), atol=1e-5, rtol=1e-5)
        # Row 2: all 0
        torch.testing.assert_close(result[2], torch.tensor([0.0, 0.0, 0.0]), atol=1e-5, rtol=1e-5)
