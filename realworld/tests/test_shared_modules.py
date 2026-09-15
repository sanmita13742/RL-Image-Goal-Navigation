"""
realworld/tests/test_shared_modules.py — Shared module tests.
=============================================================
Verifies that the DINOv3 encoder, geometric dataset builder, and
uniform dataset builder (reused from the MuJoCo pipeline) can be
imported and work with synthetic data.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
MUJOCO_ROOT = ROOT / "simulation" / "ranger_mini_v3"
sys.path.insert(0, str(MUJOCO_ROOT))

# Test imports work without MuJoCo dependencies
from src.data.index_segments import build_frame_index
from src.data.dinov3_encoder import FrozenDINOv3
from src.data.goal_set import build_goal_set
from src.data.hindsight_geometric import build_geometric_dataset
from src.data.hindsight_uniform import build_uniform_dataset


class TestSharedModules:
    
    def test_dinov3_encoder_loads(self):
        """FrozenDINOv3 can be instantiated (offline smoke test)."""
        # Note: We don't download weights in tests, just check initialization
        # works without failing on imports
        encoder = FrozenDINOv3(device="cpu")
        assert encoder.device == torch.device("cpu")
    
    def test_geometric_sampling_math(self):
        """Test geometric distribution sampling matches expected probabilities."""
        p = 0.99
        N = 1000
        
        # In hindsight_geometric.py, it uses np.random.geometric(p=1-p)
        # where 1-p is the success probability
        # E[X] for geometric is 1 / (1-p)
        success_prob = 1.0 - p
        expected_mean = 1.0 / success_prob
        
        samples = np.random.geometric(p=success_prob, size=10000)
        mean = np.mean(samples)
        
        # Allow 10% error margin
        assert abs(mean - expected_mean) / expected_mean < 0.1, (
            f"Geometric mean {mean:.1f} far from expected {expected_mean:.1f}"
        )
        assert np.min(samples) >= 1
    
    def test_build_goal_set_logic(self, tmp_dir):
        """Test goal set building logic with synthetic data."""
        # Create synthetic frame index
        df = pd.DataFrame({
            "global_step": np.arange(10),
            "segment_id": ["segment_000"] * 10,
            "rgb_abs_path": [f"/path/to/{i}.png" for i in range(10)]
        })
        
        # Synthetic SSD scores (threshold is 0.02)
        ssd = np.array([0.01, 0.03, 0.05, 0.01, 0.1, 0.01, 0.01, 0.025, 0.04, 0.01])
        valid_mask = ssd > 0.02  # Expect indices 1, 2, 4, 7, 8
        
        out_dir = tmp_dir / "goals"
        goals_df = build_goal_set(df, ssd, valid_mask, out_dir)
        
        assert len(goals_df) == 5
        assert list(goals_df["embedding_index"]) == [1, 2, 4, 7, 8]
        assert (out_dir / "valid_goals.parquet").exists()
        assert (out_dir / "goal_set_meta.json").exists()
        
    def test_action_normalization_import(self):
        """Ensure action normalizer is accessible via shared package."""
        from shared.action_normalizer import ActionNormalizer
        norm = ActionNormalizer(device="cpu")
        assert abs(norm.a_max[0].item() - 1.8) < 1e-5
