#!/usr/bin/env python3
"""
realworld/tests/test_fused_pipeline.py — Test Fused ONNX & OpenVINO Policy Engine.
=================================================================================
Validates the fused Image -> DINOv3 -> Actor model export and runtime execution.
Runs offline without hardware or ROS 2 dependencies.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from realworld.src.fused_policy_engine import FusedPolicyEngine

MODELS_DIR = ROOT / "realworld" / "models" / "fused"
STEP_XML = MODELS_DIR / "minav_fused_step_policy.xml"
STEP_ONNX = MODELS_DIR / "minav_fused_step_policy.onnx"


@pytest.mark.skipif(not STEP_XML.exists(), reason="Fused OpenVINO step model does not exist yet")
def test_openvino_step_inference():
    """Test FusedPolicyEngine running OpenVINO step-wise inference."""
    engine = FusedPolicyEngine(STEP_XML, backend="openvino")
    engine.reset()

    # Set synthetic unit goal embedding
    dummy_goal = np.random.randn(384).astype(np.float32)
    dummy_goal /= np.linalg.norm(dummy_goal)
    engine.set_goal(dummy_goal)

    # Simulate 6 control cycles with raw camera resolution (480, 640, 3)
    for step_idx in range(6):
        raw_rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        action, similarity, is_ready = engine.step(raw_rgb)

        assert action.shape == (3,), f"Expected action shape (3,), got {action.shape}"
        assert np.all(action >= -1.0) and np.all(action <= 1.0), f"Action out of [-1, 1] bounds: {action}"
        assert -1.05 <= similarity <= 1.05, f"Similarity out of cosine bounds: {similarity}"

        if step_idx < 3:
            assert not is_ready, f"Buffer reported ready too early at step {step_idx}"
        else:
            assert is_ready, f"Buffer should be ready at step {step_idx}"


@pytest.mark.skipif(not STEP_ONNX.exists(), reason="Fused ONNX step model does not exist yet")
def test_onnx_step_inference():
    """Test FusedPolicyEngine running ONNX Runtime step-wise inference."""
    engine = FusedPolicyEngine(STEP_ONNX, backend="onnx")
    engine.reset()

    dummy_goal = np.random.randn(384).astype(np.float32)
    dummy_goal /= np.linalg.norm(dummy_goal)
    engine.set_goal(dummy_goal)

    raw_rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    action, similarity, _ = engine.step(raw_rgb)

    assert action.shape == (3,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


@pytest.mark.skipif(not STEP_XML.exists(), reason="Fused OpenVINO step model does not exist yet")
def test_goal_setting_from_image():
    """Test setting goal directly from a synthetic RGB image."""
    engine = FusedPolicyEngine(STEP_XML, backend="openvino")
    engine.reset()

    goal_rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    engine.set_goal_from_image(goal_rgb)

    assert engine.current_goal_phi is not None
    assert engine.current_goal_phi.shape == (1, 384)
    norm = np.linalg.norm(engine.current_goal_phi)
    assert np.isclose(norm, 1.0, atol=1e-3), f"Goal embedding norm should be 1.0, got {norm}"
