"""
realworld/tests/conftest.py — Pytest fixtures for real-world tests.
====================================================================
Provides shared fixtures for both offline tests (no ROS needed)
and online tests (require ROS 2 environment).

Environment variables:
  MINAV_ROS2_AVAILABLE=1  — enable tests that require ROS 2
  MINAV_HARDWARE=1        — enable tests that require real hardware
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

import pytest
import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


# ── Markers ──

def pytest_configure(config):
    config.addinivalue_line("markers", "ros2: requires ROS 2 environment")
    config.addinivalue_line("markers", "hardware: requires real robot hardware")


# ── Skip conditions ──

ros2_available = os.environ.get("MINAV_ROS2_AVAILABLE", "0") == "1"
hardware_available = os.environ.get("MINAV_HARDWARE", "0") == "1"


def pytest_collection_modifyitems(config, items):
    """Skip ROS2/hardware tests if not available."""
    for item in items:
        if "ros2" in item.keywords and not ros2_available:
            item.add_marker(pytest.mark.skip(reason="ROS 2 not available (set MINAV_ROS2_AVAILABLE=1)"))
        if "hardware" in item.keywords and not hardware_available:
            item.add_marker(pytest.mark.skip(reason="Hardware not available (set MINAV_HARDWARE=1)"))


# ── Offline fixtures (always available) ──

@pytest.fixture
def tmp_dir():
    """Create a temporary directory, cleaned up after test."""
    d = Path(tempfile.mkdtemp(prefix="minav_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_rgb_frame():
    """Generate a random 480×640×3 uint8 RGB image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_depth_image():
    """Generate a synthetic depth image (60×640) with some obstacles."""
    depth = np.full((60, 640), 5.0, dtype=np.float32)
    # Add a close obstacle in the center
    depth[:, 280:360] = 0.4
    return depth


@pytest.fixture
def sample_config():
    """Minimal pipeline config for testing."""
    return {
        "run": {"id": "test_run", "seed": 42, "device": "cpu"},
        "robot": {
            "cmd_vel_topic": "/cmd_vel",
            "odom_topic": "/odom",
            "max_linear_vel": 1.8,
            "min_linear_vel": -0.9,
            "max_lateral_vel": 1.0,
            "max_angular_vel": 1.5,
            "dry_run": True,
        },
        "camera": {
            "topic": "/camera/color/image_raw",
            "encoding": "rgb8",
            "width": 640,
            "height": 480,
        },
        "lidar": {
            "topic": "/rslidar_points",
            "projection_width": 640,
            "projection_height": 60,
            "max_range": 10.0,
        },
        "exploration": {
            "duration_minutes": 0.5,
            "control_freq_hz": 10,
            "segment_size": 100,
            "output_subdir": "exploration",
            "pink_noise_beta": 1,
        },
        "safety": {
            "camera_timeout_s": 2.0,
            "odom_timeout_s": 2.0,
            "lidar_timeout_s": 3.0,
        },
    }
