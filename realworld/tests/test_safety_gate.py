import numpy as np
import pytest
from unittest.mock import Mock

from realworld.src.exploration_runner import ExplorationRunner
from shared.drive_command import DriveCommand
from shared.pink_uniform_policy import PolicyConfig


def test_directional_safety_gate():
    """Verify that the safety gate clamps directional velocities based on obstacles,
    but always preserves angular velocity to allow escapes.
    """
    policy_config = PolicyConfig(
        beta=1.0,
        policy_freq_hz=2.0,
        control_freq_hz=20.0,
        smoothing_alpha=0.2,
        vx_range=[0.0, 0.6],
        vy_range=[-0.3, 0.3],
        wz_range=[-1.0, 1.0],
        seed=42
    )

    runner = ExplorationRunner(
        robot=Mock(),
        camera=Mock(),
        lidar=Mock(),
        safety=Mock(),
        recorder=Mock(),
        policy_config=policy_config,
        safety_gate_min_depth=0.5,
        safety_gate_min_pixels=75
    )

    h, w = 60, 640

    # Case 1: No obstacles -> No clamping
    depth = np.full((h, w), 10.0, dtype=np.float32)
    cmd = DriveCommand(v_linear=0.5, v_lateral=0.2, v_angular=0.8)
    safe_cmd, blocked = runner._safety_gate(cmd, depth)
    assert not blocked
    assert safe_cmd.v_linear == 0.5
    assert safe_cmd.v_lateral == 0.2
    assert safe_cmd.v_angular == 0.8

    # Case 2: Front obstacle -> Clamp forward motion only
    depth = np.full((h, w), 10.0, dtype=np.float32)
    depth[:, 250:350] = 0.4  # Obstacle in the Front sector (w//3 to 2*w//3)
    cmd = DriveCommand(v_linear=0.5, v_lateral=0.2, v_angular=0.8)
    safe_cmd, blocked = runner._safety_gate(cmd, depth)
    assert blocked
    assert safe_cmd.v_linear == 0.0  # Clamped
    assert safe_cmd.v_lateral == 0.2 # Preserved
    assert safe_cmd.v_angular == 0.8 # Preserved

    # Case 3: Front obstacle -> Do not clamp reverse motion
    cmd = DriveCommand(v_linear=-0.5, v_lateral=0.2, v_angular=0.8)
    safe_cmd, blocked = runner._safety_gate(cmd, depth)
    assert not blocked
    assert safe_cmd.v_linear == -0.5

    # Case 4: Right obstacle -> Clamp rightward crab walk (vy < 0)
    depth = np.full((h, w), 10.0, dtype=np.float32)
    depth[:, 100:200] = 0.4  # Obstacle in the Right sector (0 to w//3)
    cmd = DriveCommand(v_linear=0.5, v_lateral=-0.3, v_angular=0.8)
    safe_cmd, blocked = runner._safety_gate(cmd, depth)
    assert blocked
    assert safe_cmd.v_linear == 0.5   # Preserved
    assert safe_cmd.v_lateral == 0.0  # Clamped
    assert safe_cmd.v_angular == 0.8  # Preserved

    # Case 5: Right obstacle -> Do not clamp leftward crab walk
    cmd = DriveCommand(v_linear=0.5, v_lateral=0.3, v_angular=0.8)
    safe_cmd, blocked = runner._safety_gate(cmd, depth)
    assert not blocked
    assert safe_cmd.v_lateral == 0.3

    # Case 6: Left obstacle -> Clamp leftward crab walk (vy > 0)
    depth = np.full((h, w), 10.0, dtype=np.float32)
    depth[:, 500:600] = 0.4  # Obstacle in the Left sector (2*w//3 to w)
    cmd = DriveCommand(v_linear=0.5, v_lateral=0.3, v_angular=0.8)
    safe_cmd, blocked = runner._safety_gate(cmd, depth)
    assert blocked
    assert safe_cmd.v_linear == 0.5   # Preserved
    assert safe_cmd.v_lateral == 0.0  # Clamped
    assert safe_cmd.v_angular == 0.8  # Preserved

    # Case 7: Obstacles everywhere -> Clamp translational but preserve angular (wz)
    depth = np.full((h, w), 0.1, dtype=np.float32) # Obstacles in all sectors
    cmd = DriveCommand(v_linear=0.5, v_lateral=0.3, v_angular=0.8)
    safe_cmd, blocked = runner._safety_gate(cmd, depth)
    assert blocked
    assert safe_cmd.v_linear == 0.0   # Clamped
    assert safe_cmd.v_lateral == 0.0  # Clamped
    assert safe_cmd.v_angular == 0.8  # PRESERVED - the escape mechanism
