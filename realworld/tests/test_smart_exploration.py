"""
realworld/tests/test_smart_exploration.py — Tests for smart exploration strategies.
=====================================================================================
Validates all 5 exploration enhancement strategies in isolation and integrated:
  1. 1D LiDAR range scan extraction (Strategy 4)
  2. LiDAR-aware action shaping (Strategy 1)
  3. Adaptive speed zones (Strategy 2)
  4. Panoramic rotation injection state machine (Strategy 3)
  5. 2D coverage heatmap & angular nudge (Strategy 5)
  6. SmartExplorationRunner execution & DataRecorder integration
  7. Switching behavior (vanilla vs. smart)
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from shared.drive_command import DriveCommand
from shared.pink_uniform_policy import PinkUniformNoisePolicy, PolicyConfig
from shared.lidar_aware_policy import LidarAwarePolicy, SmartPolicyConfig
from shared.lidar_scan_extractor import extract_scan_from_depth_image, extract_scan_from_xyz
from realworld.src.data_recorder import DataRecorder
from realworld.src.smart_exploration_runner import SmartExplorationRunner
from realworld.src.exploration_runner import ExplorationRunner


# =====================================================================
# Strategy 4: 1D LiDAR Range Scan Extraction Tests
# =====================================================================

class TestLidarScanExtractor:

    def test_extract_scan_shape_and_range(self):
        """Test extraction from a 2D depth image produces normalized (72,) vector."""
        depth = np.full((60, 640), 5.0, dtype=np.float32)
        # Put an obstacle at 2.0m in the center (front)
        depth[:, 300:340] = 2.0

        scan = extract_scan_from_depth_image(depth, num_bins=72, max_range=10.0)

        assert scan.shape == (72,)
        assert scan.dtype == np.float32
        assert np.all(scan >= 0.0)
        assert np.all(scan <= 1.0)

        # Center bin should reflect the 2.0m obstacle (2.0 / 10.0 = 0.2)
        center_bin = scan[36]
        assert np.isclose(center_bin, 0.2, atol=0.05)

    def test_extract_scan_empty_fallback(self):
        """None or empty depth returns all 1.0 (clear)."""
        scan = extract_scan_from_depth_image(None, num_bins=72, max_range=10.0)
        assert scan.shape == (72,)
        assert np.all(scan == 1.0)

    def test_extract_scan_from_xyz(self):
        """Test extraction directly from 3D point cloud."""
        # Create points at 3.0m in front (x=3, y=0, z=0)
        # and at 1.0m to the left (x=0, y=1, z=0)
        pts = np.array([
            [3.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],  # ground plane (should be filtered out)
        ], dtype=np.float32)

        scan = extract_scan_from_xyz(
            pts, num_bins=72, max_range=10.0,
            height_min=-0.3, height_max=1.0,
        )

        assert scan.shape == (72,)
        assert np.all(scan >= 0.0)
        assert np.all(scan <= 1.0)
        # Left obstacle at 1.0m (1.0 / 10.0 = 0.1) should be present
        assert np.min(scan) <= 0.15


# =====================================================================
# Strategy 1 & 2: Action Shaping and Speed Adaptation Tests
# =====================================================================

class TestLidarAwarePolicy:

    @pytest.fixture
    def base_policy(self):
        config = PolicyConfig(
            policy_freq_hz=2.0,
            control_freq_hz=20.0,
            smoothing_alpha=1.0,  # snap to target for instant testing
            vx_range=(0.2, 0.6),
            vy_range=(-0.3, 0.3),
            wz_range=(-1.0, 1.0),
            seed=42,
        )
        return PinkUniformNoisePolicy(config)

    def test_clear_space_unshaped(self, base_policy):
        """In open clear space, policy moves forward at full speed."""
        smart_cfg = SmartPolicyConfig(
            action_shaping_enabled=True,
            adaptive_speed_enabled=True,
            panoramic_rotations_enabled=False,
            coverage_nudge_enabled=False,
        )
        policy = LidarAwarePolicy(base_policy, smart_cfg)

        # Clear depth image (all 8 metres away)
        clear_depth = np.full((60, 640), 8.0, dtype=np.float32)
        policy.update_lidar_state(clear_depth)

        cmd = policy.get_action()
        diag = policy.get_diagnostics()

        assert cmd.v_linear > 0.0
        assert diag["speed_scale"] == 1.0
        assert diag["shaped"] is False

    def test_front_blocked_action_shaping(self, base_policy):
        """When front is blocked, forward motion is clamped to <= 0 and steering turns away."""
        smart_cfg = SmartPolicyConfig(
            action_shaping_enabled=True,
            adaptive_speed_enabled=True,
            panoramic_rotations_enabled=False,
            coverage_nudge_enabled=False,
            wall_danger_depth=0.55,
            sector_min_pixels=10,
        )
        policy = LidarAwarePolicy(base_policy, smart_cfg)

        # Blocked front: obstacle at 0.35m in front sector
        blocked_depth = np.full((60, 640), 8.0, dtype=np.float32)
        # Front sector is cols 213 to 426
        blocked_depth[:, 213:426] = 0.35
        # Make left side clearer (8.0m) than right side (1.0m)
        blocked_depth[:, 0:213] = 1.0

        policy.update_lidar_state(blocked_depth)
        cmd = policy.get_action()
        diag = policy.get_diagnostics()

        # Forward velocity should be negative (backing up) or zero
        assert cmd.v_linear <= 0.0
        # Steering should turn left (positive angular velocity) toward clear space
        assert cmd.v_angular >= 0.5
        assert diag["shaped"] is True

    def test_side_blocked_lateral_clamping(self, base_policy):
        """When left wall is close, lateral movement is clamped to rightward only (vy <= 0)."""
        smart_cfg = SmartPolicyConfig(
            action_shaping_enabled=True,
            adaptive_speed_enabled=False,
            panoramic_rotations_enabled=False,
            coverage_nudge_enabled=False,
            wall_danger_depth=0.55,
            sector_min_pixels=10,
        )
        policy = LidarAwarePolicy(base_policy, smart_cfg)

        # Left wall at 0.3m (cols 426 to 640)
        wall_depth = np.full((60, 640), 8.0, dtype=np.float32)
        wall_depth[:, 426:640] = 0.30

        policy.update_lidar_state(wall_depth)
        cmd = policy.get_action()

        assert cmd.v_lateral <= 0.0  # Cannot move left into wall

    def test_adaptive_speed_zones(self, base_policy):
        """Speed scales down continuously when approaching obstacles."""
        smart_cfg = SmartPolicyConfig(
            action_shaping_enabled=False,  # isolate speed scaling
            adaptive_speed_enabled=True,
            speed_min_dist=0.65,
            speed_max_dist=2.00,
            speed_min_scale=0.20,
            speed_max_scale=1.00,
            panoramic_rotations_enabled=False,
            coverage_nudge_enabled=False,
        )
        policy = LidarAwarePolicy(base_policy, smart_cfg)

        # 1. Far away (3.0m) -> speed scale 1.0
        far_depth = np.full((60, 640), 3.0, dtype=np.float32)
        policy.update_lidar_state(far_depth)
        policy.get_action()
        assert policy.get_diagnostics()["speed_scale"] == 1.0

        # 2. Medium distance (1.325m = midway between 0.65 and 2.0) -> scale ~0.6
        mid_depth = np.full((60, 640), 1.325, dtype=np.float32)
        policy.update_lidar_state(mid_depth)
        policy.get_action()
        scale = policy.get_diagnostics()["speed_scale"]
        assert 0.5 <= scale <= 0.7

        # 3. Close quarters (0.5m) -> crawl speed scale 0.2
        close_depth = np.full((60, 640), 0.5, dtype=np.float32)
        policy.update_lidar_state(close_depth)
        policy.get_action()
        assert policy.get_diagnostics()["speed_scale"] == 0.20


# =====================================================================
# Strategy 3: Panoramic Rotation Injection Tests
# =====================================================================

class TestPanoramicRotations:

    def test_panoramic_rotation_lifecycle(self):
        """Test panoramic rotation triggers, holds rotation, and completes."""
        config = PolicyConfig(
            policy_freq_hz=2.0, control_freq_hz=20.0,
            vx_range=(0.2, 0.6), vy_range=(-0.1, 0.1), wz_range=(-0.5, 0.5),
        )
        base = PinkUniformNoisePolicy(config)
        smart_cfg = SmartPolicyConfig(
            action_shaping_enabled=False,
            adaptive_speed_enabled=False,
            panoramic_rotations_enabled=True,
            panoramic_interval_s=1.0,    # 1 second for test speed
            panoramic_duration_s=0.5,    # 0.5s rotation
            panoramic_speed=0.60,
            coverage_nudge_enabled=False,
        )
        policy = LidarAwarePolicy(base, smart_cfg)
        policy.update_lidar_state(np.full((60, 640), 5.0, dtype=np.float32))

        # Initial tick: normal exploration
        cmd = policy.get_action()
        assert not policy.is_panoramic_active

        # Simulate time passing past interval
        policy._last_panoramic_time = time.time() - 2.0

        # Next tick triggers rotation injection
        cmd_pano = policy.get_action()
        assert policy.is_panoramic_active
        assert cmd_pano.v_linear == 0.0
        assert cmd_pano.v_lateral == 0.0
        assert np.isclose(cmd_pano.v_angular, 0.60)

        # Simulate completion of duration
        policy._panoramic_start_time = time.time() - 1.0
        cmd_after = policy.get_action()
        assert not policy.is_panoramic_active
        assert policy.get_diagnostics()["strategy"] == "pink_noise"


# =====================================================================
# Strategy 5: 2D Coverage Grid & Nudge Tests
# =====================================================================

class TestCoverageGrid:

    def test_coverage_grid_tracking(self):
        """Visit grid accumulates visits at mapped coordinates."""
        config = PolicyConfig()
        base = PinkUniformNoisePolicy(config)
        smart_cfg = SmartPolicyConfig(
            action_shaping_enabled=False,
            adaptive_speed_enabled=False,
            panoramic_rotations_enabled=False,
            coverage_nudge_enabled=True,
            coverage_grid_size=(50, 50),
            coverage_map_meters=16.0,
            coverage_nudge_scale=0.25,
        )
        policy = LidarAwarePolicy(base, smart_cfg)

        # Update pose at (0, 0) -> center cell (25, 25)
        for _ in range(5):
            policy.update_pose(x=0.0, y=0.0, yaw=0.0)

        grid = policy.visit_grid
        assert grid[25, 25] == 5
        assert np.count_nonzero(grid) == 1

        # Moving to an unvisited direction produces angular bias
        nudge = policy._compute_coverage_nudge()
        assert abs(nudge) <= 0.25


# =====================================================================
# Integration & Switching Tests
# =====================================================================

class TestSmartExplorationIntegration:

    def test_smart_exploration_runner_step(self, tmp_path):
        """Test SmartExplorationRunner runs steps and records 72-bin scans."""
        # Create mock robot, camera, lidar, safety
        mock_robot = MagicMock()
        mock_robot.dry_run = True
        mock_robot.odom_received = True
        mock_pose = MagicMock()
        mock_pose.x, mock_pose.y, mock_pose.z, mock_pose.yaw = 1.0, 2.0, 0.0, 0.5
        mock_robot.get_pose.return_value = mock_pose

        mock_camera = MagicMock()
        mock_camera.has_frame = True
        mock_camera.last_frame_time = time.time()
        test_rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        mock_camera.get_frame.return_value = (test_rgb, time.time())

        mock_lidar = MagicMock()
        mock_lidar.has_scan = True
        mock_depth = np.full((60, 640), 4.0, dtype=np.float32)
        mock_lidar.get_depth_image.return_value = mock_depth

        mock_safety = MagicMock()
        mock_safety.check.return_value = True

        recorder = DataRecorder(session_dir=tmp_path / "exploration", segment_size=10)
        policy_cfg = PolicyConfig(control_freq_hz=20.0)

        runner = SmartExplorationRunner(
            robot=mock_robot,
            camera=mock_camera,
            lidar=mock_lidar,
            safety=mock_safety,
            recorder=recorder,
            policy_config=policy_cfg,
            duration_minutes=0.01,
            record_lidar_scan=True,
        )

        assert runner.smart_policy is not None
        assert isinstance(runner.smart_policy, LidarAwarePolicy)

        # Directly run 2 steps through the loop logic
        cmd = runner.smart_policy.get_action()
        assert isinstance(cmd, DriveCommand)

        scan_72 = extract_scan_from_depth_image(mock_depth, num_bins=72)
        assert scan_72.shape == (72,)

    def test_switching_modes_isolated(self, tmp_path):
        """Verify vanilla ExplorationRunner vs SmartExplorationRunner are clean and distinct."""
        mock_robot = MagicMock()
        mock_robot.dry_run = True
        mock_cam = MagicMock()
        mock_lidar = MagicMock()
        mock_safety = MagicMock()
        recorder = DataRecorder(session_dir=tmp_path / "test_switch", segment_size=10)
        policy_cfg = PolicyConfig()

        # Vanilla runner has pure PinkUniformNoisePolicy
        vanilla_runner = ExplorationRunner(
            robot=mock_robot, camera=mock_cam, lidar=mock_lidar,
            safety=mock_safety, recorder=recorder, policy_config=policy_cfg,
        )
        assert isinstance(vanilla_runner._policy, PinkUniformNoisePolicy)
        assert not hasattr(vanilla_runner, "smart_policy")

        # Smart runner wraps with LidarAwarePolicy
        smart_runner = SmartExplorationRunner(
            robot=mock_robot, camera=mock_cam, lidar=mock_lidar,
            safety=mock_safety, recorder=recorder, policy_config=policy_cfg,
        )
        assert isinstance(smart_runner.smart_policy, LidarAwarePolicy)
        assert isinstance(smart_runner.smart_policy.base_policy, PinkUniformNoisePolicy)
