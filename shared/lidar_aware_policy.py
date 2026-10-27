"""
shared/lidar_aware_policy.py — Intelligent LiDAR-aware exploration policy.
==========================================================================
Wraps the MINav-faithful PinkUniformNoisePolicy without modifying its source.
Provides 4 additive, switchable exploration strategies:
  1. LiDAR-Aware Action Shaping: Pre-emptively narrows action ranges away
     from obstacles instead of blindly sampling into walls and getting blocked.
  2. Adaptive Speed Zones: Scales linear velocity based on clearance (fast in
     open corridors, slow crawl in tight areas for dense visual mapping).
  3. Panoramic Rotation Injections: Periodically performs 360° in-place yaw
     rotations (every N minutes) to capture full-circle visual views.
  4. Coverage Heatmap & Nudge: Maintains a 2D visit grid from odometry and
     adds a gentle angular bias toward unexplored space.

All strategies default to True when using smart mode, but each can be
toggled independently via SmartPolicyConfig.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from shared.drive_command import DriveCommand
from shared.pink_uniform_policy import PinkUniformNoisePolicy, PolicyConfig

logger = logging.getLogger(__name__)


@dataclass
class SmartPolicyConfig:
    """Configuration for smart exploration enhancements."""

    # Strategy 1: LiDAR-Aware Action Shaping
    action_shaping_enabled: bool = True
    obstacle_clearance_threshold: float = 0.65  # metres
    wall_danger_depth: float = 0.55             # metres
    tight_space_depth: float = 0.50             # metres (distance below which reverse evasion triggers)
    sector_min_pixels: int = 75                 # min near pixels to confirm obstacle
    # Minimum pixel count for the distance-only shaping path.
    # Prevents single noisy LiDAR returns from triggering jittery shaping.
    sector_min_pixels_distance: int = 25
    evasion_reverse_speed: float = -0.15        # m/s reverse when tight front
    evasion_turn_speed: float = 0.8             # rad/s turn when avoiding obstacle

    # Strategy 2: Adaptive Speed Zones
    adaptive_speed_enabled: bool = True
    speed_min_dist: float = 0.65                # metres (crawl speed threshold)
    speed_max_dist: float = 2.00                # metres (full speed threshold)
    speed_min_scale: float = 0.20               # min speed factor (crawl)
    speed_max_scale: float = 1.00               # max speed factor (open space)
    wz_tight_boost: float = 1.30                # turn rate boost in tight quarters

    # Strategy 3: Panoramic Rotation Injections
    # NOTE: Disabled by default — panoramic frames produce zero-displacement
    # transitions that dilute hindsight relabeling quality. Enable after
    # confirming the relabeler filters |pos_s - pos_g| < threshold pairs.
    panoramic_rotations_enabled: bool = False
    panoramic_interval_s: float = 180.0         # seconds between injections (3 min)
    panoramic_duration_s: float = 10.5          # duration (~360 deg at 0.6 rad/s)
    panoramic_speed: float = 0.60               # rad/s rotation speed

    # Strategy 5: Coverage Heatmap & Angular Nudge
    coverage_nudge_enabled: bool = True
    coverage_grid_size: Tuple[int, int] = (50, 50)
    coverage_map_meters: float = 16.0           # 16m x 16m area
    coverage_nudge_scale: float = 0.25          # max rad/s angular nudge
    coverage_probe_radius_cells: int = 4        # radius in cells to check for open space


class LidarAwarePolicy:
    """Wraps PinkUniformNoisePolicy with intelligent real-world exploration.

    Parameters
    ----------
    base_policy : PinkUniformNoisePolicy
        The unmodified underlying pink uniform noise policy.
    config : SmartPolicyConfig, optional
        Configuration for smart strategies. If None, default config is used.
    """

    def __init__(
        self,
        base_policy: PinkUniformNoisePolicy,
        config: Optional[SmartPolicyConfig] = None,
    ):
        self._base = base_policy
        self.config = config or SmartPolicyConfig()

        # Cached sensor inputs
        self._depth: Optional[np.ndarray] = None
        self._pose_x: float = 0.0
        self._pose_y: float = 0.0
        self._pose_yaw: float = 0.0

        # Sector metrics
        self._front_min: float = 10.0
        self._left_min: float = 10.0
        self._right_min: float = 10.0
        self._front_px: int = 0
        self._left_px: int = 0
        self._right_px: int = 0

        # Strategy 3: Panoramic rotation state
        self._panoramic_active: bool = False
        self._panoramic_start_time: float = 0.0
        self._last_panoramic_time: float = time.time()

        # Strategy 5: 2D Coverage Grid
        gx, gy = self.config.coverage_grid_size
        self._visit_grid = np.zeros((gx, gy), dtype=np.int32)
        self._cell_size_m = self.config.coverage_map_meters / float(gx)
        self._map_origin_m = self.config.coverage_map_meters / 2.0

        # Action history & diagnostics
        self._last_raw_action: DriveCommand = DriveCommand(0.0, 0.0, 0.0)
        self._last_smart_action: DriveCommand = DriveCommand(0.0, 0.0, 0.0)
        self._diagnostics: dict = {}

    @property
    def base_policy(self) -> PinkUniformNoisePolicy:
        """Reference to the underlying pink noise policy."""
        return self._base

    @property
    def target_action(self) -> DriveCommand:
        """Target action from the underlying policy."""
        return self._base.target_action

    @property
    def smoothed_action(self) -> DriveCommand:
        """Current smoothed action output."""
        return self._last_smart_action

    @property
    def visit_grid(self) -> np.ndarray:
        """Copy of the 2D exploration visit grid."""
        return self._visit_grid.copy()

    @property
    def is_panoramic_active(self) -> bool:
        """Whether a panoramic rotation injection is currently executing."""
        return self._panoramic_active

    def update_lidar_state(self, depth: np.ndarray) -> None:
        """Update cached LiDAR depth projection and compute sector distances.

        Parameters
        ----------
        depth : np.ndarray
            (H, W) float32 depth image spanning 360° horizontally.
        """
        if depth is None or depth.size == 0:
            return

        self._depth = depth
        h, w = depth.shape
        threshold = self.config.wall_danger_depth

        # Sectors: RoboSense atan2 mapping:
        # Col 0 to W//3: right (-180 to -60 deg)
        # Col W//3 to 2*W//3: front (-60 to +60 deg)
        # Col 2*W//3 to W: left (+60 to +180 deg)
        right_sec = depth[:, 0 : w // 3]
        front_sec = depth[:, w // 3 : 2 * w // 3]
        left_sec = depth[:, 2 * w // 3 : w]

        self._front_min = float(front_sec.min()) if front_sec.size > 0 else 10.0
        self._right_min = float(right_sec.min()) if right_sec.size > 0 else 10.0
        self._left_min = float(left_sec.min()) if left_sec.size > 0 else 10.0

        self._front_px = int((front_sec < threshold).sum())
        self._right_px = int((right_sec < threshold).sum())
        self._left_px = int((left_sec < threshold).sum())

    def update_pose(self, x: float, y: float, yaw: float) -> None:
        """Update current robot pose and mark visit on coverage grid.

        Parameters
        ----------
        x, y : float
            Odometry position in metres.
        yaw : float
            Odometry heading in radians.
        """
        self._pose_x = x
        self._pose_y = y
        self._pose_yaw = yaw

        # Update visit grid (Strategy 5)
        if self.config.coverage_nudge_enabled:
            gx_dim, gy_dim = self.config.coverage_grid_size
            gx = int((x + self._map_origin_m) / self._cell_size_m)
            gy = int((y + self._map_origin_m) / self._cell_size_m)
            if 0 <= gx < gx_dim and 0 <= gy < gy_dim:
                self._visit_grid[gx, gy] += 1

    def _compute_coverage_nudge(self) -> float:
        """Find least-visited adjacent sector and return angular nudge (rad/s)."""
        if not self.config.coverage_nudge_enabled:
            return 0.0

        gx_dim, gy_dim = self.config.coverage_grid_size
        curr_gx = int((self._pose_x + self._map_origin_m) / self._cell_size_m)
        curr_gy = int((self._pose_y + self._map_origin_m) / self._cell_size_m)

        r = self.config.coverage_probe_radius_cells
        # Probe 8 directions: angles 0, 45, 90, 135, 180, -135, -90, -45 deg
        angles = np.linspace(-np.pi, np.pi, 8, endpoint=False)
        visit_scores = []

        for ang in angles:
            px = int(curr_gx + r * np.cos(ang))
            py = int(curr_gy + r * np.sin(ang))
            if 0 <= px < gx_dim and 0 <= py < gy_dim:
                score = self._visit_grid[px, py]
            else:
                score = 999999  # Penalize outside map boundary
            visit_scores.append(score)

        least_idx = int(np.argmin(visit_scores))
        best_angle = angles[least_idx]

        # Angular difference between current heading and best angle
        diff = (best_angle - self._pose_yaw + np.pi) % (2.0 * np.pi) - np.pi
        nudge = np.clip(diff * 0.4, -self.config.coverage_nudge_scale, self.config.coverage_nudge_scale)
        return float(nudge)

    def _check_panoramic_rotation(self, current_time: float) -> Optional[DriveCommand]:
        """Handle state machine for Strategy 3: Panoramic Rotation Injections."""
        if not self.config.panoramic_rotations_enabled:
            return None

        # Check if currently in rotation
        if self._panoramic_active:
            elapsed = current_time - self._panoramic_start_time
            if elapsed < self.config.panoramic_duration_s:
                # Still rotating
                return DriveCommand(
                    v_linear=0.0,
                    v_lateral=0.0,
                    v_angular=self.config.panoramic_speed,
                )
            else:
                # Finished rotation
                self._panoramic_active = False
                self._last_panoramic_time = current_time
                logger.info(f"[SMART POLICY] Panoramic 360° rotation completed in {elapsed:.1f}s.")
                return None

        # Check if interval has elapsed to start new rotation
        if (current_time - self._last_panoramic_time) >= self.config.panoramic_interval_s:
            # Only trigger rotation if we are not in severe collision danger
            if (self._front_min >= self.config.tight_space_depth
                and self._left_min >= self.config.wall_danger_depth
                and self._right_min >= self.config.wall_danger_depth):
                self._panoramic_active = True
                self._panoramic_start_time = current_time
                logger.info(
                    f"[SMART POLICY] Triggering panoramic 360° rotation injection at pos "
                    f"({self._pose_x:.2f}, {self._pose_y:.2f})"
                )
                return DriveCommand(
                    v_linear=0.0,
                    v_lateral=0.0,
                    v_angular=self.config.panoramic_speed,
                )

        return None

    def get_action(self) -> DriveCommand:
        """Sample and compute the next intelligent exploration action.

        Returns
        -------
        DriveCommand
            Action with applied shaping, speed adaptation, and coverage nudge.
        """
        now = time.time()

        # 1. Check for active panoramic rotation injection (Strategy 3)
        panoramic_cmd = self._check_panoramic_rotation(now)
        if panoramic_cmd is not None:
            self._last_smart_action = panoramic_cmd
            self._diagnostics = {
                "strategy": "panoramic_rotation",
                "speed_scale": 0.0,
                "front_min": self._front_min,
                "panoramic_active": True,
                "coverage_nudge": 0.0,
            }
            return panoramic_cmd

        # 2. Get base action from unmodified PinkUniformNoisePolicy
        raw_cmd = self._base.get_action()
        self._last_raw_action = raw_cmd

        vx = raw_cmd.v_linear
        vy = raw_cmd.v_lateral
        wz = raw_cmd.v_angular

        # 3. Strategy 5: Add coverage nudge to wz
        nudge_wz = 0.0
        if self.config.coverage_nudge_enabled:
            nudge_wz = self._compute_coverage_nudge()
            wz += nudge_wz

        # 4. Strategy 1: LiDAR-Aware Action Shaping
        # Instead of letting the safety gate reject blocked commands, reshape them
        shaped = False
        if self.config.action_shaping_enabled:
            # Use AND logic: require BOTH sufficient pixel count AND close
            # distance to confirm a real obstacle.  The pixel threshold for
            # the distance-only path (sector_min_pixels_distance) is much
            # lower than the full sector_min_pixels, but still prevents
            # single noisy LiDAR returns from triggering jittery shaping.
            min_px_dist = self.config.sector_min_pixels_distance
            front_blocked = (
                self._front_px >= self.config.sector_min_pixels
                or (self._front_min < self.config.obstacle_clearance_threshold
                    and self._front_px >= min_px_dist)
            )
            left_blocked = (
                self._left_px >= self.config.sector_min_pixels
                or (self._left_min < self.config.wall_danger_depth
                    and self._left_px >= min_px_dist)
            )
            right_blocked = (
                self._right_px >= self.config.sector_min_pixels
                or (self._right_min < self.config.wall_danger_depth
                    and self._right_px >= min_px_dist)
            )

            # If front is blocked, avoid driving into obstacle
            if front_blocked:
                shaped = True
                if self._front_min < self.config.tight_space_depth:
                    # Critical tight space: back up gently
                    vx = self.config.evasion_reverse_speed
                else:
                    # Narrow clearance: stop forward surge
                    vx = min(vx, 0.0)

                # Steer toward direction of greatest clearance
                if self._left_min > self._right_min:
                    wz = max(wz, self.config.evasion_turn_speed)
                else:
                    wz = min(wz, -self.config.evasion_turn_speed)

            # Clamp lateral crab-walk away from side obstacles
            if left_blocked:
                # Obstacle on left: clamp to moving right only (vy <= 0)
                vy = min(vy, 0.0)
                shaped = True
            if right_blocked:
                # Obstacle on right: clamp to moving left only (vy >= 0)
                vy = max(vy, 0.0)
                shaped = True

        # 5. Strategy 2: Adaptive Speed Zones
        speed_scale = 1.0
        if self.config.adaptive_speed_enabled and vx > 0:
            d_min = self.config.speed_min_dist
            d_max = self.config.speed_max_dist
            s_min = self.config.speed_min_scale
            s_max = self.config.speed_max_scale

            # Smooth linear interpolation between d_min and d_max
            if self._front_min >= d_max:
                speed_scale = s_max
            elif self._front_min <= d_min:
                speed_scale = s_min
                # Boost turn rate in tight quarters to quickly escape
                wz *= self.config.wz_tight_boost
            else:
                frac = (self._front_min - d_min) / (d_max - d_min)
                speed_scale = s_min + frac * (s_max - s_min)

            vx *= speed_scale

        # 6. Physical limit clamping (Ranger Mini V3 bounds)
        vx = float(np.clip(vx, -0.9, 1.8))
        vy = float(np.clip(vy, -1.0, 1.0))
        wz = float(np.clip(wz, -1.5, 1.5))

        cmd = DriveCommand(v_linear=vx, v_lateral=vy, v_angular=wz)
        self._last_smart_action = cmd

        self._diagnostics = {
            "strategy": "shaped_pink_noise" if shaped else "pink_noise",
            "speed_scale": speed_scale,
            "front_min": self._front_min,
            "panoramic_active": False,
            "coverage_nudge": nudge_wz,
            "shaped": shaped,
        }

        return cmd

    def get_diagnostics(self) -> dict:
        """Return diagnostic metrics from the latest action step."""
        return dict(self._diagnostics)
