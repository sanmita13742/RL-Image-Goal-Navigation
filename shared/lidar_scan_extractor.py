"""
shared/lidar_scan_extractor.py — 1D LiDAR range scan vector extraction.
========================================================================
Extracts a compact 1D range scan vector (default: 72 bins = 5° resolution over 360°)
from either:
  1. A projected 2D LiDAR depth image (H, W) float32 (ultra-fast, <0.1ms)
  2. Raw 3D point cloud XYZ coordinates (N, 3) float32

Values are normalized to [0.0, 1.0] by dividing by max_range:
  - 0.0 = obstacle in contact (0 metres)
  - 1.0 = clear to max_range (e.g. 10.0 metres) or no return

This 1D geometric vector conditions the offline RL policy (Strategy 4),
providing geometric awareness that disambiguates visual aliasing traps.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def extract_scan_from_depth_image(
    depth_image: np.ndarray,
    num_bins: int = 72,
    max_range: float = 10.0,
) -> np.ndarray:
    """Extract a 1D normalized range scan from a 2D depth image.

    Parameters
    ----------
    depth_image : np.ndarray
        (H, W) float32 depth image spanning 360° horizontally.
    num_bins : int
        Number of angular bins (default: 72 -> 5° per bin).
    max_range : float
        Maximum LiDAR range in metres used for normalization.

    Returns
    -------
    np.ndarray
        (num_bins,) float32 vector in [0.0, 1.0].
    """
    if depth_image is None or depth_image.size == 0:
        return np.ones(num_bins, dtype=np.float32)

    h, w = depth_image.shape
    # Minimum distance across all elevation bands for each azimuth column
    azimuth_min = np.min(depth_image, axis=0)  # shape (W,)

    # Bin into num_bins
    col_boundaries = np.linspace(0, w, num_bins + 1, endpoint=True).astype(np.int32)
    scan_1d = np.empty(num_bins, dtype=np.float32)

    for i in range(num_bins):
        c0, c1 = col_boundaries[i], col_boundaries[i + 1]
        if c1 > c0:
            sector_min = np.min(azimuth_min[c0:c1])
        else:
            sector_min = max_range
        # Handle nan/inf
        if not np.isfinite(sector_min):
            sector_min = max_range
        scan_1d[i] = sector_min

    # Normalize to [0, 1]
    scan_norm = np.clip(scan_1d / float(max_range), 0.0, 1.0).astype(np.float32)
    return scan_norm


def extract_scan_from_xyz(
    xyz: np.ndarray,
    num_bins: int = 72,
    max_range: float = 10.0,
    min_range: float = 0.1,
    height_min: float = -0.3,
    height_max: float = 1.0,
) -> np.ndarray:
    """Extract a 1D normalized range scan directly from 3D XYZ points.

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) float32 coordinates relative to robot base_link.
    num_bins : int
        Number of angular bins (default: 72).
    max_range : float
        Maximum range in metres.
    min_range : float
        Minimum valid range (filters chassis self-hits).
    height_min : float
        Minimum height relative to base_link (filters ground plane).
    height_max : float
        Maximum height relative to base_link (filters ceiling).

    Returns
    -------
    np.ndarray
        (num_bins,) float32 vector in [0.0, 1.0].
    """
    if xyz is None or len(xyz) == 0:
        return np.ones(num_bins, dtype=np.float32)

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    # Filter by height and 2D range
    dist_2d = np.sqrt(x**2 + y**2)
    valid = (
        (z >= height_min)
        & (z <= height_max)
        & (dist_2d >= min_range)
        & (dist_2d <= max_range)
        & np.isfinite(dist_2d)
    )

    scan_1d = np.full(num_bins, max_range, dtype=np.float32)
    if not np.any(valid):
        return np.ones(num_bins, dtype=np.float32)

    x_v, y_v, dist_v = x[valid], y[valid], dist_2d[valid]
    angles = np.arctan2(y_v, x_v)  # in [-pi, pi]

    # Map [-pi, pi] to [0, num_bins)
    bin_idx = ((angles + np.pi) / (2.0 * np.pi) * num_bins).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)

    np.minimum.at(scan_1d, bin_idx, dist_v)

    scan_norm = np.clip(scan_1d / float(max_range), 0.0, 1.0).astype(np.float32)
    return scan_norm
