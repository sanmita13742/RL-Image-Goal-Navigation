"""
realworld/src/ros2_lidar.py — ROS 2 RoboSense LiDAR → depth image projection.
===============================================================================
Subscribes to /scanner/cloud (sensor_msgs/PointCloud2) and projects the
3D point cloud into a 2D obstacle proximity image that matches the interface
expected by the exploration policy's collision avoidance.

The MuJoCo pipeline uses a depth renderer (60×640) from a virtual lidar_cam.
This module produces a compatible (H, W) numpy array where lower values mean
closer obstacles, so the same thresholds (0.3m, 0.6m, 1.0m) work.

Confirmed from rosbag:
  /scanner/cloud  → sensor_msgs/PointCloud2 (~10 Hz, 730 msgs / 73s)
"""

from __future__ import annotations

import logging
import struct
import threading
import time
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2

logger = logging.getLogger(__name__)


def _parse_pointcloud2_xyz(msg: PointCloud2) -> np.ndarray:
    """Extract (N, 3) float32 XYZ array from a PointCloud2 message.

    Handles typical field layouts from RoboSense LiDARs (x, y, z, intensity, ...).
    Only extracts the first three float32 fields (x, y, z).
    """
    # Find x, y, z field offsets
    field_map = {f.name: f for f in msg.fields}

    if "x" not in field_map or "y" not in field_map or "z" not in field_map:
        raise ValueError(f"PointCloud2 missing x/y/z fields. Available: {list(field_map.keys())}")

    ox = field_map["x"].offset
    oy = field_map["y"].offset
    oz = field_map["z"].offset
    point_step = msg.point_step

    n_points = msg.width * msg.height
    data = np.frombuffer(msg.data, dtype=np.uint8)

    # Extract x, y, z as float32 views
    xs = np.ndarray(n_points, dtype=np.float32,
                    buffer=data, offset=ox, strides=(point_step,))
    ys = np.ndarray(n_points, dtype=np.float32,
                    buffer=data, offset=oy, strides=(point_step,))
    zs = np.ndarray(n_points, dtype=np.float32,
                    buffer=data, offset=oz, strides=(point_step,))

    return np.column_stack([xs, ys, zs])


class ROS2LiDAR(Node):
    """ROS 2 subscriber for RoboSense LiDAR with 2D depth projection.

    Projects the 3D point cloud into a 2D obstacle proximity image
    compatible with the exploration policy's collision avoidance.

    The projection works by:
    1. Filtering points by height (remove ground and ceiling).
    2. Computing horizontal angle (atan2) and distance for each point.
    3. Binning into a (projection_height, projection_width) grid.
    4. Storing minimum distance per bin.

    Parameters
    ----------
    topic : str
        LiDAR topic. Default: /scanner/cloud.
    projection_width : int
        Output depth image width. Default: 640.
    projection_height : int
        Output depth image height. Default: 60.
    max_range : float
        Maximum range in metres. Default: 10.0.
    min_range : float
        Minimum valid range in metres. Default: 0.1.
    height_min : float
        Min height relative to base_link to include. Default: -0.3 (ground filter).
    height_max : float
        Max height relative to base_link to include. Default: 1.0 (ceiling filter).
    fov_deg : float
        Horizontal field of view. Default: 360 (full RoboSense scan).
    """

    def __init__(
        self,
        topic: str = "/scanner/cloud",
        projection_width: int = 640,
        projection_height: int = 60,
        max_range: float = 10.0,
        min_range: float = 0.1,
        height_min: float = -0.3,
        height_max: float = 1.0,
        fov_deg: float = 360.0,
        node_name: str = "minav_lidar",
    ):
        super().__init__(node_name)

        self._proj_w = projection_width
        self._proj_h = projection_height
        self._max_range = max_range
        self._min_range = min_range
        self._height_min = height_min
        self._height_max = height_max
        self._fov_rad = np.radians(fov_deg)

        # Thread-safe depth image buffer
        self._depth_img: Optional[np.ndarray] = None
        self._depth_lock = threading.Lock()
        self._scan_count = 0
        self._last_scan_time: Optional[float] = None

        # QoS — match rosbag profile (reliable, volatile)
        lidar_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )
        self._sub = self.create_subscription(
            PointCloud2, topic, self._lidar_callback, lidar_qos
        )

        logger.info(f"ROS2LiDAR subscribing to: {topic}")
        logger.info(f"  Projection: {projection_width}×{projection_height}, "
                     f"range: [{min_range}, {max_range}]m, "
                     f"height: [{height_min}, {height_max}]m")

    def _project_to_depth(self, xyz: np.ndarray) -> np.ndarray:
        """Project (N, 3) XYZ points into a (H, W) depth image.

        The output image has the same semantics as the MuJoCo depth render:
        each pixel stores the minimum distance to an obstacle in that angular
        bin. Pixels with no points are filled with max_range.

        The horizontal axis spans the full FOV; the vertical axis spans
        different elevation bands within [height_min, height_max].
        """
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        # Filter by height (relative to base_link)
        height_mask = (z >= self._height_min) & (z <= self._height_max)

        # Filter by range
        dist_2d = np.sqrt(x**2 + y**2)
        range_mask = (dist_2d >= self._min_range) & (dist_2d <= self._max_range)

        valid = height_mask & range_mask
        if valid.sum() == 0:
            return np.full((self._proj_h, self._proj_w), self._max_range, dtype=np.float32)

        x_v, y_v, z_v = x[valid], y[valid], z[valid]
        dist_v = dist_2d[valid]

        # Horizontal angle: atan2(y, x), mapped to [0, W)
        angles = np.arctan2(y_v, x_v)  # [-π, π]
        # Normalize to [0, 1] within FOV
        half_fov = self._fov_rad / 2.0
        col_frac = (angles + half_fov) / self._fov_rad
        col_idx = np.clip((col_frac * self._proj_w).astype(np.int32), 0, self._proj_w - 1)

        # Vertical bin: map height within [height_min, height_max] to [0, H)
        z_frac = (z_v - self._height_min) / (self._height_max - self._height_min)
        row_idx = np.clip((z_frac * self._proj_h).astype(np.int32), 0, self._proj_h - 1)

        # Build depth image: minimum distance per bin
        depth_img = np.full((self._proj_h, self._proj_w), self._max_range, dtype=np.float32)
        np.minimum.at(depth_img, (row_idx, col_idx), dist_v)

        return depth_img

    def _lidar_callback(self, msg: PointCloud2) -> None:
        """Parse PointCloud2 and project to depth image."""
        try:
            xyz = _parse_pointcloud2_xyz(msg)
            depth_img = self._project_to_depth(xyz)

            with self._depth_lock:
                self._depth_img = depth_img
                self._scan_count += 1
                self._last_scan_time = time.time()

                if self._scan_count == 1:
                    n_points = len(xyz)
                    logger.info(
                        f"First LiDAR scan: {n_points} points → "
                        f"depth image {depth_img.shape}"
                    )

        except Exception as e:
            logger.error(f"LiDAR processing error: {e}")

    def get_depth_image(self) -> Optional[np.ndarray]:
        """Return the latest depth image (H, W) float32, or None.

        Values represent minimum obstacle distance in metres per angular bin.
        Lower values = closer obstacles (same semantics as MuJoCo depth render).
        """
        with self._depth_lock:
            if self._depth_img is None:
                return None
            return self._depth_img.copy()

    @property
    def scan_count(self) -> int:
        """Total number of LiDAR scans received."""
        with self._depth_lock:
            return self._scan_count

    @property
    def last_scan_time(self) -> Optional[float]:
        """Wall-clock time of the last received scan."""
        with self._depth_lock:
            return self._last_scan_time

    @property
    def has_scan(self) -> bool:
        """Whether at least one scan has been received."""
        with self._depth_lock:
            return self._depth_img is not None
