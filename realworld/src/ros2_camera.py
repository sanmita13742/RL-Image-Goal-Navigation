"""
realworld/src/ros2_camera.py — ROS 2 RealSense camera subscriber.
=================================================================
Subscribes to the Intel RealSense RGB topic and provides thread-safe
access to the latest frame as a numpy array.

Confirmed from launch script:
  ros2 launch realsense2_camera rs_launch.py
  Default topic: /camera/color/image_raw (sensor_msgs/Image, rgb8)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image

logger = logging.getLogger(__name__)


class ROS2Camera(Node):
    """ROS 2 subscriber for the Intel RealSense RGB camera.

    Maintains a thread-safe buffer of the latest frame as a numpy array.

    Parameters
    ----------
    topic : str
        Camera image topic. Default: /camera/color/image_raw.
    encoding : str
        Expected image encoding. Default: rgb8.
    expected_width : int
        Expected image width for validation. Default: 640.
    expected_height : int
        Expected image height for validation. Default: 480.
    """

    def __init__(
        self,
        topic: str = "/camera/color/image_raw",
        encoding: str = "rgb8",
        expected_width: int = 640,
        expected_height: int = 480,
        node_name: str = "minav_camera",
    ):
        super().__init__(node_name)

        self._expected_encoding = encoding
        self._expected_width = expected_width
        self._expected_height = expected_height

        # Thread-safe frame buffer
        self._frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._frame_count = 0
        self._last_frame_time: Optional[float] = None
        self._frame_consumed = False

        # QoS — match publisher exactly based on ros2 topic info
        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10,
        )
        self._sub = self.create_subscription(
            Image, topic, self._image_callback, cam_qos
        )

        logger.info(f"ROS2Camera subscribing to: {topic} (encoding: {encoding})")

    def _image_callback(self, msg: Image) -> None:
        """Convert sensor_msgs/Image to numpy array and store."""
        try:
            # Validate encoding
            if msg.encoding != self._expected_encoding:
                # Handle common encoding variations
                if msg.encoding == "bgr8" and self._expected_encoding == "rgb8":
                    channels = 3
                elif msg.encoding in ("rgb8", "bgr8"):
                    channels = 3
                else:
                    logger.warning(f"Unexpected encoding: {msg.encoding}")
                    return

            channels = 3  # RGB or BGR
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, channels
            )

            # Convert BGR → RGB if needed
            if msg.encoding == "bgr8":
                frame = frame[:, :, ::-1].copy()

            with self._frame_lock:
                self._frame = frame
                self._frame_count += 1
                self._last_frame_time = time.time()
                self._frame_consumed = False

                if self._frame_count == 1:
                    logger.info(
                        f"First camera frame received: {frame.shape} "
                        f"({msg.encoding})"
                    )

        except Exception as e:
            logger.error(f"Camera frame decode error: {e}")

    def get_frame(self) -> tuple[Optional[np.ndarray], Optional[float]]:
        """Return the latest unconsumed camera frame and its timestamp.
        
        Returns (None, None) if no new frame has arrived since the last call.
        """
        with self._frame_lock:
            if self._frame is None or self._frame_consumed:
                return None, None
            self._frame_consumed = True
            return self._frame.copy(), self._last_frame_time

    @property
    def frame_count(self) -> int:
        """Total number of frames received."""
        with self._frame_lock:
            return self._frame_count

    @property
    def last_frame_time(self) -> Optional[float]:
        """Wall-clock time of the last received frame."""
        with self._frame_lock:
            return self._last_frame_time

    @property
    def has_frame(self) -> bool:
        """Whether at least one frame has been received."""
        with self._frame_lock:
            return self._frame is not None
