"""
realworld/ — Real-world MINav pipeline for Ranger Mini V3 via ROS 2.

This package implements the real-robot counterpart of the MuJoCo simulation
pipeline. It collects data from the physical robot using ROS 2 (Foxy),
and feeds it into the exact same downstream processing (DINOv3 → Hindsight
→ TD3+BC training).

Hardware stack:
  - Robot:  Ranger Mini V3 via CAN bus (ranger_bringup)
  - Camera: Intel RealSense (realsense2_camera)
  - LiDAR:  RoboSense (/rslidar_points)
  - Odom:   ranger_base /odom at ~50 Hz
"""
