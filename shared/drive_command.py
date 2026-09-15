"""
shared/drive_command.py — Platform-independent motion command dataclass.
===================================================================
Extracted from simulation/robot_base.py so that both MuJoCo and
real-world ROS 2 pipelines can use the same DriveCommand type
without importing MuJoCo.
"""

from __future__ import annotations
import dataclasses


@dataclasses.dataclass
class DriveCommand:
    """Unified motion command for the Ranger Mini V3.

    Maps directly to geometry_msgs/Twist in ROS 2:
        linear.x  = v_linear   (m/s, + forward)
        linear.y  = v_lateral  (m/s, + left / crab walk)
        angular.z = v_angular  (rad/s, + turn left / CCW)
    """
    v_linear:  float = 0.0   # m/s  (+ forward)
    v_lateral: float = 0.0   # m/s  (+ left, crab walk)
    v_angular: float = 0.0   # rad/s (+ turn left, CCW)
