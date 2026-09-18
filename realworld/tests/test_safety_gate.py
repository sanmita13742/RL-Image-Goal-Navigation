import pytest
import rclpy
import struct
import time
from unittest.mock import MagicMock
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from realworld.src.lidar_safety_gate import LidarSafetyGate

def create_cloud(points):
    """Create a basic PointCloud2 message for testing."""
    msg = PointCloud2()
    msg.point_step = 16
    data = bytearray()
    for p in points:
        data.extend(struct.pack('4f', p[0], p[1], p[2], 1.0))
    msg.data = data
    return msg

@pytest.fixture(scope="module")
def ros_init():
    rclpy.init()
    yield
    rclpy.shutdown()

@pytest.fixture
def gate(ros_init):
    from rclpy.parameter import Parameter
    overrides = [
        Parameter('minimum_points', Parameter.Type.INTEGER, 1),
        Parameter('robot_length', Parameter.Type.DOUBLE, 0.751),
        Parameter('robot_width', Parameter.Type.DOUBLE, 0.60),
        Parameter('safety_margin', Parameter.Type.DOUBLE, 0.1),
        Parameter('lidar_offset_x', Parameter.Type.DOUBLE, 0.0508),
        Parameter('lidar_offset_y', Parameter.Type.DOUBLE, 0.0),
        Parameter('stop_distance', Parameter.Type.DOUBLE, 0.5),
        Parameter('blindspot_radius', Parameter.Type.DOUBLE, 0.15)
    ]
    node = LidarSafetyGate(parameter_overrides=overrides)
    node.cmd_pub.publish = MagicMock()
    node.last_cloud_time = time.time()
    yield node
    node.destroy_node()

def test_forward_obstacle_blocks_forward(gate):
    # base_link x bound = 0.3755
    # lidar_offset = 0.0508
    # If point is at lidar x = 0.6, base_link x = 0.6508
    # distance from chassis = 0.6508 - 0.3755 = 0.2753 < stop_distance (0.5)
    msg = create_cloud([[0.6, 0.0, 0.0]])
    gate.lidar_callback(msg)
    gate.last_cloud_time = time.time()
    
    cmd = Twist()
    cmd.linear.x = 0.5
    gate.cmd_callback(cmd)
    
    published = gate.cmd_pub.publish.call_args[0][0]
    assert published.linear.x == 0.0

def test_backward_obstacle_blocks_backward(gate):
    # base_link x bound = -0.3755
    # If point is at lidar x = -0.6, base_link x = -0.5492
    # parallel projection (dir_x = -1) = 0.5492
    # distance from chassis = 0.5492 - 0.3755 = 0.1737 < 0.5
    msg = create_cloud([[-0.6, 0.0, 0.0]])
    gate.lidar_callback(msg)
    gate.last_cloud_time = time.time()
    
    cmd = Twist()
    cmd.linear.x = -0.5
    gate.cmd_callback(cmd)
    
    published = gate.cmd_pub.publish.call_args[0][0]
    assert published.linear.x == 0.0

def test_left_obstacle_blocks_left(gate):
    # base_link y bound = 0.3
    # If point is at lidar y = 0.6, base_link y = 0.6
    # distance from chassis = 0.6 - 0.3 = 0.3 < 0.5
    msg = create_cloud([[0.0, 0.6, 0.0]])
    gate.lidar_callback(msg)
    gate.last_cloud_time = time.time()
    
    cmd = Twist()
    cmd.linear.y = 0.5
    gate.cmd_callback(cmd)
    
    published = gate.cmd_pub.publish.call_args[0][0]
    assert published.linear.y == 0.0

def test_diagonal_obstacle_blocks_diagonal(gate):
    # moving at 45 deg (dir_x=0.707, dir_y=0.707)
    # max_para = 0.3755*0.707 + 0.3*0.707 = 0.4776
    # base_link x=0.6, y=0.6
    # parallel = 0.6*0.707 + 0.6*0.707 = 0.8484
    # distance from chassis = 0.8484 - 0.4776 = 0.3708 < 0.5
    # lidar x = 0.6 - 0.0508 = 0.5492, y = 0.6
    msg = create_cloud([[0.5492, 0.6, 0.0]])
    gate.lidar_callback(msg)
    gate.last_cloud_time = time.time()
    
    cmd = Twist()
    cmd.linear.x = 0.5
    cmd.linear.y = 0.5
    gate.cmd_callback(cmd)
    
    published = gate.cmd_pub.publish.call_args[0][0]
    assert published.linear.x == 0.0
    assert published.linear.y == 0.0

def test_obstacle_outside_chassis_corridor_allows_motion(gate):
    # Forward motion (dir_x=1, dir_y=0)
    # corridor half width = 0.3 + 0.1(margin) = 0.4
    # Obstacle at y = 0.6 (outside corridor)
    msg = create_cloud([[0.6, 0.6, 0.0]])
    gate.lidar_callback(msg)
    gate.last_cloud_time = time.time()
    
    cmd = Twist()
    cmd.linear.x = 0.5
    gate.cmd_callback(cmd)
    
    published = gate.cmd_pub.publish.call_args[0][0]
    assert published.linear.x == 0.5

def test_corner_obstacle_blocks_diagonal(gate):
    # Front-left corner of chassis is approx x=0.3755, y=0.3 in base_link
    # Point just outside corner: base_link x=0.45, y=0.4
    # moving diagonal front-left
    msg = create_cloud([[0.45 - 0.0508, 0.4, 0.0]])
    gate.lidar_callback(msg)
    gate.last_cloud_time = time.time()
    
    cmd = Twist()
    cmd.linear.x = 0.5
    cmd.linear.y = 0.5
    gate.cmd_callback(cmd)
    
    published = gate.cmd_pub.publish.call_args[0][0]
    assert published.linear.x == 0.0
    assert published.linear.y == 0.0

def test_stale_cloud_fails_closed(gate):
    msg = create_cloud([[2.0, 0.0, 0.0]])
    gate.lidar_callback(msg)
    
    cmd = Twist()
    cmd.linear.x = 0.5
    
    # Artificially age the cloud
    gate.last_cloud_time = time.time() - 2.0
    gate.cmd_callback(cmd)
    
    published = gate.cmd_pub.publish.call_args[0][0]
    assert published.linear.x == 0.0
    assert published.linear.y == 0.0
