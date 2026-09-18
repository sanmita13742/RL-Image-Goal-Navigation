import pytest
import rclpy
import struct
import time
from unittest.mock import MagicMock
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2

# Adjust path to import the node
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
        # x, y, z, intensity
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
    # Set parameters to lower minimum_points to 1 for easy testing
    from rclpy.parameter import Parameter
    overrides = [
        Parameter('minimum_points', Parameter.Type.INTEGER, 1)
    ]
    node = LidarSafetyGate(parameter_overrides=overrides)
    
    # Mock the publisher to capture output
    node.cmd_pub.publish = MagicMock()
    
    # Mock time so cloud is never stale
    node.last_cloud_time = time.time()
    
    yield node
    node.destroy_node()

def test_forward_obstacle_blocks_forward(gate):
    # Obstacle directly in front
    msg = create_cloud([[0.4, 0.0, 0.0]])
    gate.lidar_callback(msg)
    gate.last_cloud_time = time.time()
    
    # Command forward
    cmd = Twist()
    cmd.linear.x = 0.5
    gate.cmd_callback(cmd)
    
    # Should be blocked
    published = gate.cmd_pub.publish.call_args[0][0]
    assert published.linear.x == 0.0

def test_forward_obstacle_allows_backward(gate):
    # Obstacle directly in front
    msg = create_cloud([[0.4, 0.0, 0.0]])
    gate.lidar_callback(msg)
    gate.last_cloud_time = time.time()
    
    # Command backward
    cmd = Twist()
    cmd.linear.x = -0.5
    gate.cmd_callback(cmd)
    
    # Should be allowed
    published = gate.cmd_pub.publish.call_args[0][0]
    assert published.linear.x == -0.5

def test_backward_obstacle_blocks_backward(gate):
    # Obstacle behind
    msg = create_cloud([[-0.4, 0.0, 0.0]])
    gate.lidar_callback(msg)
    gate.last_cloud_time = time.time()
    
    # Command backward
    cmd = Twist()
    cmd.linear.x = -0.5
    gate.cmd_callback(cmd)
    
    # Should be blocked
    published = gate.cmd_pub.publish.call_args[0][0]
    assert published.linear.x == 0.0

def test_left_obstacle_blocks_left(gate):
    # Obstacle to the left
    msg = create_cloud([[0.0, 0.4, 0.0]])
    gate.lidar_callback(msg)
    gate.last_cloud_time = time.time()
    
    # Command left (crab walk)
    cmd = Twist()
    cmd.linear.y = 0.5
    gate.cmd_callback(cmd)
    
    # Should be blocked
    published = gate.cmd_pub.publish.call_args[0][0]
    assert published.linear.y == 0.0

def test_diagonal_obstacle_blocks_diagonal(gate):
    # Obstacle front-left (at 45 degrees, r=0.42 => x=0.3, y=0.3)
    msg = create_cloud([[0.3, 0.3, 0.0]])
    gate.lidar_callback(msg)
    gate.last_cloud_time = time.time()
    
    # Command diagonal front-left
    cmd = Twist()
    cmd.linear.x = 0.5
    cmd.linear.y = 0.5
    gate.cmd_callback(cmd)
    
    # Should be blocked
    published = gate.cmd_pub.publish.call_args[0][0]
    assert published.linear.x == 0.0
    assert published.linear.y == 0.0

def test_obstacle_outside_corridor_allows_motion(gate):
    # Obstacle far to the right (x=0.4, y=-1.0)
    msg = create_cloud([[0.4, -1.0, 0.0]])
    gate.lidar_callback(msg)
    gate.last_cloud_time = time.time()
    
    # Command forward
    cmd = Twist()
    cmd.linear.x = 0.5
    gate.cmd_callback(cmd)
    
    # Should be allowed because it's outside the corridor width
    published = gate.cmd_pub.publish.call_args[0][0]
    assert published.linear.x == 0.5

def test_stale_cloud_fails_closed(gate):
    # Setup cloud
    msg = create_cloud([[2.0, 0.0, 0.0]]) # Safe distance
    gate.lidar_callback(msg)
    
    # Command forward (normally safe)
    cmd = Twist()
    cmd.linear.x = 0.5
    
    # Artificially age the cloud
    gate.last_cloud_time = time.time() - 2.0 # Older than timeout of 1.0s
    gate.cmd_callback(cmd)
    
    # Should be blocked
    published = gate.cmd_pub.publish.call_args[0][0]
    assert published.linear.x == 0.0
