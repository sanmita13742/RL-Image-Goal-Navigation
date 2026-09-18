import time
import logging
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, Float32, Int32

class LidarSafetyGate(Node):
    """
    Chassis-Aware Omnidirectional LiDAR Safety Gate Node for the Ranger Mini V3.
    Consumes /rslidar_points and /minav/cmd_vel.
    Projects the point cloud onto the commanded velocity vector.
    Calculates the exact swept footprint based on the robot's physical dimensions.
    Blocks motion if an obstacle lies in this dynamic swept corridor.
    Publishes safe commands to /cmd_vel.
    """
    def __init__(self, **kwargs):
        super().__init__('lidar_safety_gate', **kwargs)

        # Parameters
        self.declare_parameter('input_cmd_topic', '/minav/cmd_vel')
        self.declare_parameter('output_cmd_topic', '/cmd_vel')
        self.declare_parameter('lidar_topic', '/rslidar_points')
        
        # New chassis-aware parameters
        self.declare_parameter('robot_length', 0.751)
        self.declare_parameter('robot_width', 0.60)
        self.declare_parameter('safety_margin', 0.1)
        self.declare_parameter('lidar_offset_x', 0.0508)
        self.declare_parameter('lidar_offset_y', 0.0)
        self.declare_parameter('blindspot_radius', 0.15)
        
        self.declare_parameter('min_z', -0.2)
        self.declare_parameter('max_z', 0.5)
        self.declare_parameter('stop_distance', 0.5)
        self.declare_parameter('minimum_points', 10)
        self.declare_parameter('cloud_timeout_s', 1.0)
        self.declare_parameter('min_command_speed', 0.01)

        self.input_cmd_topic = self.get_parameter('input_cmd_topic').value
        self.output_cmd_topic = self.get_parameter('output_cmd_topic').value
        self.lidar_topic = self.get_parameter('lidar_topic').value
        
        self.robot_length = self.get_parameter('robot_length').value
        self.robot_width = self.get_parameter('robot_width').value
        self.safety_margin = self.get_parameter('safety_margin').value
        self.lidar_offset_x = self.get_parameter('lidar_offset_x').value
        self.lidar_offset_y = self.get_parameter('lidar_offset_y').value
        self.blindspot_radius = self.get_parameter('blindspot_radius').value
        
        self.min_z = self.get_parameter('min_z').value
        self.max_z = self.get_parameter('max_z').value
        self.stop_distance = self.get_parameter('stop_distance').value
        self.minimum_points = self.get_parameter('minimum_points').value
        self.cloud_timeout_s = self.get_parameter('cloud_timeout_s').value
        self.min_command_speed = self.get_parameter('min_command_speed').value

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, self.output_cmd_topic, 10)
        
        # Diagnostic Publishers
        self.dist_pub = self.create_publisher(Float32, '/safety_gate/obstacle_distance', 10)
        self.blocked_pub = self.create_publisher(Bool, '/safety_gate/blocked', 10)
        self.pts_pub = self.create_publisher(Int32, '/safety_gate/point_count', 10)
        self.age_pub = self.create_publisher(Float32, '/safety_gate/cloud_age', 10)

        # Subscribers
        from rclpy.qos import qos_profile_sensor_data
        self.lidar_sub = self.create_subscription(
            PointCloud2, self.lidar_topic, self.lidar_callback, qos_profile_sensor_data
        )
        self.cmd_sub = self.create_subscription(
            Twist, self.input_cmd_topic, self.cmd_callback, 10
        )

        # State
        self.last_cloud_time = 0.0
        self.cached_cloud = None

        self.get_logger().info(f"Chassis-Aware LidarSafetyGate initialized.")
        self.get_logger().info(f"Robot Dims: L={self.robot_length:.3f}m, W={self.robot_width:.3f}m. Lidar offset: x={self.lidar_offset_x:.3f}m")

    def lidar_callback(self, msg: PointCloud2):
        """Process point cloud, filter vertically and radially, and cache it."""
        self.last_cloud_time = time.time()
        
        if msg.point_step != 16:
            self.get_logger().warning(f"Unexpected point_step: {msg.point_step}. Expected 16.", throttle_duration_sec=5.0)
            return

        dtype_list = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32)]
        
        try:
            cloud_bytes = bytes(msg.data)
            cloud_arr = np.frombuffer(cloud_bytes, dtype=dtype_list)
        except Exception as e:
            self.get_logger().error(f"Error parsing pointcloud: {e}", throttle_duration_sec=5.0)
            return

        valid_mask = ~np.isnan(cloud_arr['x']) & ~np.isnan(cloud_arr['y']) & ~np.isnan(cloud_arr['z'])
        cloud_arr = cloud_arr[valid_mask]

        # Apply Z filter and blindspot filter
        # The raw lidar points are in rslidar frame. Wait to apply base_link offsets until cmd_callback.
        r_sq = cloud_arr['x']**2 + cloud_arr['y']**2
        box_mask = (
            (cloud_arr['z'] >= self.min_z) &
            (cloud_arr['z'] <= self.max_z) &
            (r_sq >= self.blindspot_radius**2)
        )
        
        self.cached_cloud = cloud_arr[box_mask]

    def cmd_callback(self, msg: Twist):
        """Intercept command, compute dynamic chassis footprint in command direction, and check for obstacles."""
        cloud_age = time.time() - self.last_cloud_time
        self.age_pub.publish(Float32(data=cloud_age))

        safe_msg = Twist()
        safe_msg.angular.x = msg.angular.x
        safe_msg.angular.y = msg.angular.y
        safe_msg.angular.z = msg.angular.z

        if cloud_age > self.cloud_timeout_s:
            self.get_logger().error(f"STALE CLOUD: age {cloud_age:.2f}s > {self.cloud_timeout_s}s. Failing closed.", throttle_duration_sec=2.0)
            safe_msg.linear.x = 0.0
            safe_msg.linear.y = 0.0
            safe_msg.linear.z = 0.0
            self.cmd_pub.publish(safe_msg)
            self.blocked_pub.publish(Bool(data=True))
            return

        vx = msg.linear.x
        vy = msg.linear.y
        speed = math.hypot(vx, vy)

        if speed < self.min_command_speed or self.cached_cloud is None:
            safe_msg.linear.x = vx
            safe_msg.linear.y = vy
            safe_msg.linear.z = msg.linear.z
            self.cmd_pub.publish(safe_msg)
            self.blocked_pub.publish(Bool(data=False))
            return

        dir_x = vx / speed
        dir_y = vy / speed

        # 1. Transform points from rslidar to base_link
        # A point px, py in rslidar is at (px + offset_x, py + offset_y) in base_link.
        bx = self.cached_cloud['x'] + self.lidar_offset_x
        by = self.cached_cloud['y'] + self.lidar_offset_y
        
        # 2. Project points onto the movement direction and its perpendicular
        parallel = bx * dir_x + by * dir_y
        perpendicular = -bx * dir_y + by * dir_x
        
        # 3. Calculate the swept footprint bounds based on the chassis size
        L_half = self.robot_length / 2.0
        W_half = self.robot_width / 2.0
        
        # The perpendicular bounds of the swept footprint
        max_perp = L_half * abs(dir_y) + W_half * abs(dir_x)
        # The parallel offset representing the "front" edge of the robot in the direction of motion
        max_para = L_half * abs(dir_x) + W_half * abs(dir_y)
        
        # A point is an obstacle if it lies within the swept corridor width (plus margin) 
        # AND it is in front of the robot (parallel > max_para)
        # AND it is within the stop distance (parallel <= max_para + stop_distance)
        corridor_mask = (
            (parallel > max_para) & 
            (parallel <= max_para + self.stop_distance) & 
            (np.abs(perpendicular) <= max_perp + self.safety_margin)
        )
        
        relevant_points = self.cached_cloud[corridor_mask]
        point_count = len(relevant_points)
        
        if point_count >= self.minimum_points:
            relevant_parallel = parallel[corridor_mask]
            # True distance to the obstacle from the chassis edge
            distance_from_chassis = float(np.percentile(relevant_parallel, 5)) - max_para
            
            self.get_logger().warn(
                f"SAFETY BLOCK: Suppressing translation! Dir=[{dir_x:.2f}, {dir_y:.2f}] "
                f"Obstacle at {distance_from_chassis:.2f}m from chassis (pts: {point_count})",
                throttle_duration_sec=1.0
            )
            
            safe_msg.linear.x = 0.0
            safe_msg.linear.y = 0.0
            safe_msg.linear.z = 0.0
            is_blocked = True
            obstacle_distance = distance_from_chassis
        else:
            obstacle_distance = float('inf')
            safe_msg.linear.x = vx
            safe_msg.linear.y = vy
            safe_msg.linear.z = msg.linear.z
            is_blocked = False

        self.cmd_pub.publish(safe_msg)

        self.pts_pub.publish(Int32(data=point_count))
        self.dist_pub.publish(Float32(data=obstacle_distance if obstacle_distance != float('inf') else 999.0))
        self.blocked_pub.publish(Bool(data=is_blocked))

def main(args=None):
    rclpy.init(args=args)
    node = LidarSafetyGate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt")
    finally:
        zero_msg = Twist()
        node.cmd_pub.publish(zero_msg)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
