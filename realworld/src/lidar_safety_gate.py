import time
import logging
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, Float32, Int32

class LidarSafetyGate(Node):
    """
    LiDAR Safety Gate Node for the Ranger Mini V3.
    Directly consumes /rslidar_points and /minav/cmd_vel.
    Publishes safe commands to /cmd_vel.
    """
    def __init__(self, **kwargs):
        super().__init__('lidar_safety_gate', **kwargs)

        # Parameters
        self.declare_parameter('input_cmd_topic', '/minav/cmd_vel')
        self.declare_parameter('output_cmd_topic', '/cmd_vel')
        self.declare_parameter('lidar_topic', '/rslidar_points')
        self.declare_parameter('front_min_x', 0.15)
        self.declare_parameter('front_max_x', 0.8)
        self.declare_parameter('lateral_min_y', -0.4)
        self.declare_parameter('lateral_max_y', 0.4)
        self.declare_parameter('min_z', -0.2)
        self.declare_parameter('max_z', 0.5)
        self.declare_parameter('stop_distance', 0.5)
        self.declare_parameter('warning_distance', 0.8)
        self.declare_parameter('minimum_points', 10)
        self.declare_parameter('cloud_timeout_s', 1.0)

        self.input_cmd_topic = self.get_parameter('input_cmd_topic').value
        self.output_cmd_topic = self.get_parameter('output_cmd_topic').value
        self.lidar_topic = self.get_parameter('lidar_topic').value
        self.front_min_x = self.get_parameter('front_min_x').value
        self.front_max_x = self.get_parameter('front_max_x').value
        self.lateral_min_y = self.get_parameter('lateral_min_y').value
        self.lateral_max_y = self.get_parameter('lateral_max_y').value
        self.min_z = self.get_parameter('min_z').value
        self.max_z = self.get_parameter('max_z').value
        self.stop_distance = self.get_parameter('stop_distance').value
        self.warning_distance = self.get_parameter('warning_distance').value
        self.minimum_points = self.get_parameter('minimum_points').value
        self.cloud_timeout_s = self.get_parameter('cloud_timeout_s').value

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, self.output_cmd_topic, 10)
        
        # Diagnostic Publishers
        self.dist_pub = self.create_publisher(Float32, '/safety_gate/obstacle_distance', 10)
        self.blocked_pub = self.create_publisher(Bool, '/safety_gate/blocked', 10)
        self.pts_pub = self.create_publisher(Int32, '/safety_gate/point_count', 10)
        self.age_pub = self.create_publisher(Float32, '/safety_gate/cloud_age', 10)

        # Subscribers
        lidar_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=5
        )
        # Attempt to match RSLidar best-effort or reliable. If it's reliable, rclpy will adapt if we use default 10,
        # but PointCloud2 is often BEST_EFFORT. Let's subscribe with best_effort to be safe, or SYSTEM_DEFAULT.
        # RSLidar SDK often uses SYSTEM_DEFAULT.
        from rclpy.qos import qos_profile_sensor_data
        self.lidar_sub = self.create_subscription(
            PointCloud2, self.lidar_topic, self.lidar_callback, qos_profile_sensor_data
        )
        self.cmd_sub = self.create_subscription(
            Twist, self.input_cmd_topic, self.cmd_callback, 10
        )

        # State
        self.last_cloud_time = 0.0
        self.obstacle_distance = float('inf')
        self.point_count = 0
        self.is_blocked = True # Fail closed initially

        self.get_logger().info(f"LidarSafetyGate initialized. Listening to {self.lidar_topic} and {self.input_cmd_topic}")
        self.get_logger().info(f"Publishing safe commands to {self.output_cmd_topic}")

    def lidar_callback(self, msg: PointCloud2):
        """Process point cloud and determine safety state."""
        self.last_cloud_time = time.time()
        
        if msg.point_step != 16:
            self.get_logger().warning(f"Unexpected point_step: {msg.point_step}. Expected 16 (x,y,z,intensity).", throttle_duration_sec=5.0)
            return

        # Fast parsing using numpy
        # RSLidar point_step=16 is [x (float32), y (float32), z (float32), intensity (float32)]
        dtype_list = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32)]
        
        # In ROS 2, Python msg.data is array.array or bytes. Convert to bytes for numpy.
        try:
            cloud_bytes = bytes(msg.data)
            cloud_arr = np.frombuffer(cloud_bytes, dtype=dtype_list)
        except Exception as e:
            self.get_logger().error(f"Error parsing pointcloud: {e}", throttle_duration_sec=5.0)
            return

        # Handle NaNs
        valid_mask = ~np.isnan(cloud_arr['x']) & ~np.isnan(cloud_arr['y']) & ~np.isnan(cloud_arr['z'])
        cloud_arr = cloud_arr[valid_mask]

        # Apply geometric envelope filter
        # Front region box
        box_mask = (
            (cloud_arr['x'] >= self.front_min_x) &
            (cloud_arr['x'] <= self.front_max_x) &
            (cloud_arr['y'] >= self.lateral_min_y) &
            (cloud_arr['y'] <= self.lateral_max_y) &
            (cloud_arr['z'] >= self.min_z) &
            (cloud_arr['z'] <= self.max_z)
        )
        
        points_in_box = cloud_arr[box_mask]
        self.point_count = len(points_in_box)

        if self.point_count >= self.minimum_points:
            # Use 5th percentile of X distance to avoid single-point noise false positives
            self.obstacle_distance = float(np.percentile(points_in_box['x'], 5))
            self.is_blocked = (self.obstacle_distance < self.stop_distance)
        else:
            self.obstacle_distance = float('inf')
            self.is_blocked = False

        # Publish diagnostics
        self.pts_pub.publish(Int32(data=self.point_count))
        self.dist_pub.publish(Float32(data=self.obstacle_distance if self.obstacle_distance != float('inf') else 999.0))
        self.blocked_pub.publish(Bool(data=self.is_blocked))

    def cmd_callback(self, msg: Twist):
        """Intercept command, apply safety logic, and republish."""
        cloud_age = time.time() - self.last_cloud_time
        self.age_pub.publish(Float32(data=cloud_age))

        if cloud_age > self.cloud_timeout_s:
            self.get_logger().error(f"STALE CLOUD: age {cloud_age:.2f}s > {self.cloud_timeout_s}s. Failing closed.", throttle_duration_sec=2.0)
            # Override to block state if stale
            self.is_blocked = True
            # We don't have valid obstacle distance, so just force block forward motion.
        
        safe_msg = Twist()
        safe_msg.linear.x = msg.linear.x
        safe_msg.linear.y = msg.linear.y
        safe_msg.linear.z = msg.linear.z
        safe_msg.angular.x = msg.angular.x
        safe_msg.angular.y = msg.angular.y
        safe_msg.angular.z = msg.angular.z

        if self.is_blocked:
            if msg.linear.x > 0.0:
                self.get_logger().warn(
                    f"SAFETY BLOCK: Suppressing forward motion! Obstacle at {self.obstacle_distance:.2f}m (pts: {self.point_count})",
                    throttle_duration_sec=1.0
                )
                safe_msg.linear.x = 0.0
                # Note: We leave lateral (crab walk) and angular (turn) alone to allow MINav to spin away or strafe.
                # However, if lateral/angular commands would cause a collision, they might need blocking too.
                # Per requirements: "Do not change lateral/angular commands unless required to prevent forward collision."
                # We strictly suppress forward motion for the minimal safety gate.
        
        # Publish the command
        self.cmd_pub.publish(safe_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LidarSafetyGate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt")
    finally:
        # Publish 0 velocity on exit
        zero_msg = Twist()
        node.cmd_pub.publish(zero_msg)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
