import sys
from pathlib import Path
from rosbags.rosbag2 import Reader
from collections import defaultdict
import json

def analyze_bag(bag_path):
    print(f"Analyzing ROS 2 bag at: {bag_path}")
    bag_path = Path(bag_path)
    
    if not bag_path.exists():
        print(f"Error: {bag_path} does not exist.")
        return
    
    topics = defaultdict(int)
    topic_types = {}
    total_messages = 0
    start_time = float('inf')
    end_time = 0.0
    
    try:
        with Reader(bag_path) as reader:
            for connection in reader.connections:
                topic_types[connection.topic] = connection.msgtype
            
            for connection, timestamp, rawdata in reader.messages():
                topics[connection.topic] += 1
                total_messages += 1
                
                t_sec = timestamp / 1e9
                if t_sec < start_time:
                    start_time = t_sec
                if t_sec > end_time:
                    end_time = t_sec
                    
        duration = end_time - start_time if end_time > start_time else 0
        
        print("\n=== ROS 2 Bag Report ===")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Total Messages: {total_messages}")
        
        print("\n=== Topics ===")
        for topic, count in sorted(topics.items()):
            msg_type = topic_types.get(topic, "Unknown")
            freq = count / duration if duration > 0 else 0
            print(f"{topic:<40} {msg_type:<35} {count:>8} msgs  (~{freq:>6.2f} Hz)")
            
        print("\n=== Sensor Detection ===")
        sensors = []
        for topic, msg_type in topic_types.items():
            if 'sensor_msgs/msg/Image' in msg_type:
                sensors.append(f"Camera (RGB/Depth) on topic: {topic}")
            elif 'sensor_msgs/msg/PointCloud2' in msg_type:
                sensors.append(f"LiDAR/PointCloud on topic: {topic}")
            elif 'nav_msgs/msg/Odometry' in msg_type:
                sensors.append(f"Odometry on topic: {topic}")
            elif 'sensor_msgs/msg/Imu' in msg_type:
                sensors.append(f"IMU on topic: {topic}")
            elif 'sensor_msgs/msg/NavSatFix' in msg_type:
                sensors.append(f"GPS on topic: {topic}")
            elif 'tf2_msgs/msg/TFMessage' in msg_type:
                sensors.append(f"TF (Transforms) on topic: {topic}")
            elif 'geometry_msgs/msg/Twist' in msg_type:
                sensors.append(f"Twist (Velocity/Cmd) on topic: {topic}")
            elif 'sensor_msgs/msg/JointState' in msg_type:
                sensors.append(f"Joint States on topic: {topic}")
                
        for s in sensors:
            print(f" - {s}")
            
    except Exception as e:
        print(f"Failed to read bag: {e}")

if __name__ == "__main__":
    analyze_bag(r"C:\Users\sanmi\Desktop\projects\RL\simulation\ranger_mini_v3\test_recording8")
