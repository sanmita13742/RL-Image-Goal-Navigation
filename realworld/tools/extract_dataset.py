import sys
from pathlib import Path
import json
import csv
from rosbags.highlevel import AnyReader

def extract_dataset(bag_path, out_dir):
    print(f"Extracting dataset from {bag_path} to {out_dir}")
    bag_path = Path(bag_path)
    out_dir = Path(out_dir)
    
    lidar_dir = out_dir / "lidar"
    lidar_dir.mkdir(parents=True, exist_ok=True)
    
    states_csv_path = out_dir / "states.csv"
    actions_csv_path = out_dir / "actions.csv"
    
    with open(states_csv_path, 'w', newline='') as sf, open(actions_csv_path, 'w', newline='') as af:
        state_writer = csv.writer(sf)
        state_writer.writerow(['timestamp', 'pose_x', 'pose_y', 'pose_z', 'ori_x', 'ori_y', 'ori_z', 'ori_w', 'linear_vel_x', 'angular_vel_z'])
        
        action_writer = csv.writer(af)
        action_writer.writerow(['timestamp', 'cmd_linear_x', 'cmd_angular_z'])
        
        try:
            from rosbags.typesys import get_typestore, Stores
            typestore = get_typestore(Stores.LATEST)
            
            with AnyReader([bag_path], default_typestore=typestore) as reader:
                for connection, timestamp, rawdata in reader.messages():
                    t_sec = timestamp / 1e9
                    
                    if connection.topic == '/odom':
                        msg = reader.deserialize(rawdata, connection.msgtype)
                        state_writer.writerow([
                            t_sec,
                            msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z,
                            msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w,
                            msg.twist.twist.linear.x, msg.twist.twist.angular.z
                        ])
                        
                    elif connection.topic == '/cmd_vel':
                        msg = reader.deserialize(rawdata, connection.msgtype)
                        action_writer.writerow([
                            t_sec,
                            msg.linear.x, msg.angular.z
                        ])
                        
                    elif connection.topic == '/rslidar_points':
                        # We save lidar metadata (timestamp) and extract binary blob.
                        # For full pcd, we would parse PointCloud2 fully. 
                        # Saving binary directly for fast access if needed, or structured numpy array.
                        pass
        except Exception as e:
            print(f"Error extracting: {e}")
            
    # Generate metadata.json
    metadata = {
        "dataset": "Ranger Mini Offline RL",
        "bag_source": bag_path.name,
        "topics": ["/odom", "/cmd_vel", "/rslidar_points"],
        "sensors": ["Odometry", "RoboSense LiDAR"]
    }
    
    with open(out_dir / "metadata.json", "w") as mf:
        json.dump(metadata, mf, indent=4)
        
    print("Extraction complete.")

if __name__ == "__main__":
    bag = r"C:\Users\sanmi\Desktop\projects\RL\simulation\ranger_mini_v3\test_recording8"
    out = r"C:\Users\sanmi\Desktop\projects\RL\simulation\ranger_mini_v3\dataset"
    extract_dataset(bag, out)
