import os
import subprocess
import time
import signal
import sys
import yaml
from datetime import datetime

def main():
    print("=" * 60)
    print("MINav 2-Hour Dataset Bag Recorder (QoS Fix)")
    print("=" * 60)

    # 1. Create a QoS Override file. 
    # ROS 2 Foxy rosbag2 defaults to "Reliable" QoS. Camera topics use "Best Effort".
    # Without this override, rosbag2 silently ignores the camera topic!
    qos_file = "camera_qos_override.yaml"
    qos_config = {
        "/camera/color/image_raw": {
            "reliability": "best_effort",
            "durability": "volatile",
            "history": "keep_last",
            "depth": 10
        }
    }
    
    with open(qos_file, "w") as f:
        yaml.dump(qos_config, f)
        
    print(f"Created QoS override file: {qos_file} (Allows recording Best-Effort camera feeds)")

    topics = [
        "/camera/color/image_raw",
        "/cmd_vel"
    ]
    
    print("\nTopics to record:")
    print(" - /camera/color/image_raw (RGB Image, 1280x720 native)")
    print(" - /cmd_vel (FINAL Executed Actions)")
    print(" - Note: /rslidar_points is intentionally EXCLUDED.")
    
    print("\nExpected Rates & Storage (Estimates):")
    print(" - RGB (30 Hz): ~83 MB/s -> ~300 GB/hr -> ~600 GB for 2 hours")
    print(" - /cmd_vel (~10 Hz): Negligible")
    print(" - Total Expected Size: ~600 GB (Ensure sufficient disk space!)")

    # Generate timestamped bag name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_name = f"MINav_Dataset_{timestamp}"
    
    print(f"\nOutput Directory: {bag_name}")
    print("Duration: 2 hours (7200 seconds)")
    print("\nStarting recording in 5 seconds... Press Ctrl+C to stop early.")
    
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("Aborted before starting.")
        return

    # Build recording command with QoS override
    cmd = [
        "ros2", "bag", "record", 
        "-o", bag_name,
        "--qos-profile-overrides-path", qos_file
    ] + topics
    
    print(f"\nRunning recording: {' '.join(cmd)}")
    
    is_windows = sys.platform == "win32"
    
    if is_windows:
        process = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        process = subprocess.Popen(cmd, preexec_fn=os.setsid)
    
    duration = 2 * 60 * 60  # 2 hours in seconds
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            remaining = duration - elapsed
            sys.stdout.write(f"\rRecording... {elapsed/60:.1f} min elapsed, {remaining/60:.1f} min remaining. (Ctrl+C to stop)")
            sys.stdout.flush()
            time.sleep(1)
            
            # Check if process died
            if process.poll() is not None:
                print("\n\nros2 bag process terminated unexpectedly.")
                break
                
        if time.time() - start_time >= duration:
            print("\n\n2 hours reached! Stopping recording automatically.")
            
    except KeyboardInterrupt:
        print("\n\nManual stop requested. Terminating recording gracefully...")
        
    finally:
        # Stop the process gracefully to ensure metadata.yaml is saved
        if process.poll() is None:
            if is_windows:
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGINT)
            
            # Wait for graceful shutdown
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("Process didn't terminate gracefully, forcing kill...")
                process.terminate()
                
        print(f"\nBag saved to: {os.path.abspath(bag_name)}")
        print("Done.")

if __name__ == '__main__':
    main()
