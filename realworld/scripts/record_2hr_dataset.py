import os
import subprocess
import time
import signal
import sys
from datetime import datetime

def main():
    print("=" * 60)
    print("MINav 2-Hour Dataset Bag Recorder")
    print("=" * 60)

    topics = [
        "/camera/color/image_raw_10hz",
        "/cmd_vel"
    ]
    
    print("Topics to record:")
    print(" - /camera/color/image_raw_10hz (RGB Image, 1280x720 native, Throttled to 10 Hz)")
    print(" - /cmd_vel (FINAL Executed Actions)")
    print(" - Note: /rslidar_points is intentionally EXCLUDED.")
    
    print("\nExpected Rates & Storage (Estimates):")
    print(" - RGB (10 Hz): ~27 MB/s -> ~100 GB/hr -> ~200 GB for 2 hours")
    print(" - /cmd_vel (~10 Hz): Negligible")
    print(" - Total Expected Size: ~200 GB (Ensure sufficient disk space!)")

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

    # Start the 10 Hz throttle node
    throttle_cmd = [
        "ros2", "run", "topic_tools", "throttle", "messages",
        "/camera/color/image_raw", "10.0", "/camera/color/image_raw_10hz"
    ]
    print(f"\nStarting throttle: {' '.join(throttle_cmd)}")
    
    is_windows = sys.platform == "win32"
    
    if is_windows:
        throttle_process = subprocess.Popen(throttle_cmd, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        throttle_process = subprocess.Popen(throttle_cmd, preexec_fn=os.setsid)

    # Give throttle node a second to spin up
    time.sleep(1)

    # Build recording command
    cmd = ["ros2", "bag", "record", "-o", bag_name] + topics
    
    print(f"\nRunning recording: {' '.join(cmd)}")
    
    # On Windows, CREATE_NEW_PROCESS_GROUP allows sending CTRL_BREAK_EVENT
    # On Linux, preexec_fn=os.setsid allows sending SIGINT to the process group
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
            
            # Kill the throttle process
            if throttle_process.poll() is None:
                if is_windows:
                    throttle_process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(os.getpgid(throttle_process.pid), signal.SIGINT)
                try:
                    throttle_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    throttle_process.terminate()
                
        print(f"\nBag saved to: {os.path.abspath(bag_name)}")
        print("Done.")

if __name__ == '__main__':
    main()
