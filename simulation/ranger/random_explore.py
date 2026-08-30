"""
ranger_mujoco/random_explore.py
============================================================
Runs a random exploration policy in the test environment and 
logs the dataset (RGB, LiDAR depth, Pose, Actions, Timestamp).

Outputs are saved in the `dataset/` directory:
- dataset/rgb/       : Front RGB camera frames (.png)
- dataset/depth/     : LiDAR depth camera frames (.png)
- dataset/log.csv    : Timestamps, Actions, and Robot Pose

Run:  python random_explore.py
"""

import sys
import os
import csv
import time
import math
import random
import numpy as np
from pathlib import Path

# Fix Windows terminal encoding just in case
sys.stdout.reconfigure(encoding="utf-8")

try:
    from PIL import Image
except ImportError:
    print("FATAL: Pillow is required to save images. Please run: pip install Pillow")
    sys.exit(1)

import mujoco
import mujoco.viewer

# Allow importing from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from robot import RangerRobot
from robot_base import DriveCommand
from test_env import build_world_xml
from exploration_policies import LinearExploration, WhiteNoiseExploration

def euler_from_quaternion(w, x, y, z):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z

def save_depth(arr: np.ndarray, path: str) -> None:
    """Normalize and save depth as grayscale PNG."""
    d_min, d_max = arr.min(), arr.max()
    if d_max > d_min:
        norm = ((arr - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(arr, dtype=np.uint8)
    Image.fromarray(norm, mode="L").save(path)

def main():
    # ── Setup Logging Directory ────────────────────────────────────────────────
    out_dir = Path(__file__).parent / "dataset"
    rgb_dir = out_dir / "rgb"
    depth_dir = out_dir / "depth"
    
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    
    csv_file = out_dir / "log.csv"
    file_exists = csv_file.exists()
    
    # ── Configuration ────────────────────────────────────────────────────────
    CONTROL_FREQ = 10.0            # Hz (Log and change action 10 times per second)
    MAX_EPISODE_STEPS = 1000       # Collect 1000 steps (100 seconds) for this test
    ACTION_HOLD_STEPS = 10         # Hold action for N control steps (e.g., 1 second)
    SHOW_VIEWER = True             # Set to True to watch the robot live in a 3D window
    
    # ── Setup MuJoCo World ───────────────────────────────────────────────────
    world_xml = Path(__file__).parent / "_random_world.xml"
    world_xml.write_text(build_world_xml(), encoding="utf-8")

    robot = RangerRobot()
    robot.load(world_xml)
    
    # Clean up temp file
    try:
        world_xml.unlink()
    except Exception:
        pass

    # Create renderer for cameras
    renderer_rgb = mujoco.Renderer(robot.model, height=240, width=320)
    renderer_depth = mujoco.Renderer(robot.model, height=240, width=320)
    renderer_depth.enable_depth_rendering()

    print("============================================================")
    print("  Ranger Mini  --  Random Exploration Data Collection")
    print("============================================================")
    print(f"  Logging to       : {out_dir}")
    print(f"  Control Freq     : {CONTROL_FREQ} Hz")
    print(f"  Max Steps        : {MAX_EPISODE_STEPS}")
    print("============================================================")

    # Sim timestep
    sim_dt = robot.model.opt.timestep
    sim_steps_per_control = int(1.0 / (CONTROL_FREQ * sim_dt))

    with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "frame", "timestamp", 
                "linear_vel_cmd", "angular_vel_cmd",
                "pos_x", "pos_y", "yaw",
                "rgb_path", "depth_path"
            ])

        frame_idx = 0
        start_time = time.time()
        
        # Initialize exploration policy
        # Options: LinearExploration(CONTROL_FREQ, hold_time=1.0) or WhiteNoiseExploration(CONTROL_FREQ)
        policy = WhiteNoiseExploration(CONTROL_FREQ, v_range=(-0.1, 1.5), w_range=(-1.0, 1.0))
        
        # Optional live viewer
        viewer = None
        if SHOW_VIEWER:
            viewer = mujoco.viewer.launch_passive(robot.model, robot.data)
            
        try:
            while frame_idx < MAX_EPISODE_STEPS:
                # 1. Render Cameras first to get LiDAR depth for collision avoidance
                # RGB
                renderer_rgb.update_scene(robot.data, camera="front_cam")
                rgb_img = renderer_rgb.render()
                
                # Depth
                renderer_depth.update_scene(robot.data, camera="lidar_cam")
                depth_img = renderer_depth.render()
    
                # 2. Extract Pose
                pos_x, pos_y, pos_z = robot.data.qpos[0:3]
                qw, qx, qy, qz = robot.data.qpos[3:7]
                _, _, yaw = euler_from_quaternion(qw, qx, qy, qz)
    
                # 3. Action Selection (using our chosen policy)
                min_depth = np.min(depth_img)
                cmd = policy.get_action(min_depth)
                
                # Extract velocities for logging
                current_v_linear = cmd.v_linear
                current_v_angular = cmd.v_angular
    
                # 4. Save and Log Data
                rgb_filename = f"frame_{frame_idx:06d}.png"
                Image.fromarray(rgb_img).save(rgb_dir / rgb_filename)
                
                depth_filename = f"frame_{frame_idx:06d}.png"
                save_depth(depth_img, str(depth_dir / depth_filename))
    
                sim_time = robot.data.time
    
                writer.writerow([
                    frame_idx,
                    f"{sim_time:.3f}",
                    f"{current_v_linear:.3f}",
                    f"{current_v_angular:.3f}",
                    f"{pos_x:.4f}",
                    f"{pos_y:.4f}",
                    f"{yaw:.4f}",
                    f"rgb/{rgb_filename}",
                    f"depth/{depth_filename}"
                ])
                f.flush()
    
                # CLI Feedback
                if frame_idx % 10 == 0:
                    print(f"  [Log] Frame {frame_idx:04d}/{MAX_EPISODE_STEPS} | "
                          f"Action: (v={current_v_linear:+.2f}, w={current_v_angular:+.2f}) | "
                          f"Pose: (x={pos_x:+.2f}, y={pos_y:+.2f}, yaw={yaw:+.2f}) | MinDepth: {min_depth:.2f}m")
    
                # 5. Physics steps
                for _ in range(sim_steps_per_control):
                    robot.apply_command(cmd)
                    robot.step()
                    
                if viewer is not None:
                    if not viewer.is_running():
                        print("Viewer closed by user. Stopping data collection early.")
                        break
                    viewer.sync()
                    
        except KeyboardInterrupt:
            print("Data collection interrupted by user.")
            
        finally:
            if viewer is not None:
                viewer.close()

    elapsed = time.time() - start_time
    print("============================================================")
    print(f"  [PASS] Data collection complete!")
    print(f"         Total frames: {MAX_EPISODE_STEPS}")
    print(f"         Time elapsed: {elapsed:.2f} s")
    print(f"         Saved to    : {out_dir}")
    print("============================================================")

if __name__ == "__main__":
    main()
