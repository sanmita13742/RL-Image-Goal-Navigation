import sys
import time
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import mujoco
import mujoco.viewer

sys.path.insert(0, str(Path(__file__).parent.parent))

from robot import RangerMiniV3Robot
from robot_base import DriveCommand
from exploration_policies import PrimitiveExplorationPolicy

def save_depth(arr: np.ndarray, path: str) -> None:
    d_min, d_max = arr.min(), arr.max()
    norm = ((arr - d_min) / (d_max - d_min) * 255).astype(np.uint8) \
           if d_max > d_min else np.zeros_like(arr, dtype=np.uint8)
    Image.fromarray(norm, mode="L").save(path)

def main():
    parser = argparse.ArgumentParser(description="Test/Inspect the new MINav navigation map.")
    parser.add_argument("--viewer", action="store_true", help="Launch the MuJoCo viewer")
    parser.add_argument("--steps", type=int, default=1000, help="Number of exploration steps to run")
    parser.add_argument("--save-images", action="store_true", help="Save RGB and Depth images to test_output/")
    args = parser.parse_args()

    map_path = Path(__file__).parent.parent / "maps" / "new_navigation_map" / "scene.xml"
    if not map_path.exists():
        print(f"Error: Map not found at {map_path}")
        sys.exit(1)

    print(f"Loading map: {map_path}")
    robot = RangerMiniV3Robot()
    robot.load(map_path)
    
    # Init pose
    robot.data.qpos[0] = -7.0
    robot.data.qpos[1] = -7.0
    robot.data.qpos[3] = 1.0  # qw
    robot.data.qpos[4] = 0.0
    robot.data.qpos[5] = 0.0
    robot.data.qpos[6] = 0.0  # qz
    mujoco.mj_forward(robot.model, robot.data)

    out_dir = Path(__file__).parent.parent / "test_output"
    if args.save_images:
        out_dir.mkdir(exist_ok=True)
        (out_dir / "rgb").mkdir(exist_ok=True)
        (out_dir / "depth").mkdir(exist_ok=True)
        print(f"Images will be saved to: {out_dir}")

    renderer_rgb = mujoco.Renderer(robot.model, height=240, width=320) if args.save_images else None
    renderer_depth = mujoco.Renderer(robot.model, height=60, width=640)
    renderer_depth.enable_depth_rendering()

    control_freq = 10.0
    sim_dt = robot.model.opt.timestep
    sim_steps_per_control = int(1.0 / (control_freq * sim_dt))
    
    policy = PrimitiveExplorationPolicy(control_freq, beta=0)

    viewer = mujoco.viewer.launch_passive(robot.model, robot.data) if args.viewer else None

    print(f"Running exploration for {args.steps} steps...")
    
    for step in range(args.steps):
        if args.save_images:
            renderer_rgb.update_scene(robot.data, camera="front_cam")
            rgb_img = renderer_rgb.render()
            Image.fromarray(rgb_img).save(out_dir / "rgb" / f"{step:04d}.png")
            
        renderer_depth.update_scene(robot.data, camera="lidar_cam")
        depth_img = renderer_depth.render()
        
        if args.save_images:
            save_depth(depth_img, str(out_dir / "depth" / f"{step:04d}.png"))
            
        pos_x, pos_y, _ = robot.data.qpos[0:3]
        qw, qx, qy, qz  = robot.data.qpos[3:7]
        yaw = 2.0 * np.arctan2(qz, qw)
        
        cmd, prim = policy.get_action(depth_img, pos_x, pos_y, yaw)
        
        for _ in range(sim_steps_per_control):
            robot.apply_command(cmd)
            robot.step()
            
        if viewer:
            if not viewer.is_running():
                print("Viewer closed.")
                break
            viewer.sync()
            
        if step % 100 == 0:
            print(f"Step {step:04d} | Pos: ({pos_x:+.2f}, {pos_y:+.2f}) | Action: {prim.name}")

    if viewer:
        viewer.close()

    print("Test completed.")

if __name__ == "__main__":
    main()
