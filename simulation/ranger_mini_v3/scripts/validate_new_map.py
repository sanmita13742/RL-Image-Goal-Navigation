import sys
import numpy as np
from pathlib import Path
import mujoco
import mujoco.viewer
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from robot import RangerMiniV3Robot
from robot_base import DriveCommand
from exploration_policies import PrimitiveExplorationPolicy


def validate_environment(map_path: Path):
    print(f"Loading map: {map_path}")
    
    # 1. Load Model & Robot
    try:
        robot = RangerMiniV3Robot()
        robot.load(map_path)
        print("[OK] MuJoCo model and robot loaded successfully.")
    except Exception as e:
        print(f"[FAIL] Failed to load model: {e}")
        return False
        
    # Initialization pose
    robot.data.qpos[0] = -7.0
    robot.data.qpos[1] = -7.0
    robot.data.qpos[3] = 1.0  # qw
    robot.data.qpos[4] = 0.0
    robot.data.qpos[5] = 0.0
    robot.data.qpos[6] = 0.0  # qz
    mujoco.mj_forward(robot.model, robot.data)
    
    # Renderers
    renderer_rgb = mujoco.Renderer(robot.model, height=240, width=320)
    renderer_depth = mujoco.Renderer(robot.model, height=60, width=640)
    renderer_depth.enable_depth_rendering()
    
    # Test RGB / Depth rendering sizes
    renderer_rgb.update_scene(robot.data, camera="front_cam")
    rgb_img = renderer_rgb.render()
    if rgb_img.shape == (240, 320, 3):
        print("[OK] RGB rendering works and has expected dimensions.")
    else:
        print(f"[FAIL] RGB rendering failed or bad dimensions: {rgb_img.shape}")
        return False
        
    renderer_depth.update_scene(robot.data, camera="lidar_cam")
    depth_img = renderer_depth.render()
    if depth_img.shape == (60, 640):
        print("[OK] Depth rendering works and has expected dimensions.")
    else:
        print(f"[FAIL] Depth rendering failed or bad dimensions: {depth_img.shape}")
        return False

    # Simulate 1000 steps with exploration policy
    control_freq = 10.0
    sim_dt = robot.model.opt.timestep
    sim_steps_per_control = int(1.0 / (control_freq * sim_dt))
    
    policy = PrimitiveExplorationPolicy(control_freq, beta=0)
    
    print("Running 1000 steps of simulation...")
    
    positions = []
    yaws = []
    
    for step in range(1000):
        renderer_depth.update_scene(robot.data, camera="lidar_cam")
        depth_img = renderer_depth.render()
        
        pos_x, pos_y, _ = robot.data.qpos[0:3]
        qw, qx, qy, qz  = robot.data.qpos[3:7]
        # simplified yaw calculation for metric tracking
        yaw = 2.0 * np.arctan2(qz, qw)
        
        positions.append((pos_x, pos_y))
        yaws.append(yaw)
        
        cmd, _ = policy.get_action(depth_img, pos_x, pos_y, yaw)
        
        for _ in range(sim_steps_per_control):
            robot.apply_command(cmd)
            robot.step()
            
        if np.isnan(robot.data.qpos).any() or np.isinf(robot.data.qpos).any():
            print(f"[FAIL] NaN or Inf encountered in physics state at step {step}")
            return False
            
    pos_array = np.array(positions)
    yaw_array = np.array(yaws)
    
    pos_spread = np.max(pos_array, axis=0) - np.min(pos_array, axis=0)
    yaw_spread = np.max(yaw_array) - np.min(yaw_array)
    
    print(f"Movement spread: X={pos_spread[0]:.2f}m, Y={pos_spread[1]:.2f}m, Yaw={yaw_spread:.2f}rad")
    
    if pos_spread[0] > 0.1 or pos_spread[1] > 0.1:
        print("[OK] Robot can move forward and translate.")
    else:
        print("[FAIL] Robot translation is minimal, might be stuck.")
        
    if yaw_spread > 0.1:
        print("[OK] Robot can rotate.")
    else:
        print("[FAIL] Robot rotation is minimal.")
        
    print("[OK] Validation passed!")
    return True

if __name__ == "__main__":
    map_path = Path(__file__).parent.parent / "maps" / "new_navigation_map" / "scene.xml"
    success = validate_environment(map_path)
    if not success:
        sys.exit(1)
