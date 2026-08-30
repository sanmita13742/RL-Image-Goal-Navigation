import sys
import time
import torch
import numpy as np
from PIL import Image
import cv2
import pandas as pd
from collections import deque
from pathlib import Path
import mujoco
import mujoco.viewer

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from test_env import build_world_xml
from robot import RangerMiniV3Robot
from robot_base import DriveCommand
from src.rl.td3_bc import TD3_BC
from src.data.dinov3_encoder import FrozenDINOv3

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Loading Encoder...")
    encoder = FrozenDINOv3(device=device)
    
    print("Loading TD3_BC...")
    agent = TD3_BC(state_dim=1536, action_dim=3, goal_dim=384, device=device)
    ckpt_path = ROOT / "data" / "models" / "td3_bc_debug" / "checkpoints" / "latest.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found at {ckpt_path}")
        return
    agent.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    agent.actor.eval()
    
    print("Loading Goal...")
    # Load first valid goal from dataset for evaluation
    phi_cache = np.load(ROOT / "data" / "processed" / "dinov3" / "phi_cache.npy")
    valid_goals = np.load(ROOT / "data" / "processed" / "dinov3" / "valid_goals.npy")
    index_df = pd.read_parquet(ROOT / "data" / "processed" / "dinov3" / "embedding_index.parquet")
    valid_indices = np.where(valid_goals)[0]
    if len(valid_indices) == 0:
        print("No valid goals found.")
        return
    goal_idx = valid_indices[-1] # take a valid goal
    goal_phi = torch.tensor(phi_cache[goal_idx], dtype=torch.float32, device=device).unsqueeze(0)
    print(f"Using goal index: {goal_idx}")
    
    # Extract local path for the goal image
    goal_kaggle_path = index_df.iloc[goal_idx]['rgb_abs_path']
    # Format usually looks like: /kaggle/input/datasets/.../20260811_234812/segment_.../rgb/...
    parts = Path(goal_kaggle_path).parts
    session_idx = -1
    for i, p in enumerate(parts):
        if p == "20260811_234812":
            session_idx = i
            break
    if session_idx != -1:
        local_rel_path = Path(*parts[session_idx:])
        goal_img_path = ROOT / "dataset" / local_rel_path
    else:
        # Fallback if the path structure is different
        local_rel_path = Path(*parts[-4:])
        goal_img_path = ROOT / "dataset" / local_rel_path
        
    goal_bgr = cv2.imread(str(goal_img_path))
    if goal_bgr is not None:
        goal_bgr = cv2.resize(goal_bgr, (784, 448))
    else:
        print(f"Warning: Could not load goal image from {goal_img_path}")
        goal_bgr = np.zeros((448, 784, 3), dtype=np.uint8)
    
    print("Initializing Simulation...")
    world_xml_str = build_world_xml()
    # Inject higher framebuffer resolution for 784x448 rendering
    world_xml_str = world_xml_str.replace(
        '<mujoco model="ranger_test_env">',
        '<mujoco model="ranger_test_env">\n  <visual>\n    <global offwidth="784" offheight="448"/>\n  </visual>'
    )
    world_xml = ROOT / "_eval_world.xml"
    world_xml.write_text(world_xml_str, encoding="utf-8")
    
    robot = RangerMiniV3Robot()
    robot.load(world_xml)
    
    # Set initial pose
    robot.data.qpos[0] = -4.5
    robot.data.qpos[1] = 0.0
    robot.data.qpos[3] = 0.7071068
    robot.data.qpos[4] = 0.0
    robot.data.qpos[5] = 0.0
    robot.data.qpos[6] = 0.7071068
    mujoco.mj_forward(robot.model, robot.data)
    
    renderer_rgb = mujoco.Renderer(robot.model, height=448, width=784)
    
    try:
        world_xml.unlink()
    except Exception:
        pass
        
    history = deque(maxlen=4)
    
    # Fill history initially
    renderer_rgb.update_scene(robot.data, camera="front_cam")
    rgb_img = renderer_rgb.render()
    img = Image.fromarray(rgb_img).convert("RGB")
    tensor = encoder.transform(img).unsqueeze(0).to(device)
    _, phi, _ = encoder._forward_batch(tensor)
    phi = phi.to(device)
    
    for _ in range(4):
        history.append(phi)
        
    CONTROL_FREQ = 10.0
    sim_dt = robot.model.opt.timestep
    sim_steps_per_control = int(1.0 / (CONTROL_FREQ * sim_dt))
    
    print("Starting evaluation loop...")
    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        viewer.cam.azimuth = 150
        viewer.cam.elevation = -25
        viewer.cam.distance = 8.0
        viewer.cam.lookat[:] = [0, 0, 0.5]
        
        for step in range(1000):
            if not viewer.is_running():
                break
                
            step_start = time.time()
            
            # 1. Capture Image
            renderer_rgb.update_scene(robot.data, camera="front_cam")
            rgb_img = renderer_rgb.render()
            img = Image.fromarray(rgb_img).convert("RGB")
            
            # 2. Encode
            tensor = encoder.transform(img).unsqueeze(0).to(device)
            _, phi, _ = encoder._forward_batch(tensor)
            history.append(phi.to(device))
            
            # 3. Predict Action
            state = torch.cat(list(history), dim=-1) # [1, 1536]
            with torch.no_grad():
                action_norm = agent.actor(state, goal_phi).squeeze(0) # [3]
                action = agent.normalizer.denormalize(action_norm).cpu().numpy()
                
            cmd = DriveCommand(
                v_linear=float(action[0]),
                v_lateral=float(action[1]),
                v_angular=float(action[2])
            )
            
            # 4. Step Environment
            for _ in range(sim_steps_per_control):
                robot.apply_command(cmd)
                robot.step()
                
            viewer.sync()
            
            # Show side-by-side OpenCV visualization
            curr_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            combined = cv2.hconcat([curr_bgr, goal_bgr])
            cv2.imshow("TD3_BC Evaluation: Current State (Left) vs Goal State (Right)", combined)
            cv2.waitKey(1)
            
            elapsed = time.time() - step_start
            sleep_for = (1.0 / CONTROL_FREQ) - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
                
            if step % 100 == 0:
                pos_x, pos_y = robot.data.qpos[0:2]
                print(f"Step {step}/1000 | pos: ({pos_x:.2f}, {pos_y:.2f}) | Action: v={cmd.v_linear:.2f}, lat={cmd.v_lateral:.2f}, w={cmd.v_angular:.2f}")
                
    cv2.destroyAllWindows()
    print("Evaluation completed.")

if __name__ == "__main__":
    main()
