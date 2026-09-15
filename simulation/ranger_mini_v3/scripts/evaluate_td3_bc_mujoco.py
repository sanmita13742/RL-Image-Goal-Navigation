import sys
import time
import argparse
import torch
import numpy as np
from PIL import Image
import cv2
import pandas as pd
from collections import deque
from pathlib import Path
import mujoco
import mujoco.viewer
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


from robot import RangerMiniV3Robot
from robot_base import DriveCommand
from src.rl.td3_bc import TD3_BC
from src.data.dinov3_encoder import FrozenDINOv3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--render", action="store_true", help="Enable interactive MuJoCo viewer")
    parser.add_argument("--num_episodes", type=int, default=20, help="Number of episodes to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for episode goal selection")
    args = parser.parse_args()

    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Loading Encoder...")
    encoder = FrozenDINOv3(device=device)
    
    print(f"Loading TD3_BC from {args.checkpoint}...")
    agent = TD3_BC(state_dim=1536, action_dim=3, goal_dim=384, device=device)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found at {ckpt_path}")
        return
    agent.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    agent.actor.eval()
    
    print("Loading Dataset goals...")
    phi_cache = np.load(ROOT / "data" / "processed" / "dinov3" / "phi_cache.npy")
    valid_goals = np.load(ROOT / "data" / "processed" / "dinov3" / "valid_goals.npy")
    index_df = pd.read_parquet(ROOT / "data" / "processed" / "dinov3" / "embedding_index.parquet")
    valid_indices = np.where(valid_goals)[0]
    
    if len(valid_indices) == 0:
        print("No valid goals found.")
        return
        
    # Select random goals for evaluation
    eval_goal_indices = np.random.choice(valid_indices, size=args.num_episodes, replace=False)
    
    # Setup Simulation World
    print("Initializing Simulation with Complex Map...")
    complex_map_path = ROOT / "maps" / "complex" / "scene.xml"
    if not complex_map_path.exists():
        print(f"Error: Complex map not found at {complex_map_path}")
        return
        
    world_xml_str = complex_map_path.read_text(encoding="utf-8")
    
    # Fix the relative path to ranger_mini_v3.xml since we are saving the temp xml to ROOT
    world_xml_str = world_xml_str.replace('../../ranger_mini_v3.xml', 'ranger_mini_v3.xml')
    
    # Inject higher framebuffer resolution for 784x448 rendering
    if "<visual>" in world_xml_str:
        world_xml_str = world_xml_str.replace(
            '<visual>',
            '<visual>\n    <global offwidth="784" offheight="448"/>'
        )
    else:
        world_xml_str = world_xml_str.replace(
            '</mujoco>',
            '  <visual>\n    <global offwidth="784" offheight="448"/>\n  </visual>\n</mujoco>'
        )
        
    world_xml = ROOT / "_eval_world.xml"
    world_xml.write_text(world_xml_str, encoding="utf-8")
    
    robot = RangerMiniV3Robot()
    robot.load(world_xml)
    
    renderer_rgb = mujoco.Renderer(robot.model, height=448, width=784)
    
    try:
        world_xml.unlink()
    except Exception:
        pass
        
    CONTROL_FREQ = 10.0
    sim_dt = robot.model.opt.timestep
    sim_steps_per_control = int(1.0 / (CONTROL_FREQ * sim_dt))
    MAX_STEPS = 1000
    SUCCESS_THRESHOLD = 0.8
    
    results = []
    
    for ep, goal_idx in enumerate(eval_goal_indices):
        print(f"\n--- Episode {ep+1}/{args.num_episodes} | Goal index: {goal_idx} ---")
        goal_phi = torch.tensor(phi_cache[goal_idx], dtype=torch.float32, device=device).unsqueeze(0)
        
        # Extract goal image for visualization if rendering
        goal_bgr = np.zeros((448, 784, 3), dtype=np.uint8)
        if args.render:
            goal_kaggle_path = index_df.iloc[goal_idx]['rgb_abs_path']
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
                local_rel_path = Path(*parts[-4:])
                goal_img_path = ROOT / "dataset" / local_rel_path
                
            goal_bgr = cv2.imread(str(goal_img_path))
            if goal_bgr is not None:
                goal_bgr = cv2.resize(goal_bgr, (784, 448))
        
        # Reset robot to fixed start
        mujoco.mj_resetData(robot.model, robot.data)
        robot.data.qpos[0] = -4.5
        robot.data.qpos[1] = 0.0
        robot.data.qpos[3] = 0.7071068
        robot.data.qpos[4] = 0.0
        robot.data.qpos[5] = 0.0
        robot.data.qpos[6] = 0.7071068
        mujoco.mj_forward(robot.model, robot.data)
        
        history = deque(maxlen=4)
        renderer_rgb.update_scene(robot.data, camera="front_cam")
        rgb_img = renderer_rgb.render()
        img = Image.fromarray(rgb_img).convert("RGB")
        tensor = encoder.transform(img).unsqueeze(0).to(device)
        _, phi, _ = encoder._forward_batch(tensor)
        phi = phi.to(device)
        for _ in range(4):
            history.append(phi)
            
        success = False
        steps_taken = MAX_STEPS
        collision_count = 0
        
        def episode_loop(viewer=None):
            nonlocal success, steps_taken, collision_count, history
            
            for step in range(MAX_STEPS):
                if viewer and not viewer.is_running():
                    break
                    
                step_start = time.time()
                
                # Check similarity
                current_phi = history[-1]
                sim = F.cosine_similarity(current_phi, goal_phi, dim=-1).item()
                if sim >= SUCCESS_THRESHOLD:
                    success = True
                    steps_taken = step
                    print(f"Goal reached at step {step}! Similarity: {sim:.3f}")
                    
                    # Save the image comparison
                    try:
                        save_dir = ROOT / "eval_results"
                        save_dir.mkdir(exist_ok=True)
                        save_path = save_dir / f"success_ckpt{Path(args.checkpoint).stem}_ep{ep}_step{step}.png"
                        curr_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                        if goal_bgr is not None and goal_bgr.shape == curr_bgr.shape:
                            combined = cv2.hconcat([curr_bgr, goal_bgr])
                            cv2.imwrite(str(save_path), combined)
                        else:
                            cv2.imwrite(str(save_path), curr_bgr)
                        print(f"Saved success frame comparison to {save_path}")
                    except Exception as e:
                        print(f"Failed to save success image: {e}")
                    
                    break
                
                # Predict Action
                state = torch.cat(list(history), dim=-1) # [1, 1536]
                with torch.no_grad():
                    action_norm = agent.actor(state, goal_phi).squeeze(0) # [3]
                    action = agent.normalizer.denormalize(action_norm).cpu().numpy()
                    
                cmd = DriveCommand(
                    v_linear=float(action[0]),
                    v_lateral=float(action[1]),
                    v_angular=float(action[2])
                )
                
                # Step Environment
                for _ in range(sim_steps_per_control):
                    robot.apply_command(cmd)
                    robot.step()
                    
                if viewer:
                    viewer.sync()
                
                # Check simple collision heuristic (if v_linear > 0.5 but velocity is ~0)
                vel = np.linalg.norm(robot.data.qvel[0:2])
                if abs(cmd.v_linear) > 0.5 and vel < 0.05:
                    collision_count += 1
                
                # Update observation
                renderer_rgb.update_scene(robot.data, camera="front_cam")
                rgb_img = renderer_rgb.render()
                img = Image.fromarray(rgb_img).convert("RGB")
                tensor = encoder.transform(img).unsqueeze(0).to(device)
                _, new_phi, _ = encoder._forward_batch(tensor)
                history.append(new_phi.to(device))
                
                if viewer:
                    curr_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                    combined = cv2.hconcat([curr_bgr, goal_bgr])
                    cv2.imshow("TD3_BC Evaluation", combined)
                    cv2.waitKey(1)
                    
                    elapsed = time.time() - step_start
                    sleep_for = (1.0 / CONTROL_FREQ) - elapsed
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                        
                if step % 100 == 0 and not viewer:
                    pos_x, pos_y = robot.data.qpos[0:2]
                    print(f"Step {step}/{MAX_STEPS} | pos: ({pos_x:.2f}, {pos_y:.2f}) | Sim: {sim:.3f}")

            if viewer:
                cv2.destroyAllWindows()

        if args.render:
            with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
                viewer.cam.azimuth = 150
                viewer.cam.elevation = -25
                viewer.cam.distance = 8.0
                viewer.cam.lookat[:] = [0, 0, 0.5]
                episode_loop(viewer)
        else:
            episode_loop(viewer=None)
            
        stl = int(success) * (1.0 - (steps_taken / MAX_STEPS))
        print(f"Episode {ep+1} Result - Success: {success}, Steps: {steps_taken}, STL: {stl:.3f}, Stuck/Collisions: {collision_count}")
        results.append({
            "episode": ep + 1,
            "goal_idx": goal_idx,
            "success": success,
            "steps": steps_taken,
            "stl": stl,
            "collisions": collision_count
        })
        
    df = pd.DataFrame(results)
    success_rate = df["success"].mean() * 100
    avg_steps = df[df["success"] == True]["steps"].mean() if success_rate > 0 else MAX_STEPS
    avg_stl = df["stl"].mean()
    avg_collisions = df["collisions"].mean()
    
    print("\n=================================")
    print(f"EVALUATION SUMMARY: {Path(args.checkpoint).name}")
    print("=================================")
    print(f"Success Rate : {success_rate:.1f}%")
    print(f"Average STL  : {avg_stl:.3f}")
    print(f"Avg Time     : {avg_steps:.1f} steps (successful eps only)")
    print(f"Avg Collision: {avg_collisions:.1f} frames stuck")
    print("=================================\n")

if __name__ == "__main__":
    main()
