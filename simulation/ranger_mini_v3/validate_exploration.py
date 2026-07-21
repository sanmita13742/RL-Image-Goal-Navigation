import mujoco
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
from collections import defaultdict
import math

sys.path.append(str(Path(__file__).resolve().parent.parent))
from robot_base import DriveCommand
from ranger_mini_v3.robot import RangerMiniV3Robot
from exploration_policies import WhiteNoiseExploration, OUExploration, TruePinkExploration, UniformPinkExploration
from test_env import build_world_xml

def run_policy(robot, policy, steps, world_size=15.0):
    mujoco.mj_resetData(robot.model, robot.data)
    
    # Start at known safe spot
    robot.data.qpos[0] = -4.5
    robot.data.qpos[1] = 0.0
    
    xs, ys = [], []
    grid_counts = defaultdict(int)
    grid_res = 0.5 # 0.5m grid squares
    
    # Needs renderer for depth based collision avoidance
    renderer_depth = mujoco.Renderer(robot.model, height=60, width=640)
    renderer_depth.enable_depth_rendering()
    
    for i in range(steps):
        # 10 Hz control
        if i % 10 == 0:
            renderer_depth.update_scene(robot.data, camera="lidar_cam")
            depth_img = renderer_depth.render()
            min_depth = np.min(depth_img)
            
            cmd = policy.get_action(min_depth)
            
        robot.apply_command(cmd)
        mujoco.mj_step(robot.model, robot.data)
        
        # Record trajectory at 10 Hz
        if i % 10 == 0:
            x, y = robot.data.qpos[0], robot.data.qpos[1]
            xs.append(x)
            ys.append(y)
            
            gx, gy = int(x / grid_res), int(y / grid_res)
            grid_counts[(gx, gy)] += 1
            
    # Calculate Coverage % (assuming a 30x30 world -> 60x60 grid = 3600 cells)
    # We'll just count unique visited cells relative to the max observed
    unique_cells = len(grid_counts)
    
    # Calculate Shannon Entropy of state visitations
    total_visits = sum(grid_counts.values())
    entropy = 0.0
    if total_visits > 0:
        for count in grid_counts.values():
            p = count / total_visits
            entropy -= p * math.log2(p)
            
    return xs, ys, unique_cells, entropy

def main():
    world_xml = Path(__file__).parent / "_val_world.xml"
    world_xml.write_text(build_world_xml(), encoding="utf-8")

    robot = RangerMiniV3Robot()
    robot.load(world_xml)
    
    try:
        world_xml.unlink()
    except Exception:
        pass

    STEPS = 20000 # 80 seconds
    CONTROL_FREQ = 10.0
    
    policies = {
        "Baseline (White Noise)": WhiteNoiseExploration(CONTROL_FREQ),
        "Previous Baseline (OU Noise)": OUExploration(CONTROL_FREQ),
        "Optimized (Uniform Pink Noise)": UniformPinkExploration(CONTROL_FREQ)
    }
    
    results = {}
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (name, policy) in enumerate(policies.items()):
        print(f"Running {name}...")
        xs, ys, unique, ent = run_policy(robot, policy, STEPS)
        results[name] = {"coverage": unique, "entropy": ent}
        
        ax = axes[idx]
        ax.plot(xs, ys, alpha=0.5, color='b')
        ax.scatter(xs[0], ys[0], color='g', marker='o', s=100, label='Start')
        ax.scatter(xs[-1], ys[-1], color='r', marker='x', s=100, label='End')
        ax.set_title(f"{name}\nEntropy: {ent:.2f} bits | Cells: {unique}")
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.grid(True)
        
    axes[0].legend()
    plt.tight_layout()
    
    out_path = Path(__file__).parent / "exploration_comparison.png"
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")
    
    print("\n--- Results ---")
    for name, res in results.items():
        print(f"{name}:")
        print(f"  Unique Cells Visited : {res['coverage']}")
        print(f"  State Entropy        : {res['entropy']:.2f} bits")

if __name__ == "__main__":
    main()
