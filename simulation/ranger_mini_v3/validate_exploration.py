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
from exploration_policies import PrimitiveExplorationPolicy, Primitive
from test_env import build_world_xml

def euler_from_quaternion(w, x, y, z):
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
    
    recoveries = 0
    distance = 0.0
    last_x, last_y = -4.5, 0.0
    
    for i in range(steps):
        # 10 Hz control
        if i % 10 == 0:
            renderer_depth.update_scene(robot.data, camera="lidar_cam")
            depth_img = renderer_depth.render()
            
            x, y = robot.data.qpos[0], robot.data.qpos[1]
            qw, qx, qy, qz = robot.data.qpos[3:7]
            _, _, yaw = euler_from_quaternion(qw, qx, qy, qz)
            
            cmd, prim = policy.get_action(depth_img, x, y, yaw)
            
            if policy.recovery_timer == int(policy.control_freq * 1.5): # Just entered recovery
                recoveries += 1
            
        robot.apply_command(cmd)
        mujoco.mj_step(robot.model, robot.data)
        
        # Record trajectory at 10 Hz
        if i % 10 == 0:
            x, y = robot.data.qpos[0], robot.data.qpos[1]
            xs.append(x)
            ys.append(y)
            distance += math.hypot(x - last_x, y - last_y)
            last_x, last_y = x, y
            
            gx, gy = int(x / grid_res), int(y / grid_res)
            grid_counts[(gx, gy)] += 1
            
    unique_cells = len(grid_counts)
    total_visits = sum(grid_counts.values())
    entropy = 0.0
    revisitation_rate = 0.0
    
    if total_visits > 0:
        for count in grid_counts.values():
            p = count / total_visits
            entropy -= p * math.log2(p)
        revisitation_rate = total_visits / unique_cells if unique_cells > 0 else 0
            
    return {
        "xs": xs,
        "ys": ys,
        "unique_cells": unique_cells,
        "entropy": entropy,
        "distance": distance,
        "recoveries": recoveries,
        "revisitation_rate": revisitation_rate,
        "primitive_counts": policy.primitive_counts,
        "primitive_durations": policy.primitive_duration_sum,
        "primitive_transitions": policy.primitive_transitions
    }

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
    SEEDS = 3
    
    betas = {
        "White Noise (beta=0)": 0,
        "Pink Noise (beta=1)": 1,
        "Brown Noise (beta=2)": 2
    }
    
    aggregated_results = {}
    
    # Plotting setup
    fig_traj, axes_traj = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (name, beta) in enumerate(betas.items()):
        print(f"\n=================================")
        print(f"Evaluating {name}")
        print(f"=================================")
        
        runs = []
        for seed in range(SEEDS):
            np.random.seed(seed)
            import random
            random.seed(seed)
            
            policy = PrimitiveExplorationPolicy(CONTROL_FREQ, beta=beta)
            res = run_policy(robot, policy, STEPS)
            runs.append(res)
            print(f"  Seed {seed}: Coverage={res['unique_cells']}, Entropy={res['entropy']:.2f}")
            
        # Aggregate stats
        avg_cells = np.mean([r['unique_cells'] for r in runs])
        std_cells = np.std([r['unique_cells'] for r in runs])
        avg_entropy = np.mean([r['entropy'] for r in runs])
        std_entropy = np.std([r['entropy'] for r in runs])
        avg_dist = np.mean([r['distance'] for r in runs])
        avg_recoveries = np.mean([r['recoveries'] for r in runs])
        
        print(f"\nMetrics for {name}:")
        print(f"  Cells Visited: {avg_cells:.1f} ± {std_cells:.1f}")
        print(f"  State Entropy: {avg_entropy:.2f} ± {std_entropy:.2f} bits")
        print(f"  Distance:      {avg_dist:.1f} m")
        print(f"  Collisions:    {avg_recoveries:.1f}")
        
        # Plot trajectory for the first seed
        ax = axes_traj[idx]
        xs = runs[0]['xs']
        ys = runs[0]['ys']
        ax.plot(xs, ys, alpha=0.5, color='b')
        ax.scatter(xs[0], ys[0], color='g', marker='o', s=100, label='Start')
        ax.scatter(xs[-1], ys[-1], color='r', marker='x', s=100, label='End')
        ax.set_title(f"{name}\nEntropy: {runs[0]['entropy']:.2f} | Cells: {runs[0]['unique_cells']}")
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.grid(True)
        if idx == 0:
            ax.legend()
            
    plt.tight_layout()
    out_traj = Path(__file__).parent / "exploration_comparison_primitives.png"
    plt.savefig(out_traj)
    print(f"\nSaved trajectory plots to {out_traj}")

if __name__ == "__main__":
    main()
