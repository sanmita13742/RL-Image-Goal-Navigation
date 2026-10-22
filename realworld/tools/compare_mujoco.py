import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from pathlib import Path

def compute_yaw(ori_x, ori_y, ori_z, ori_w):
    rot = R.from_quat(np.column_stack([ori_x, ori_y, ori_z, ori_w]))
    return rot.as_euler('zyx')[:, 0]

def calculate_metrics(real_val, sim_val):
    rmse = np.sqrt(np.mean((real_val - sim_val)**2))
    return rmse

def compare_mujoco_and_real(real_csv, sim_csv, out_dir):
    print("Loading datasets...")
    df_real = pd.read_csv(real_csv)
    df_sim = pd.read_csv(sim_csv, on_bad_lines='skip')
    
    # Normalize timestamps to start at 0
    df_real['time'] = df_real['timestamp'] - df_real['timestamp'].iloc[0]
    df_sim['time'] = df_sim['timestamp'] - df_sim['timestamp'].iloc[0]
    
    # Calculate yaw for real robot from quaternion
    df_real['yaw'] = compute_yaw(df_real['ori_x'], df_real['ori_y'], df_real['ori_z'], df_real['ori_w'])
    
    # Interpolate simulation data to match real time steps
    print("Aligning timestamps via interpolation...")
    common_time = df_real['time'].values
    max_sim_time = df_sim['time'].max()
    
    # Only keep common time up to the end of simulation if sim is shorter
    common_time = common_time[common_time <= max_sim_time]
    
    interp_sim_x = interp1d(df_sim['time'], df_sim['pos_x'], kind='linear')(common_time)
    interp_sim_y = interp1d(df_sim['time'], df_sim['pos_y'], kind='linear')(common_time)
    interp_sim_yaw = interp1d(df_sim['time'], df_sim['yaw'], kind='linear')(common_time)
    
    real_x = df_real['pose_x'].values[:len(common_time)].copy()
    real_y = df_real['pose_y'].values[:len(common_time)].copy()
    real_yaw = df_real['yaw'].values[:len(common_time)].copy()
    
    # Offset both to start at origin for fair trajectory comparison
    real_x -= real_x[0]
    real_y -= real_y[0]
    interp_sim_x -= interp_sim_x[0]
    interp_sim_y -= interp_sim_y[0]
    
    # Compute RMSE
    print("Computing statistics...")
    rmse_x = calculate_metrics(real_x, interp_sim_x)
    rmse_y = calculate_metrics(real_y, interp_sim_y)
    rmse_yaw = calculate_metrics(real_yaw, interp_sim_yaw)
    
    path_len_real = np.sum(np.sqrt(np.diff(real_x)**2 + np.diff(real_y)**2))
    path_len_sim = np.sum(np.sqrt(np.diff(interp_sim_x)**2 + np.diff(interp_sim_y)**2))
    
    # Plot Trajectory
    print("Generating plots...")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    plt.plot(real_x, real_y, label='Real Robot', color='blue', linewidth=2)
    plt.plot(interp_sim_x, interp_sim_y, label='MuJoCo Sim', color='red', linestyle='--', linewidth=2)
    plt.title('Trajectory Comparison: Real vs Sim')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir / 'trajectory_comparison.png')
    plt.close()
    
    # Plot Yaw
    plt.figure(figsize=(8, 4))
    plt.plot(common_time, real_yaw, label='Real Yaw', color='blue')
    plt.plot(common_time, interp_sim_yaw, label='Sim Yaw', color='red', linestyle='--')
    plt.title('Yaw Comparison Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Yaw (rad)')
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir / 'yaw_comparison.png')
    plt.close()
    
    # Save Report
    report = f"""# MuJoCo vs Real Robot Comparison Report

## Time Alignment
- The datasets were aligned via linear interpolation using the real robot timestamps as the base.
- Comparison duration: {common_time[-1]:.2f} seconds

## Trajectory Statistics
- **Path Length (Real):** {path_len_real:.2f} m
- **Path Length (Sim):** {path_len_sim:.2f} m
- **Trajectory RMSE (X):** {rmse_x:.4f} m
- **Trajectory RMSE (Y):** {rmse_y:.4f} m

## Orientation Statistics
- **Yaw RMSE:** {rmse_yaw:.4f} rad

Plots have been saved to the `{out_dir}` directory.
"""
    with open(out_dir / "comparison_report.md", "w") as f:
        f.write(report)
        
    print(f"Comparison complete. Report saved to {out_dir / 'comparison_report.md'}")

if __name__ == "__main__":
    real_csv = r"C:\Users\sanmi\Desktop\projects\RL\simulation\ranger_mini_v3\dataset\states.csv"
    sim_csv = r"C:\Users\sanmi\Desktop\projects\RL\simulation\ranger_mini_v3\dataset\log.csv"
    out_dir = r"C:\Users\sanmi\Desktop\projects\RL\simulation\ranger_mini_v3\comparison"
    compare_mujoco_and_real(real_csv, sim_csv, out_dir)
