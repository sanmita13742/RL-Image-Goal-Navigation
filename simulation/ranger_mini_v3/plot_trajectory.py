"""
ranger_mujoco/plot_trajectory.py
============================================================
Reads the dataset log and plots the robot's exploration trajectory 
as well as a heatmap of state visitation for the Ranger Mini V3.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    log_path = Path(__file__).parent / "dataset" / "log.csv"
    if not log_path.exists():
        print(f"Error: Log file not found at {log_path}")
        sys.exit(1)
        
    df = pd.read_csv(log_path, on_bad_lines='skip')
    
    if len(df) == 0:
        print("Dataset is empty. Run random_explore.py first.")
        sys.exit(1)
        
    x = df['pos_x'].values
    y = df['pos_y'].values
    
    # Environment dimensions (based on 12x8 world, centered at 0,0)
    # X from -6 to 6, Y from -4 to 4
    extent = [-6.5, 6.5, -4.5, 4.5]

    plt.figure(figsize=(14, 6))
    
    # 1. Trajectory Plot
    plt.subplot(1, 2, 1)
    plt.title("Ranger Mini V3 Trajectory (LiDAR Avoidance)")
    plt.plot(x, y, 'b-', linewidth=1.5, alpha=0.6, label='Path')
    plt.plot(x[0], y[0], 'go', markersize=8, label="Start")
    plt.plot(x[-1], y[-1], 'ro', markersize=8, label="End")
    
    # Optional: draw rough inner boundary
    plt.plot([-3, 3, 3, -3, -3], [-1.5, -1.5, 1.5, 1.5, -1.5], 'k--', alpha=0.3, label='Obstacle Core')
    
    plt.xlim(extent[0], extent[1])
    plt.ylim(extent[2], extent[3])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    
    # 2. State Visitation Heatmap
    plt.subplot(1, 2, 2)
    plt.title("State Visitation Heatmap")
    hist, xedges, yedges = np.histogram2d(x, y, bins=[40, 40], range=[[extent[0], extent[1]], [extent[2], extent[3]]])
    plt.imshow(hist.T, origin='lower', extent=[extent[0], extent[1], extent[2], extent[3]], cmap='hot', interpolation='nearest')
    plt.colorbar(label="Visits")
    
    plt.xlim(extent[0], extent[1])
    plt.ylim(extent[2], extent[3])
    plt.xlabel("X Position (m)")
    
    plt.tight_layout()
    out_img = Path(__file__).parent / "trajectory_heatmap.png"
    plt.savefig(out_img, dpi=150)
    print(f"Saved plot to {out_img}")
    
    try:
        plt.show()
    except Exception as e:
        print(f"Could not open display window: {e}")

if __name__ == "__main__":
    main()
