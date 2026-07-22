import argparse
import os
import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from scipy.stats import skew, kurtosis

try:
    import imagehash
    from PIL import Image
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ---------------------------------------------------------
# Base Analyzer
# ---------------------------------------------------------

class BaseAnalyzer:
    def __init__(self, df: pd.DataFrame, out_dir: Path, image_dir: Path = None):
        self.df = df
        self.out_dir = out_dir
        self.plots_dir = out_dir / "plots"
        self.stats_dir = out_dir / "statistics"
        self.image_dir = image_dir
        self.stats = {}

        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.stats_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        raise NotImplementedError

    def save_fig(self, fig, filename: str):
        path = self.plots_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def save_stats(self, filename: str):
        path = self.stats_dir / filename
        pd.DataFrame([self.stats]).T.to_csv(path, header=["Value"])


# ---------------------------------------------------------
# 1. Trajectory Analyzer
# ---------------------------------------------------------

class TrajectoryAnalyzer(BaseAnalyzer):
    def run(self):
        print("Running TrajectoryAnalyzer...")
        if 'pos_x' not in self.df.columns or 'pos_y' not in self.df.columns:
            print("Missing pos_x or pos_y, skipping TrajectoryAnalyzer.")
            return

        x = self.df['pos_x'].values
        y = self.df['pos_y'].values
        
        # Plot XY Trajectory
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(x, y, alpha=0.6, label="Trajectory", color='b')
        ax.scatter(x[0], y[0], color='g', marker='o', s=100, label='Start', zorder=5)
        ax.scatter(x[-1], y[-1], color='r', marker='x', s=100, label='End', zorder=5)
        
        if 'yaw' in self.df.columns:
            # Subsample headings to avoid clutter
            step = max(1, len(x) // 50)
            u = np.cos(self.df['yaw'].values[::step])
            v = np.sin(self.df['yaw'].values[::step])
            ax.quiver(x[::step], y[::step], u, v, alpha=0.3, color='k', scale=20)
            
        ax.set_aspect('equal', 'box')
        ax.set_title("XY Trajectory")
        ax.legend()
        ax.grid(True)
        ax.set_xlim(-6.5, 6.5)
        ax.set_ylim(-4.5, 4.5)
        self.save_fig(fig, "xy_trajectory.png")
        
        # Compute Stats
        dx = np.diff(x)
        dy = np.diff(y)
        dists = np.hypot(dx, dy)
        total_distance = np.sum(dists)
        
        duration = len(self.df) * 0.1 # assuming 10Hz if no timestamp
        if 'timestamp' in self.df.columns:
            ts = pd.to_numeric(self.df['timestamp'], errors='coerce').dropna()
            if len(ts) > 1:
                duration = ts.iloc[-1] - ts.iloc[0]
                
        avg_speed = total_distance / duration if duration > 0 else 0
        
        speeds = dists / 0.1 # approx speeds if timestamp missing, but better to use velocities if present
        if 'linear_vel_cmd' in self.df.columns and 'lateral_vel_cmd' in self.df.columns:
            speeds = np.hypot(self.df['linear_vel_cmd'], self.df['lateral_vel_cmd'])
            
        max_speed = np.max(speeds) if len(speeds) > 0 else 0
        
        self.stats = {
            "Total_Distance_m": total_distance,
            "Exploration_Duration_s": duration,
            "Average_Speed_m_s": avg_speed,
            "Max_Speed_m_s": max_speed
        }
        self.save_stats("trajectory_stats.csv")


# ---------------------------------------------------------
# 2. Occupancy Analyzer
# ---------------------------------------------------------

class OccupancyAnalyzer(BaseAnalyzer):
    def run(self, grid_res=0.5):
        print("Running OccupancyAnalyzer...")
        if 'pos_x' not in self.df.columns or 'pos_y' not in self.df.columns:
            return

        x = self.df['pos_x'].values
        y = self.df['pos_y'].values
        
        # Convert to grid
        gx = np.floor(x / grid_res).astype(int)
        gy = np.floor(y / grid_res).astype(int)
        
        # Compute visits
        coords = list(zip(gx, gy))
        visit_counts = pd.Series(coords).value_counts()
        
        unique_cells = len(visit_counts)
        avg_visits = visit_counts.mean()
        max_visits = visit_counts.max()
        
        # Trajectory Coverage over time
        unique_over_time = []
        seen = set()
        for c in coords:
            seen.add(c)
            unique_over_time.append(len(seen))
            
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(unique_over_time)
        ax.set_title("Coverage vs Time")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Unique Cells Visited")
        ax.grid(True)
        self.save_fig(fig, "coverage_over_time.png")
        
        # Revisit Heatmap
        # Build 2D array and ensure it covers at least the map bounds (-6 to 6, -4 to 4)
        map_min_x, map_max_x = -6.0, 6.0
        map_min_y, map_max_y = -4.0, 4.0
        
        min_gx = min(int(np.floor(map_min_x / grid_res)), min(gx))
        max_gx = max(int(np.floor(map_max_x / grid_res)), max(gx))
        min_gy = min(int(np.floor(map_min_y / grid_res)), min(gy))
        max_gy = max(int(np.floor(map_max_y / grid_res)), max(gy))
        
        width = max_gx - min_gx + 1
        height = max_gy - min_gy + 1
        heatmap = np.zeros((height, width))
        for (cx, cy), count in visit_counts.items():
            heatmap[cy - min_gy, cx - min_gx] = count
            
        fig, ax = plt.subplots(figsize=(8, 8))
        extent = [min_gx * grid_res, (max_gx + 1) * grid_res, min_gy * grid_res, (max_gy + 1) * grid_res]
        im = ax.imshow(heatmap, origin='lower', cmap='hot', interpolation='nearest', extent=extent)
        plt.colorbar(im, ax=ax, label="Visit Count")
        ax.set_title("Occupancy / Revisit Heatmap")
        ax.set_xlim(-6.5, 6.5)
        ax.set_ylim(-4.5, 4.5)
        self.save_fig(fig, "revisit_heatmap.png")
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(visit_counts.values, bins=50)
        ax.set_title("Visit Count Histogram")
        ax.set_yscale('log')
        self.save_fig(fig, "visit_histogram.png")

        # 95% Saturation
        target = 0.95 * unique_cells
        time_to_95 = 0
        for i, val in enumerate(unique_over_time):
            if val >= target:
                time_to_95 = i
                break

        self.stats = {
            "Unique_Cells_Visited": unique_cells,
            "Average_Visits_Per_Cell": avg_visits,
            "Max_Visits": max_visits,
            "Revisitation_Ratio": (len(self.df) - unique_cells) / len(self.df) if len(self.df) > 0 else 0,
            "Steps_to_95_Percent_Coverage": time_to_95
        }
        self.save_stats("occupancy_stats.csv")


# ---------------------------------------------------------
# 3. Motion Analyzer
# ---------------------------------------------------------

class MotionAnalyzer(BaseAnalyzer):
    def run(self):
        print("Running MotionAnalyzer...")
        cols = [c for c in ['linear_vel_cmd', 'lateral_vel_cmd', 'angular_vel_cmd'] if c in self.df.columns]
        if not cols:
            return
            
        fig, axes = plt.subplots(1, len(cols), figsize=(5*len(cols), 4))
        if len(cols) == 1:
            axes = [axes]
            
        for ax, col in zip(axes, cols):
            sns.histplot(self.df[col], kde=True, ax=ax)
            ax.set_title(f"Histogram of {col}")
            
            self.stats[f"{col}_mean"] = self.df[col].mean()
            self.stats[f"{col}_std"] = self.df[col].std()
            self.stats[f"{col}_max"] = self.df[col].max()
            self.stats[f"{col}_min"] = self.df[col].min()
            
        self.save_fig(fig, "motion_histograms.png")
        self.save_stats("motion_stats.csv")


# ---------------------------------------------------------
# 4. Primitive Analyzer
# ---------------------------------------------------------

class PrimitiveAnalyzer(BaseAnalyzer):
    def run(self):
        print("Running PrimitiveAnalyzer...")
        # Since standard log might not have 'primitive', let's look for it
        if 'primitive' not in self.df.columns:
            print("No 'primitive' column found, skipping PrimitiveAnalyzer.")
            return
            
        prims = self.df['primitive'].values
        unique, counts = np.unique(prims, return_counts=True)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(counts, labels=unique, autopct='%1.1f%%')
        ax.set_title("Primitive Usage")
        self.save_fig(fig, "primitive_usage.png")
        
        # Transition Matrix
        transitions = pd.DataFrame(index=unique, columns=unique).fillna(0)
        for i in range(len(prims)-1):
            transitions.loc[prims[i], prims[i+1]] += 1
            
        # Normalize
        t_sum = transitions.sum(axis=1)
        # Avoid div by zero
        transitions_norm = transitions.div(t_sum.where(t_sum != 0, 1), axis=0)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(transitions_norm, annot=True, fmt=".2f", cmap='Blues', ax=ax)
        ax.set_title("Primitive Transition Matrix")
        self.save_fig(fig, "primitive_transitions.png")
        
        # Durations
        durations = []
        curr = prims[0]
        length = 1
        for p in prims[1:]:
            if p == curr:
                length += 1
            else:
                durations.append((curr, length))
                curr = p
                length = 1
        durations.append((curr, length))
        
        df_dur = pd.DataFrame(durations, columns=['Primitive', 'Duration'])
        avg_dur = df_dur.groupby('Primitive').mean()
        
        for idx, row in avg_dur.iterrows():
            self.stats[f"{idx}_Avg_Duration"] = row['Duration']
            
        self.save_stats("primitive_stats.csv")


# ---------------------------------------------------------
# 5. Safety Analyzer
# ---------------------------------------------------------

class SafetyAnalyzer(BaseAnalyzer):
    def run(self):
        print("Running SafetyAnalyzer...")
        if 'recovery_state' not in self.df.columns:
            return
            
        is_recovery = self.df['recovery_state'] != 'NORMAL'
        recovery_events = (is_recovery != is_recovery.shift(1)) & is_recovery
        
        recovery_count = recovery_events.sum()
        total_recovery_steps = is_recovery.sum()
        
        self.stats = {
            "Recovery_Count": recovery_count,
            "Avg_Recovery_Duration": total_recovery_steps / recovery_count if recovery_count > 0 else 0,
            "Percent_Time_In_Recovery": (total_recovery_steps / len(self.df)) * 100
        }
        self.save_stats("safety_stats.csv")


# ---------------------------------------------------------
# 6. Loop Analyzer
# ---------------------------------------------------------

class LoopAnalyzer(BaseAnalyzer):
    def run(self, window_size=100, displacement_thresh=1.0, distance_thresh=5.0):
        print("Running LoopAnalyzer...")
        if 'pos_x' not in self.df.columns or 'pos_y' not in self.df.columns:
            return
            
        x = self.df['pos_x'].values
        y = self.df['pos_y'].values
        
        dx = np.diff(x)
        dy = np.diff(y)
        dists = np.hypot(dx, dy)
        dists = np.insert(dists, 0, 0)
        cum_dist = np.cumsum(dists)
        
        loops_detected = 0
        loop_flags = np.zeros(len(x), dtype=bool)
        
        for i in range(len(x) - window_size):
            j = i + window_size
            displacement = math.hypot(x[j] - x[i], y[j] - y[i])
            travelled = cum_dist[j] - cum_dist[i]
            
            if travelled > distance_thresh and displacement < displacement_thresh:
                loops_detected += 1
                loop_flags[i:j] = True
                
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(x, y, alpha=0.5, color='gray', label="Trajectory")
        if loops_detected > 0:
            ax.scatter(x[loop_flags], y[loop_flags], color='red', s=5, label="Detected Loops")
        
        ax.set_title("Loop Detection")
        ax.set_aspect('equal')
        ax.legend()
        self.save_fig(fig, "loop_detection.png")
        
        self.stats = {
            "Loops_Detected": loops_detected,
            "Percent_Time_Looping": (loop_flags.sum() / len(x)) * 100
        }
        self.save_stats("loop_stats.csv")


# ---------------------------------------------------------
# 7. Information Metrics (Entropy)
# ---------------------------------------------------------

class InformationAnalyzer(BaseAnalyzer):
    def run(self):
        print("Running InformationAnalyzer...")
        cols = ['pos_x', 'pos_y', 'linear_vel_cmd', 'angular_vel_cmd']
        valid_cols = [c for c in cols if c in self.df.columns]
        
        if not valid_cols:
            return
            
        data = self.df[valid_cols].dropna().values
        if len(data) == 0:
            return
            
        # Compute joint entropy using histogram
        # Discretize each dim into 10 bins
        H, edges = np.histogramdd(data, bins=10)
        p = H / np.sum(H)
        p = p[p > 0]
        joint_entropy = -np.sum(p * np.log2(p))
        
        self.stats["Joint_StateAction_Entropy_bits"] = joint_entropy
        
        if 'pos_x' in self.df.columns and 'pos_y' in self.df.columns:
            H, _, _ = np.histogram2d(self.df['pos_x'], self.df['pos_y'], bins=20)
            p = H / np.sum(H)
            p = p[p > 0]
            self.stats["State_Entropy_bits"] = -np.sum(p * np.log2(p))
            
        self.save_stats("information_stats.csv")


# ---------------------------------------------------------
# 8. Image Quality Analyzer
# ---------------------------------------------------------

class ImageQualityAnalyzer(BaseAnalyzer):
    def run(self):
        print("Running ImageQualityAnalyzer...")
        if not self.image_dir or not self.image_dir.exists():
            print("No image directory provided or exists. Skipping.")
            return
            
        if not HAS_IMAGEHASH or not HAS_CV2:
            print("Missing Pillow/imagehash/cv2. Image quality metrics limited or skipped.")
            return
            
        images = list(self.image_dir.glob("*.png")) + list(self.image_dir.glob("*.jpg"))
        if len(images) > 200:
            # Sample 200 images to save time if dataset is huge
            images = np.random.choice(images, 200, replace=False)
            
        if not images:
            return
            
        hashes = set()
        duplicates = 0
        blurs = []
        brights = []
        
        for img_path in tqdm(images, desc="Analyzing Images"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            
            brights.append(np.mean(img))
            blurs.append(np.var(cv2.Laplacian(img, cv2.CV_64F)))
            
            pil_img = Image.fromarray(img)
            h = imagehash.average_hash(pil_img)
            if h in hashes:
                duplicates += 1
            else:
                hashes.add(h)
                
        self.stats = {
            "Sampled_Images": len(images),
            "Avg_Brightness": np.mean(brights) if brights else 0,
            "Avg_Blur_Variance": np.mean(blurs) if blurs else 0,
            "Duplicate_Percentage": (duplicates / len(images)) * 100 if images else 0
        }
        self.save_stats("image_quality_stats.csv")


# ---------------------------------------------------------
# Report Generator
# ---------------------------------------------------------

class ReportGenerator:
    def __init__(self, out_dir: Path, all_stats: dict):
        self.out_dir = out_dir
        self.all_stats = all_stats
        
    def run(self):
        report_path = self.out_dir / "report.md"
        
        # Simple Logic
        cov = self.all_stats.get('Unique_Cells_Visited', 0)
        dup = self.all_stats.get('Duplicate_Percentage', 0)
        loop_pct = self.all_stats.get('Percent_Time_Looping', 0)
        
        if cov > 40 and loop_pct < 10 and dup < 5:
            rec = "READY FOR RL"
        elif cov > 20 and loop_pct < 20:
            rec = "READY WITH MINOR IMPROVEMENTS"
        else:
            rec = "COLLECT MORE DATA"
            
        md = f"""# Dataset Analysis Report

## Overall Recommendation: {rec}

### Justification
- Unique Coverage: {cov} cells
- Looping Percentage: {loop_pct:.1f}%
- Duplicate Images: {dup:.1f}%

### Metrics Summary
"""
        for k, v in self.all_stats.items():
            if isinstance(v, float):
                md += f"- **{k}**: {v:.3f}\n"
            else:
                md += f"- **{k}**: {v}\n"
                
        report_path.write_text(md)
        print(f"\nSaved Report to {report_path}")
        
        # Global Summary CSV
        summary_path = self.out_dir.parent / "summary.csv"
        df_new = pd.DataFrame([self.all_stats])
        if summary_path.exists():
            df_new.to_csv(summary_path, mode='a', header=False, index=False)
        else:
            df_new.to_csv(summary_path, index=False)
        print(f"Appended to {summary_path}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RL Dataset Analysis & Validation")
    parser.add_argument("--csv", type=str, required=True, help="Path to log.csv")
    parser.add_argument("--images", type=str, default=None, help="Path to image directory")
    parser.add_argument("--outdir", type=str, default="analysis_output", help="Output directory")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path, on_bad_lines='skip')
    out_dir = Path(args.outdir)
    image_dir = Path(args.images) if args.images else None

    # Instantiate Analyzers
    analyzers = [
        TrajectoryAnalyzer(df, out_dir, image_dir),
        OccupancyAnalyzer(df, out_dir, image_dir),
        MotionAnalyzer(df, out_dir, image_dir),
        PrimitiveAnalyzer(df, out_dir, image_dir),
        SafetyAnalyzer(df, out_dir, image_dir),
        LoopAnalyzer(df, out_dir, image_dir),
        InformationAnalyzer(df, out_dir, image_dir),
        ImageQualityAnalyzer(df, out_dir, image_dir)
    ]

    all_stats = {}
    
    # Run all
    for analyzer in analyzers:
        try:
            analyzer.run()
            all_stats.update(analyzer.stats)
        except Exception as e:
            print(f"Error running {analyzer.__class__.__name__}: {e}")

    # Generate Report
    rg = ReportGenerator(out_dir, all_stats)
    rg.run()


if __name__ == "__main__":
    main()
