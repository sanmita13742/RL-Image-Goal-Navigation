"""
analyze_dataset.py
============================================================
RL Dataset Analysis & Validation.

Supports two input modes:

  1. LEGACY (flat CSV):
       python analyze_dataset.py --csv path/to/log.csv --outdir out/

  2. SESSION (segment-based, new format):
       python analyze_dataset.py --session path/to/dataset/<YYYYMMDD_HHMMSS>/ --outdir out/

     Automatically discovers all segment_NNN/segment.csv files, concatenates
     them into one global DataFrame using global_step as the ordering key,
     then runs all existing session-level analyses PLUS new segment-level
     analyses (per-segment coverage, cross-segment continuity, etc.).

Analyzers (all existing, unchanged):
  1. TrajectoryAnalyzer   -- XY path, total distance, speed
  2. OccupancyAnalyzer    -- Heatmap, coverage over time, revisitation ratio
  3. MotionAnalyzer       -- Action histograms (linear/lateral/angular vel)
  4. PrimitiveAnalyzer    -- Primitive usage pie + transition matrix (if col present)
  5. SafetyAnalyzer       -- Recovery state machine events (if col present)
  6. LoopAnalyzer         -- Behavioral loop detection
  7. InformationAnalyzer  -- Joint state-action entropy
  8. ImageQualityAnalyzer -- Blur / brightness / duplicate detection (optional)

New (session mode only):
  9. SegmentAnalyzer      -- Per-segment coverage, distance, position continuity,
                             sim_time continuity, global_step monotonicity
"""

import argparse
import math
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
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


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_session(session_dir: Path) -> tuple[pd.DataFrame, dict]:
    """
    Discover all segment_NNN/segment.csv files in session_dir.
    Concatenate them into one DataFrame sorted by global_step.
    Returns (df, metadata_dict).
    """
    seg_dirs = sorted([d for d in session_dir.iterdir()
                       if d.is_dir() and d.name.startswith("segment_")])
    if not seg_dirs:
        raise ValueError(f"No segment_NNN directories found in {session_dir}")

    frames = []
    for sd in seg_dirs:
        csv_path = sd / "segment.csv"
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found, skipping segment {sd.name}")
            continue
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        df["_segment_name"] = sd.name   # internal tag for segment analysis
        frames.append(df)

    if not frames:
        raise ValueError(f"No readable segment CSVs found in {session_dir}")

    df_all = pd.concat(frames, ignore_index=True)

    # Sort by global_step to guarantee order even if files were read out-of-order
    if "global_step" in df_all.columns:
        df_all = df_all.sort_values("global_step").reset_index(drop=True)

    # Rename columns to match what existing analyzers expect
    # segment.csv uses "sim_time"; legacy log.csv used "timestamp"
    if "sim_time" in df_all.columns and "timestamp" not in df_all.columns:
        df_all = df_all.rename(columns={"sim_time": "timestamp"})

    # Load metadata if present
    meta = {}
    meta_path = session_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    print(f"  Loaded {len(seg_dirs)} segments, {len(df_all)} total rows from {session_dir.name}")
    return df_all, meta


# ─────────────────────────────────────────────────────────────────────────────
# Base Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class BaseAnalyzer:
    def __init__(self, df: pd.DataFrame, out_dir: Path, image_dir: Path = None):
        self.df        = df
        self.out_dir   = out_dir
        self.plots_dir = out_dir / "plots"
        self.stats_dir = out_dir / "statistics"
        self.image_dir = image_dir
        self.stats     = {}

        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.stats_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        raise NotImplementedError

    def save_fig(self, fig, filename: str):
        path = self.plots_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def save_stats(self, filename: str):
        path = self.stats_dir / filename
        pd.DataFrame([self.stats]).T.to_csv(path, header=["Value"])


# ─────────────────────────────────────────────────────────────────────────────
# 1. Trajectory Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryAnalyzer(BaseAnalyzer):
    def run(self):
        print("Running TrajectoryAnalyzer...")
        if 'pos_x' not in self.df.columns or 'pos_y' not in self.df.columns:
            print("  Missing pos_x or pos_y, skipping.")
            return

        x = self.df['pos_x'].values
        y = self.df['pos_y'].values

        fig, ax = plt.subplots(figsize=(8, 8))
        # Color-code by time to show temporal progression
        sc = ax.scatter(x, y, c=np.arange(len(x)), cmap='plasma', s=1, alpha=0.5)
        plt.colorbar(sc, ax=ax, label="Step")
        ax.scatter(x[0],  y[0],  color='lime',  marker='o', s=120, label='Start', zorder=5)
        ax.scatter(x[-1], y[-1], color='red',   marker='x', s=120, label='End',   zorder=5)

        if 'yaw' in self.df.columns:
            step = max(1, len(x) // 60)
            u = np.cos(self.df['yaw'].values[::step])
            v = np.sin(self.df['yaw'].values[::step])
            ax.quiver(x[::step], y[::step], u, v, alpha=0.25, color='k', scale=25)

        ax.set_aspect('equal', 'box')
        ax.set_title(f"XY Trajectory  ({len(x):,} steps)")
        ax.legend()
        ax.grid(True)
        ax.set_xlim(-6.5, 6.5)
        ax.set_ylim(-4.5, 4.5)
        self.save_fig(fig, "xy_trajectory.png")

        dx = np.diff(x)
        dy = np.diff(y)
        dists = np.hypot(dx, dy)
        total_distance = float(np.sum(dists))

        duration = len(self.df) * 0.1
        if 'timestamp' in self.df.columns:
            ts = pd.to_numeric(self.df['timestamp'], errors='coerce').dropna()
            if len(ts) > 1:
                duration = float(ts.iloc[-1] - ts.iloc[0])

        avg_speed = total_distance / duration if duration > 0 else 0

        if 'linear_vel_cmd' in self.df.columns and 'lateral_vel_cmd' in self.df.columns:
            speeds = np.hypot(self.df['linear_vel_cmd'], self.df['lateral_vel_cmd'])
        else:
            speeds = dists / 0.1

        self.stats = {
            "Total_Distance_m":      total_distance,
            "Exploration_Duration_s": duration,
            "Average_Speed_m_s":     avg_speed,
            "Max_Speed_m_s":         float(np.max(speeds)) if len(speeds) > 0 else 0,
        }
        self.save_stats("trajectory_stats.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Occupancy Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class OccupancyAnalyzer(BaseAnalyzer):
    def run(self, grid_res=0.5):
        print("Running OccupancyAnalyzer...")
        if 'pos_x' not in self.df.columns or 'pos_y' not in self.df.columns:
            return

        x = self.df['pos_x'].values
        y = self.df['pos_y'].values
        gx = np.floor(x / grid_res).astype(int)
        gy = np.floor(y / grid_res).astype(int)
        coords = list(zip(gx, gy))
        visit_counts = pd.Series(coords).value_counts()

        unique_cells = len(visit_counts)
        avg_visits   = float(visit_counts.mean())
        max_visits   = int(visit_counts.max())

        # Coverage over time
        unique_over_time = []
        seen = set()
        for c in coords:
            seen.add(c)
            unique_over_time.append(len(seen))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(unique_over_time, linewidth=0.8)
        ax.set_title(f"Coverage vs Global Step  (final={unique_cells} unique cells)")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Unique 0.5m Cells Visited")
        ax.grid(True)
        self.save_fig(fig, "coverage_over_time.png")

        # Heatmap
        map_min_x, map_max_x = -6.0, 6.0
        map_min_y, map_max_y = -4.0, 4.0
        min_gx = min(int(np.floor(map_min_x / grid_res)), int(gx.min()))
        max_gx = max(int(np.floor(map_max_x / grid_res)), int(gx.max()))
        min_gy = min(int(np.floor(map_min_y / grid_res)), int(gy.min()))
        max_gy = max(int(np.floor(map_max_y / grid_res)), int(gy.max()))
        width  = max_gx - min_gx + 1
        height = max_gy - min_gy + 1
        heatmap = np.zeros((height, width))
        for (cx, cy), count in visit_counts.items():
            heatmap[cy - min_gy, cx - min_gx] = count

        fig, ax = plt.subplots(figsize=(10, 8))
        extent = [min_gx*grid_res, (max_gx+1)*grid_res, min_gy*grid_res, (max_gy+1)*grid_res]
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
        ax.set_xlabel("Visits per Cell")
        self.save_fig(fig, "visit_histogram.png")

        target = 0.95 * unique_cells
        time_to_95 = next((i for i, v in enumerate(unique_over_time) if v >= target), len(unique_over_time))

        self.stats = {
            "Unique_Cells_Visited":       unique_cells,
            "Average_Visits_Per_Cell":    avg_visits,
            "Max_Visits":                 max_visits,
            "Revisitation_Ratio":         (len(self.df) - unique_cells) / len(self.df),
            "Steps_to_95_Percent_Coverage": time_to_95,
        }
        self.save_stats("occupancy_stats.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Motion Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class MotionAnalyzer(BaseAnalyzer):
    def run(self):
        print("Running MotionAnalyzer...")
        cols = [c for c in ['linear_vel_cmd', 'lateral_vel_cmd', 'angular_vel_cmd']
                if c in self.df.columns]
        if not cols:
            return

        fig, axes = plt.subplots(1, len(cols), figsize=(5*len(cols), 4))
        if len(cols) == 1:
            axes = [axes]

        for ax, col in zip(axes, cols):
            sns.histplot(self.df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"{col}")
            self.stats[f"{col}_mean"] = float(self.df[col].mean())
            self.stats[f"{col}_std"]  = float(self.df[col].std())
            self.stats[f"{col}_max"]  = float(self.df[col].max())
            self.stats[f"{col}_min"]  = float(self.df[col].min())

        self.save_fig(fig, "motion_histograms.png")
        self.save_stats("motion_stats.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Primitive Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class PrimitiveAnalyzer(BaseAnalyzer):
    def run(self):
        print("Running PrimitiveAnalyzer...")
        if 'primitive' not in self.df.columns:
            print("  No 'primitive' column, skipping.")
            return

        prims  = self.df['primitive'].values
        unique, counts = np.unique(prims, return_counts=True)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(counts, labels=unique, autopct='%1.1f%%')
        ax.set_title("Primitive Usage")
        self.save_fig(fig, "primitive_usage.png")

        transitions = pd.DataFrame(index=unique, columns=unique).fillna(0)
        for i in range(len(prims)-1):
            transitions.loc[prims[i], prims[i+1]] += 1
        t_sum = transitions.sum(axis=1)
        transitions_norm = transitions.div(t_sum.where(t_sum != 0, 1), axis=0)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(transitions_norm.astype(float), annot=True, fmt=".2f", cmap='Blues', ax=ax)
        ax.set_title("Primitive Transition Matrix")
        self.save_fig(fig, "primitive_transitions.png")
        self.save_stats("primitive_stats.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Safety Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class SafetyAnalyzer(BaseAnalyzer):
    def run(self):
        print("Running SafetyAnalyzer...")
        if 'recovery_state' not in self.df.columns:
            return
        is_recovery   = self.df['recovery_state'] != 'NORMAL'
        recovery_events = (is_recovery != is_recovery.shift(1)) & is_recovery
        recovery_count  = int(recovery_events.sum())
        total_rec_steps = int(is_recovery.sum())
        self.stats = {
            "Recovery_Count":           recovery_count,
            "Avg_Recovery_Duration":    total_rec_steps / recovery_count if recovery_count > 0 else 0,
            "Percent_Time_In_Recovery": (total_rec_steps / len(self.df)) * 100,
        }
        self.save_stats("safety_stats.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Loop Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class LoopAnalyzer(BaseAnalyzer):
    def run(self, window_size=100, displacement_thresh=1.0, distance_thresh=5.0):
        print("Running LoopAnalyzer...")
        if 'pos_x' not in self.df.columns or 'pos_y' not in self.df.columns:
            return

        x = self.df['pos_x'].values
        y = self.df['pos_y'].values
        dx = np.diff(x); dy = np.diff(y)
        dists    = np.insert(np.hypot(dx, dy), 0, 0)
        cum_dist = np.cumsum(dists)

        loops_detected = 0
        loop_flags = np.zeros(len(x), dtype=bool)

        for i in range(len(x) - window_size):
            j = i + window_size
            displacement = math.hypot(x[j]-x[i], y[j]-y[i])
            travelled    = cum_dist[j] - cum_dist[i]
            if travelled > distance_thresh and displacement < displacement_thresh:
                loops_detected += 1
                loop_flags[i:j] = True

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(x, y, alpha=0.4, color='gray', linewidth=0.5, label="Trajectory")
        if loops_detected > 0:
            ax.scatter(x[loop_flags], y[loop_flags], color='red', s=2, label="Loops")
        ax.set_title(f"Loop Detection  ({loops_detected} events)")
        ax.set_aspect('equal')
        ax.legend()
        ax.set_xlim(-6.5, 6.5)
        ax.set_ylim(-4.5, 4.5)
        self.save_fig(fig, "loop_detection.png")

        self.stats = {
            "Loops_Detected":        loops_detected,
            "Percent_Time_Looping":  (loop_flags.sum() / len(x)) * 100,
        }
        self.save_stats("loop_stats.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Information Metrics
# ─────────────────────────────────────────────────────────────────────────────

class InformationAnalyzer(BaseAnalyzer):
    def run(self):
        print("Running InformationAnalyzer...")
        cols = [c for c in ['pos_x', 'pos_y', 'linear_vel_cmd', 'angular_vel_cmd']
                if c in self.df.columns]
        if not cols:
            return
        data = self.df[cols].dropna().values
        if len(data) == 0:
            return

        H, _ = np.histogramdd(data, bins=10)
        p = H / np.sum(H)
        p = p[p > 0]
        self.stats["Joint_StateAction_Entropy_bits"] = float(-np.sum(p * np.log2(p)))

        if 'pos_x' in self.df.columns and 'pos_y' in self.df.columns:
            H2, _, _ = np.histogram2d(self.df['pos_x'], self.df['pos_y'], bins=20)
            p2 = H2 / np.sum(H2)
            p2 = p2[p2 > 0]
            self.stats["State_Entropy_bits"] = float(-np.sum(p2 * np.log2(p2)))

        self.save_stats("information_stats.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Image Quality Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class ImageQualityAnalyzer(BaseAnalyzer):
    def run(self):
        print("Running ImageQualityAnalyzer...")
        if not self.image_dir or not self.image_dir.exists():
            print("  No image directory provided or exists. Skipping.")
            return
        if not HAS_IMAGEHASH or not HAS_CV2:
            print("  Missing imagehash/cv2. Skipping image quality metrics.")
            return

        images = list(self.image_dir.glob("*.png")) + list(self.image_dir.glob("*.jpg"))
        if len(images) > 200:
            images = list(np.random.choice(images, 200, replace=False))
        if not images:
            return

        hashes = set(); duplicates = 0; blurs = []; brights = []
        for img_path in tqdm(images, desc="Analyzing Images"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            brights.append(float(np.mean(img)))
            blurs.append(float(np.var(cv2.Laplacian(img, cv2.CV_64F))))
            pil_img = Image.fromarray(img)
            h = imagehash.average_hash(pil_img)
            if h in hashes:
                duplicates += 1
            else:
                hashes.add(h)

        self.stats = {
            "Sampled_Images":       len(images),
            "Avg_Brightness":       float(np.mean(brights)) if brights else 0,
            "Avg_Blur_Variance":    float(np.mean(blurs))   if blurs   else 0,
            "Duplicate_Percentage": (duplicates / len(images)) * 100 if images else 0,
        }
        self.save_stats("image_quality_stats.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Segment Analyzer  (session mode only)
# ─────────────────────────────────────────────────────────────────────────────

class SegmentAnalyzer(BaseAnalyzer):
    """
    Per-segment breakdown analysis for the segment-based continuous dataset.

    Checks and plots:
    - Per-segment distance traveled
    - Per-segment area covered (unique 0.5m cells)
    - Position continuity at segment boundaries (dist between last/first rows)
    - sim_time continuity (should be strictly monotonic)
    - global_step continuity (should be 0, 1, 2, ... with no gaps)
    - Segment-by-segment coverage accumulation
    """

    def run(self, grid_res=0.5):
        print("Running SegmentAnalyzer...")
        if "_segment_name" not in self.df.columns:
            print("  No _segment_name column — not a session dataset. Skipping.")
            return

        seg_names = sorted(self.df["_segment_name"].unique())
        n_segs    = len(seg_names)

        seg_distances   = []
        seg_unique_cells = []
        seg_step_counts  = []
        boundary_dists   = []
        sim_time_gaps    = []
        global_step_gaps = []

        prev_last_row = None

        for seg in seg_names:
            sdf = self.df[self.df["_segment_name"] == seg].copy()
            seg_step_counts.append(len(sdf))

            if 'pos_x' in sdf.columns and 'pos_y' in sdf.columns:
                dx = np.diff(sdf['pos_x'].values)
                dy = np.diff(sdf['pos_y'].values)
                seg_distances.append(float(np.sum(np.hypot(dx, dy))))

                gx = np.floor(sdf['pos_x'].values / grid_res).astype(int)
                gy = np.floor(sdf['pos_y'].values / grid_res).astype(int)
                seg_unique_cells.append(len(set(zip(gx, gy))))
            else:
                seg_distances.append(0.0)
                seg_unique_cells.append(0)

            if prev_last_row is not None:
                first_row = sdf.iloc[0]
                # Position boundary distance
                if 'pos_x' in sdf.columns:
                    d = math.hypot(
                        float(first_row['pos_x']) - float(prev_last_row['pos_x']),
                        float(first_row['pos_y']) - float(prev_last_row['pos_y'])
                    )
                    boundary_dists.append(d)
                # sim_time gap
                if 'timestamp' in sdf.columns:
                    gap = float(first_row['timestamp']) - float(prev_last_row['timestamp'])
                    sim_time_gaps.append(gap)
                # global_step gap
                if 'global_step' in sdf.columns:
                    gap_g = int(first_row['global_step']) - int(prev_last_row['global_step'])
                    global_step_gaps.append(gap_g)

            prev_last_row = sdf.iloc[-1]

        # ── Plot 1: per-segment distance ──────────────────────────────────────
        fig, ax = plt.subplots(figsize=(max(8, n_segs // 4), 4))
        ax.bar(range(n_segs), seg_distances, width=0.8, color='steelblue')
        ax.set_title("Distance Traveled per Segment")
        ax.set_xlabel("Segment Index")
        ax.set_ylabel("Distance (m)")
        ax.grid(axis='y')
        self.save_fig(fig, "segment_distances.png")

        # ── Plot 2: per-segment unique cells ──────────────────────────────────
        fig, ax = plt.subplots(figsize=(max(8, n_segs // 4), 4))
        ax.bar(range(n_segs), seg_unique_cells, width=0.8, color='darkorange')
        ax.set_title("Unique Cells Visited per Segment")
        ax.set_xlabel("Segment Index")
        ax.set_ylabel("Unique 0.5m Cells")
        ax.grid(axis='y')
        self.save_fig(fig, "segment_unique_cells.png")

        # ── Plot 3: cumulative unique cells (exploration progress) ────────────
        cumulative_cells = []
        seen_all = set()
        for seg in seg_names:
            sdf = self.df[self.df["_segment_name"] == seg]
            if 'pos_x' in sdf.columns:
                gx = np.floor(sdf['pos_x'].values / grid_res).astype(int)
                gy = np.floor(sdf['pos_y'].values / grid_res).astype(int)
                seen_all.update(zip(gx, gy))
            cumulative_cells.append(len(seen_all))

        fig, ax = plt.subplots(figsize=(max(8, n_segs // 4), 4))
        ax.plot(range(n_segs), cumulative_cells, marker='o', markersize=3, linewidth=1.5)
        ax.set_title("Cumulative Coverage vs Segment Index")
        ax.set_xlabel("Segment Index")
        ax.set_ylabel("Total Unique 0.5m Cells")
        ax.grid(True)
        self.save_fig(fig, "segment_cumulative_coverage.png")

        # ── Plot 4: boundary position distances ───────────────────────────────
        if boundary_dists:
            fig, ax = plt.subplots(figsize=(max(8, n_segs // 4), 4))
            ax.bar(range(len(boundary_dists)), boundary_dists, width=0.8, color='crimson')
            ax.axhline(y=1.0, color='k', linestyle='--', linewidth=1,
                       label="1m (suspicious reset threshold)")
            ax.set_title("Position Jump at Segment Boundaries\n(should be < 1m for continuous exploration)")
            ax.set_xlabel("Boundary Index (seg N → N+1)")
            ax.set_ylabel("Distance (m)")
            ax.legend()
            ax.grid(axis='y')
            self.save_fig(fig, "segment_boundary_distances.png")

        # ── Plot 5: sim_time gap at boundaries ────────────────────────────────
        if sim_time_gaps:
            expected_gap = 1.0 / 10.0  # 0.1s at 10 Hz
            fig, ax = plt.subplots(figsize=(max(8, n_segs // 4), 4))
            ax.bar(range(len(sim_time_gaps)), sim_time_gaps, width=0.8, color='mediumseagreen')
            ax.axhline(y=expected_gap, color='k', linestyle='--', linewidth=1,
                       label=f"Expected {expected_gap:.3f}s")
            ax.set_title("sim_time Gap at Segment Boundaries  (should ≈ 0.1s)")
            ax.set_xlabel("Boundary Index")
            ax.set_ylabel("Δ sim_time (s)")
            ax.legend()
            ax.grid(axis='y')
            self.save_fig(fig, "segment_simtime_gaps.png")

        # ── Stats ─────────────────────────────────────────────────────────────
        n_resets = sum(1 for d in boundary_dists if d > 1.0)

        self.stats = {
            "Num_Segments":                n_segs,
            "Total_Steps":                 int(sum(seg_step_counts)),
            "Avg_Steps_Per_Segment":       float(np.mean(seg_step_counts)),
            "Avg_Distance_Per_Segment_m":  float(np.mean(seg_distances)),
            "Avg_Unique_Cells_Per_Segment": float(np.mean(seg_unique_cells)),
            "Final_Cumulative_Cells":      int(cumulative_cells[-1]) if cumulative_cells else 0,
            "Suspicious_Resets_Detected":  n_resets,
            "Max_Boundary_Jump_m":         float(max(boundary_dists)) if boundary_dists else 0.0,
            "Avg_Boundary_Jump_m":         float(np.mean(boundary_dists)) if boundary_dists else 0.0,
            "global_step_gaps_all_1":      all(g == 1 for g in global_step_gaps),
        }

        # Save per-segment detail table
        seg_detail = pd.DataFrame({
            "segment":      seg_names,
            "steps":        seg_step_counts,
            "distance_m":   seg_distances,
            "unique_cells": seg_unique_cells,
        })
        seg_detail.to_csv(self.stats_dir / "segment_detail.csv", index=False)

        self.save_stats("segment_stats.csv")
        print(
            f"  Segments: {n_segs} | "
            f"Total steps: {sum(seg_step_counts):,} | "
            f"Final coverage: {cumulative_cells[-1] if cumulative_cells else 0} cells | "
            f"Resets detected: {n_resets}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Report Generator
# ─────────────────────────────────────────────────────────────────────────────

class ReportGenerator:
    def __init__(self, out_dir: Path, all_stats: dict, session_meta: dict = None):
        self.out_dir      = out_dir
        self.all_stats    = all_stats
        self.session_meta = session_meta or {}

    def run(self):
        cov      = self.all_stats.get('Unique_Cells_Visited', 0)
        dup      = self.all_stats.get('Duplicate_Percentage', 0)
        loop_pct = self.all_stats.get('Percent_Time_Looping', 0)
        n_resets = self.all_stats.get('Suspicious_Resets_Detected', 0)

        if cov > 40 and loop_pct < 10 and dup < 5 and n_resets == 0:
            rec = "READY FOR RL"
        elif cov > 20 and loop_pct < 20:
            rec = "READY WITH MINOR IMPROVEMENTS"
        else:
            rec = "COLLECT MORE DATA"

        # Session meta section
        meta_md = ""
        if self.session_meta:
            meta_md = "\n## Session Metadata\n"
            for k, v in self.session_meta.items():
                if k != "segments":   # skip huge list
                    meta_md += f"- **{k}**: {v}\n"

        md = f"""# Dataset Analysis Report

## Overall Recommendation: {rec}

### Justification
- Unique Coverage: {cov} cells
- Looping Percentage: {loop_pct:.1f}%
- Duplicate Images: {dup:.1f}%
- Suspicious Resets at Boundaries: {n_resets}
{meta_md}
## Metrics Summary
"""
        for k, v in self.all_stats.items():
            if isinstance(v, float):
                md += f"- **{k}**: {v:.4f}\n"
            else:
                md += f"- **{k}**: {v}\n"

        report_path = self.out_dir / "report.md"
        report_path.write_text(md)
        print(f"\nSaved Report to {report_path}")

        summary_path = self.out_dir.parent / "summary.csv"
        df_new = pd.DataFrame([self.all_stats])
        if summary_path.exists():
            df_new.to_csv(summary_path, mode='a', header=False, index=False)
        else:
            df_new.to_csv(summary_path, index=False)
        print(f"Appended to {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RL Dataset Analysis & Validation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv",     type=str, help="Path to a single flat log.csv (legacy mode)")
    group.add_argument("--session", type=str, help="Path to session directory containing segment_NNN/ subdirs (new mode)")
    parser.add_argument("--images", type=str, default=None,
                        help="Path to an image directory for quality analysis (optional)")
    parser.add_argument("--outdir", type=str, default="analysis_output",
                        help="Output directory for plots, stats, report")
    args = parser.parse_args()

    session_meta = {}

    if args.session:
        # ── Session (segment-based) mode ──────────────────────────────────────
        session_dir = Path(args.session)
        if not session_dir.exists():
            print(f"Error: session directory {session_dir} does not exist.")
            return
        df, session_meta = load_session(session_dir)
        # Default outdir inside the session dir if not overridden
        out_dir = Path(args.outdir) if args.outdir != "analysis_output" \
                  else session_dir / "analysis_output"
        # Default image dir: first segment's rgb/ if not specified
        if args.images:
            image_dir = Path(args.images)
        else:
            first_seg = sorted([d for d in session_dir.iterdir()
                                 if d.is_dir() and d.name.startswith("segment_")])
            image_dir = (first_seg[0] / "rgb") if first_seg else None

    else:
        # ── Legacy (flat CSV) mode ────────────────────────────────────────────
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"Error: {csv_path} does not exist.")
            return
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        out_dir   = Path(args.outdir)
        image_dir = Path(args.images) if args.images else None

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output dir: {out_dir}")
    print(f"  Rows loaded: {len(df):,}")

    # ── Build analyzer list ───────────────────────────────────────────────────
    analyzers = [
        TrajectoryAnalyzer(df, out_dir, image_dir),
        OccupancyAnalyzer(df, out_dir, image_dir),
        MotionAnalyzer(df, out_dir, image_dir),
        PrimitiveAnalyzer(df, out_dir, image_dir),
        SafetyAnalyzer(df, out_dir, image_dir),
        LoopAnalyzer(df, out_dir, image_dir),
        InformationAnalyzer(df, out_dir, image_dir),
        ImageQualityAnalyzer(df, out_dir, image_dir),
    ]

    # Add SegmentAnalyzer only in session mode
    if args.session:
        analyzers.append(SegmentAnalyzer(df, out_dir, image_dir))

    all_stats = {}
    for analyzer in analyzers:
        try:
            analyzer.run()
            all_stats.update(analyzer.stats)
        except Exception as e:
            print(f"  ERROR in {analyzer.__class__.__name__}: {e}")
            import traceback; traceback.print_exc()

    rg = ReportGenerator(out_dir, all_stats, session_meta)
    rg.run()

    print("\nDone. Output layout:")
    print(f"  {out_dir}/plots/        — trajectory, heatmap, motion histograms, segment charts")
    print(f"  {out_dir}/statistics/   — CSV stats tables + segment_detail.csv")
    print(f"  {out_dir}/report.md     — overall recommendation")


if __name__ == "__main__":
    main()
