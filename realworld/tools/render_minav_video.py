#!/usr/bin/env python3
"""
realworld/tools/render_minav_video.py — Render MINav Presentation Video with Live HUD.
=====================================================================================
Generates an MP4 video file featuring the exact visual HUD layout from the MINav paper:
  - Main third-person / onboard camera feed
  - Dual Picture-in-Picture: Real-time Frame & Goal Image
  - Live dynamic Cosine Similarity graph with 0.8 success threshold line
  - Animated "Success!" and "Switching to next goal" splash overlays
  - Estimated distance and goal banners

Usage:
  python realworld/tools/render_minav_video.py --output output-realworld/minav_demo.mp4
"""

import sys
import argparse
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

VIEWER_DIR = ROOT / "realworld" / "viewer"
DATA_DIR = ROOT / "run_full_main" / "run_full_main" / "exploration" / "segment_000"
PHI_PATH = ROOT / "output-realworld" / "processed" / "dinov3" / "phi_vectors.npy"


def parse_args():
    parser = argparse.ArgumentParser(description="Render MINav Paper-Style Presentation Video")
    parser.add_argument("--output", type=str, default="output-realworld/minav_demo.mp4", help="Output MP4 path")
    parser.add_argument("--fps", type=float, default=15.0, help="Output video FPS")
    parser.add_argument("--width", type=int, default=1280, help="Video width")
    parser.add_argument("--height", type=int, default=720, help="Video height")
    parser.add_argument("--max-frames", type=int, default=300, help="Number of frames to render")
    return parser.parse_args()


def draw_hud(frame, current_idx, total_frames, current_sim, sim_history, goal_img, goal_name, est_dist, is_success):
    h, w, _ = frame.shape
    overlay = frame.copy()

    # 1. Top Header & Goal Notification Banner
    cv2.putText(overlay, "MINav demo", (30, 42), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, "Ranger Mini V3 Offline RL", (32, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    banner_w, banner_h = 380, 52
    bx = (w - banner_w) // 2
    by = 18
    cv2.rectangle(overlay, (bx, by), (bx + banner_w, by + banner_h), (15, 20, 32), -1)
    cv2.rectangle(overlay, (bx, by), (bx + banner_w, by + banner_h), (80, 80, 80), 1)
    cv2.putText(overlay, f"Navigating to {goal_name}", (bx + 20, by + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, f"Estimated distance: {est_dist:.1f} m.", (bx + 20, by + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 100, 240), 1, cv2.LINE_AA)

    # 2. Dual Picture-in-Picture (Top Right)
    pip_w, pip_h = 240, 140
    pip_x = w - pip_w - 24
    
    # PIP 1: Real-time Frame
    pip_y1 = 20
    pip_rt = cv2.resize(frame, (pip_w, pip_h))
    overlay[pip_y1:pip_y1 + pip_h, pip_x:pip_x + pip_w] = pip_rt
    cv2.rectangle(overlay, (pip_x, pip_y1), (pip_x + pip_w, pip_y1 + pip_h), (255, 255, 255), 2)
    cv2.rectangle(overlay, (pip_x, pip_y1 + pip_h - 26), (pip_x + pip_w, pip_y1 + pip_h), (240, 240, 240), -1)
    cv2.putText(overlay, "Real-time Frame", (pip_x + 8, pip_y1 + pip_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)

    # PIP 2: Goal Image
    pip_y2 = pip_y1 + pip_h + 14
    if goal_img is not None:
        pip_g = cv2.resize(goal_img, (pip_w, pip_h))
        overlay[pip_y2:pip_y2 + pip_h, pip_x:pip_x + pip_w] = pip_g
    cv2.rectangle(overlay, (pip_x, pip_y2), (pip_x + pip_w, pip_y2 + pip_h), (255, 255, 255), 2)
    cv2.rectangle(overlay, (pip_x, pip_y2 + pip_h - 26), (pip_x + pip_w, pip_y2 + pip_h), (240, 240, 240), -1)
    cv2.putText(overlay, "Goal Image", (pip_x + 8, pip_y2 + pip_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)

    # 3. Success Splash Banner
    if is_success:
        cv2.putText(overlay, "Success!", (w // 2 - 160, h // 2 + 10), cv2.FONT_HERSHEY_DUPLEX, 2.2, (20, 20, 240), 6, cv2.LINE_AA)
        cv2.putText(overlay, "Success!", (w // 2 - 160, h // 2 + 10), cv2.FONT_HERSHEY_DUPLEX, 2.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, "Switching to the next goal", (w // 2 - 180, h // 2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    # 4. Bottom Cosine Similarity Live Graph
    gh = 130
    gy = h - gh
    # Semi-transparent dark background band
    graph_bg = overlay[gy:h, 0:w].copy()
    cv2.rectangle(graph_bg, (0, 0), (w, gh), (12, 16, 26), -1)
    cv2.addWeighted(graph_bg, 0.85, overlay[gy:h, 0:w], 0.15, 0, overlay[gy:h, 0:w])

    # Y-axis Labels
    cv2.putText(overlay, "Similarity", (24, gy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(overlay, "1.0", (32, gy + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(overlay, "0.8", (32, gy + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 240), 1, cv2.LINE_AA)
    cv2.putText(overlay, "0.5", (32, gy + 88), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(overlay, "0.0", (32, gy + 118), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

    gx_start = 80
    gx_end = w - 40
    gw = gx_end - gx_start

    # Reference grid lines
    def sim_to_y(s):
        s_clamped = max(0.0, min(1.0, s))
        return int(gy + 28 + (1.0 - s_clamped) * 85)

    # 0.8 Red Threshold Line
    y_thresh = sim_to_y(0.80)
    cv2.line(overlay, (gx_start, y_thresh), (gx_end, y_thresh), (70, 70, 240), 2, cv2.LINE_AA)

    # Draw Yellow Trajectory Line
    pts = []
    for i, s in enumerate(sim_history):
        px = int(gx_start + (i / max(1, total_frames - 1)) * gw)
        py = sim_to_y(s)
        pts.append((px, py))

    if len(pts) > 1:
        for i in range(len(pts) - 1):
            cv2.line(overlay, pts[i], pts[i + 1], (20, 220, 250), 3, cv2.LINE_AA)

    # Leading Red Dot on Playhead
    if len(pts) > 0:
        cur_pt = pts[-1]
        cv2.circle(overlay, cur_pt, 6, (40, 40, 250), -1)
        cv2.circle(overlay, cur_pt, 6, (255, 255, 255), 2)

    # Graph Legend at bottom
    cv2.line(overlay, (gx_start, h - 10), (gx_start + 25, h - 10), (20, 220, 250), 2)
    cv2.putText(overlay, "Yellow line: Cosine similarity between current frame and goal", (gx_start + 32, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 220, 250), 1, cv2.LINE_AA)

    cv2.line(overlay, (gx_start + 500, h - 10), (gx_start + 525, h - 10), (70, 70, 240), 2)
    cv2.putText(overlay, "Red line: Predefined success threshold (0.8)", (gx_start + 532, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 70, 240), 1, cv2.LINE_AA)

    return overlay


def main():
    args = parse_args()
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("MINAV VIDEO RENDERER")
    print("=" * 65)
    print(f"  Output video  : {output_path}")
    print(f"  Resolution    : {args.width}x{args.height} @ {args.fps} FPS")

    # Load Embeddings & Goal
    phi_vectors = np.load(PHI_PATH)
    goal_frame_idx = 480
    goal_phi = phi_vectors[goal_frame_idx]

    # Pre-calculate similarity trajectory for first max_frames
    start_frame = 350
    total_frames = min(args.max_frames, len(phi_vectors) - start_frame)
    sims = (phi_vectors[start_frame:start_frame + total_frames] @ goal_phi).clip(-1.0, 1.0)

    # Load Goal Image
    rgb_dir = DATA_DIR / "rgb"
    goal_img_path = rgb_dir / f"{goal_frame_idx:06d}.jpg"
    goal_img = cv2.imread(str(goal_img_path)) if goal_img_path.exists() else None

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, args.fps, (args.width, args.height))

    sim_history = []
    print(f"Rendering {total_frames} frames...")

    for i in range(total_frames):
        cur_idx = start_frame + i
        cur_sim = float(sims[i])
        sim_history.append(cur_sim)

        img_path = rgb_dir / f"{cur_idx:06d}.jpg"
        if img_path.exists():
            frame = cv2.imread(str(img_path))
            frame = cv2.resize(frame, (args.width, args.height))
        else:
            frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)

        is_success = cur_sim >= 0.80
        est_dist = max(0.0, (1.0 - cur_sim) * 12.0)

        rendered = draw_hud(
            frame=frame,
            current_idx=i,
            total_frames=total_frames,
            current_sim=cur_sim,
            sim_history=sim_history,
            goal_img=goal_img,
            goal_name="Goal 2 (Atrium Doors)",
            est_dist=est_dist,
            is_success=is_success,
        )

        out.write(rendered)
        if i % 50 == 0 or i == total_frames - 1:
            print(f"  Progress: frame {i+1}/{total_frames} (sim={cur_sim:.3f})")

    out.release()
    print("=" * 65)
    print(f"[OK] Video successfully rendered to: {output_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
