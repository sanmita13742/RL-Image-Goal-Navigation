"""
ranger_mujoco/random_explore.py
============================================================
Runs the pink-uniform exploration policy in the MuJoCo test environment
and logs the dataset with SEGMENT-BASED file organization.

The robot explores CONTINUOUSLY for the full session.
Segment directories are purely a file-management boundary —
the robot is NEVER teleported or reset at segment boundaries.

Session + Segment Output Layout
---------------------------------
    dataset/
        <YYYYMMDD_HHMMSS>/
            metadata.json           -- Session-level provenance
            segment_000/
                rgb/
                    000000.png      -- segment_step counter, starts at 0
                    000001.png
                    ...
                depth/
                    000000.png
                    ...
                segment.csv         -- Per-segment log, one row per step
            segment_001/
                ...

segment.csv columns
--------------------
    trajectory_id  -- Always 0; marks all segments as one continuous rollout
    global_step    -- Monotonically increasing across the entire session (0 … TOTAL_STEPS-1)
    segment_step   -- Local step within this segment file (0 … SEGMENT_SIZE-1)
    sim_time       -- MuJoCo simulation time (seconds, continuous, never reset)
    linear_vel_cmd -- Applied linear velocity command (m/s)
    lateral_vel_cmd-- Applied lateral velocity command (m/s)
    angular_vel_cmd-- Applied angular velocity command (rad/s)
    pos_x          -- Robot x position (m)
    pos_y          -- Robot y position (m)
    yaw            -- Robot yaw (rad)
    rgb_path       -- Relative path to RGB image: rgb/000000.png
    depth_path     -- Relative path to depth image: depth/000000.png

NOTE: terminated/truncated are intentionally absent.
Segment boundaries are NOT RL episode boundaries. The robot physically
continues from its current state across every segment transition.
Use global_step to reconstruct the full continuous sequence.

Image/Action alignment
-----------------------
At each global_step t the logger records:
    observation at t  (render BEFORE physics step)
    action at t       (policy output based on obs_t)
The physics step is applied AFTER logging, producing state t+1.
This is Option A: (obs_t, action_t) per row.
Row t+1 holds obs_{t+1}.

Cross-segment continuity:
    last row of segment_000: global_step = SEGMENT_SIZE - 1
    first row of segment_001: global_step = SEGMENT_SIZE
    robot physically continued — NO teleport between them.

Restart/overwrite safety
-------------------------
Session directory is timestamped to second precision.
If it already exists, the script aborts. Existing segment
directories are never overwritten.

Run:  python random_explore.py
"""

import sys
import csv
import json
import time
import math
import numpy as np
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

try:
    from PIL import Image
except ImportError:
    print("FATAL: Pillow is required. pip install Pillow")
    sys.exit(1)

import mujoco
import mujoco.viewer

sys.path.insert(0, str(Path(__file__).parent.parent))

from robot import RangerMiniV3Robot
from robot_base import DriveCommand
from test_env import build_world_xml
from exploration_policies import PrimitiveExplorationPolicy


# ── Helpers ───────────────────────────────────────────────────────────────────

def euler_from_quaternion(w, x, y, z):
    """Quaternion → (roll, pitch, yaw) in radians."""
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = max(-1.0, min(1.0, t2))
    pitch_y = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return roll_x, pitch_y, yaw_z


def save_depth(arr: np.ndarray, path: str) -> None:
    """Normalize and save depth array as grayscale PNG."""
    d_min, d_max = arr.min(), arr.max()
    norm = ((arr - d_min) / (d_max - d_min) * 255).astype(np.uint8) \
           if d_max > d_min else np.zeros_like(arr, dtype=np.uint8)
    Image.fromarray(norm, mode="L").save(path)


def find_next_segment_id(session_dir: Path) -> int:
    """Return the next unused segment ID. Never overwrites existing segments."""
    idx = 0
    while (session_dir / f"segment_{idx:03d}").exists():
        idx += 1
    return idx


def open_segment(session_dir: Path, seg_id: int, save_images: bool):
    """
    Create segment directory, open its CSV, write header.
    Returns (seg_dir, rgb_dir, depth_dir, csv_file_handle, csv_writer).
    """
    seg_name  = f"segment_{seg_id:03d}"
    seg_dir   = session_dir / seg_name
    rgb_dir   = seg_dir / "rgb"
    depth_dir = seg_dir / "depth"
    rgb_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)

    csv_file = open(seg_dir / "segment.csv", mode="w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow([
        "trajectory_id",
        "global_step",
        "segment_step",
        "sim_time",
        "linear_vel_cmd",
        "lateral_vel_cmd",
        "angular_vel_cmd",
        "pos_x",
        "pos_y",
        "yaw",
        "rgb_path",
        "depth_path",
    ])
    return seg_dir, rgb_dir, depth_dir, csv_file, writer


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Configuration ────────────────────────────────────────────────────────
    CONTROL_FREQ  = 10.0      # Hz  — do not change
    SEGMENT_SIZE  = 500       # Steps per segment file
    TOTAL_STEPS   = 72_000    # 2 hours at 10 Hz
    BETA          = 1         # Noise exponent: 1 = Pink (MINav paper)
    BUFFER_SIZE   = 8192      # FFT buffer size per noise channel
    SHOW_VIEWER   = True      # Watch the robot live in 3D
    SAVE_IMAGES   = True      # Save RGB/Depth PNGs

    TRAJECTORY_ID = 0         # Always 0: entire session is one continuous rollout

    # ── Session directory ────────────────────────────────────────────────────
    session_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir    = Path(__file__).parent / "dataset"
    session_dir = base_dir / session_id

    if session_dir.exists():
        print(f"ERROR: Session directory already exists: {session_dir}")
        print("Wait one second and re-run, or delete the directory manually.")
        sys.exit(1)

    session_dir.mkdir(parents=True)

    # ── MuJoCo world ─────────────────────────────────────────────────────────
    world_xml = Path(__file__).parent / "_random_world.xml"
    world_xml.write_text(build_world_xml(), encoding="utf-8")

    robot = RangerMiniV3Robot()
    robot.load(world_xml)

    # Set initial pose and capture the model's natural z ground-clearance.
    # This is called ONCE before the loop — never again during collection.
    robot.data.qpos[0] = -4.5
    robot.data.qpos[1] = 0.0
    robot.data.qpos[3] = 0.7071068   # qw
    robot.data.qpos[4] = 0.0
    robot.data.qpos[5] = 0.0
    robot.data.qpos[6] = 0.7071068   # qz  → 90° yaw
    mujoco.mj_forward(robot.model, robot.data)

    try:
        world_xml.unlink()
    except Exception:
        pass

    # ── Renderers ─────────────────────────────────────────────────────────────
    if SAVE_IMAGES:
        renderer_rgb = mujoco.Renderer(robot.model, height=240, width=320)
    renderer_depth = mujoco.Renderer(robot.model, height=60, width=640)
    renderer_depth.enable_depth_rendering()

    sim_dt = robot.model.opt.timestep
    sim_steps_per_control = int(1.0 / (CONTROL_FREQ * sim_dt))

    num_segments = (TOTAL_STEPS + SEGMENT_SIZE - 1) // SEGMENT_SIZE

    print("============================================================")
    print("  Ranger Mini  --  Segment-Based Continuous Data Collection")
    print("============================================================")
    print(f"  Session ID     : {session_id}")
    print(f"  Logging to     : {session_dir}")
    print(f"  Control Freq   : {CONTROL_FREQ} Hz")
    print(f"  Total steps    : {TOTAL_STEPS}  (~{TOTAL_STEPS/CONTROL_FREQ/3600:.1f} hrs)")
    print(f"  Segment size   : {SEGMENT_SIZE} steps")
    print(f"  Num segments   : {num_segments}")
    print(f"  Noise          : FFT Pink Uniform (beta={BETA}, buf={BUFFER_SIZE})")
    print(f"  Robot resets   : NONE (continuous exploration)")
    print("============================================================")

    # ── Policy — constructed ONCE, never reset ────────────────────────────────
    # Pink-noise buffers, visit grid, loop detector all run for the full session.
    policy = PrimitiveExplorationPolicy(CONTROL_FREQ, beta=BETA)

    # ── Viewer ────────────────────────────────────────────────────────────────
    viewer = None
    if SHOW_VIEWER:
        viewer = mujoco.viewer.launch_passive(robot.model, robot.data)

    # ── Segment metadata accumulator ──────────────────────────────────────────
    segment_meta = []
    start_time   = time.time()

    # Start from the next unused segment ID (supports partial-run resume)
    seg_id = find_next_segment_id(session_dir)

    # Open the first segment
    seg_dir, rgb_dir, depth_dir, csv_file, writer = open_segment(
        session_dir, seg_id, SAVE_IMAGES
    )
    seg_step    = 0   # local counter within current segment
    seg_start_global = 0

    try:
        for global_step in range(TOTAL_STEPS):

            # ── Roll over to next segment ────────────────────────────────────
            # This is a pure FILE boundary. Robot is NOT touched.
            if seg_step == SEGMENT_SIZE:
                csv_file.close()
                n_imgs = len(list(rgb_dir.glob("*.png"))) if SAVE_IMAGES else SEGMENT_SIZE
                segment_meta.append({
                    "segment_id":   f"segment_{seg_id:03d}",
                    "global_start": seg_start_global,
                    "global_end":   global_step - 1,   # inclusive
                    "num_steps":    SEGMENT_SIZE,
                    "num_images":   n_imgs,
                })
                print(
                    f"  [SEG DONE] segment_{seg_id:03d} | "
                    f"global {seg_start_global}–{global_step-1} | "
                    f"{SEGMENT_SIZE} steps | {n_imgs} images"
                )

                seg_id          += 1
                seg_step         = 0
                seg_start_global = global_step

                seg_dir, rgb_dir, depth_dir, csv_file, writer = open_segment(
                    session_dir, seg_id, SAVE_IMAGES
                )
                # ── NO reset_robot() here ─────────────────────────────────────

            # ── 1. Render cameras (obs_t, BEFORE physics step) ───────────────
            if SAVE_IMAGES:
                renderer_rgb.update_scene(robot.data, camera="front_cam")
                rgb_img = renderer_rgb.render()

            renderer_depth.update_scene(robot.data, camera="lidar_cam")
            depth_img = renderer_depth.render()

            # ── 2. Extract pose ──────────────────────────────────────────────
            pos_x, pos_y, _ = robot.data.qpos[0:3]
            qw, qx, qy, qz  = robot.data.qpos[3:7]
            _, _, yaw = euler_from_quaternion(qw, qx, qy, qz)

            # ── 3. Action (EXISTING pink-uniform policy — zero changes) ──────
            cmd, prim = policy.get_action(depth_img, pos_x, pos_y, yaw)
            min_depth = float(np.min(depth_img))

            # ── 4. Save images ───────────────────────────────────────────────
            img_filename = f"{seg_step:06d}.png"
            if SAVE_IMAGES:
                Image.fromarray(rgb_img).save(rgb_dir / img_filename)
                save_depth(depth_img, str(depth_dir / img_filename))

            # ── 5. Log row ───────────────────────────────────────────────────
            sim_time = robot.data.time
            writer.writerow([
                TRAJECTORY_ID,
                global_step,
                seg_step,
                f"{sim_time:.3f}",
                f"{cmd.v_linear:.3f}",
                f"{cmd.v_lateral:.3f}",
                f"{cmd.v_angular:.3f}",
                f"{pos_x:.4f}",
                f"{pos_y:.4f}",
                f"{yaw:.4f}",
                f"rgb/{img_filename}",
                f"depth/{img_filename}",
            ])
            csv_file.flush()

            # ── 6. CLI feedback ───────────────────────────────────────────────
            if global_step % 100 == 0:
                elapsed = time.time() - start_time
                print(
                    f"  [g={global_step:06d} seg={seg_id:03d}/{seg_step:03d}] "
                    f"{prim.name} | v={cmd.v_linear:+.2f} w={cmd.v_angular:+.2f} | "
                    f"({pos_x:+.2f},{pos_y:+.2f}) yaw={yaw:+.2f} | "
                    f"minD={min_depth:.2f}m | {elapsed:.0f}s"
                )

            # ── 7. Physics step (action_t → produces state_{t+1}) ────────────
            for _ in range(sim_steps_per_control):
                robot.apply_command(cmd)
                robot.step()

            if viewer is not None:
                if not viewer.is_running():
                    print("Viewer closed. Stopping collection.")
                    break
                viewer.sync()

            seg_step += 1

    except KeyboardInterrupt:
        print("\nCollection interrupted by user.")

    finally:
        # Close the currently open segment (may be partial)
        try:
            csv_file.close()
        except Exception:
            pass

        if viewer is not None:
            viewer.close()

    # Record the final (possibly partial) segment
    n_imgs = len(list(rgb_dir.glob("*.png"))) if SAVE_IMAGES else seg_step
    segment_meta.append({
        "segment_id":   f"segment_{seg_id:03d}",
        "global_start": seg_start_global,
        "global_end":   seg_start_global + seg_step - 1,   # inclusive
        "num_steps":    seg_step,
        "num_images":   n_imgs,
    })

    total_recorded = sum(m["num_steps"] for m in segment_meta)

    # ── Write metadata.json ───────────────────────────────────────────────────
    session_meta = {
        "session_id":        session_id,
        "date":              datetime.now().isoformat(),
        "robot":             "Ranger Mini V3",
        "policy":            "PrimitiveExplorationPolicy",
        "noise":             "FFT Pink Uniform",
        "beta":              float(BETA),
        "buffer_size":       BUFFER_SIZE,
        "control_frequency": CONTROL_FREQ,
        "segment_size":      SEGMENT_SIZE,
        "total_steps_planned": TOTAL_STEPS,
        "total_steps_recorded": total_recorded,
        "num_segments":      len(segment_meta),
        "trajectory_id":     TRAJECTORY_ID,
        "robot_resets":      0,
        "simulator":         "MuJoCo",
        "segments":          segment_meta,
    }
    with open(session_dir / "metadata.json", "w", encoding="utf-8") as mf:
        json.dump(session_meta, mf, indent=4)

    elapsed = time.time() - start_time
    print("============================================================")
    print(f"  [DONE] Continuous collection complete!")
    print(f"         Session ID      : {session_id}")
    print(f"         Segments        : {len(segment_meta)}")
    print(f"         Steps recorded  : {total_recorded}")
    print(f"         Robot resets    : 0")
    print(f"         Time elapsed    : {elapsed:.1f} s")
    print(f"         Saved to        : {session_dir}")
    print("============================================================")


if __name__ == "__main__":
    main()
