"""
validate_trajectories.py
============================================================
Runs a SHORT test collection (configurable number of trajectories,
configurable steps per trajectory) and validates all structural
invariants of the trajectory-aware dataset.

Usage:
    python validate_trajectories.py

Checks:
    1.  Number of trajectory directories == number of episodes collected
    2.  Every trajectory has an rgb/ directory
    3.  Every trajectory has a trajectory.csv
    4.  Number of CSV rows == expected recorded timesteps
    5.  Every CSV image_path exists on disk
    6.  No image belongs to two trajectories (global uniqueness)
    7.  timestep starts at 0 for each trajectory
    8.  timestep is strictly monotonically increasing within each trajectory
    9.  No cross-trajectory transition exists in any single CSV
   10.  terminated/truncated correctly identifies episode boundaries
   11.  Action values are finite (not NaN/Inf)
   12.  Image resolution is consistent (240x320 RGB, 60x640 depth)
   13.  metadata.json exists and contains expected keys
"""

import sys
import os
import csv
import json
import time
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

# ── Test configuration ────────────────────────────────────────────────────────
TEST_NUM_TRAJECTORIES    = 4      # Short: 4 trajectories
TEST_STEPS_PER_EPISODE   = 30     # Short: 30 steps each
CONTROL_FREQ             = 10.0
BETA                     = 1
BUFFER_SIZE              = 8192
SAVE_IMAGES              = True

try:
    from PIL import Image
except ImportError:
    print("FATAL: Pillow required. pip install Pillow")
    sys.exit(1)

import mujoco

sys.path.insert(0, str(Path(__file__).parent.parent))
from robot import RangerMiniV3Robot
from robot_base import DriveCommand
from test_env import build_world_xml
from exploration_policies import PrimitiveExplorationPolicy

import math


def euler_from_quaternion(w, x, y, z):
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


def save_depth(arr, path):
    d_min, d_max = arr.min(), arr.max()
    if d_max > d_min:
        norm = ((arr - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(arr, dtype=np.uint8)
    Image.fromarray(norm, mode="L").save(path)


def reset_robot(robot, model, start_z: float):
    robot.data.qpos[0] = -4.5
    robot.data.qpos[1] = 0.0
    robot.data.qpos[2] = start_z   # model's natural ground-clearance height
    robot.data.qpos[3] = 0.7071068
    robot.data.qpos[4] = 0.0
    robot.data.qpos[5] = 0.0
    robot.data.qpos[6] = 0.7071068
    robot.data.qvel[:] = 0.0
    robot.data.ctrl[:] = 0.0
    mujoco.mj_forward(model, robot.data)


# ── Step 1: Run a short collection ───────────────────────────────────────────

def run_short_collection():
    session_id  = "VALIDATION_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir    = Path(__file__).parent / "dataset"
    session_dir = base_dir / session_id
    session_dir.mkdir(parents=True)

    print(f"\n{'='*60}")
    print(f"  STEP 1: Short collection ({TEST_NUM_TRAJECTORIES} trajectories × {TEST_STEPS_PER_EPISODE} steps)")
    print(f"  Output: {session_dir}")
    print(f"{'='*60}")

    world_xml = Path(__file__).parent / "_val_world.xml"
    world_xml.write_text(build_world_xml(), encoding="utf-8")
    robot = RangerMiniV3Robot()
    robot.load(world_xml)
    # Capture the model's natural ground-clearance z after first mj_forward
    robot.data.qpos[0] = -4.5
    robot.data.qpos[1] = 0.0
    robot.data.qpos[3] = 0.7071068
    robot.data.qpos[4] = 0.0
    robot.data.qpos[5] = 0.0
    robot.data.qpos[6] = 0.7071068
    mujoco.mj_forward(robot.model, robot.data)
    START_Z = float(robot.data.qpos[2])
    try:
        world_xml.unlink()
    except Exception:
        pass

    if SAVE_IMAGES:
        renderer_rgb = mujoco.Renderer(robot.model, height=240, width=320)
    renderer_depth = mujoco.Renderer(robot.model, height=60, width=640)
    renderer_depth.enable_depth_rendering()

    sim_dt = robot.model.opt.timestep
    sim_steps_per_control = int(1.0 / (CONTROL_FREQ * sim_dt))

    policy = PrimitiveExplorationPolicy(CONTROL_FREQ, beta=BETA)

    trajectory_meta = []
    total_steps = 0

    for traj_num in range(TEST_NUM_TRAJECTORIES):
        traj_name = f"trajectory_{traj_num:03d}"
        traj_dir  = session_dir / traj_name
        rgb_dir   = traj_dir / "rgb"
        depth_dir = traj_dir / "depth"
        rgb_dir.mkdir(parents=True)
        depth_dir.mkdir(parents=True)

        reset_robot(robot, robot.model, START_Z)

        csv_path = traj_dir / "trajectory.csv"
        with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                "timestep", "sim_time",
                "linear_vel_cmd", "lateral_vel_cmd", "angular_vel_cmd",
                "pos_x", "pos_y", "yaw",
                "rgb_path", "depth_path",
                "terminated", "truncated",
            ])

            for step in range(TEST_STEPS_PER_EPISODE):
                is_last_step = (step == TEST_STEPS_PER_EPISODE - 1)

                if SAVE_IMAGES:
                    renderer_rgb.update_scene(robot.data, camera="front_cam")
                    rgb_img = renderer_rgb.render()

                renderer_depth.update_scene(robot.data, camera="lidar_cam")
                depth_img = renderer_depth.render()

                pos_x, pos_y, pos_z = robot.data.qpos[0:3]
                qw, qx, qy, qz      = robot.data.qpos[3:7]
                _, _, yaw = euler_from_quaternion(qw, qx, qy, qz)

                cmd, prim = policy.get_action(depth_img, pos_x, pos_y, yaw)

                img_filename = f"{step:06d}.png"
                if SAVE_IMAGES:
                    Image.fromarray(rgb_img).save(rgb_dir / img_filename)
                    save_depth(depth_img, str(depth_dir / img_filename))

                sim_time = robot.data.time
                terminated = False
                truncated  = is_last_step

                writer.writerow([
                    step,
                    f"{sim_time:.3f}",
                    f"{cmd.v_linear:.3f}",
                    f"{cmd.v_lateral:.3f}",
                    f"{cmd.v_angular:.3f}",
                    f"{pos_x:.4f}",
                    f"{pos_y:.4f}",
                    f"{yaw:.4f}",
                    f"rgb/{img_filename}",
                    f"depth/{img_filename}",
                    terminated,
                    truncated,
                ])

                for _ in range(sim_steps_per_control):
                    robot.apply_command(cmd)
                    robot.step()

        num_images = len(list(rgb_dir.glob("*.png"))) if SAVE_IMAGES else TEST_STEPS_PER_EPISODE
        trajectory_meta.append({
            "trajectory_id": traj_name,
            "num_steps":     TEST_STEPS_PER_EPISODE,
            "num_images":    num_images,
            "terminated":    False,
            "truncated":     True,
        })
        total_steps += TEST_STEPS_PER_EPISODE
        print(f"  Collected {traj_name}: {TEST_STEPS_PER_EPISODE} steps")

    session_meta = {
        "session_id":            session_id,
        "date":                  datetime.now().isoformat(),
        "robot":                 "Ranger Mini V3",
        "policy":                "PrimitiveExplorationPolicy",
        "noise":                 "FFT Pink Uniform",
        "beta":                  float(BETA),
        "buffer_size":           BUFFER_SIZE,
        "control_frequency":     CONTROL_FREQ,
        "max_steps_per_episode": TEST_STEPS_PER_EPISODE,
        "num_trajectories":      len(trajectory_meta),
        "total_steps":           total_steps,
        "random_seed":           None,
        "simulator":             "MuJoCo",
        "trajectories":          trajectory_meta,
    }
    with open(session_dir / "metadata.json", "w", encoding="utf-8") as mf:
        json.dump(session_meta, mf, indent=4)

    print(f"\n  Collection done. Validating...")
    return session_dir


# ── Step 2: Validate ──────────────────────────────────────────────────────────

def validate(session_dir: Path):
    print(f"\n{'='*60}")
    print(f"  STEP 2: Validation of {session_dir.name}")
    print(f"{'='*60}")

    failures = []
    all_image_paths = set()  # For global uniqueness check (Check 6)

    # ── Check 13: metadata.json ───────────────────────────────────────────────
    meta_path = session_dir / "metadata.json"
    if not meta_path.exists():
        failures.append("metadata.json does not exist")
        meta = {}
    else:
        with open(meta_path) as f:
            meta = json.load(f)
        required_keys = ["session_id", "num_trajectories", "total_steps", "trajectories"]
        for k in required_keys:
            if k not in meta:
                failures.append(f"metadata.json missing key: {k}")

    # ── Check 1: trajectory directory count ──────────────────────────────────
    traj_dirs = sorted([d for d in session_dir.iterdir()
                        if d.is_dir() and d.name.startswith("trajectory_")])
    expected_count = TEST_NUM_TRAJECTORIES

    if len(traj_dirs) != expected_count:
        failures.append(
            f"Check 1 FAIL: expected {expected_count} trajectory dirs, "
            f"found {len(traj_dirs)}"
        )
    else:
        print(f"  [OK] Check  1: {len(traj_dirs)} trajectory directories found")

    # ── Per-trajectory checks ─────────────────────────────────────────────────
    traj_report = []

    for traj_dir in traj_dirs:
        traj_name = traj_dir.name
        row_errors = []

        # Check 2: rgb/ exists
        if not (traj_dir / "rgb").exists():
            row_errors.append("rgb/ directory missing")

        # Check 3: trajectory.csv exists
        csv_path = traj_dir / "trajectory.csv"
        if not csv_path.exists():
            row_errors.append("trajectory.csv missing")
            traj_report.append((traj_name, 0, 0, False, False, row_errors))
            continue

        with open(csv_path, newline="", encoding="utf-8") as cf:
            rows = list(csv.DictReader(cf))

        num_rows = len(rows)

        # Check 4: row count == expected
        if num_rows != TEST_STEPS_PER_EPISODE:
            row_errors.append(
                f"Check 4 FAIL: expected {TEST_STEPS_PER_EPISODE} rows, got {num_rows}"
            )

        # Per-row checks
        prev_timestep = -1
        for i, row in enumerate(rows):
            ts = int(row["timestep"])

            # Check 7: starts at 0
            if i == 0 and ts != 0:
                row_errors.append(f"Check 7 FAIL: first timestep is {ts}, expected 0")

            # Check 8: monotonically increasing
            if ts <= prev_timestep and i > 0:
                row_errors.append(
                    f"Check 8 FAIL: timestep {ts} at row {i} is not > {prev_timestep}"
                )
            prev_timestep = ts

            # Check 5: image path exists
            rgb_rel  = row["rgb_path"]
            depth_rel = row["depth_path"]
            rgb_full  = traj_dir / rgb_rel
            depth_full = traj_dir / depth_rel

            if SAVE_IMAGES and not rgb_full.exists():
                row_errors.append(f"Check 5 FAIL: {rgb_full} not found (row {i})")
            if SAVE_IMAGES and not depth_full.exists():
                row_errors.append(f"Check 5 FAIL: {depth_full} not found (row {i})")

            # Check 6: global uniqueness — track absolute paths
            abs_rgb = str(rgb_full.resolve())
            if abs_rgb in all_image_paths:
                row_errors.append(f"Check 6 FAIL: duplicate image path {abs_rgb}")
            all_image_paths.add(abs_rgb)

            # Check 10: terminated/truncated correctness
            is_last = (i == num_rows - 1)
            terminated = row["terminated"] in ("True", "true", "1")
            truncated  = row["truncated"]  in ("True", "true", "1")

            if is_last and not truncated and not terminated:
                row_errors.append(
                    f"Check 10 FAIL: last row (step {ts}) has terminated=False truncated=False"
                )
            if not is_last and (terminated or truncated):
                row_errors.append(
                    f"Check 10 FAIL: non-last row (step {ts}) has terminated={terminated} truncated={truncated}"
                )

            # Check 11: finite action values
            for col in ("linear_vel_cmd", "lateral_vel_cmd", "angular_vel_cmd"):
                val = float(row[col])
                if not math.isfinite(val):
                    row_errors.append(f"Check 11 FAIL: {col}={val} at row {i}")

        # Check 9: no cross-trajectory transitions (all timesteps within 1 CSV)
        # Guaranteed by structure, but verify that timestep never resets mid-file
        tss = [int(r["timestep"]) for r in rows]
        for i in range(1, len(tss)):
            if tss[i] <= tss[i-1]:
                row_errors.append(
                    f"Check 9 FAIL: timestep regression at row {i}: "
                    f"{tss[i-1]} -> {tss[i]}"
                )

        # Check 12: image resolution
        if SAVE_IMAGES:
            rgb_imgs  = sorted((traj_dir / "rgb").glob("*.png"))
            depth_imgs = sorted((traj_dir / "depth").glob("*.png"))

            if rgb_imgs:
                img = Image.open(rgb_imgs[0])
                if img.size != (320, 240):
                    row_errors.append(
                        f"Check 12 FAIL: RGB resolution {img.size}, expected (320, 240)"
                    )
                else:
                    pass  # OK

            if depth_imgs:
                img = Image.open(depth_imgs[0])
                if img.size != (640, 60):
                    row_errors.append(
                        f"Check 12 FAIL: Depth resolution {img.size}, expected (640, 60)"
                    )

        n_imgs = len(list((traj_dir / "rgb").glob("*.png"))) if SAVE_IMAGES else num_rows
        final_row = rows[-1] if rows else {}
        terminated_final = final_row.get("terminated", "False") in ("True", "true", "1")
        truncated_final  = final_row.get("truncated", "True")   in ("True", "true", "1")

        traj_report.append((traj_name, num_rows, n_imgs, terminated_final, truncated_final, row_errors))

    # ── Check 2+3 summary ────────────────────────────────────────────────────
    for traj_name, num_rows, n_imgs, terminated, truncated, errs in traj_report:
        status = "OK" if not errs else "FAIL"
        print(
            f"  [{status:4s}] {traj_name}: {num_rows} steps, {n_imgs} images, "
            f"terminated={terminated}, truncated={truncated}"
        )
        for e in errs:
            print(f"         ↳ {e}")
            failures.append(f"{traj_name}: {e}")

    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if not failures:
        print("  ALL CHECKS PASSED ✓")
    else:
        print(f"  {len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print(f"    ✗ {f}")
    print(f"{'='*60}\n")

    return len(failures) == 0


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    session_dir = run_short_collection()
    ok = validate(session_dir)
    sys.exit(0 if ok else 1)
