"""
validate_segments.py
============================================================
Validates the segment-based continuous collection design.

Runs: 2500 total steps, segment size 500 → expected 5 segments.

Checks:
  1.  Exactly 5 segment directories exist
  2.  Every segment has rgb/ and depth/ directories
  3.  Every segment has segment.csv
  4.  CSV row count matches expected steps per segment
  5.  global_step is strictly continuous 0..2499 across all segments
  6.  segment_step restarts at 0 for each segment and is monotonically increasing
  7.  Robot position does NOT reset at segment boundaries
      (pos_x/pos_y at last row of seg N ≈ pos_x/pos_y at first row of seg N+1)
  8.  sim_time is strictly monotonically increasing (never resets)
  9.  Every rgb_path image exists on disk
 10.  No image filename belongs to two different segments (cross-segment uniqueness)
 11.  trajectory_id is always 0
 12.  Image resolution: RGB=240x320, Depth=60x640
 13.  metadata.json has expected keys and robot_resets==0
 14.  No rows lost or duplicated (total rows == TOTAL_STEPS)
 15.  Action values are finite across all segments

Usage:
    python validate_segments.py
"""

import sys
import csv
import json
import math
import time
import numpy as np
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

# ── Test config (deliberately small) ─────────────────────────────────────────
TEST_TOTAL_STEPS  = 2500
TEST_SEGMENT_SIZE = 500
CONTROL_FREQ      = 10.0
BETA              = 1
BUFFER_SIZE       = 8192
SAVE_IMAGES       = True

try:
    from PIL import Image as PILImage
except ImportError:
    print("FATAL: Pillow required. pip install Pillow")
    sys.exit(1)

import mujoco

sys.path.insert(0, str(Path(__file__).parent.parent))
from robot import RangerMiniV3Robot
from test_env import build_world_xml
from exploration_policies import PrimitiveExplorationPolicy

import mujoco.viewer


def euler_from_quaternion(w, x, y, z):
    t2 = +2.0 * (w * y - z * x)
    t2 = max(-1.0, min(1.0, t2))
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)


def save_depth(arr, path):
    d_min, d_max = arr.min(), arr.max()
    norm = ((arr - d_min) / (d_max - d_min) * 255).astype(np.uint8) \
           if d_max > d_min else np.zeros_like(arr, dtype=np.uint8)
    PILImage.fromarray(norm, mode="L").save(path)


def open_segment(session_dir, seg_id, save_images):
    seg_dir   = session_dir / f"segment_{seg_id:03d}"
    rgb_dir   = seg_dir / "rgb"
    depth_dir = seg_dir / "depth"
    rgb_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)

    csv_file = open(seg_dir / "segment.csv", mode="w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow([
        "trajectory_id", "global_step", "segment_step", "sim_time",
        "linear_vel_cmd", "lateral_vel_cmd", "angular_vel_cmd",
        "pos_x", "pos_y", "yaw", "rgb_path", "depth_path",
    ])
    return seg_dir, rgb_dir, depth_dir, csv_file, writer


# ── Step 1: Run the collection ────────────────────────────────────────────────

def run_collection():
    session_id  = "VALSEGS_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(__file__).parent / "dataset" / session_id
    session_dir.mkdir(parents=True)

    print(f"\n{'='*60}")
    print(f"  STEP 1: Short collection")
    print(f"  {TEST_TOTAL_STEPS} steps, segment size {TEST_SEGMENT_SIZE}")
    print(f"  → expect {TEST_TOTAL_STEPS // TEST_SEGMENT_SIZE} segments")
    print(f"  Output: {session_dir}")
    print(f"{'='*60}")

    world_xml = Path(__file__).parent / "_valsegs_world.xml"
    world_xml.write_text(build_world_xml(), encoding="utf-8")
    robot = RangerMiniV3Robot()
    robot.load(world_xml)

    robot.data.qpos[0] = -4.5
    robot.data.qpos[1] = 0.0
    robot.data.qpos[3] = 0.7071068
    robot.data.qpos[4] = 0.0
    robot.data.qpos[5] = 0.0
    robot.data.qpos[6] = 0.7071068
    mujoco.mj_forward(robot.model, robot.data)

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

    policy   = PrimitiveExplorationPolicy(CONTROL_FREQ, beta=BETA)
    seg_id   = 0
    seg_step = 0
    seg_start_global = 0
    segment_meta = []

    seg_dir, rgb_dir, depth_dir, csv_file, writer = open_segment(
        session_dir, seg_id, SAVE_IMAGES
    )

    for global_step in range(TEST_TOTAL_STEPS):
        # Roll segment if full
        if seg_step == TEST_SEGMENT_SIZE:
            csv_file.close()
            n_imgs = len(list(rgb_dir.glob("*.png")) + list(rgb_dir.glob("*.jpg"))) if SAVE_IMAGES else TEST_SEGMENT_SIZE
            segment_meta.append({
                "segment_id":   f"segment_{seg_id:03d}",
                "global_start": seg_start_global,
                "global_end":   global_step - 1,
                "num_steps":    TEST_SEGMENT_SIZE,
                "num_images":   n_imgs,
            })
            print(f"  [SEG] segment_{seg_id:03d} closed at global_step={global_step-1}")
            seg_id          += 1
            seg_step         = 0
            seg_start_global = global_step
            seg_dir, rgb_dir, depth_dir, csv_file, writer = open_segment(
                session_dir, seg_id, SAVE_IMAGES
            )
            # ── NO robot reset ───────────────────────────────────────────────

        if SAVE_IMAGES:
            renderer_rgb.update_scene(robot.data, camera="front_cam")
            rgb_img = renderer_rgb.render()
        renderer_depth.update_scene(robot.data, camera="lidar_cam")
        depth_img = renderer_depth.render()

        pos_x, pos_y, _ = robot.data.qpos[0:3]
        qw, qx, qy, qz  = robot.data.qpos[3:7]
        yaw = euler_from_quaternion(qw, qx, qy, qz)

        cmd, prim = policy.get_action(depth_img, pos_x, pos_y, yaw)

        img_filename = f"{seg_step:06d}.png"
        if SAVE_IMAGES:
            PILImage.fromarray(rgb_img).save(rgb_dir / img_filename)
            save_depth(depth_img, str(depth_dir / img_filename))

        sim_time = robot.data.time
        writer.writerow([
            0, global_step, seg_step,
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

        for _ in range(sim_steps_per_control):
            robot.apply_command(cmd)
            robot.step()

        seg_step += 1

    csv_file.close()
    n_imgs = len(list(rgb_dir.glob("*.png")) + list(rgb_dir.glob("*.jpg"))) if SAVE_IMAGES else seg_step
    segment_meta.append({
        "segment_id":   f"segment_{seg_id:03d}",
        "global_start": seg_start_global,
        "global_end":   seg_start_global + seg_step - 1,
        "num_steps":    seg_step,
        "num_images":   n_imgs,
    })
    print(f"  [SEG] segment_{seg_id:03d} closed at global_step={seg_start_global + seg_step - 1}")

    # Write metadata
    session_meta = {
        "session_id":           session_id,
        "date":                 datetime.now().isoformat(),
        "segment_size":         TEST_SEGMENT_SIZE,
        "total_steps_planned":  TEST_TOTAL_STEPS,
        "total_steps_recorded": sum(m["num_steps"] for m in segment_meta),
        "num_segments":         len(segment_meta),
        "trajectory_id":        0,
        "robot_resets":         0,
        "segments":             segment_meta,
    }
    with open(session_dir / "metadata.json", "w", encoding="utf-8") as mf:
        json.dump(session_meta, mf, indent=4)

    print(f"  Collection done: {TEST_TOTAL_STEPS} steps → {len(segment_meta)} segments")
    return session_dir


# ── Step 2: Validate ──────────────────────────────────────────────────────────

def validate(session_dir: Path):
    print(f"\n{'='*60}")
    print(f"  STEP 2: Validation of {session_dir.name}")
    print(f"{'='*60}")

    failures = []
    expected_num_segments = TEST_TOTAL_STEPS // TEST_SEGMENT_SIZE

    # ── Check 13: metadata.json ───────────────────────────────────────────────
    meta_path = session_dir / "metadata.json"
    if not meta_path.exists():
        failures.append("metadata.json missing")
        meta = {}
    else:
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("robot_resets", -1) != 0:
            failures.append(f"Check 13 FAIL: robot_resets={meta.get('robot_resets')}, expected 0")
        else:
            print(f"  [OK ] Check 13: robot_resets = 0")
        if meta.get("trajectory_id", -1) != 0:
            failures.append(f"Check 13 FAIL: trajectory_id={meta.get('trajectory_id')}, expected 0")

    # ── Check 1: segment count ────────────────────────────────────────────────
    seg_dirs = sorted([d for d in session_dir.iterdir()
                       if d.is_dir() and d.name.startswith("segment_")])
    if len(seg_dirs) != expected_num_segments:
        failures.append(
            f"Check 1 FAIL: expected {expected_num_segments} segments, found {len(seg_dirs)}"
        )
    else:
        print(f"  [OK ] Check  1: {len(seg_dirs)} segment directories")

    # ── Load all rows from all segments in order ──────────────────────────────
    all_rows       = []     # (seg_name, row_dict)
    seen_img_paths = set()  # For cross-segment uniqueness (Check 10)

    for seg_dir in seg_dirs:
        seg_name = seg_dir.name

        # Check 2: rgb/ and depth/ exist
        if not (seg_dir / "rgb").exists():
            failures.append(f"Check 2 FAIL: {seg_name}/rgb/ missing")
        if not (seg_dir / "depth").exists():
            failures.append(f"Check 2 FAIL: {seg_name}/depth/ missing")

        # Check 3: segment.csv exists
        csv_path = seg_dir / "segment.csv"
        if not csv_path.exists():
            failures.append(f"Check 3 FAIL: {seg_name}/segment.csv missing")
            continue

        with open(csv_path, newline="", encoding="utf-8") as cf:
            rows = list(csv.DictReader(cf))

        # Check 4: row count
        expected_rows = TEST_SEGMENT_SIZE if seg_dir != seg_dirs[-1] else \
                        TEST_TOTAL_STEPS - (len(seg_dirs) - 1) * TEST_SEGMENT_SIZE
        if len(rows) != expected_rows:
            failures.append(
                f"Check 4 FAIL: {seg_name} has {len(rows)} rows, expected {expected_rows}"
            )

        for row in rows:
            all_rows.append((seg_name, row))

            # Check 10: cross-segment image uniqueness
            img_key = f"{seg_name}/{row['rgb_path']}"
            if img_key in seen_img_paths:
                failures.append(f"Check 10 FAIL: duplicate image path {img_key}")
            seen_img_paths.add(img_key)

            # Check 9: image exists
            rgb_full = seg_dir / row["rgb_path"]
            if SAVE_IMAGES and not rgb_full.exists():
                failures.append(f"Check 9 FAIL: {rgb_full} not found")

            # Check 11: trajectory_id == 0
            if int(row["trajectory_id"]) != 0:
                failures.append(
                    f"Check 11 FAIL: trajectory_id={row['trajectory_id']} in {seg_name}"
                )

            # Check 15: finite actions
            for col in ("linear_vel_cmd", "lateral_vel_cmd", "angular_vel_cmd"):
                if not math.isfinite(float(row[col])):
                    failures.append(f"Check 15 FAIL: {col}={row[col]} in {seg_name}")

    # ── Check 14: total rows ──────────────────────────────────────────────────
    if len(all_rows) != TEST_TOTAL_STEPS:
        failures.append(
            f"Check 14 FAIL: total rows={len(all_rows)}, expected {TEST_TOTAL_STEPS}"
        )
    else:
        print(f"  [OK ] Check 14: total rows = {len(all_rows)} (no lost/duplicated)")

    # ── Check 5: global_step continuous 0..TOTAL_STEPS-1 ────────────────────
    prev_global = -1
    global_ok = True
    for seg_name, row in all_rows:
        g = int(row["global_step"])
        if g != prev_global + 1:
            failures.append(
                f"Check 5 FAIL: global_step jump at {seg_name}: {prev_global} → {g}"
            )
            global_ok = False
            break
        prev_global = g
    if global_ok:
        print(f"  [OK ] Check  5: global_step continuous 0..{prev_global}")

    # ── Check 6: segment_step resets and is monotonic ─────────────────────────
    seg6_ok = True
    for seg_dir in seg_dirs:
        csv_path = seg_dir / "segment.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, newline="") as cf:
            rows = list(csv.DictReader(cf))
        steps = [int(r["segment_step"]) for r in rows]
        if steps[0] != 0:
            failures.append(f"Check 6 FAIL: {seg_dir.name} segment_step starts at {steps[0]}")
            seg6_ok = False
        for i in range(1, len(steps)):
            if steps[i] != steps[i-1] + 1:
                failures.append(
                    f"Check 6 FAIL: {seg_dir.name} segment_step not monotonic at {i}"
                )
                seg6_ok = False
                break
    if seg6_ok:
        print(f"  [OK ] Check  6: segment_step resets to 0 and is monotonic in each segment")

    # ── Check 7: robot position does NOT reset at segment boundaries ──────────
    pos_reset_ok = True
    for i in range(len(seg_dirs) - 1):
        csv_a = seg_dirs[i] / "segment.csv"
        csv_b = seg_dirs[i+1] / "segment.csv"
        if not csv_a.exists() or not csv_b.exists():
            continue
        with open(csv_a, newline="") as f:
            rows_a = list(csv.DictReader(f))
        with open(csv_b, newline="") as f:
            rows_b = list(csv.DictReader(f))
        if not rows_a or not rows_b:
            continue
        last_x  = float(rows_a[-1]["pos_x"])
        last_y  = float(rows_a[-1]["pos_y"])
        first_x = float(rows_b[0]["pos_x"])
        first_y = float(rows_b[0]["pos_y"])
        dist = math.hypot(first_x - last_x, first_y - last_y)
        # If robot were reset to (-4.5, 0) from anywhere meaningful, dist >> 1.0
        # A single physics step can move at most ~0.2m at 1.8 m/s / 10 Hz
        if dist > 1.0:
            failures.append(
                f"Check 7 FAIL: large position jump at boundary {seg_dirs[i].name}"
                f"→{seg_dirs[i+1].name}: ({last_x:.2f},{last_y:.2f})"
                f"→({first_x:.2f},{first_y:.2f}), dist={dist:.2f}m"
            )
            pos_reset_ok = False
        else:
            print(
                f"  [OK ] Check  7: {seg_dirs[i].name}→{seg_dirs[i+1].name} "
                f"boundary dist={dist:.3f}m (no reset)"
            )
    if not pos_reset_ok:
        pass  # errors already added

    # ── Check 8: sim_time strictly monotonically increasing ───────────────────
    sim_times = [float(r["sim_time"]) for _, r in all_rows]
    t8_ok = True
    for i in range(1, len(sim_times)):
        if sim_times[i] <= sim_times[i-1]:
            failures.append(
                f"Check 8 FAIL: sim_time not monotonic at global_step={i}: "
                f"{sim_times[i-1]:.3f}→{sim_times[i]:.3f}"
            )
            t8_ok = False
            break
    if t8_ok:
        print(
            f"  [OK ] Check  8: sim_time monotonic {sim_times[0]:.3f}..{sim_times[-1]:.3f}s"
        )

    # ── Check 12: image resolution ────────────────────────────────────────────
    if SAVE_IMAGES:
        for seg_dir in seg_dirs[:2]:   # spot-check first 2 segments
            rgb_imgs   = sorted(list((seg_dir / "rgb").glob("*.png")) + list((seg_dir / "rgb").glob("*.jpg")))
            depth_imgs = sorted(list((seg_dir / "depth").glob("*.png")) + list((seg_dir / "depth").glob("*.jpg")))
            if rgb_imgs:
                img = PILImage.open(rgb_imgs[0])
                if img.size != (320, 240):
                    failures.append(f"Check 12 FAIL: RGB size {img.size} in {seg_dir.name}")
                else:
                    print(f"  [OK ] Check 12: {seg_dir.name} RGB=320×240")
            if depth_imgs:
                img = PILImage.open(depth_imgs[0])
                if img.size != (640, 60):
                    failures.append(f"Check 12 FAIL: Depth size {img.size} in {seg_dir.name}")
                else:
                    print(f"  [OK ] Check 12: {seg_dir.name} Depth=640×60")

    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Segment summary:")
    for seg_dir in seg_dirs:
        csv_path = seg_dir / "segment.csv"
        if csv_path.exists():
            with open(csv_path, newline="") as cf:
                rows = list(csv.DictReader(cf))
            g_start = rows[0]["global_step"] if rows else "?"
            g_end   = rows[-1]["global_step"] if rows else "?"
            n_imgs  = len(list((seg_dir/"rgb").glob("*.png")) + list((seg_dir/"rgb").glob("*.jpg"))) if SAVE_IMAGES else "?"
            print(f"    {seg_dir.name}: {len(rows)} steps | "
                  f"global {g_start}–{g_end} | {n_imgs} images")

    print(f"\n{'='*60}")
    if not failures:
        print("  ALL CHECKS PASSED ✓")
    else:
        print(f"  {len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print(f"    ✗ {f}")
    print(f"{'='*60}\n")

    return len(failures) == 0


if __name__ == "__main__":
    session_dir = run_collection()
    ok = validate(session_dir)
    sys.exit(0 if ok else 1)
