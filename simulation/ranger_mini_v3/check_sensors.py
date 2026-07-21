"""
ranger_mujoco/check_sensors.py  --  Phase 4: Verify Sensors
============================================================
Tests:
  [OK] Front RGB camera renders correctly -> front_cam_frame.png
  [OK] LiDAR depth camera renders -> lidar_depth.png
  [OK] IMU accelerometer shows ~9.81 m/s2 on Z-axis when stationary
  [OK] IMU gyroscope reads ~0 rad/s when stationary
  [OK] Wheel velocity sensors read 0 when stopped
  [OK] Steering angle sensors read 0 at rest

Run:  python check_sensors.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco
import numpy as np

from robot import RangerMiniV3Robot

try:
    from PIL import Image
    USE_PIL = True
except ImportError:
    USE_PIL = False


def save_rgb(arr: np.ndarray, path: str) -> None:
    if USE_PIL:
        Image.fromarray(arr).save(path)
        print(f"  Saved RGB -> {path}  shape={arr.shape}  max={arr.max()}")
    else:
        raw = path.replace(".png", ".raw")
        with open(raw, "wb") as f:
            f.write(arr.tobytes())
        print(f"  PIL unavailable -- raw bytes -> {raw}  (install Pillow)")


def save_depth(arr: np.ndarray, path: str) -> None:
    """Normalise and save depth as grayscale PNG."""
    if USE_PIL:
        d_min, d_max = arr.min(), arr.max()
        if d_max > d_min:
            norm = ((arr - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            norm = np.zeros_like(arr, dtype=np.uint8)
        Image.fromarray(norm, mode="L").save(path)
        print(f"  Saved depth -> {path}  min={d_min:.2f}m  max={d_max:.2f}m")
    else:
        print("  PIL unavailable -- depth not saved (install Pillow)")


def check(label: str, passed: bool, detail: str = "") -> bool:
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status}  {label}", end="")
    if detail:
        print(f"  [{detail}]", end="")
    print()
    return passed


def section(title: str) -> None:
    print(f"\n-- {title} " + "-" * (54 - len(title)))


def main() -> None:
    xml = Path(__file__).parent / "ranger_mini_v3.xml"
    robot = RangerMiniV3Robot()
    robot.load(xml)

    print("=" * 60)
    print("  Ranger Mini  --  Phase 4: Sensor Verification")
    print("=" * 60)

    # Let robot settle on the ground
    print("\n  Stepping 200 frames to let robot settle...")
    for _ in range(200):
        mujoco.mj_step(robot.model, robot.data)

    all_passed = True

    # -- IMU ------------------------------------------------------------------
    section("IMU Checks")
    imu = robot.read_imu()
    ax, ay, az = imu["accel"]
    gx, gy, gz = imu["gyro"]

    ok = check("Accel-Z ~9.81 m/s2 (gravity)", 8.0 < abs(az) < 11.5,
               f"az = {az:.3f} m/s2")
    all_passed = all_passed and ok
    print(f"  IMU accel: [{ax:+.3f}, {ay:+.3f}, {az:+.3f}] m/s2")

    ok = check("Gyro all axes ~0 rad/s (stationary)",
               all(abs(g) < 0.5 for g in [gx, gy, gz]),
               f"gyro=[{gx:+.3f}, {gy:+.3f}, {gz:+.3f}]")
    all_passed = all_passed and ok

    # -- Wheel velocity sensors -----------------------------------------------
    section("Wheel Velocity Sensors")
    wv = robot.read_wheel_velocities()
    for name, val in wv.items():
        ok = check(f"{name} ~0 rad/s (stopped)", abs(val) < 0.5,
                   f"{val:.4f} rad/s")
        all_passed = all_passed and ok

    # -- Steering angle sensors -----------------------------------------------
    section("Steering Angle Sensors")
    sa = robot.read_steering_angles()
    for name, val in sa.items():
        ok = check(f"{name} ~0 rad (at rest)", abs(val) < 0.05,
                   f"{math.degrees(val):.2f} deg")
        all_passed = all_passed and ok

    # -- Camera: Front RGB ----------------------------------------------------
    section("Front RGB Camera")
    renderer = mujoco.Renderer(robot.model, height=480, width=640)
    renderer.update_scene(robot.data, camera="front_cam")
    rgb = renderer.render()
    ok = check("RGB frame rendered", rgb is not None and rgb.size > 0,
               f"shape={rgb.shape}")
    all_passed = all_passed and ok
    ok = check("RGB has pixel variation (not blank)", rgb.std() > 5.0,
               f"std={rgb.std():.1f}")
    all_passed = all_passed and ok
    out_rgb = Path(__file__).parent / "front_cam_frame.png"
    save_rgb(rgb, str(out_rgb))

    # -- Camera: Depth / LiDAR ------------------------------------------------
    section("LiDAR Depth Camera")
    renderer.enable_depth_rendering()
    renderer.update_scene(robot.data, camera="lidar_cam")
    depth = renderer.render()
    ok = check("Depth frame rendered", depth is not None and depth.size > 0,
               f"shape={depth.shape}")
    all_passed = all_passed and ok
    finite_depth = depth[np.isfinite(depth)]
    if finite_depth.size > 0:
        ok = check("Depth has finite values", True,
                   f"min={finite_depth.min():.2f}m  max={finite_depth.max():.2f}m")
    else:
        ok = check("Depth has finite values", False, "all values are inf/nan")
    all_passed = all_passed and ok
    out_depth = Path(__file__).parent / "lidar_depth.png"
    save_depth(depth, str(out_depth))

    # -- Summary --------------------------------------------------------------
    print()
    print("=" * 60)
    if all_passed:
        print("  [PASS]  ALL SENSOR CHECKS PASSED")
    else:
        print("  [FAIL]  SOME SENSOR CHECKS FAILED -- see above")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
