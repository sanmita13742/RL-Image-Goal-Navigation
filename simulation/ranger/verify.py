"""
ranger_mujoco/verify.py  --  Phase 1: Verify Ranger Mini loads correctly
=======================================================================
Checklist:
  [OK] XML parses without errors
  [OK] All bodies present
  [OK] All joints correct
  [OK] All actuators correct
  [OK] All sensors correct
  [OK] Physics stable for 500 steps
  [OK] Robot dimensions reported

Run:  python verify.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco
from robot import RangerRobot


# ─────────────────────────────────────────────────────────────────────────────
# Expected topology
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_BODIES = {
    "base_link",
    "camera_link",
    "lidar_link",
    "fr_steering_link",
    "fr_wheel_link",
    "fl_steering_wheel_link",
    "fl_wheel_link",
    "rl_steering_wheel_link",
    "rl_wheel_link",
    "rr_steering_wheel_link",
    "rr_wheel_link",
}

EXPECTED_JOINTS = [
    "root",              # freejoint
    "fl_steering_joint",
    "fr_steering_joint",
    "rl_steering_joint",
    "rr_steering_joint",
    "fl_wheel_joint",
    "fr_wheel_joint",
    "rl_wheel_joint",
    "rr_wheel_joint",
]

EXPECTED_ACTUATORS = [
    "act_fl_steer", "act_fr_steer", "act_rl_steer", "act_rr_steer",
    "act_fl_drive", "act_fr_drive", "act_rl_drive", "act_rr_drive",
]

EXPECTED_SENSORS = [
    "imu_accel", "imu_gyro",
    "fl_wheel_vel", "fr_wheel_vel", "rl_wheel_vel", "rr_wheel_vel",
    "fl_steer_pos", "fr_steer_pos", "rl_steer_pos", "rr_steer_pos",
]

EXPECTED_CAMERAS = ["front_cam", "lidar_cam"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def check(label: str, passed: bool, detail: str = "") -> bool:
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status}  {label}", end="")
    if detail:
        print(f"  [{detail}]", end="")
    print()
    return passed


def section(title: str) -> None:
    print()
    print(f"-- {title} " + "-" * (54 - len(title)))


# ─────────────────────────────────────────────────────────────────────────────
# Main verification
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    xml = Path(__file__).parent / "ranger.xml"
    all_passed = True

    print("=" * 60)
    print("  Ranger Mini  --  Phase 1 Verification")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────────
    section("Load Model")
    robot = RangerRobot()
    try:
        robot.load(xml)
        ok = check("XML parsed without exception", True)
    except Exception as e:
        ok = check("XML parsed without exception", False, str(e))
        print("\nFATAL: model failed to load -- cannot continue.")
        sys.exit(1)
    all_passed = all_passed and ok

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    robot.print_summary()

    # ── Body hierarchy ────────────────────────────────────────────────────────
    section("Body Hierarchy")
    actual_bodies = set(robot.body_names())
    missing = EXPECTED_BODIES - actual_bodies
    extra   = actual_bodies - EXPECTED_BODIES
    ok1 = check("All expected bodies present", not missing,
                f"missing: {missing}" if missing else "")
    ok2 = check("No unexpected extra bodies",  not extra,
                f"extra: {extra}"     if extra   else "")
    all_passed = all_passed and ok1 and ok2

    # ── Joints ────────────────────────────────────────────────────────────────
    section("Joints")
    actual_joints = robot.joint_names()
    for j in EXPECTED_JOINTS:
        ok = check(f"Joint: {j}", j in actual_joints)
        all_passed = all_passed and ok
    print(f"  Total joints: {robot.model.njnt}")

    # ── Actuators ─────────────────────────────────────────────────────────────
    section("Actuators")
    actual_acts = robot.actuator_names()
    for a in EXPECTED_ACTUATORS:
        ok = check(f"Actuator: {a}", a in actual_acts)
        all_passed = all_passed and ok
    ok = check("Actuator count = 8", robot.model.nu == 8, f"got {robot.model.nu}")
    all_passed = all_passed and ok

    # ── Sensors ───────────────────────────────────────────────────────────────
    section("Sensors")
    actual_sensors = robot.sensor_names()
    for s in EXPECTED_SENSORS:
        ok = check(f"Sensor: {s}", s in actual_sensors)
        all_passed = all_passed and ok

    # ── Cameras ───────────────────────────────────────────────────────────────
    section("Cameras")
    actual_cams = robot.camera_names()
    for c in EXPECTED_CAMERAS:
        ok = check(f"Camera: {c}", c in actual_cams)
        all_passed = all_passed and ok

    # ── Physics stability ─────────────────────────────────────────────────────
    section("Physics Stability (500 steps)")
    start = time.perf_counter()
    energy_ok = True
    pos_ok = True
    import numpy as np

    for step_i in range(500):
        mujoco.mj_step(robot.model, robot.data)
        # Check for NaN/Inf in state
        if not np.all(np.isfinite(robot.data.qpos)):
            energy_ok = False
            print(f"  [FAIL]  NaN/Inf in qpos at step {step_i}")
            break
        if not np.all(np.isfinite(robot.data.qvel)):
            energy_ok = False
            print(f"  [FAIL]  NaN/Inf in qvel at step {step_i}")
            break

    elapsed = time.perf_counter() - start
    ok1 = check("No NaN/Inf in state vectors", energy_ok)
    all_passed = all_passed and ok1

    # Check robot hasn't sunk through floor
    base_z = robot.data.qpos[2]
    ok2 = check("Base stays above ground (z > -0.1 m)", base_z > -0.1,
                f"z = {base_z:.3f}")
    all_passed = all_passed and ok2
    print(f"  500 steps in {elapsed*1000:.1f} ms  ({elapsed/0.5:.2f}x real-time for dt=0.001)")

    # ── IMU sanity ────────────────────────────────────────────────────────────
    section("IMU Sanity")
    imu = robot.read_imu()
    az = imu["accel"][2]
    ok = check("IMU accel-Z ~9.81 m/s2 (gravity)", 8.0 < abs(az) < 11.0,
               f"az = {az:.2f}")
    all_passed = all_passed and ok

    # ── Dimensions ────────────────────────────────────────────────────────────
    section("Robot Dimensions")
    dims = robot.get_dimensions()
    print(f"  Wheel radius  : {dims.wheel_radius:.4f} m")
    print(f"  Track width   : {dims.track_width:.4f} m   ({dims.track_width*100:.1f} cm)")
    print(f"  Wheelbase     : {dims.wheelbase:.4f} m   ({dims.wheelbase*100:.1f} cm)")
    print(f"  Total mass    : {dims.mass_total:.2f} kg")

    # ── Result ────────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    if all_passed:
        print("  [PASS]  ALL CHECKS PASSED -- Ranger Mini loads correctly!")
    else:
        print("  [FAIL]  SOME CHECKS FAILED -- see details above")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
