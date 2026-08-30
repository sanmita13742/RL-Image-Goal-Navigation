"""
ranger_mujoco/drive.py  —  Ranger Mini keyboard teleoperation (Phase 3)
=======================================================================
Controls (click viewer window first to focus it):
  ↑ / ↓        forward / backward
  ← / →        turn left / right
  Enter         stop
  [ / ]         decrease / increase steering sensitivity
  Escape        quit

Drive model: Ackermann 4-Wheel Steering
  All four wheels steer individually for smooth radius turns.
  Steering angles computed via ackermann_angles() in robot_base.py.

Requires:  pip install mujoco
Run:       python drive.py
"""

import sys
import time
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco
import mujoco.viewer

from robot import RangerMiniV3Robot
from robot_base import DriveCommand

# ── GLFW key codes ────────────────────────────────────────────────────────────
GLFW_KEY_UP     = 265
GLFW_KEY_DOWN   = 264
GLFW_KEY_LEFT   = 263
GLFW_KEY_RIGHT  = 262
GLFW_KEY_ENTER  = 257   # stop
GLFW_KEY_ESCAPE = 256   # quit
GLFW_KEY_LBRACKET = 91  # [ — decrease steer sensitivity
GLFW_KEY_RBRACKET = 93  # ] — increase steer sensitivity


# ── Shared command state ──────────────────────────────────────────────────────
cmd       = DriveCommand()
quit_flag = False
steer_scale = 1.0   # multiplier for turning speed

# Speed presets (m/s linear, rad/s angular)
FORWARD_SPEED  = 1.5    # m/s
ANGULAR_SPEED  = 0.8    # rad/s

KEY_MAP = {
    GLFW_KEY_UP:    DriveCommand(v_linear= FORWARD_SPEED, v_angular= 0.0),
    GLFW_KEY_DOWN:  DriveCommand(v_linear=-FORWARD_SPEED, v_angular= 0.0),
    GLFW_KEY_LEFT:  DriveCommand(v_linear= 0.0,           v_angular= ANGULAR_SPEED),
    GLFW_KEY_RIGHT: DriveCommand(v_linear= 0.0,           v_angular=-ANGULAR_SPEED),
    GLFW_KEY_ENTER: DriveCommand(v_linear= 0.0,           v_angular= 0.0),
}


def on_key(keycode: int) -> None:
    """MuJoCo viewer key callback."""
    global cmd, quit_flag, steer_scale
    if keycode == GLFW_KEY_ESCAPE:
        quit_flag = True
        return
    if keycode == GLFW_KEY_LBRACKET:
        steer_scale = max(0.2, steer_scale - 0.1)
        print(f"\r  Steer scale: {steer_scale:.1f}  ", end="", flush=True)
        return
    if keycode == GLFW_KEY_RBRACKET:
        steer_scale = min(2.0, steer_scale + 0.1)
        print(f"\r  Steer scale: {steer_scale:.1f}  ", end="", flush=True)
        return
    if keycode in KEY_MAP:
        preset = KEY_MAP[keycode]
        cmd = DriveCommand(preset.v_linear, preset.v_angular * steer_scale)


def print_hud(robot: RangerMiniV3Robot) -> None:
    """Print live telemetry to terminal (overwrites same line)."""
    angles = robot.read_steering_angles()
    vels   = robot.read_wheel_velocities()
    imu    = robot.read_imu()
    fl_deg = math.degrees(angles["fl_steer_pos"])
    fr_deg = math.degrees(angles["fr_steer_pos"])
    rl_deg = math.degrees(angles["rl_steer_pos"])
    rr_deg = math.degrees(angles["rr_steer_pos"])
    avg_vel = (abs(vels["fl_wheel_vel"]) + abs(vels["fr_wheel_vel"]) +
               abs(vels["rl_wheel_vel"]) + abs(vels["rr_wheel_vel"])) / 4.0
    speed_ms = avg_vel * robot.WHEEL_RADIUS
    gz = imu["gyro"][2]
    print(
        f"\r  v={cmd.v_linear:+5.2f}m/s  ω={cmd.v_angular:+5.2f}rad/s"
        f"  |  steer FL={fl_deg:+5.1f}° FR={fr_deg:+5.1f}°"
        f"  RL={rl_deg:+5.1f}° RR={rr_deg:+5.1f}°"
        f"  |  spd={speed_ms:.2f}m/s  yaw_rate={gz:+.2f}rad/s",
        end="", flush=True
    )


def main() -> None:
    global quit_flag

    xml = Path(__file__).parent / "ranger_mini_v3.xml"
    robot = RangerMiniV3Robot()
    robot.load(xml)

    print("═" * 70)
    print("  Ranger Mini  —  MuJoCo Keyboard Teleoperation (Phase 3)")
    print("═" * 70)
    print("  ↑ / ↓        : forward / backward")
    print("  ← / →        : turn left / right  (Ackermann 4WS)")
    print("  Enter         : stop")
    print("  [ / ]         : decrease / increase steering sensitivity")
    print("  Escape        : quit")
    print()
    print("  ⚠  Click the viewer window to focus it first!")
    print("═" * 70)

    with mujoco.viewer.launch_passive(
        robot.model, robot.data, key_callback=on_key
    ) as viewer:

        viewer.cam.azimuth   = 140
        viewer.cam.elevation = -22
        viewer.cam.distance  = 5.0

        hud_every = 20   # print every N steps
        step_count = 0

        while viewer.is_running() and not quit_flag:
            step_start = time.time()

            robot.apply_command(cmd)
            robot.step()
            viewer.sync()

            step_count += 1
            if step_count % hud_every == 0:
                print_hud(robot)

            elapsed   = time.time() - step_start
            sleep_for = robot.model.opt.timestep - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    print("\nSimulation ended.")


if __name__ == "__main__":
    main()
