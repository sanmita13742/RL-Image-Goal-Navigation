"""
drive.py  —  Scout Mini keyboard teleoperation for MuJoCo
─────────────────────────────────────────────────────────
NOTE: W/A/S/D are reserved by the MuJoCo viewer for camera navigation.
      We use ARROW KEYS instead, which do not conflict.

Controls (click the viewer window first to focus it)
  ↑  /  ↓      forward / backward
  ←  /  →      turn left / turn right
  Enter         stop
  Escape        quit

Requires:  pip install mujoco
Run:       python drive.py
"""

import mujoco
import mujoco.viewer
import time

# ── Robot constants (from scout_mini.urdf) ──────────────────────────────────
WHEEL_RADIUS = 0.08      # metres
TRACK_WIDTH  = 0.4165    # metres  (left-right wheel centre distance)

# ── GLFW key codes  (these are NOT ASCII — MuJoCo passes raw GLFW codes) ────
# MuJoCo viewer already uses W/A/S/D for camera, so we use arrow keys.
GLFW_KEY_UP     = 265
GLFW_KEY_DOWN   = 264
GLFW_KEY_LEFT   = 263
GLFW_KEY_RIGHT  = 262
GLFW_KEY_ENTER  = 257   # stop
GLFW_KEY_ESCAPE = 256   # quit


def skid_steer(v_linear: float, v_angular: float):
    """Convert (m/s, rad/s) twist to wheel angular velocities (rad/s)."""
    v_left  = (v_linear - v_angular * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
    v_right = (v_linear + v_angular * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
    return v_left, v_right


def apply_command(data, v_lin, v_ang):
    v_left, v_right = skid_steer(v_lin, v_ang)
    data.ctrl[0] = v_left   # front-left
    data.ctrl[1] = v_left   # rear-left
    data.ctrl[2] = v_right  # front-right
    data.ctrl[3] = v_right  # rear-right


# ── Shared command state ─────────────────────────────────────────────────────
v_linear  = 0.0
v_angular = 0.0
quit_flag = False

KEY_MAP = {
    GLFW_KEY_UP:    ( 1.5,  0.0),   # forward
    GLFW_KEY_DOWN:  (-1.5,  0.0),   # backward
    GLFW_KEY_LEFT:  ( 0.0,  1.2),   # turn left
    GLFW_KEY_RIGHT: ( 0.0, -1.2),   # turn right
    GLFW_KEY_ENTER: ( 0.0,  0.0),   # stop
}


def on_key(keycode):
    """Called by the MuJoCo viewer on every key press."""
    global v_linear, v_angular, quit_flag
    if keycode == GLFW_KEY_ESCAPE:
        quit_flag = True
        return
    if keycode in KEY_MAP:
        v_linear, v_angular = KEY_MAP[keycode]


def main():
    global quit_flag

    model = mujoco.MjModel.from_xml_path("scout_mini.xml")
    data  = mujoco.MjData(model)

    print("=" * 52)
    print("  Scout Mini  —  MuJoCo Keyboard Teleop")
    print("=" * 52)
    print("  ↑ / ↓    : forward / backward")
    print("  ← / →    : turn left / right")
    print("  Enter     : stop")
    print("  Escape    : quit")
    print()
    print("  ⚠  Click the viewer window to focus it first!")
    print("=" * 52)

    with mujoco.viewer.launch_passive(
        model, data, key_callback=on_key
    ) as viewer:

        # Pull back to a good overview angle
        viewer.cam.azimuth   = 135
        viewer.cam.elevation = -25
        viewer.cam.distance  = 3.5

        while viewer.is_running() and not quit_flag:
            step_start = time.time()

            apply_command(data, v_linear, v_angular)
            mujoco.mj_step(model, data)
            viewer.sync()

            # Sleep the remainder of the timestep to stay roughly real-time
            elapsed   = time.time() - step_start
            sleep_for = model.opt.timestep - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    print("Simulation ended.")


if __name__ == "__main__":
    main()
