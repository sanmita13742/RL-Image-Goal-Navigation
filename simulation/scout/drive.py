"""
drive.py  —  Scout Mini keyboard teleoperation for MuJoCo
─────────────────────────────────────────────────────────
Refactored to use the RobotBase abstraction layer.
Identical behaviour to the original — skid-steer controls.

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

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco
import mujoco.viewer

from robot import ScoutMiniRobot
from robot_base import DriveCommand

# ── GLFW key codes  (raw GLFW codes, NOT ASCII) ──────────────────────────────
GLFW_KEY_UP     = 265
GLFW_KEY_DOWN   = 264
GLFW_KEY_LEFT   = 263
GLFW_KEY_RIGHT  = 262
GLFW_KEY_ENTER  = 257   # stop
GLFW_KEY_ESCAPE = 256   # quit

# ── Shared command state ─────────────────────────────────────────────────────
cmd       = DriveCommand()
quit_flag = False

KEY_MAP = {
    GLFW_KEY_UP:    DriveCommand( 1.5,  0.0),   # forward
    GLFW_KEY_DOWN:  DriveCommand(-1.5,  0.0),   # backward
    GLFW_KEY_LEFT:  DriveCommand( 0.0,  1.2),   # turn left
    GLFW_KEY_RIGHT: DriveCommand( 0.0, -1.2),   # turn right
    GLFW_KEY_ENTER: DriveCommand( 0.0,  0.0),   # stop
}


def on_key(keycode: int) -> None:
    """Called by the MuJoCo viewer on every key press."""
    global cmd, quit_flag
    if keycode == GLFW_KEY_ESCAPE:
        quit_flag = True
        return
    if keycode in KEY_MAP:
        cmd = KEY_MAP[keycode]


def main() -> None:
    global quit_flag

    xml = Path(__file__).parent / "scout_mini.xml"
    robot = ScoutMiniRobot()
    robot.load(xml)

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
        robot.model, robot.data, key_callback=on_key
    ) as viewer:

        # Pull back to a good overview angle
        viewer.cam.azimuth   = 135
        viewer.cam.elevation = -25
        viewer.cam.distance  = 3.5

        while viewer.is_running() and not quit_flag:
            step_start = time.time()

            robot.apply_command(cmd)
            robot.step()
            viewer.sync()

            # Sleep the remainder of the timestep to stay roughly real-time
            elapsed   = time.time() - step_start
            sleep_for = robot.model.opt.timestep - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    print("Simulation ended.")


if __name__ == "__main__":
    main()
