"""
ranger_mujoco/test_env.py  —  Phase 5: Test Environment
========================================================
Loads a richer world with:
  • Checker-board floor
  • Perimeter walls
  • Obstacle course (boxes + cylinders)
  • Start/finish lines
  • Ranger Mini robot with keyboard teleop

Controls:
  ↑ / ↓    : forward / backward
  ← / →    : turn left / right
  Enter     : stop
  Escape    : quit

Run:  python test_env.py
"""

import sys
import time
import math
import tempfile
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
GLFW_KEY_ENTER  = 257
GLFW_KEY_ESCAPE = 256

# ── Shared state ──────────────────────────────────────────────────────────────
cmd       = DriveCommand()
quit_flag = False

FORWARD_SPEED = 1.5
ANGULAR_SPEED = 0.8

KEY_MAP = {
    GLFW_KEY_UP:    DriveCommand(v_linear= FORWARD_SPEED, v_angular= 0.0),
    GLFW_KEY_DOWN:  DriveCommand(v_linear=-FORWARD_SPEED, v_angular= 0.0),
    GLFW_KEY_LEFT:  DriveCommand(v_linear= 0.0,           v_angular= ANGULAR_SPEED),
    GLFW_KEY_RIGHT: DriveCommand(v_linear= 0.0,           v_angular=-ANGULAR_SPEED),
    GLFW_KEY_ENTER: DriveCommand(v_linear= 0.0,           v_angular= 0.0),
}


def on_key(keycode: int) -> None:
    global cmd, quit_flag
    if keycode == GLFW_KEY_ESCAPE:
        quit_flag = True
        return
    if keycode in KEY_MAP:
        cmd = KEY_MAP[keycode]


# ── World XML builder ─────────────────────────────────────────────────────────

def build_world_xml() -> str:
    """
    Build a complete MJCF world XML that includes:
    - ranger_mini_v3.xml content inlined via <include>
    - Richer environment: walls, ramps, obstacles
    """
    ranger_xml = (Path(__file__).parent / "ranger_mini_v3.xml").resolve()

    xml = f"""<mujoco model="ranger_test_env">

  <compiler angle="radian"/>
  <option timestep="0.004" integrator="RK4" gravity="0 0 -9.81"/>

  <default>
    <joint damping="1.0" armature="0.01"/>
    <geom friction="1.2 0.01 0.001" condim="4" solimp="0.9 0.95 0.001" solref="0.02 1"/>
  </default>

  <!-- Include the base robot model -->
  <include file="{ranger_xml}"/>

  <asset>
    <!-- Start / Finish lines -->
    <material name="start_mat"  rgba="0.0 0.8 0.2 1"/>
    <material name="finish_mat" rgba="0.9 0.2 0.1 1"/>
    <material name="wall_mat"   rgba="0.55 0.55 0.60 1"/>
    <material name="box_mat"    rgba="0.85 0.55 0.15 1"/>
    <material name="cone_mat"   rgba="0.95 0.3  0.1  1"/>
    <material name="ramp_mat"   rgba="0.4  0.6  0.8  1"/>
  </asset>

  <worldbody>

    <!-- ── Lighting ────────────────────────────────────────────────────── -->
    <!-- Different lighting: Warm and cool lights in the scene -->
    <light name="warm_fill" pos="-4 -2 4" dir="1 0.5 -1" diffuse="0.8 0.5 0.3" castshadow="false"/>
    <light name="cool_fill" pos="4 2 4" dir="-1 -0.5 -1" diffuse="0.3 0.5 0.8" castshadow="false"/>
    <light name="green_fill" pos="0 3 4" dir="0 -1 -1" diffuse="0.2 0.6 0.3" castshadow="false"/>

    <!-- ── Perimeter walls (12x8 meters) ───────────────────────────────── -->
    <geom name="wall_n" type="box" size="6 0.2 1.25" pos="0  4 1.25" material="wall_mat" contype="1" conaffinity="1"/>
    <geom name="wall_s" type="box" size="6 0.2 1.25" pos="0 -4 1.25" material="wall_mat" contype="1" conaffinity="1"/>
    <geom name="wall_e" type="box" size="0.2 4 1.25" pos=" 6 0 1.25" material="wall_mat" contype="1" conaffinity="1"/>
    <geom name="wall_w" type="box" size="0.2 4 1.25" pos="-6 0 1.25" material="wall_mat" contype="1" conaffinity="1"/>

    <!-- ── Obstacles matching the map layout ───────────────────────────── -->
    <!-- Central block -->
    <geom name="obs_center" type="box" size="1.5 1.0 1.0" pos="0 0 1.0" material="box_mat"/>
    <geom name="obs_center_l" type="box" size="0.8 0.5 0.8" pos="-2.3 0 0.8" material="ramp_mat"/>
    <geom name="obs_center_r" type="box" size="0.8 0.5 0.8" pos="2.3 0 0.8" material="ramp_mat"/>
    
    <!-- Top-left structure -->
    <geom name="obs_tl1" type="box" size="1.2 0.6 1.0" pos="-4.8 3.4 1.0" material="start_mat"/>
    <geom name="obs_tl2" type="box" size="0.5 1.0 1.0" pos="-5.5 2.0 1.0" material="wall_mat"/>

    <!-- Bottom-right structure -->
    <geom name="obs_br1" type="box" size="1.0 1.0 1.0" pos="5.0 -3.0 1.0" material="finish_mat"/>
    <geom name="obs_br2" type="box" size="0.5 0.5 1.0" pos="3.5 -3.5 1.0" material="finish_mat"/>

    <!-- Top-right structure -->
    <geom name="obs_tr1" type="box" size="1.5 0.5 1.0" pos="4.5 3.5 1.0" material="cone_mat"/>
    
    <!-- Bottom-left structure -->
    <geom name="obs_bl1" type="box" size="1.5 0.5 1.0" pos="-4.5 -3.5 1.0" material="box_mat"/>
    <geom name="obs_bl2" type="box" size="0.5 0.5 1.0" pos="-5.5 -2.5 1.0" material="cone_mat"/>

  </worldbody>
</mujoco>"""
    return xml


def main() -> None:
    global quit_flag

    # Write the merged world XML to a temp file
    world_xml = Path(__file__).parent / "_test_world.xml"
    world_xml.write_text(build_world_xml(), encoding="utf-8")

    print("═" * 70)
    print("  Ranger Mini V3  —  Phase 5: Test Environment")
    print("═" * 70)
    print("  World: obstacle course with slalom boxes, cones, ramp, corridor")
    print("  ↑ / ↓    : forward / backward")
    print("  ← / →    : turn left / right  (Ackermann 4WS)")
    print("  Enter     : stop  |  Escape : quit")
    print("  ⚠  Click the viewer window to focus it first!")
    print("═" * 70)

    robot = RangerMiniV3Robot()
    robot.load(world_xml)
    
    # Set start pose (-4.5, 0) facing +Y (yaw = 90 deg)
    robot.data.qpos[0] = -4.5
    robot.data.qpos[1] = 0.0
    robot.data.qpos[3] = 0.7071068
    robot.data.qpos[4] = 0.0
    robot.data.qpos[5] = 0.0
    robot.data.qpos[6] = 0.7071068
    mujoco.mj_forward(robot.model, robot.data)

    with mujoco.viewer.launch_passive(
        robot.model, robot.data, key_callback=on_key
    ) as viewer:

        viewer.cam.azimuth   = 150
        viewer.cam.elevation = -25
        viewer.cam.distance  = 8.0
        viewer.cam.lookat[:] = [0, 0, 0.5]

        while viewer.is_running() and not quit_flag:
            step_start = time.time()
            robot.apply_command(cmd)
            robot.step()
            viewer.sync()
            elapsed = time.time() - step_start
            sleep_for = robot.model.opt.timestep - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    # Clean up temp file
    try:
        world_xml.unlink()
    except Exception:
        pass

    print("\nSimulation ended.")


if __name__ == "__main__":
    main()
