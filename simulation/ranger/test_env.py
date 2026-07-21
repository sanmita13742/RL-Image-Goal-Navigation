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

from robot import RangerRobot
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
    - ranger.xml content inlined via <include>  (MuJoCo supports file includes)
    - Richer environment: walls, ramps, obstacles
    """
    ranger_xml = (Path(__file__).parent / "ranger.xml").resolve()

    xml = f"""<mujoco model="ranger_test_env">

  <compiler angle="radian" meshdir="{ranger_xml.parent / 'meshes'}/"/>
  <option timestep="0.004" integrator="RK4" gravity="0 0 -9.81"/>

  <default>
    <joint damping="1.0" armature="0.01"/>
    <geom friction="1.2 0.01 0.001" condim="4" solimp="0.9 0.95 0.001" solref="0.02 1"/>
  </default>

  <asset>
    <!-- Floor textures -->
    <texture name="floor_tex" type="2d" builtin="checker"
             rgb1="0.22 0.22 0.22" rgb2="0.32 0.32 0.32" width="512" height="512"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="20 20"/>

    <!-- Start / Finish lines -->
    <material name="start_mat"  rgba="0.0 0.8 0.2 1"/>
    <material name="finish_mat" rgba="0.9 0.2 0.1 1"/>
    <material name="wall_mat"   rgba="0.55 0.55 0.60 1"/>
    <material name="box_mat"    rgba="0.85 0.55 0.15 1"/>
    <material name="cone_mat"   rgba="0.95 0.3  0.1  1"/>
    <material name="ramp_mat"   rgba="0.4  0.6  0.8  1"/>

    <!-- Robot materials -->
    <material name="chassis_mat"  rgba="0.18 0.18 0.18 1"/>
    <material name="wheel_mat"    rgba="0.12 0.12 0.12 1"/>
    <material name="steering_mat" rgba="0.30 0.30 0.35 1"/>
    <material name="cam_mat"      rgba="0.9  0.2  0.1  1"/>
    <material name="lidar_mat"    rgba="0.1  0.6  0.9  1"/>
  </asset>

  <worldbody>

    <!-- ── Lighting ────────────────────────────────────────────────────── -->
    <!-- Different lighting: Warm and cool lights in the scene -->
    <light name="sun"       pos="0 0 12" dir="0 0 -1"
           diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2" castshadow="true"/>
    <light name="warm_fill" pos="-4 -2 4" dir="1 0.5 -1"
           diffuse="0.8 0.5 0.3" castshadow="false"/>
    <light name="cool_fill" pos="4 2 4" dir="-1 -0.5 -1"
           diffuse="0.3 0.5 0.8" castshadow="false"/>
    <light name="green_fill" pos="0 3 4" dir="0 -1 -1"
           diffuse="0.2 0.6 0.3" castshadow="false"/>

    <!-- ── Ground (12x8 environment -> half sizes 6x4) ─────────────────── -->
    <geom name="floor" type="plane" size="6 4 0.1"
          material="floor_mat" friction="1.2 0.01 0.001"/>

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

    <!-- ── RANGER MINI ROBOT ────────────────────────────────────────────── -->
    <!--
      Spawned in left corridor, facing North (+Y direction)
    -->
    <body name="base_link" pos="-4.5 0 0.34" euler="0 0 1.5708">
      <freejoint name="root"/>
      <inertial pos="-0.0169 0.0068 0.0578"
                mass="88.757"
                fullinertia="1.7123 4.9003 6.3943 0.0111 -0.0194 -0.0001"/>
      <geom name="chassis_box" type="box" size="0.42 0.22 0.12" pos="0 0 0.02"
            material="chassis_mat"/>
      <site name="imu_site" pos="0 0 0" size="0.01"/>

      <body name="camera_link" pos="0.65 0 0.20">
        <geom name="cam_marker" type="box" size="0.025 0.035 0.020"
              material="cam_mat" contype="0" conaffinity="0"/>
        <camera name="front_cam" fovy="80" xyaxes="0 -1 0 0 0 1"/>
      </body>

      <body name="lidar_link" pos="0.65 0 0.25">
        <geom name="lidar_marker" type="cylinder" size="0.05 0.03"
              material="lidar_mat" contype="0" conaffinity="0"/>
        <camera name="lidar_cam" fovy="1.0" xyaxes="0 -1 0 0 0 1"/>
      </body>

      <!-- FR corner -->
      <body name="fr_steering_link" pos="0.445 -0.280 0.0335">
        <joint name="fr_steering_joint" type="hinge" axis="0 0 -1"
               range="-0.6109 0.6109" damping="5.0" armature="0.05"/>
        <inertial pos="0.0001118 -0.0073218 -0.085228" mass="2.0957"
                  fullinertia="0.0077827 0.0012664 0.0079221 1.29e-8 7.56e-7 -3.99e-6"/>
        <geom name="fr_knuckle" type="cylinder" size="0.025 0.06" pos="0 0 -0.06"
              material="steering_mat" contype="0" conaffinity="0"/>
        <body name="fr_wheel_link" pos="0 0.001 -0.2918">
          <joint name="fr_wheel_joint" type="hinge" axis="0 1 0" limited="false" damping="0.5"/>
          <inertial pos="-0.00095 0 0.0021" mass="11.468"
                    fullinertia="0.05332 0.08008 0.11423 3.09e-9 8.27e-7 2.15e-7"/>
          <geom name="fr_wheel_col" type="cylinder" size="0.160 0.055"
                euler="1.5708 0 0" material="wheel_mat"/>
        </body>
      </body>

      <!-- FL corner -->
      <body name="fl_steering_wheel_link" pos="0.445 0.280 0.0335">
        <joint name="fl_steering_joint" type="hinge" axis="0 0 -1"
               range="-0.6109 0.6109" damping="5.0" armature="0.05"/>
        <inertial pos="0.00010411 0.0077919 -0.086394" mass="2.1046"
                  fullinertia="0.0077836 0.0012676 0.007923 1.57e-8 1.35e-6 3.33e-6"/>
        <geom name="fl_knuckle" type="cylinder" size="0.025 0.06" pos="0 0 -0.06"
              material="steering_mat" contype="0" conaffinity="0"/>
        <body name="fl_wheel_link" pos="0 -0.001 -0.29345">
          <joint name="fl_wheel_joint" type="hinge" axis="0 1 0" limited="false" damping="0.5"/>
          <inertial pos="0.000311 0 -0.002099" mass="11.4679"
                    fullinertia="0.05332 0.08008 0.11423 3.01e-9 8.27e-7 2.15e-7"/>
          <geom name="fl_wheel_col" type="cylinder" size="0.160 0.055"
                euler="1.5708 0 0" material="wheel_mat"/>
        </body>
      </body>

      <!-- RL corner -->
      <body name="rl_steering_wheel_link" pos="-0.445 0.280 0.0335">
        <joint name="rl_steering_joint" type="hinge" axis="0 0 -1"
               range="-0.6109 0.6109" damping="5.0" armature="0.05"/>
        <inertial pos="0.000105 0.007473 -0.085165" mass="2.09214"
                  fullinertia="0.007783 0.001266 0.007922 1.57e-8 1.35e-6 3.33e-6"/>
        <geom name="rl_knuckle" type="cylinder" size="0.025 0.06" pos="0 0 -0.06"
              material="steering_mat" contype="0" conaffinity="0"/>
        <body name="rl_wheel_link" pos="0 -0.001 -0.29345">
          <joint name="rl_wheel_joint" type="hinge" axis="0 1 0" limited="false" damping="0.5"/>
          <inertial pos="0.000311 0 -0.002099" mass="11.4679"
                    fullinertia="0.05332 0.08008 0.11423 3.01e-9 8.27e-7 2.15e-7"/>
          <geom name="rl_wheel_col" type="cylinder" size="0.160 0.055"
                euler="1.5708 0 0" material="wheel_mat"/>
        </body>
      </body>

      <!-- RR corner -->
      <body name="rr_steering_wheel_link" pos="-0.445 -0.280 0.0335">
        <joint name="rr_steering_joint" type="hinge" axis="0 0 -1"
               range="-0.6109 0.6109" damping="5.0" armature="0.05"/>
        <inertial pos="0.000105 0.007473 -0.085165" mass="2.09214"
                  fullinertia="0.007783 0.001266 0.007922 1.57e-8 1.35e-6 3.33e-6"/>
        <geom name="rr_knuckle" type="cylinder" size="0.025 0.06" pos="0 0 -0.06"
              material="steering_mat" contype="0" conaffinity="0"/>
        <body name="rr_wheel_link" pos="0 0.001 -0.2918">
          <joint name="rr_wheel_joint" type="hinge" axis="0 1 0" limited="false" damping="0.5"/>
          <inertial pos="-0.000951 0 0.002095" mass="11.4679"
                    fullinertia="0.05332 0.08008 0.11423 3.09e-9 8.27e-7 2.15e-7"/>
          <geom name="rr_wheel_col" type="cylinder" size="0.160 0.055"
                euler="1.5708 0 0" material="wheel_mat"/>
        </body>
      </body>

    </body> <!-- /base_link -->

  </worldbody>

  <actuator>
    <position name="act_fl_steer" joint="fl_steering_joint" kp="100" ctrlrange="-0.6109 0.6109"/>
    <position name="act_fr_steer" joint="fr_steering_joint" kp="100" ctrlrange="-0.6109 0.6109"/>
    <position name="act_rl_steer" joint="rl_steering_joint" kp="100" ctrlrange="-0.6109 0.6109"/>
    <position name="act_rr_steer" joint="rr_steering_joint" kp="100" ctrlrange="-0.6109 0.6109"/>
    <velocity name="act_fl_drive" joint="fl_wheel_joint" kv="60"/>
    <velocity name="act_fr_drive" joint="fr_wheel_joint" kv="60"/>
    <velocity name="act_rl_drive" joint="rl_wheel_joint" kv="60"/>
    <velocity name="act_rr_drive" joint="rr_wheel_joint" kv="60"/>
  </actuator>

  <sensor>
    <accelerometer name="imu_accel" site="imu_site"/>
    <gyro          name="imu_gyro"  site="imu_site"/>
    <jointvel name="fl_wheel_vel" joint="fl_wheel_joint"/>
    <jointvel name="fr_wheel_vel" joint="fr_wheel_joint"/>
    <jointvel name="rl_wheel_vel" joint="rl_wheel_joint"/>
    <jointvel name="rr_wheel_vel" joint="rr_wheel_joint"/>
    <jointpos name="fl_steer_pos" joint="fl_steering_joint"/>
    <jointpos name="fr_steer_pos" joint="fr_steering_joint"/>
    <jointpos name="rl_steer_pos" joint="rl_steering_joint"/>
    <jointpos name="rr_steer_pos" joint="rr_steering_joint"/>
  </sensor>

</mujoco>"""
    return xml


def main() -> None:
    global quit_flag

    # Write the merged world XML to a temp file
    world_xml = Path(__file__).parent / "_test_world.xml"
    world_xml.write_text(build_world_xml(), encoding="utf-8")

    print("═" * 70)
    print("  Ranger Mini  —  Phase 5: Test Environment")
    print("═" * 70)
    print("  World: obstacle course with slalom boxes, cones, ramp, corridor")
    print("  ↑ / ↓    : forward / backward")
    print("  ← / →    : turn left / right  (Ackermann 4WS)")
    print("  Enter     : stop  |  Escape : quit")
    print("  ⚠  Click the viewer window to focus it first!")
    print("═" * 70)

    robot = RangerRobot()
    robot.load(world_xml)

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
