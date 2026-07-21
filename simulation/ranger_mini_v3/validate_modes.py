import mujoco
import numpy as np
from pathlib import Path
import sys
import math

sys.path.append(str(Path(__file__).resolve().parent.parent))
from robot_base import DriveCommand, RobotDimensions, compute_4ws_ik
from ranger_mini_v3.robot import RangerMiniV3Robot

def print_diff(name, actual, expected, unit):
    diff = actual - expected
    status = "OK" if abs(diff) < 0.05 else "WARN"
    print(f"  {name:15}: {actual:7.3f} {unit} (Expected: {expected:7.3f}, Diff: {diff:7.3f}) [{status}]")

def test_mode(robot, mode_name, cmd, duration=3.0):
    print(f"\n=========================================")
    print(f"Mode: {mode_name}")
    print(f"Command: v_linear={cmd.v_linear}, v_lateral={cmd.v_lateral}, v_angular={cmd.v_angular}")
    print(f"=========================================")
    
    mujoco.mj_resetData(robot.model, robot.data)
    
    # Calculate Expected Kinematics
    dims = robot.get_dimensions()
    current_angles = (0.0, 0.0, 0.0, 0.0)
    expected_angles, expected_speeds = compute_4ws_ik(cmd, dims, current_angles)
    
    dt = robot.model.opt.timestep
    steps = int(duration / dt)
    
    # Run simulation to settle
    for _ in range(steps):
        robot.apply_command(cmd)
        mujoco.mj_step(robot.model, robot.data)
        
    # Take measurements
    vel = np.zeros(6)
    mujoco.mj_objectVelocity(robot.model, robot.data, mujoco.mjtObj.mjOBJ_BODY, 
                            mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"), 
                            vel, 1) # 1 = local frame
                            
    actual_linear = vel[3]
    actual_lateral = vel[4]
    actual_yaw_rate = vel[2]
    
    print("--- Chassis Velocities ---")
    print_diff("Linear X", actual_linear, cmd.v_linear, "m/s")
    print_diff("Lateral Y", actual_lateral, cmd.v_lateral, "m/s")
    print_diff("Yaw Rate", actual_yaw_rate, cmd.v_angular, "rad/s")
    
    if cmd.v_angular != 0:
        expected_radius = math.hypot(cmd.v_linear, cmd.v_lateral) / abs(cmd.v_angular)
        actual_radius = math.hypot(actual_linear, actual_lateral) / abs(actual_yaw_rate) if abs(actual_yaw_rate) > 1e-3 else float('inf')
        print_diff("Turn Radius", actual_radius, expected_radius, "m")
        
    print("\n--- Steering Angles ---")
    sensor_fl_steer = robot.data.sensor("fl_steer_pos").data[0]
    sensor_fr_steer = robot.data.sensor("fr_steer_pos").data[0]
    sensor_rl_steer = robot.data.sensor("rl_steer_pos").data[0]
    sensor_rr_steer = robot.data.sensor("rr_steer_pos").data[0]
    
    print_diff("FL Angle", math.degrees(sensor_fl_steer), math.degrees(expected_angles[0]), "deg")
    print_diff("FR Angle", math.degrees(sensor_fr_steer), math.degrees(expected_angles[1]), "deg")
    print_diff("RL Angle", math.degrees(sensor_rl_steer), math.degrees(expected_angles[2]), "deg")
    print_diff("RR Angle", math.degrees(sensor_rr_steer), math.degrees(expected_angles[3]), "deg")
    
    print("\n--- Wheel Speeds ---")
    sensor_fl_vel = robot.data.sensor("fl_wheel_vel").data[0]
    sensor_fr_vel = robot.data.sensor("fr_wheel_vel").data[0]
    sensor_rl_vel = robot.data.sensor("rl_wheel_vel").data[0]
    sensor_rr_vel = robot.data.sensor("rr_wheel_vel").data[0]
    
    print_diff("FL Speed", sensor_fl_vel, expected_speeds[0], "rad/s")
    print_diff("FR Speed", sensor_fr_vel, expected_speeds[1], "rad/s")
    print_diff("RL Speed", sensor_rl_vel, expected_speeds[2], "rad/s")
    print_diff("RR Speed", sensor_rr_vel, expected_speeds[3], "rad/s")

def main():
    robot = RangerMiniV3Robot()
    
    # 1. Straight Drive
    test_mode(robot, "Straight Drive", DriveCommand(v_linear=1.5, v_lateral=0.0, v_angular=0.0))
    
    # 2. Constant Turn
    test_mode(robot, "Constant Turn", DriveCommand(v_linear=1.0, v_lateral=0.0, v_angular=1.0))
    
    # 3. Spin Mode
    test_mode(robot, "Spin Mode", DriveCommand(v_linear=0.0, v_lateral=0.0, v_angular=1.5))
    
    # 4. Traverse Mode
    test_mode(robot, "Traverse Mode", DriveCommand(v_linear=0.0, v_lateral=1.0, v_angular=0.0))
    
    # 5. Diagonal Mode
    test_mode(robot, "Diagonal Mode", DriveCommand(v_linear=1.0, v_lateral=1.0, v_angular=0.0))

if __name__ == "__main__":
    main()
