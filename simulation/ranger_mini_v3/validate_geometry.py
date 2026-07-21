import mujoco
from pathlib import Path
import sys

# Ensure simulation package is in path
sys.path.append(str(Path(__file__).resolve().parent))
from robot import RangerMiniV3Robot

def main():
    print("=== Ranger Mini V3 Geometry Validation ===")
    
    # Initialize the robot
    robot = RangerMiniV3Robot()
    
    # Print dynamic constants extracted from MJCF
    print(f"\n[Extracted from MJCF]")
    print(f"WHEELBASE:    {robot.WHEELBASE:.4f} m (Expected: 0.494 m)")
    print(f"TRACK_WIDTH:  {robot.TRACK_WIDTH:.4f} m (Expected: 0.364 m)")
    print(f"WHEEL_RADIUS: {robot.WHEEL_RADIUS:.4f} m (Expected: 0.100 m)")
    
    # Run a single step to update forward kinematics (global positions)
    robot.step()
    
    # Check Chassis footprint
    geom_chassis_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_GEOM, "chassis_box")
    chassis_size = robot.model.geom_size[geom_chassis_id]
    print(f"\n[Chassis Dimensions]")
    print(f"Length: {chassis_size[0] * 2:.3f} m (Expected: 0.720 m)")
    print(f"Width:  {chassis_size[1] * 2:.3f} m (Expected: 0.500 m)")
    print(f"Height: {chassis_size[2] * 2:.3f} m (Expected: 0.345 m)")
    
    # Check Ground Clearance
    # The bottom of the chassis box is pos_z - half_height
    chassis_z = robot.data.geom_xpos[geom_chassis_id][2]
    clearance = chassis_z - chassis_size[2]
    print(f"\n[Ground Clearance]")
    print(f"Clearance: {clearance:.4f} m (Expected: 0.1050 m)")
    
    # Verify Wheel Positions and Ground Contact
    print(f"\n[Wheel Positions & Contact]")
    for wheel_name in ["fl_wheel_col", "fr_wheel_col", "rl_wheel_col", "rr_wheel_col"]:
        geom_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_GEOM, wheel_name)
        pos = robot.data.geom_xpos[geom_id]
        
        # Bottom of the wheel
        radius = robot.model.geom_size[geom_id][0]
        contact_z = pos[2] - radius
        
        print(f" - {wheel_name}: center=(X:{pos[0]:.4f}, Y:{pos[1]:.4f}, Z:{pos[2]:.4f}), Ground Contact Z={contact_z:.4f} m")

if __name__ == "__main__":
    main()
