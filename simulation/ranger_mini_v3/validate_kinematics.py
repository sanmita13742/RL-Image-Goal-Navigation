import sys
from pathlib import Path
import math

sys.path.append(str(Path(__file__).resolve().parent.parent))
from robot_base import DriveCommand, RobotDimensions, compute_4ws_ik

def print_result(title, angles, speeds):
    print(f"--- {title} ---")
    names = ["FL", "FR", "RL", "RR"]
    for name, a, s in zip(names, angles, speeds):
        print(f"{name}: Angle={math.degrees(a):8.2f} deg | Speed={s:6.2f} rad/s")
    print()

def main():
    dims = RobotDimensions(
        wheel_radius=0.1,
        track_width=0.4,
        wheelbase=0.5,
        mass_total=100.0,
        description="TestRobot"
    )
    
    # 1. Straight Line (Ackermann)
    cmd_straight = DriveCommand(v_linear=1.0, v_lateral=0.0, v_angular=0.0)
    current = (0.0, 0.0, 0.0, 0.0)
    angles, speeds = compute_4ws_ik(cmd_straight, dims, current)
    print_result("1. Straight Line (v_linear=1.0)", angles, speeds)
    
    # 2. Constant Radius Turn (Ackermann)
    cmd_turn = DriveCommand(v_linear=1.0, v_lateral=0.0, v_angular=1.0) # Turning CCW
    angles, speeds = compute_4ws_ik(cmd_turn, dims, current)
    print_result("2. Ackermann Turn (v_linear=1.0, v_angular=1.0)", angles, speeds)
    
    # 3. Spin Mode
    cmd_spin = DriveCommand(v_linear=0.0, v_lateral=0.0, v_angular=2.0)
    angles, speeds = compute_4ws_ik(cmd_spin, dims, current)
    print_result("3. Spin Mode (v_angular=2.0)", angles, speeds)
    
    # 4. Traverse Mode (Pure lateral)
    cmd_traverse = DriveCommand(v_linear=0.0, v_lateral=1.0, v_angular=0.0)
    angles, speeds = compute_4ws_ik(cmd_traverse, dims, current)
    print_result("4. Traverse Mode (v_lateral=1.0)", angles, speeds)
    
    # 5. Diagonal Mode
    cmd_diagonal = DriveCommand(v_linear=1.0, v_lateral=1.0, v_angular=0.0)
    angles, speeds = compute_4ws_ik(cmd_diagonal, dims, current)
    print_result("5. Diagonal Mode (v_linear=1.0, v_lateral=1.0)", angles, speeds)
    
    # 6. Continuous Wrapping Demo
    # Start steering at 80 degrees, command a target that would be 100 degrees physical
    print("--- 6. Continuous Wrapping (No Unwinding) ---")
    current_80 = (-math.radians(80), -math.radians(80), -math.radians(80), -math.radians(80)) # -80 joint = 80 physical
    
    # Target physical is 100 deg: vx = cos(100), vy = sin(100) -> vx=-0.173, vy=0.984
    cmd_100 = DriveCommand(v_linear=-0.1736, v_lateral=0.9848, v_angular=0.0)
    angles, speeds = compute_4ws_ik(cmd_100, dims, current_80)
    
    # If stateless, it would map to -80 deg (flip 180 and drive backwards)
    # With wrapping, it should map to 100 deg physical (-100 deg joint)
    print(f"Current Joint Angle: -80 deg")
    print(f"New Joint Angle:   {math.degrees(angles[0]):.2f} deg")
    print(f"New Wheel Speed:   {speeds[0]:.2f} rad/s")
    print()

if __name__ == "__main__":
    main()
