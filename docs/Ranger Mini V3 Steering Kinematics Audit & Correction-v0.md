# Ranger Mini V3 Steering Kinematics Audit & Correction

We have successfully overhauled the steering inverse kinematics (IK) logic for the Ranger Mini V3 to guarantee smooth, continuous steering transitions without 180° unwinding snaps.

## The Mathematics Explained

A 4WD4WS robot can drive a wheel in an infinite number of configurations to satisfy a velocity vector `(Vx, Vy)`. The two physically distinct approaches are:
1. Steer towards the vector and spin the wheel forward.
2. Steer exactly opposite to the vector and spin the wheel in reverse.

The previous `_get_wheel_ik` stateless implementation forced the steering angle strictly into the `[-90°, 90°]` range using standard atan2 clamps. This caused "unwinding" where, if the steering target transitioned across the 90° threshold (e.g. from 85° to 95°), the motor would violently flip 180° back to -85° and invert the wheel speed. 

We replaced this with a **continuous shortest-path algorithm**:
1. We compute both potential physical configurations (forward-drive vs. reverse-drive).
2. We read the *current* wheel angle directly from the simulation sensors (`self.read_steering_angles()`).
3. We calculate the absolute angular distance from the current position to both target solutions, accounting for cyclic wrapping across `2π`.
4. We choose the target angle requiring the smallest physical rotation.

## Implementation Updates
- `robot_base.py`: Deprecated the stateless `ackermann_angles` and `ackermann_wheel_speeds` functions. Replaced them with the unified, state-aware `compute_4ws_ik(cmd, dims, current_angles)` function.
- `robot.py`: Modified `apply_command()` to dynamically fetch the sensor positions of the steering servos, enabling closed-loop setpoint generation.

## Validation Results

We executed the `validate_kinematics.py` script to simulate mathematical operations across the 4 major control modes, plus a dedicated continuous wrapping test.

### 1. Straight Line Mode
- All steering angles at `0.00 deg`.
- All wheel speeds equal at `10.00 rad/s`.

### 2. Constant Radius Turn (Ackermann)
- Front inner wheel steers sharpest (`-17.35 deg`).
- Front outer wheel steers less (`-11.77 deg`).
- Proper Ackermann turning center maintained mathematically.

### 3. Spin Mode (Zero-Radius Turn)
- All wheels angle tangentially around the chassis center (`±51.34 deg`).
- Left wheels spin backwards (`-6.40 rad/s`) while right wheels spin forwards (`6.40 rad/s`).

### 4. Traverse Mode (Crab)
- All wheels steer strictly to `-90.00 deg`.
- All wheels spin uniformly to create pure lateral translation.

### 5. Diagonal Mode
- Both longitudinal and lateral velocity commanded.
- All wheels perfectly synchronize to `-45.00 deg` at `14.14 rad/s`.

### 6. Continuous Wrapping (No Unwinding)
- **Scenario:** The wheel is currently steered to `80 deg`. A command requires `100 deg` steering.
- **Previous behavior:** The system would snap to `-80 deg` and reverse the drive speed (traveling 160 degrees).
- **New behavior:** The system smoothly advances `20 deg` to reach the `100 deg` setpoint, avoiding unwinding!
