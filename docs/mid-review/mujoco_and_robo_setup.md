# MuJoCo and Robot Setup

## 1. MuJoCo Architecture
The simulation architecture relies on a shared interface to simulate diverse robotic platforms easily. We defined an abstract base class in [`robot_base.py`](file:///c:/Users/sanmi/Desktop/projects/RL/simulation/robot_base.py) called `RobotBase`. This class enforces a standard API:
- `load(xml_path)`: Loads the MuJoCo model and data.
- `apply_command(v_lin, v_ang)`: A robot-specific method to translate twist commands into joint actuation.
- Dimensions like `WHEEL_RADIUS`, `TRACK_WIDTH`, and `WHEELBASE`.

## 2. Scout Mini Setup
The **Scout Mini** (found in [`simulation/scout/robot.py`](file:///c:/Users/sanmi/Desktop/projects/RL/simulation/scout/robot.py)) implements a skid-steer architecture. It uses 4 drive wheels and relies on differential velocities to turn.

## 3. Ranger Mini V3 Setup
The **Ranger Mini** (found in [`simulation/ranger_mini_v3/robot.py`](file:///c:/Users/sanmi/Desktop/projects/RL/simulation/ranger_mini_v3/robot.py)) is significantly more complex:
- **Kinematics:** Ackermann steering capable, with 4 independent steering joints and 4 drive joints (8 joints total).
- **Actuation:** Uses position actuators for the 4 steering hinges, and velocity actuators for the 4 wheels.
- **Physical Properties (from URDF):**
  - Wheelbase: 0.890m
  - Track Width: 0.560m
  - Base Mass: ~88.76 kg
  - Wheel Mass: ~11.47 kg per wheel

By leveraging the abstract `RobotBase`, the `apply_command` method in the Ranger Robot computes the exact inner and outer Ackermann turning angles required to achieve the requested `v_lin` and `v_ang`. We also modeled sensors (RGB Camera, simulated LiDAR for depth, IMU) directly on the `base_link` of the Ranger MJCF model.
