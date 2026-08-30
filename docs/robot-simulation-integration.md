# Ranger Mini MuJoCo Simulation — Implementation Plan

## Background

The Scout Mini simulation already exists in `scout_mujoco/`. We are now creating an equivalent setup for the **Ranger Mini** robot and introducing an **abstraction layer** so both robots share the same codebase.

The Ranger Mini is fundamentally different from Scout Mini:
- **Scout Mini**: Skid-steer (4 drive wheels, no steering joints)
- **Ranger Mini**: Ackermann-capable with **4 independent steering + 4 drive** (8 joints total)

## Architecture

```
RL/
├── robot_base.py              # [NEW] Abstract base class for all robots
├── scout_mujoco/              # [EXISTING] Scout Mini files
│   ├── scout_mini.xml
│   ├── drive.py               # [MODIFY] Use abstraction layer
│   ├── check_camera.py        # [MODIFY] Use abstraction layer
│   └── meshes/
└── ranger_mujoco/             # [NEW] Ranger Mini files
    ├── ranger.xml             # MJCF converted from URDF
    ├── meshes/                # Symlinked/copied STL files
    ├── drive.py               # Ranger teleop using abstraction
    ├── check_sensors.py       # Verify sensors
    └── verify.py              # Phase 1 verification script
```

## Proposed Changes

### Abstraction Layer

#### [NEW] robot_base.py (at RL/ root)
Abstract base class `RobotBase` with:
- `load(xml_path)` — load model + data
- `apply_command(v_lin, v_ang)` — abstract, robot-specific
- `get_joint_names()` — abstract
- `get_actuator_names()` — abstract
- `WHEEL_RADIUS`, `TRACK_WIDTH`, `WHEELBASE` constants

#### [NEW] ScoutMiniRobot (scout_mujoco/robot.py)
Concrete class for Scout — skid-steer `apply_command`.

#### [NEW] RangerRobot (ranger_mujoco/robot.py)
Concrete class for Ranger — Ackermann steering `apply_command`.
- Front steering: Ackermann geometry (inner/outer angles different)
- Rear steering: can be set to 0 or crab-mode

### ranger_mujoco/ Files

#### [NEW] ranger.xml
Full MJCF converted from URDF with:
- Cylinder collision geometry for each wheel (radius ~0.16m, measured from URDF joint offsets)
- Primitive box chassis (no heavy STL for physics)
- Optional STL visual meshes
- 4 steering hinge joints (position actuators)
- 4 wheel hinge joints (velocity actuators)
- IMU sensor (accelerometer + gyro on base_link)
- Front RGB camera
- Simulated LiDAR camera (depth)

#### [NEW] drive.py
Keyboard teleop using `RangerRobot` abstraction.
- Arrow keys = forward/back/turn
- `[` / `]` = increase/decrease steering sensitivity
- Ackermann angle computation shown in terminal

#### [NEW] check_sensors.py
- Load model, step 200 frames
- Print IMU readings
- Render front camera frame → save PNG
- Render depth (LiDAR proxy) frame → save PNG

#### [NEW] verify.py
Phase 1 verification:
- Load XML, check no warnings
- Print all bodies, joints, actuators, sensors
- Run 500 steps, check energy/stability
- Report robot dimensions

### Modified Scout Files

#### [MODIFY] scout_mujoco/drive.py
Refactor to use `ScoutMiniRobot` class, keeping same behavior.

## Robot Dimensions (from URDF analysis)

| Parameter | Value |
|---|---|
| Wheelbase (front↔rear) | 0.890 m (2 × 0.445 m) |
| Track width (left↔right) | 0.560 m (2 × 0.280 m) |
| Steering z-offset | 0.0335 m above base |
| Wheel drop (from steering) | ~0.292 m |
| Estimated wheel radius | ~0.16 m |
| Base mass | 88.76 kg |
| Per-wheel mass | 11.47 kg |

## Verification Plan

### Phase 1 – Model Loads
Run `python verify.py` — expect: no warnings, stable simulation.

### Phase 2 – Drive
Run `python drive.py` — expect: robot drives with Ackermann steering.

### Phase 3 – Sensors
Run `python check_sensors.py` — expect: camera PNG saved, IMU data printed.
