# Ranger Mini — MuJoCo Simulation Documentation

## Robot Hierarchy

```
Ranger Mini (ranger.xml)
├── base_link                     [free body — 88.76 kg chassis]
│   ├── camera_link               [pos: +0.38m fwd, +0.15m up]
│   │   └── front_cam             [RGB camera, fovy=80°]
│   ├── lidar_link                [pos: top-centre, +0.22m up]
│   │   └── lidar_cam             [depth camera proxy, fovy=1°]
│   │
│   ├── fr_steering_link          [pos: +0.445, -0.280, +0.0335]
│   │   ├── fr_steering_joint     [hinge, axis Z-down, ±35°]
│   │   └── fr_wheel_link         [pos: 0, +0.001, -0.2918]
│   │       └── fr_wheel_joint    [hinge, axis Y, continuous]
│   │
│   ├── fl_steering_wheel_link    [pos: +0.445, +0.280, +0.0335]
│   │   ├── fl_steering_joint     [hinge, axis Z-down, ±35°]
│   │   └── fl_wheel_link         [pos: 0, -0.001, -0.29345]
│   │       └── fl_wheel_joint    [hinge, axis Y, continuous]
│   │
│   ├── rl_steering_wheel_link    [pos: -0.445, +0.280, +0.0335]
│   │   ├── rl_steering_joint     [hinge, axis Z-down, ±35°]
│   │   └── rl_wheel_link         [pos: 0, -0.001, -0.29345]
│   │       └── rl_wheel_joint    [hinge, axis Y, continuous]
│   │
│   └── rr_steering_wheel_link    [pos: -0.445, -0.280, +0.0335]
│       ├── rr_steering_joint     [hinge, axis Z-down, ±35°]
│       └── rr_wheel_link         [pos: 0, +0.001, -0.2918]
│           └── rr_wheel_joint    [hinge, axis Y, continuous]
│
├── imu_site                      [site at base_link origin]
```

## Physical Dimensions

| Parameter | Value | Source |
|---|---|---|
| Wheelbase (front ↔ rear) | **0.890 m** (89 cm) | URDF joint x-offsets ±0.445 |
| Track width (left ↔ right) | **0.560 m** (56 cm) | URDF joint y-offsets ±0.280 |
| Steering column height | **+0.0335 m** above base | URDF joint z-offset |
| Wheel drop (steering → hub) | **~0.292 m** | URDF wheel joint z-offset |
| Wheel radius | **0.160 m** (16 cm) | Estimated from geometry |
| Steering range | **±35°** (±0.6109 rad) | Conservative (URDF has ±180°) |
| Base mass | **88.76 kg** | URDF |
| Per steering assembly | **2.095–2.105 kg** | URDF |
| Per wheel | **11.468 kg** | URDF |
| Total mass | **≈143 kg** | Base + 4×(steer+wheel) |
| Spawn height | **0.34 m** | Tuned so wheels rest on ground |

## Actuators (ctrl[] Index Map)

| Index | Name | Joint | Type | Units |
|---|---|---|---|---|
| `ctrl[0]` | `act_fl_steer` | `fl_steering_joint` | position (kp=100) | rad |
| `ctrl[1]` | `act_fr_steer` | `fr_steering_joint` | position (kp=100) | rad |
| `ctrl[2]` | `act_rl_steer` | `rl_steering_joint` | position (kp=100) | rad |
| `ctrl[3]` | `act_rr_steer` | `rr_steering_joint` | position (kp=100) | rad |
| `ctrl[4]` | `act_fl_drive` | `fl_wheel_joint` | velocity (kv=60) | rad/s |
| `ctrl[5]` | `act_fr_drive` | `fr_wheel_joint` | velocity (kv=60) | rad/s |
| `ctrl[6]` | `act_rl_drive` | `rl_wheel_joint` | velocity (kv=60) | rad/s |
| `ctrl[7]` | `act_rr_drive` | `rr_wheel_joint` | velocity (kv=60) | rad/s |

## Sensors

| Name | Type | Output | Units |
|---|---|---|---|
| `imu_accel` | accelerometer | [ax, ay, az] | m/s² |
| `imu_gyro` | gyroscope | [gx, gy, gz] | rad/s |
| `fl_wheel_vel` | jointvel | scalar | rad/s |
| `fr_wheel_vel` | jointvel | scalar | rad/s |
| `rl_wheel_vel` | jointvel | scalar | rad/s |
| `rr_wheel_vel` | jointvel | scalar | rad/s |
| `fl_steer_pos` | jointpos | scalar | rad |
| `fr_steer_pos` | jointpos | scalar | rad |
| `rl_steer_pos` | jointpos | scalar | rad |
| `rr_steer_pos` | jointpos | scalar | rad |

## Cameras

| Name | Type | FoVY | Mounted At | Purpose |
|---|---|---|---|---|
| `front_cam` | RGB | 80° | +0.38m fwd, +0.15m up | Visual navigation |
| `lidar_cam` | Depth | 1° (narrow) | top-centre, +0.22m up | LiDAR proxy |

## Drive Model: Ackermann 4WD4WS

The Ranger uses **Ackermann steering geometry** — each corner steers independently. For a turning radius R:

```
fl_angle = atan2(wheelbase,  R - track/2)
fr_angle = atan2(wheelbase,  R + track/2)
rl_angle = -atan2(wheelbase/2, R - track/2)   (rear counter-steer)
rr_angle = -atan2(wheelbase/2, R + track/2)
```

Per-wheel drive speeds differ based on each wheel's distance from the turning centre.

## Project Architecture (Abstraction Layer)

```
RL/
├── robot_base.py              # Abstract RobotBase, DriveCommand, ackermann_angles()
├── scout_mujoco/
│   ├── scout_mini.xml         # Scout MuJoCo model
│   ├── robot.py               # ScoutMiniRobot (skid-steer)
│   ├── drive.py               # Keyboard teleop (uses ScoutMiniRobot)
│   └── check_camera.py        # Camera test
└── ranger_mujoco/
    ├── ranger.xml             # Ranger MuJoCo model
    ├── robot.py               # RangerRobot (Ackermann 4WD4WS)
    ├── verify.py              # Phase 1: load + topology check
    ├── drive.py               # Phase 3: keyboard teleop
    ├── check_sensors.py       # Phase 4: sensor verification
    └── test_env.py            # Phase 5: obstacle course environment
```

### To add a new robot

1. Create `my_robot_mujoco/robot.py` subclassing `RobotBase`
2. Override `apply_command()`, `get_dimensions()`, `joint_names()`, `actuator_names()`
3. Create MJCF and scripts — `drive.py` and `check_sensors.py` patterns are reusable as-is.

## Verification Results

| Phase | Script | Status |
|---|---|---|
| Phase 1 — Load model | `verify.py` | ✅ 25/25 checks pass |
| Phase 2 — Inspect topology | `verify.py` | ✅ All bodies/joints/actuators/sensors correct |
| Phase 3 — Drive | `drive.py` | Ready (run interactively) |
| Phase 4 — Sensors | `check_sensors.py` | ✅ 14/14 checks pass — IMU accel=9.81m/s², camera 480×640 saved |
| Phase 5 — Test environment | `test_env.py` | Ready (run interactively) |
