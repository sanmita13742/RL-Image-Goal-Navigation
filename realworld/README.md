# Real-World MINav Pipeline

This directory contains the real-robot execution pipeline for the MINav project. It is designed to run on the **Ranger Mini V3** robot via **ROS 2 Foxy**, using an Intel RealSense camera for image goals and a RoboSense LiDAR for collision avoidance.

## Architecture

This pipeline is the real-world counterpart to the MuJoCo simulation in `simulation/ranger_mini_v3/`. 
- **Data Collection:** Uses ROS 2 topics instead of the MuJoCo physics engine.
- **Shared Modules:** The DINOv3 encoder, Hindsight Relabeling, and TD3+BC training code are identical to the simulation pipeline and are executed via thin wrappers in `scripts/`.
- **Safety First:** The robot starts in a `DRY-RUN` state by default. A dedicated watchdog monitors sensor liveness and publishes emergency stops on timeout.

## Hardware Stack

Confirmed via rosbag and launch scripts:
- **Robot:** Ranger Mini V3 via CAN bus (`ranger_bringup` package)
- **Camera:** Intel RealSense (`realsense2_camera` package, topic: `/camera/color/image_raw`)
- **LiDAR:** RoboSense (`/rslidar_points`, ~10 Hz)
- **Odometry:** `ranger_base` wheel odometry (`/odom`, ~50 Hz)
- **ROS 2:** Foxy

## Setup

1. Source your ROS 2 Foxy installation and the Ranger workspaces:
   ```bash
   source /opt/ros/foxy/setup.bash
   source ~/agilex_ws/install/setup.bash
   source ~/rslidar_ws/install/setup.bash
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Workflow

### 1. Mandatory Pre-Flight Checks
Before running the robot, you MUST pass the smoke test. This ensures all ROS topics are active and basic code invariants hold.
```bash
python realworld/tests/smoke_test.py --phase all
```

### 2. Full Pipeline Execution
To run the full end-to-end pipeline (Exploration → DINOv3 → Hindsight → TD3+BC Training):

**Dry-Run (Default):**
```bash
python realworld/scripts/pipeline.py
```

**Armed (Robot will move):**
```bash
python realworld/scripts/pipeline.py --arm
```

### 3. Deploy Trained Policy
To run inference with a trained actor checkpoint on the live robot:
```bash
python realworld/scripts/deploy_policy.py \
    --checkpoint path/to/actor.pth \
    --goal-image path/to/goal.png \
    --arm
```
