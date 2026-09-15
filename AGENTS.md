# MINav Project Context & State (AGENTS.md)

This file maintains the persistent context and current state of the MINav (Offline RL for Image-Goal Navigation) project. It is automatically read by the Antigravity agent when starting a new chat session in this workspace.

## 1. Project Overview
**Goal**: Build a complete, portable, end-to-end reproduction pipeline for the MINav offline RL navigation system.
**Core Tech Stack**: 
- **Environment**: MuJoCo (Ranger Mini V3 Robot)
- **Representations**: DINOv3 (ViT-Small) for visual embeddings (384-dimensional state space)
- **Algorithm**: TD3+BC for Offline Reinforcement Learning
- **Relabeling**: Hindsight Geometric and Uniform Relabeling

## 2. Directory & Pipeline Architecture
The core simulation and pipeline code is located in `simulation/ranger_mini_v3/`.

### The End-to-End Pipeline
The pipeline is executed via `./run_pipeline.sh` or `scripts/pipeline.py` and runs in three mandatory phases:
1. **Data Exploration** (`scripts/run_exploration.py`): 10Hz Pink Noise policy drives the robot around the map to collect visual and spatial data. (CPU bound).
2. **Dataset Processing & Hindsight** (`scripts/build_final_dataset.py`): Converts images to DINOv3 embeddings and generates hindsight relabeled goal transitions (Uniform & Geometric). (GPU bound).
3. **TD3+BC Training** (`scripts/train_td3_bc.py`): Trains the offline actor-critic network on the generated dataset. (GPU bound).

*Note: You can pass `--smoke` to the pipeline to run a hyper-compressed 5-minute version for rapid local debugging.*

## 3. The Environment: `maps/complex`
We have designed a custom 16x16m indoor navigation maze to rigorously stress-test the model:
- **Location**: `simulation/ranger_mini_v3/maps/complex/scene.xml`
- **Visual Aliasing**: The map features identically colored U-shaped structures (Green -> Yellow -> Orange) on opposite sides of the map to test whether DINOv3 learns Euclidean geometry or simply memorizes colors.
- **Search Cells**: The map includes 3 hidden search cells embedded in the walls containing distinct random objects (Blue sphere, Yellow box, Orange cylinder) for discovery.
- **Landmarks**: Cyan cylinder, Red cone, Purple box.

## 4. Critical Implementation Rules
- **Action Normalization**: Actions must strictly be affine normalized between their physical limits (`[-0.9, -1.0, -1.5]` to `[1.8, 1.0, 1.5]`) and `[-1, 1]`. Simple division distorts the asymmetric `vx` behavior policy.
- **No Absolute Paths**: All paths in scripts must use `pathlib.Path` resolved relative to the configuration file or workspace root.
- **Device Placement**: Tensors must explicitly use the `device` parameter parsed from arguments (`cuda` or `cpu`). `.cuda()` should never be hardcoded.

## 5. Current Project Status
- **Completed**: The `complex` navigation map is fully designed, tested, and merged into the `main` branch.
- **Completed**: The MuJoCo pipeline works end-to-end locally (CPU/GPU agnostic) and passes all validation checks in smoke mode.
- **Completed**: The real-world ROS 2 pipeline (`realworld/`) is fully implemented with LiDAR collision avoidance, safety watchdogs, and a mandatory pre-flight smoke test suite.
- **Next Steps**: Execute the full-scale (2-hour) exploration and training run on the real Ranger Mini V3 hardware.

## 6. Real-World Pipeline (ROS 2 Foxy)
- **Location**: `realworld/`
- **Hardware**: Ranger Mini V3, Intel RealSense RGB camera, RoboSense LiDAR.
- **Safety First**: The exploration script (`realworld/scripts/run_exploration.py`) operates in **DRY-RUN** mode by default. The robot will not move unless explicitly passed `--arm`.
- **Pre-flight Checks**: Before arming the robot, all 11 test suites in `realworld/tests/` MUST pass via the `smoke_test.py` script. This validates topic connectivity, camera/LiDAR feeds, and action normalizations.
- **Shared Code**: The real-world pipeline reuses the dataset building, DINOv3 encoding, hindsight relabeling, and TD3+BC training scripts from the simulation pipeline to guarantee 1:1 compatibility.

---
*Agent Note: When assisting the user in this repository, always reference this context to understand the system design, environment boundaries, and the progress made so far.*
