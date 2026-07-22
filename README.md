# Ranger Mini V3 RL & Simulation Framework

Welcome to the Ranger Mini V3 Reinforcement Learning and Simulation project! This repository contains a complete pipeline for simulating the AgileX Ranger Mini V3 robot using MuJoCo, specifically focusing on its Ackermann 4-wheel steering kinematics, data collection via random exploration, and dataset analysis tools.

## 🛠 Setup Guide

Follow these steps to pull the repository and set up your local environment.

### 1. Clone the Repository
Pull the latest code from the `data-exploration` branch (or your working branch) to your local machine:
```bash
git clone -b data-exploration https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Set Up a Virtual Environment (Recommended)
It is highly recommended to use a virtual environment to manage dependencies.
```bash
# Create a virtual environment named .venv
python -m venv .venv

# Activate it (Windows)
.\.venv\Scripts\activate

# Activate it (Linux/Mac)
source .venv/bin/activate
```

### 3. Install Dependencies
Install all the required Python packages (including `mujoco`, `pandas`, `matplotlib`, `seaborn`, etc.) via the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Simulations (Walkthrough)

The core simulation scripts are located in the `simulation/ranger_mini_v3/` directory. Ensure your terminal's current working directory is set to this folder or you are running them via their relative paths.

Navigate to the simulation directory:
```bash
cd simulation/ranger_mini_v3/
```

### 1. `drive.py` (Free Driving)
A minimal playground to verify the robot loads correctly and the Ackermann steering kinematics work. 
- **What it does:** Spawns the Ranger Mini V3 in an empty MuJoCo world.
- **How to use:** Run `python drive.py`. Click the viewer window to focus, and use the **Arrow Keys** to drive. Press `[` or `]` to adjust steering sensitivity.

### 2. `test_env.py` (Obstacle Course)
A richer environment used to test the robot's maneuverability.
- **What it does:** Spawns the robot in a 12x8 meter walled environment filled with slalom boxes, cones, and ramps.
- **How to use:** Run `python test_env.py`. Drive around using the **Arrow Keys** and test collision handling.

### 3. `random_explore.py` (Data Collection)
An autonomous exploration script that collects data for Reinforcement Learning or dataset validation.
- **What it does:** The robot runs a `PrimitiveExplorationPolicy` to autonomously navigate the `test_env` course, avoiding obstacles using a simulated LiDAR depth camera. It logs RGB frames, depth frames, poses (X, Y, Yaw), and velocity commands at 10Hz.
- **How to use:** Run `python random_explore.py`. It will run and save data to the `dataset/` folder (specifically `dataset/log.csv` and optionally images).

### 4. `analyze_dataset.py` (Data Analysis & Plotting)
A comprehensive analysis suite to evaluate the quality of the dataset collected by `random_explore.py`.
- **What it does:** Reads `dataset/log.csv` and generates analytics for:
  - Trajectory & Speed (e.g., `xy_trajectory.png`)
  - Occupancy & Revisit Heatmaps (e.g., `revisit_heatmap.png`)
  - Motion Histograms
  - Loop Closures & Safety Metrics
- **How to use:** Run `python analyze_dataset.py --csv dataset/log.csv --outdir analysis_output`. Check the `analysis_output/` folder for `.png` plots, `.csv` statistics, and a final `report.md` summarizing if the dataset is ready for RL training.

---

## 📚 Documentation Overview

The `docs/` folder contains essential engineering reports and architectural decisions made throughout the project's lifecycle:

- **`Ranger Mini V3 Geometry Update Report-v0`**
  Details the modifications made to the robot's physical model, collision meshes, and inertia properties to ensure accurate physical simulation.

- **`Ranger Mini V3 Steering Kinematics Audit & Correction-v0.md`**
  A deep dive into the Ackermann 4-wheel steering model. Explains the math and corrections applied to ensure the inner and outer wheels turn at the correct angles to prevent slipping.

- **`post-exploration-framwork-res.md` & `post-exploration-framwork-resv2.md`**
  Documentation reviewing the results of the exploration framework, detailing primitive coverage, state entropy, and future improvements for the RL pipeline.

- **`robo-ranger-mini-v3`**
  Contains additional setup notes or supplementary logs related to the Ranger Mini V3 configuration.

---

## Troubleshooting

- **MuJoCo Viewer won't respond:** Always make sure to click inside the popup 3D window before pressing the arrow keys.
- **Missing modules (e.g., `ModuleNotFoundError: No module named 'mujoco'`):** Ensure your virtual environment is activated and you have run `pip install -r requirements.txt` from the project root.
