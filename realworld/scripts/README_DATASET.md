# MINav 2-Hour Dataset Collection

This guide explains how to collect the 2-hour real-world dataset required for training the MINav offline RL policy.

## Overview

The recording script runs `ros2 bag record` specifically configured for the MINav dataset format. 

**Topics Recorded:**
- `/camera/color/image_raw` (RGB Image, 720x1280 natively)
- `/cmd_vel` (FINAL Executed Actions, after the LiDAR safety gate)

**Important Rules:**
- **No LiDAR in Training:** LiDAR (`/rslidar_points`) is intentionally NOT recorded. The MINav TD3+BC policy must learn purely from vision without depth or LiDAR inputs.
- **Executed Actions Only:** We record the final `/cmd_vel` output rather than the policy's raw proposed action. If the LiDAR safety gate blocks an action, the policy learns from the zeroed (safe) action instead.

## How to Start Collection

1. Launch the RealSense camera, LiDAR, and Robot base as usual.
2. Ensure the Exploration Runner (Pink Noise Policy) and Lidar Safety Gate are running and properly armed.
3. Start the recording script:
   ```bash
   python scripts/record_2hr_dataset.py
   ```
4. The script will output estimated sizes and begin recording. It runs for **exactly 2 hours** (7200 seconds) and then gracefully stops.

## How to Stop Early

If you need to stop the collection early (e.g., due to an emergency or battery failure), press **Ctrl+C** in the terminal running the python script. 

The script will catch the signal, send a clean shutdown request to `ros2 bag record`, and ensure the `metadata.yaml` file is properly finalized so the bag is not corrupted.

## Storage Requirements

- **RGB Images (30 Hz):** ~300 GB / hour
- **Executed Actions:** Negligible
- **Total Expected Size:** ~500-600 GB

*Ensure the host PC has at least 600 GB of free NVMe/SSD space before starting.*
