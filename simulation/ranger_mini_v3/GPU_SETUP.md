# GPU Deployment Preparation for MINav Pipeline

This document outlines the steps required to deploy the complete MINav offline RL pipeline on a Linux machine equipped with NVIDIA GPUs (e.g., at a university or cloud cluster).

## 1. Conda Environment Setup

The pipeline has been designed to be fully portable across Windows (CPU/dev) and Linux (GPU/prod). You will create a new conda environment and install the required dependencies.

```bash
# Create and activate a clean environment
conda create -n minav python=3.10 -y
conda activate minav

# Install PyTorch with CUDA support (adjust the CUDA version to match your module/drivers, e.g., 11.8 or 12.1)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install MuJoCo and scientific/data dependencies
pip install mujoco numpy scipy pandas pyarrow pyyaml Pillow timm
```

## 2. Running the Full Pipeline

The entire pipeline is wrapped by the `run_pipeline.sh` orchestrator. The orchestrator automatically sequences:
1. `run_exploration.py` (2 hours of procedural generation at 10 Hz)
2. `build_final_dataset.py` (DINOv3 encoding & hindsight relabeling)
3. `train_td3_bc.py` (Offline RL Training for 1M steps)

To submit the full run in the background and ensure it continues running even if your SSH session disconnects, use `nohup`:

```bash
# Ensure you are at the repository root
cd /path/to/RL-Image-Goal-Navigation

# Make the script executable
chmod +x run_pipeline.sh

# Run the full pipeline in the background using nohup
nohup ./run_pipeline.sh --map _random_world.xml --duration-minutes 120 --device cuda > full_pipeline.out 2>&1 &

# (Optional) Save the background process ID to monitor or kill later
echo $! > pipeline.pid
```

Alternatively, if your cluster uses **Slurm**, you can submit it via a batch script (`sbatch`):

```bash
#!/bin/bash
#SBATCH --job-name=minav_td3bc
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=48:00:00

# Load necessary modules (e.g., cuda)
# module load cuda/12.1

# Activate environment
source ~/.bashrc
conda activate minav

# Run pipeline
./run_pipeline.sh --map _random_world.xml --duration-minutes 120 --device cuda
```

## 3. Monitoring and Verifying CUDA Usage

### Checking Device Assignment
The pipeline explicitly logs the device used during both the DINOv3 encoding (Phase 2) and the TD3+BC training (Phase 3).
To verify the system is using the GPU, check the centralized log file for the run:

```bash
# Replace <run_id> with the actual ID generated (e.g., minav_20260811_234812)
tail -f runs/<run_id>/pipeline.log
```
Look for lines resembling:
- `Loading ViT on cuda`
- `Trainer initialized using device: cuda`

### Monitoring GPU Utilization
While the pipeline is running, open a new SSH terminal and use `nvidia-smi` to confirm that the memory is allocated and GPU utilization is high during the `hindsight` and `training` phases:

```bash
watch -n 1 nvidia-smi
```
- You should see `python` listed under the processes using the GPU.
- Memory usage will spike during Phase 2 (DINOv3 caching) and Phase 3 (TD3+BC mini-batch updates).
