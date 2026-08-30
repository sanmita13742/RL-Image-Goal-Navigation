"""
scripts/run_all.py
============================================================
Runs the full end-to-end pipeline automatically.
1. Prints resolved config.
2. Runs small test on 1,000 frames.
3. Validates the test dataset.
4. If successful, processes the full 72,000-frame dataset.
5. Validates the full dataset.
6. Reports final statistics.
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import yaml

ROOT = Path(__file__).parent.parent

def run_cmd(cmd):
    print(f"\n> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print("ERROR: Command failed!")
        sys.exit(result.returncode)

def main():
    print("============================================================")
    print("FINAL RESOLVED CONFIGURATION")
    print("============================================================")
    config_path = ROOT / "configs" / "minav_hindsight.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    print("DINOv3:")
    print(f"  Model: {cfg['dinov3']['model_name']}")
    print(f"  Input: {cfg['dinov3']['image_height']} x {cfg['dinov3']['image_width']}")
    print(f"  Frozen: True (implemented in python)")
    print("Representation:")
    print(f"  Type: {cfg['representation']['type']} (IMPLEMENTATION CHOICE)")
    print(f"  L2 Norm: {cfg['representation']['normalize']}")
    print("SSD:")
    print(f"  Threshold: {cfg['ssd']['threshold']}")
    print(f"  Crop: {cfg['ssd']['crop_h']} x {cfg['ssd']['crop_w']} (IMPLEMENTATION CHOICE)")
    print("Geometric:")
    print(f"  p: {cfg['geometric']['p']} (IMPLEMENTATION CHOICE)")
    print("Reward:")
    print(f"  Threshold: {cfg['reward']['similarity_threshold']}")
    
    print("\n============================================================")
    print("DRY-RUN (1,000 frames)")
    print("============================================================")
    python_exe = sys.executable
    script = "scripts/encode_dinov3.py"
    
    t0 = time.time()
    run_cmd([python_exe, script, "--session", "dataset/20260811_234812", "--test", "--batch_size", "8"])
    
    print("\nValidating test dataset...")
    run_cmd([python_exe, "scripts/validate_hindsight.py", "--dir", "data/processed/test_run"])
    
    print("\nDry-run passed successfully!")
    print("============================================================")
    print("FULL DATASET PROCESSING (72,000 frames)")
    print("============================================================")
    t1 = time.time()
    
    # Run full dataset
    run_cmd([python_exe, script, "--session", "dataset/20260811_234812", "--full", "--batch_size", "16"])
    
    print("\nValidating full dataset...")
    run_cmd([python_exe, "scripts/validate_hindsight.py", "--dir", "data/processed"])
    
    t_total = time.time() - t0
    t_full = time.time() - t1
    
    print("\n============================================================")
    print("FINAL STATISTICS")
    print("============================================================")
    import pandas as pd
    import numpy as np
    
    full_dir = ROOT / "data/processed"
    df_geom = pd.read_parquet(full_dir / "hindsight_geometric/geometric_transitions.parquet")
    df_unif = pd.read_parquet(full_dir / "hindsight_uniform/uniform_transitions.parquet")
    valid_goals = np.load(full_dir / "dinov3/valid_goals.npy")
    
    total_frames = len(valid_goals)
    total_valid = valid_goals.sum()
    pct_valid = (total_valid / total_frames) * 100
    
    print(f"Total frames:           {total_frames:,}")
    print(f"Total valid goals:      {total_valid:,}")
    print(f"Valid-goal percentage:  {pct_valid:.1f}%")
    
    print(f"\nTotal transitions:      {len(df_geom) + len(df_unif):,}")
    print(f"Geometric transitions:  {len(df_geom):,}")
    print(f"Uniform transitions:    {len(df_unif):,}")
    
    print(f"\nGeometric Reward=1:     {df_geom['reward'].sum():,} ({100*df_geom['reward'].mean():.1f}%)")
    print(f"Uniform Reward=1:       {df_unif['reward'].sum():,} ({100*df_unif['reward'].mean():.1f}%)")
    
    print(f"\nDINOv3 encoding & generation time (full): {t_full:.1f}s")
    
    # Storage size
    def get_size(path):
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
        return total / (1024*1024)
        
    print(f"Storage used (processed dir): {get_size(full_dir):.1f} MB")
    print("============================================================")

if __name__ == "__main__":
    main()
