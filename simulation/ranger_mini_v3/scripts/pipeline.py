import sys
import argparse
import subprocess
import yaml
import json
import datetime
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--map", required=False)
    parser.add_argument("--duration-minutes", type=float)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--run-id")
    parser.add_argument("--resume")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    # Determine run_id and dir
    if args.resume:
        run_id = args.resume
        run_dir = ROOT / "runs" / run_id
        if not run_dir.exists():
            print(f"ERROR: Cannot resume, run {run_id} not found.")
            sys.exit(1)
        print(f"Resuming run {run_id}")
    else:
        run_id = args.run_id or f"minav_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = ROOT / "runs" / run_id
        if run_dir.exists():
            print(f"ERROR: Run directory {run_dir} already exists.")
            sys.exit(1)
        run_dir.mkdir(parents=True)
        print(f"Starting new run {run_id}")
        
        # Copy config
        config_path = ROOT / args.config
        shutil.copy(config_path, run_dir / "config_used.yaml")

    state_file = run_dir / "pipeline_state.json"
    
    def get_state():
        if state_file.exists():
            with open(state_file) as f:
                return json.load(f)
        return {"stages": {"exploration": "pending", "hindsight": "pending", "training": "pending"}}

    state = get_state()
    
    # We must construct the args to pass to sub-scripts
    base_args = [
        "--config", f"runs/{run_id}/config_used.yaml" if not args.resume else args.config,
        "--run-dir", str(run_dir)
    ]
    if args.smoke:
        base_args.append("--smoke")
        
    def run_stage(name, script_path, extra_args=None):
        status = get_state()["stages"].get(name, "pending")
        if status == "completed":
            print(f"Skipping {name}, already completed.")
            return True
            
        cmd = [sys.executable, str(ROOT / script_path)] + base_args + (extra_args or [])
        print(f"Running {name}: {' '.join(cmd)}")
        
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"Stage {name} failed with code {res.returncode}")
            return False
        return True

    # 1. Exploration
    print("\n[1/3] DATA EXPLORATION")
    if args.map:
        explore_args = ["--map", str(Path(args.map).resolve())]
    else:
        explore_args = ["--map", str(ROOT / "_temp_world.xml")] # fallback

    if not run_stage("exploration", "scripts/run_exploration.py", explore_args):
        sys.exit(1)
        
    # Validation gate exploration could go here, or handled inside run_exploration.py
    
    # 2. Hindsight
    print("\n[2/3] HINDSIGHT RELABELING")
    if not run_stage("hindsight", "scripts/build_final_dataset.py"):
        sys.exit(1)
        
    # 3. Training
    print("\n[3/3] TD3+BC TRAINING")
    if not run_stage("training", "scripts/train_td3_bc.py", ["--device", args.device]):
        sys.exit(1)
        
    print(f"\nPipeline {run_id} finished successfully!")

if __name__ == "__main__":
    main()
