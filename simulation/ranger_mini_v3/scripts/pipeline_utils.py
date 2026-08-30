import os
import sys
import json
import logging
import datetime
from pathlib import Path

def setup_pipeline_logger(run_dir: Path, name: str = "pipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s')
    
    # Stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    # File handler
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(run_dir / "pipeline.log", mode="a", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

def update_pipeline_state(run_dir: Path, stage: str, status: str):
    """
    Valid states: 'pending', 'running', 'completed', 'failed'
    Valid stages: 'exploration', 'hindsight', 'training'
    """
    state_file = run_dir / "pipeline_state.json"
    
    # Initialize if doesn't exist
    if not state_file.exists():
        state = {
            "run_id": run_dir.name,
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "config": "configs/pipeline.yaml",
            "stages": {
                "exploration": "pending",
                "hindsight":   "pending",
                "training":    "pending"
            },
            "stage_timestamps": {
                "exploration_started":  None,
                "exploration_finished": None,
                "hindsight_started":    None,
                "hindsight_finished":   None,
                "training_started":     None,
                "training_finished":    None
            }
        }
    else:
        with open(state_file, "r") as f:
            state = json.load(f)
            
    state["stages"][stage] = status
    
    now = datetime.datetime.utcnow().isoformat() + "Z"
    if status == "running":
        if state["stage_timestamps"].get(f"{stage}_started") is None:
            state["stage_timestamps"][f"{stage}_started"] = now
    elif status in ("completed", "failed"):
        state["stage_timestamps"][f"{stage}_finished"] = now
        
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)
