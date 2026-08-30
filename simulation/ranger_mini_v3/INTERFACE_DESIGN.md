# INTERFACE_DESIGN

## 1. Exact directory structure under `runs/<run_id>/`
```text
runs/
└── <run_id>/
    ├── pipeline.log                     # Unified orchestrator + stage logs
    ├── pipeline_state.json              # Current state of the pipeline
    ├── config_used.yaml                 # The exact config used for this run
    ├── PIPELINE_REPRODUCTION_REPORT.md  # Final report generated after training
    ├── exploration/
    │   ├── exploration_metadata.json    # Phase metadata
    │   ├── segment_000/
    │   │   ├── observations.csv         
    │   │   ├── rgb/
    │   │   └── depth/
    │   └── ...
    ├── processed/
    │   ├── dinov3/
    │   │   ├── phi_cache.npy
    │   │   ├── ssd_scores.npy
    │   │   ├── valid_goals.npy
    │   │   └── encoder_meta.json
    │   ├── hindsight/
    │   │   ├── geometric_transitions.parquet
    │   │   ├── uniform_transitions.parquet
    │   │   └── all_transitions.parquet
    │   └── FINAL_HINDSIGHT_DATASET_REPORT.md
    └── training/
        ├── checkpoints/
        │   ├── checkpoint_<step>.pt
        │   └── latest.pt
        ├── metrics.csv
        ├── final_model.pt
        └── training_report.md
```

## 2. Schema of `pipeline_state.json`
```json
{
  "run_id": "string",
  "created_at": "ISO 8601 string",
  "config": "string (path to used config)",
  "stages": {
    "exploration": "pending | running | completed | failed",
    "hindsight":   "pending | running | completed | failed",
    "training":    "pending | running | completed | failed"
  },
  "stage_timestamps": {
    "exploration_started":  "ISO 8601 string | null",
    "exploration_finished": "ISO 8601 string | null",
    "hindsight_started":    "ISO 8601 string | null",
    "hindsight_finished":   "ISO 8601 string | null",
    "training_started":     "ISO 8601 string | null",
    "training_finished":    "ISO 8601 string | null"
  }
}
```

## 3. Schema of `exploration_metadata.json`
```json
{
  "run_id": "string",
  "duration_minutes": "float",
  "control_freq_hz": "float",
  "total_steps_planned": "int",
  "total_steps_recorded": "int",
  "segment_size": "int",
  "num_segments": "int",
  "robot_resets": "int (MUST BE 0)",
  "smoke_mode": "boolean"
}
```

## 4. Parquet column schema for hindsight transitions
This schema applies to both `geometric_transitions.parquet` and `uniform_transitions.parquet`.
```python
{
  "trajectory_id": "int64",       # Always 0 (continuous rollout)
  "state_t3": "int64",            # index into phi_cache (t-3)
  "state_t2": "int64",            # index into phi_cache (t-2)
  "state_t1": "int64",            # index into phi_cache (t-1)
  "state_t0": "int64",            # index into phi_cache (current)
  "action_linear": "float32",     # physical action
  "action_lateral": "float32",    # physical action
  "action_angular": "float32",    # physical action
  "next_t2": "int64",             # index into phi_cache (t-2)
  "next_t1": "int64",             # index into phi_cache (t-1)
  "next_t0": "int64",             # index into phi_cache (current)
  "next_t1f": "int64",            # index into phi_cache (next frame)
  "goal_idx": "int64",            # index into phi_cache (the goal)
  "reward": "float32",            # 0.0 or 1.0
  "done": "bool"                  # True if reward == 1.0
}
```

## 5. CLI interface for each script
All stage scripts are designed to be independently executable but will log into the centralized run directory.
*   **`scripts/run_exploration.py`**
    *   `--config <path>`: Overrides default pipeline config (optional).
    *   `--run-dir <path>`: Path to `runs/<run_id>/` (required).
    *   `--map <path>`: XML map path (required).
    *   `--smoke`: Activate smoke limits.
*   **`scripts/build_final_dataset.py`**
    *   `--config <path>`: (optional)
    *   `--run-dir <path>`: Path to `runs/<run_id>/` (required).
    *   `--smoke`: Activate smoke limits.
*   **`scripts/validate_dataset.py`**
    *   `--config <path>`: (optional)
    *   `--run-dir <path>`: (required).
*   **`scripts/train_td3_bc.py`**
    *   `--config <path>`: (optional)
    *   `--run-dir <path>`: (required).
    *   `--device <auto|cpu|cuda>`: Hardware target.
    *   `--smoke`: Activate smoke limits.
*   **`pipeline.py`** (The Orchestrator called by `run_pipeline.sh`)
    *   `--config <path>`
    *   `--map <path>`
    *   `--duration-minutes <float>`
    *   `--device <auto|cpu|cuda>`
    *   `--run-id <id>`
    *   `--resume <id>`
    *   `--smoke`

## 6. How `configs/pipeline.yaml` maps to each stage's config
*   **`run_exploration.py`**:
    *   Extracts values strictly from the `exploration` and `smoke` dictionaries.
    *   Extracts `--duration-minutes` from `exploration.duration_minutes` (unless overridden by `--smoke`).
*   **`build_final_dataset.py`**:
    *   Extracts values from the `hindsight` dictionary (e.g., `dino_model`, `ssd_threshold`).
    *   Reads `smoke.max_frames` if `--smoke` is provided.
    *   Reads the input raw dataset path from the `exploration.output_subdir`.
*   **`train_td3_bc.py`**:
    *   Extracts values from the `training` dictionary (e.g., `batch_size`, `lambda_bc`, `lr_actor`).
    *   Reads `run.device`.
    *   Reads `smoke.training_steps` if `--smoke` is provided.
    *   Input data is resolved to `processed/` using the `run-dir` and `hindsight.output_subdir`.
