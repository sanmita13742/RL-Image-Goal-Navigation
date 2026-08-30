# PIPELINE_AUDIT

## 1. Current exploration entry point
*   **File:** `random_explore.py`
*   **Function/Class:** Runs procedurally in `main()`, uses `PrimitiveExplorationPolicy`.
*   **CLI Args:** None. It currently takes no arguments and runs a hardcoded 72k steps simulation.

## 2. Current exploration configuration
*   **Config File:** None.
*   **Key Parameters:** Hardcoded inside `random_explore.py:main()`:
    *   `CONTROL_FREQ`: 10.0 Hz
    *   `SEGMENT_SIZE`: 500 steps
    *   `TOTAL_STEPS`: 72,000 (2 hours)
    *   `BETA`: 1 (Pink noise)
    *   `BUFFER_SIZE`: 8192
    *   Output path is hardcoded to `dataset/YYYYMMDD_HHMMSS`.

## 3. Current dataset output format
*   **Directory Tree:**
    ```
    dataset/<session_id>/
        metadata.json
        segment_000/
            segment.csv
            rgb/<step_id>.png
            depth/<step_id>.png
        segment_001/
            ...
    ```
*   **Key Field Names (`segment.csv`):** `trajectory_id`, `global_step`, `segment_step`, `sim_time`, `linear_vel_cmd`, `lateral_vel_cmd`, `angular_vel_cmd`, `pos_x`, `pos_y`, `yaw`, `rgb_path`, `depth_path`.

## 4. Current hindsight entry point
*   **File:** `scripts/build_final_dataset.py`
*   **Function/Class:** Uses procedural logic and `FrozenDINOv3Encoder` in `main()`.
*   **CLI Args:**
    *   `--session`: Path to the raw dataset session directory (Required).
    *   `--smoke`: Run on only the first 100 frames.
    *   `--full`: Run on the full dataset.

## 5. Current hindsight output format
*   **Directory Tree:** (Hardcoded to output into the project's `data/processed/` or similar based on script internals):
    ```
    data/processed/
        dinov3/
            phi_cache.npy        (Visual representations)
            ssd_scores.npy       (Spatial standard deviation scores)
            valid_goals.npy      (Indices of valid goals)
            encoder_meta.json    (Metadata about the encoding process)
        hindsight/
            geometric_transitions.parquet
            uniform_transitions.parquet
            all_transitions.parquet
        FINAL_HINDSIGHT_DATASET_REPORT.md
    ```

## 6. Current TD3+BC training entry point
*   **File:** `scripts/train_td3_bc.py`
*   **Function/Class:** Uses `Trainer` class from `src.rl.trainer`.
*   **CLI Args:**
    *   `--config`: Path to YAML config (Required, e.g., `configs/td3_bc.yaml`).
    *   `--device`: `{cpu, cuda, auto}` (Default: `auto`).

## 7. Current checkpoint/output format
*   **Directory Tree:** (Defined by `output_dir` in `td3_bc.yaml`):
    ```
    <output_dir>/
        config.yaml
        metrics.csv
        checkpoints/
            checkpoint_<step>.pt
            latest.pt
    ```

## 8. Dependencies required per stage
*   **Exploration:** `mujoco`, `numpy`, `scipy`, `Pillow`
*   **Hindsight Encoding:** `torch`, `torchvision`, `timm`, `pandas`, `pyarrow`, `numpy`, `Pillow`
*   **TD3+BC Training:** `torch`, `numpy`, `pandas`, `pyyaml`, `pyarrow`

## 9. What needs to change for pipeline integration
*   **Interfaces & Arguments:**
    *   `random_explore.py` must be adapted (or wrapped) into `scripts/run_exploration.py` that accepts `--config`, `--map`, and overrides like `--smoke`.
    *   `build_final_dataset.py` needs to read from the master `configs/pipeline.yaml` instead of hardcoded internal configs and handle dynamic output paths correctly.
    *   `train_td3_bc.py` needs to read from the master `configs/pipeline.yaml`.
*   **Path Assumptions:**
    *   All hardcoded `dataset/` and `data/` paths must be removed. All scripts must write to `runs/<run_id>/<stage_subdir>/`.
*   **State Management:**
    *   Every script must read/write to `runs/<run_id>/pipeline_state.json` to enable the `--resume` functionality.
    *   Every script must log to `runs/<run_id>/pipeline.log`.
*   **Validation:**
    *   We need `scripts/validate_dataset.py` and gating logic inside `pipeline.py` to abort execution if validations fail.

## 10. What must NOT change
The following components are validated research logic and must remain untouched:
*   **Exploration:** Control frequency (10 Hz), continuous segment flow without physical resets, Pink noise policy ($\beta=1$).
*   **DINOv3:** ViT-S/16 architecture, 448x784 input, frozen weights, $\phi(o)$ as normalized CLS token (384-D).
*   **SSD Filter:** Threshold 0.02, center crop 14x25 patches.
*   **Hindsight:** 4-frame stacked observation, geometric sampling ($p=0.99$) for future goals, uniform goal pool, reward/done threshold of 0.8, critic dataset split (50% geometric, 50% uniform), actor dataset split (100% uniform).
*   **TD3+BC Training:** $\lambda_{BC}=0.001$, batch size 256, $\gamma=0.99$, actor/critic LR $3\times 10^{-4}$, policy delay 2, $\tau=0.005$, target noise 0.2, target noise clip 0.5, total gradient steps 1M, seed 42, action bounds $[-0.9, 1.8], [-1.0, 1.0], [-1.5, 1.5]$.
