# MINav Exploration System Audit & Analysis Report

## 1. Executive Summary
This report details the findings from an architectural audit of the MINav real-world exploration pipeline (Ranger Mini V3). The primary concern was whether the LiDAR-based safety gate logic undermines the uniform marginal action coverage intended by the Pink Uniform Noise exploration policy.

The audit verified that while the exploration policy successfully generates uniform pink noise at 2 Hz, the independent safety gate operates aggressively as a "clamp-to-zero" directional blocker at 20 Hz. We have successfully instrumented the pipeline to capture target, requested, and executed actions to empirically measure this effect in future real-world datasets.

**Crucial Note on Existing Data:** The existing dataset at `output-realworld/exploration/segment_000/segment.csv` contains exclusively `0.0` for all requested and executed velocities, confirming it was recorded in `dry-run` mode. A new, fully armed run must be executed to gather real-world action coverage statistics.

---

## 2. System Architecture & Timing (Observed Facts)

*   **Policy Generation:** `PinkUniformNoisePolicy` runs at **2 Hz**. It generates an action target by shaping a random frequency spectrum to $1/f$ (pink noise), reconstructing it via iFFT, normalizing to $N(0,1)$, and applying a Gaussian CDF to achieve a uniform marginal distribution $U(0, 1)$. This output is then scaled to the bounds `vx` [0.0, 0.6], `vy` [-0.3, 0.3], `wz` [-1.0, 1.0].
*   **Smoothing Interpolation:** The control loop in `ExplorationRunner` runs at **20 Hz**. Actions from the 2 Hz policy are smoothed using a Linear Interpolation (LERP) parameter of $\alpha = 0.2$. This means it takes approximately 10 control ticks (0.5 seconds) to converge to the new target action.
*   **Safety Gate Logic:** The `LidarSafetyGate` executes at **20 Hz** independently of the policy. It receives a 2D depth projection of the LiDAR scan (`60x640`). It divides the forward 180° into three sectors (left, center, right). If the minimum distance in any sector is below `0.55m` and contains at least `75` points:
    *   **Forward Obstacle:** Clamps `vx = 0.0`.
    *   **Left Obstacle:** Clamps `vy = max(0.0, vy)` (prevents leftward movement).
    *   **Right Obstacle:** Clamps `vy = min(0.0, vy)` (prevents rightward movement).
    *   **Rotation:** Angular velocity `wz` is **never** clamped, ensuring the robot can spin to escape.

---

## 3. Analysis & Inferences

### The "Zero-Clamp" Hypothesis
Because the safety gate clamps translational velocities directly to exactly `0.0` rather than rescaling or rejecting samples, the executed action distribution will likely show a massive spike at `vx = 0.0` and `vy = 0.0`. This mathematically breaks the uniform distribution intended by the pink noise policy. 

*   **Inference:** The executed coverage will not be uniform. It will be a zero-inflated uniform distribution, which may result in the Offline RL dataset severely under-sampling low-but-non-zero translational speeds.

### Temporal Distortion
The safety gate can intervene in the middle of a 2 Hz pink noise macro-action. Because pink noise retains memory ($1/f$ autocorrelation), the policy is unaware that the robot is stationary.
*   **Inference:** The executed sequence loses its strict $1/f$ spectral property because continuous sweeping motions are interrupted by sudden square-wave clamps to zero.

### The "Stuck Spin" Phenomenon
When facing an obstacle, `vx` and `vy` are clamped to zero, but `wz` remains active. Since the policy doesn't explicitly respond to depth, the robot simply spins in place based on the pink noise `wz` command until it happens to face an open corridor.
*   **Inference:** The spatial coverage of the robot may heavily favor corners or dead ends, where it will spend large amounts of time spinning before a random `wz` sequence turns it completely around.

---

## 4. Instrumentation Added

To definitively measure the magnitude of these effects, the pipeline was instrumented without altering the robot's behavior:

1.  **Extended CSV Schema (`data_recorder.py`)**: Added columns for `target_vx`, `target_vy`, `target_wz` (pre-smoothing actions), `obstacle_distance`, `relevant_point_count`, and `cloud_age`.
2.  **Safety Gate Diagnostics (`exploration_runner.py`)**: The `_safety_gate` now returns metadata on why it triggered, which is piped to the recorder.
3.  **Comprehensive Analysis Tool (`tools/analyze_exploration_dataset.py`)**: Created an offline script to compute differential entropy, cross-correlation, and blocking intervention rates. This will directly compare the target (uniform) distribution vs. the executed (clamped) distribution.
4.  **Unit Test Suite (`tests/test_exploration_audit.py`)**: Wrote 21 tests validating the policy ranges, pink noise spectral properties, LERP convergence, and safety gate pass/block conditions. All tests pass.

---

## 5. Recommendations for Next Steps

1.  **Execute an Armed Run:** Run `scripts/run_exploration.py --arm` on the real Ranger Mini V3 for at least 15 minutes to collect a non-zero dataset.
2.  **Run the Analysis Tool:** Execute `python tools/analyze_exploration_dataset.py --session output-realworld/exploration/` on the new data to quantify the zero-clamp spike.
3.  **Potential Mitigation (If Uniformity is Destroyed):** If the analysis shows unacceptable coverage loss, consider replacing the `clamp-to-zero` safety gate with a **rejection sampling** approach inside the 2 Hz policy loop:
    *   If the depth image indicates an obstacle, resample the pink noise generator until it produces a safe vector (e.g., negative `vx` or pure `wz`).
    *   This preserves the smoothness of the actions while naturally biasing the probability mass away from obstacles without artificially spiking `0.0`.
