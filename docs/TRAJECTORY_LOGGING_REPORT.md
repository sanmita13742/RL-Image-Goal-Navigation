# Trajectory Logging Report

## 1. Files Modified

| File | Change |
|------|--------|
| [`random_explore.py`](file:///c:/Users/sanmi/Desktop/projects/RL/simulation/ranger_mini_v3/random_explore.py) | Replaced flat single-CSV logger with trajectory-aware per-episode logger |
| [`validate_trajectories.py`](file:///c:/Users/sanmi/Desktop/projects/RL/simulation/ranger_mini_v3/validate_trajectories.py) | New: short-run validation script |

**Not modified:**
- `exploration_policies.py` — pink noise, FFT, PIT, action scaling, policy logic
- `robot.py`, `robot_base.py`, `test_env.py` — unchanged
- MuJoCo physics, rendering, control frequency, action generation

---

## 2. Existing Logging Behavior (before)

The original collector ran as a single flat loop:

```
while frame_idx < MAX_EPISODE_STEPS:
    render → get action → save image (frame_XXXXXX.png) → append CSV row → physics steps
```

- **One flat `log.csv`** with columns: `frame, timestamp, linear_vel_cmd, lateral_vel_cmd, angular_vel_cmd, pos_x, pos_y, yaw, rgb_path, depth_path`
- **Global image counter** — all images named `frame_000000.png … frame_131999.png` in a single `rgb/` folder
- **No episode/trajectory concept** — the robot was never reset
- **No `terminated`/`truncated` columns**

---

## 3. New Trajectory Definition

A **trajectory** = one episode between robot resets.

Since this is raw MuJoCo (not a Gym env), there is no environment-side termination signal. Episodes are bounded by `MAX_STEPS_PER_EPISODE` (configurable, default 500 steps = 50 s at 10 Hz). The robot is teleported back to the start pose at the end of each episode via `reset_robot()`.

- `terminated = False` always (no terminal condition is defined in this environment)
- `truncated = True` on the final step of each episode (step-limit boundary)

---

## 4. Episode Boundary Detection

```
for traj_num in range(NUM_TRAJECTORIES):
    reset_robot()                         ← episode starts
    for step in range(MAX_STEPS_PER_EPISODE):
        render / act / log / physics
        truncated = (step == MAX_STEPS_PER_EPISODE - 1)
    close trajectory CSV                  ← episode ends
    traj_id += 1
```

The exploration policy is **not reset between trajectories** — the pink-noise state and visit grid carry over. This matches the MINav paper's "single continuous exploration" with offline episode segmentation.

---

## 5. New Directory Structure

```
dataset/
    <YYYYMMDD_HHMMSS>/            ← session (one per python run)
        metadata.json             ← session-level provenance
        trajectory_000/
            rgb/
                000000.png        ← per-trajectory counter, starts at 0
                000001.png
                ...
            depth/
                000000.png
                ...
            trajectory.csv        ← one CSV per trajectory
        trajectory_001/
            ...
```

---

## 6. CSV Schema

**`trajectory.csv`** (one per trajectory):

| Column | Type | Description |
|--------|------|-------------|
| `timestep` | int | Per-trajectory step counter, starts at 0, monotonically increasing |
| `sim_time` | float | MuJoCo simulation time (seconds) |
| `linear_vel_cmd` | float | Applied linear velocity (m/s) |
| `lateral_vel_cmd` | float | Applied lateral velocity (m/s) |
| `angular_vel_cmd` | float | Applied angular velocity (rad/s) |
| `pos_x` | float | Robot x position (m) |
| `pos_y` | float | Robot y position (m) |
| `yaw` | float | Robot yaw (rad) |
| `rgb_path` | str | Relative path to RGB image, e.g. `rgb/000000.png` |
| `depth_path` | str | Relative path to depth image, e.g. `depth/000000.png` |
| `terminated` | bool | Always `False` (no terminal condition) |
| `truncated` | bool | `True` only on final row of each trajectory |

**`metadata.json`** (session level):

```json
{
    "session_id": "20260811_230522",
    "num_trajectories": 200,
    "total_steps": 100000,
    "trajectories": [
        {
            "trajectory_id": "trajectory_000",
            "num_steps": 500,
            "num_images": 500,
            "terminated": false,
            "truncated": true
        },
        ...
    ]
}
```

---

## 7. Image Naming Convention

Images use **per-trajectory frame numbering**:

```
trajectory_000/rgb/000000.png    ← step 0 of trajectory 0
trajectory_000/rgb/000001.png    ← step 1 of trajectory 0
...
trajectory_001/rgb/000000.png    ← step 0 of trajectory 1  (restarts at 0)
```

The trajectory directory guarantees global uniqueness. No session prefix is embedded in filenames.

---

## 8. Image/Action Alignment

The convention is preserved exactly from the original collector — **Option A**:

```
timestep t:  render (obs_t) → policy(obs_t) → record (obs_t, action_t) → physics → obs_{t+1}
```

The CSV row at `timestep=t` contains the observation **before** the physics step that produces `t+1`. The action in that row was **applied to produce** the state in row `t+1`. No reordering of rows or images.

---

## 9. Restart / Overwrite Behavior

- If the **session directory** already exists (same second-precision timestamp), the script aborts with an error message. The user must wait 1 second or delete manually.
- If **individual trajectory directories** already exist within a session (e.g. from a previous partial run), `find_next_trajectory_id()` skips to the next unused ID. **No data is ever overwritten.**
- Validation test sessions are prefixed with `VALIDATION_` and are safe to delete.

---

## 10. Validation Results

Ran `validate_trajectories.py` with 4 trajectories × 30 steps.

```
============================================================
  STEP 1: Short collection (4 trajectories × 30 steps)
  Output: dataset/VALIDATION_20260811_230522
============================================================
  Collected trajectory_000: 30 steps
  Collected trajectory_001: 30 steps
  Collected trajectory_002: 30 steps
  Collected trajectory_003: 30 steps

  Collection done. Validating...

============================================================
  STEP 2: Validation of VALIDATION_20260811_230522
============================================================
  [OK] Check  1: 4 trajectory directories found
  [OK  ] trajectory_000: 30 steps, 30 images, terminated=False, truncated=True
  [OK  ] trajectory_001: 30 steps, 30 images, terminated=False, truncated=True
  [OK  ] trajectory_002: 30 steps, 30 images, terminated=False, truncated=True
  [OK  ] trajectory_003: 30 steps, 30 images, terminated=False, truncated=True

============================================================
  ALL CHECKS PASSED ✓
============================================================
```

All 13 checks passed:
1. ✅ Correct number of trajectory directories
2. ✅ Every trajectory has `rgb/`
3. ✅ Every trajectory has `trajectory.csv`
4. ✅ Row count matches expected steps
5. ✅ Every image path exists on disk
6. ✅ No image belongs to two trajectories
7. ✅ `timestep` starts at 0 per trajectory
8. ✅ `timestep` monotonically increasing
9. ✅ No cross-trajectory transitions
10. ✅ `terminated`/`truncated` correct at boundaries
11. ✅ All action values finite
12. ✅ RGB=240×320, Depth=60×640
13. ✅ `metadata.json` present with required keys

---

## 11. Assumptions and Deviations

- **No Gym env** — the raw MuJoCo robot has no `env.step()` / `env.reset()`. Episodes are bounded purely by `MAX_STEPS_PER_EPISODE`. `terminated` is always `False`.
- **Policy not reset between trajectories** — the `PrimitiveExplorationPolicy` (visit grid, noise state, loop detector) persists across episodes. This is intentional and matches the MINav "continuous exploration" setting.
- **`sim_time` is not reset** — MuJoCo's internal clock is not reset between trajectories. It continues incrementing. Use `timestep` (per-trajectory counter) for within-trajectory ordering.
- **`depth/` images preserved** — even though DINOv2 only uses RGB, depth images are kept for completeness (collision avoidance was already using them).
