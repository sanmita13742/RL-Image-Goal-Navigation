"""
src/data/hindsight_geometric.py
============================================================
Phase 8: Geometric Future Sampling.

PAPER SPECIFIES:
  For transition at timestep t, sample k from:
    P(K=k) = p^(k-1) * (1-p)  for k >= 1  (geometric distribution)
  Then: goal = embedding[min(t+k, T)]
  Goal must be STRICTLY FUTURE: goal_global_step > current_global_step

PAPER DOES NOT SPECIFY:
  Numerical value of p.
  IMPLEMENTATION CHOICE: p is loaded from config.
  Default p=0.99 (long-horizon). Explicitly documented as NOT paper-specified.

IMPLEMENTATION NOTES:
  - Uses GLOBAL temporal order (global_step), NOT segment-local order.
  - A state may span segment boundaries (four frames across two segments).
  - Only drop a state if an ACTUAL missing frame exists (not at segment boundaries).
  - The final frame (global_step = N-1) cannot be a current state because
    next_state would be out of bounds.
  - goal_global_step must be <= final_global_step.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from src.data.reward import batch_compute_reward_done


FRAME_STACK = 4    # PAPER SPECIFIES: 4-frame state


def geometric_sample_k(p: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample k from geometric distribution P(K=k) = p^(k-1) * (1-p), k >= 1.

    PAPER SPECIFIES this distribution.
    PAPER DOES NOT SPECIFY the value of p.

    Equivalent to: k = geometric(1-p), using numpy convention where k >= 1.
    """
    return rng.geometric(p=1 - p, size=size)


def build_geometric_dataset(
    phi_vectors:   np.ndarray,    # [N, 384]
    frame_index:   pd.DataFrame,  # N rows, sorted by global_step
    p:             float,         # geometric sampling parameter (NOT paper-specified)
    num_goals:     int,           # goals per transition (NOT paper-specified)
    out_dir:       Path,
    max_frames:    int = None,    # for small test; None = use all
    seed:          int = 42,
) -> pd.DataFrame:
    """
    Generate geometric hindsight transitions.

    For each valid current state t (requires frames t-3, t-2, t-1, t, t+1 to exist):
      - state      = phi_vectors[t-3:t+1]    shape [4, D]
      - next_state = phi_vectors[t-2:t+2]    shape [4, D]
      - action     = from frame_index row t
      - goal       = phi_vectors[t + k]      where k ~ Geometric(p), clamped to T

    Returns DataFrame of transitions.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    N = len(phi_vectors)
    if max_frames is not None:
        N = min(N, max_frames)
        phi_vectors  = phi_vectors[:N]
        frame_index  = frame_index.iloc[:N].reset_index(drop=True)

    # Valid current states: need frames [t-3 .. t+1], so t in [3, N-2]
    # t-3 >= 0  ->  t >= 3
    # t+1 <= N-1  ->  t <= N-2
    valid_t = np.arange(FRAME_STACK - 1, N - 1)   # t in [3, N-2]

    records = []

    for t in valid_t:
        for _ in range(num_goals):
            k = int(geometric_sample_k(p=p, size=1, rng=rng)[0])
            goal_t = min(t + k, N - 1)

            # Enforce: goal_global_step > current_global_step
            if goal_t <= t:
                goal_t = t + 1

            state_phis      = phi_vectors[t-3 : t+1]    # [4, D]
            next_state_phis = phi_vectors[t-2 : t+2]    # [4, D]
            goal_phi        = phi_vectors[goal_t]        # [D]

            sim = float((state_phis @ goal_phi).mean())
            reward = int(sim >= 0.8)
            done   = reward

            row = frame_index.iloc[t]
            records.append({
                "current_global_step":   int(t),
                "goal_global_step":      int(goal_t),
                "offset_k":              int(goal_t - t),
                "sampling_method":       "geometric",
                "segment_id":            row["segment_id"],
                "action_linear":         float(row["linear_vel_cmd"]),
                "action_lateral":        float(row["lateral_vel_cmd"]),
                "action_angular":        float(row["angular_vel_cmd"]),
                # Store embedding indices — do NOT copy pixel data
                "state_idx_t3":          int(t - 3),
                "state_idx_t2":          int(t - 2),
                "state_idx_t1":          int(t - 1),
                "state_idx_t0":          int(t),
                "next_state_idx_t2":     int(t - 2),
                "next_state_idx_t1":     int(t - 1),
                "next_state_idx_t0":     int(t),
                "next_state_idx_t1_fw":  int(t + 1),
                "goal_embedding_idx":    int(goal_t),
                "goal_similarity":       float(sim),
                "reward":                reward,
                "done":                  done,
                # Metadata
                "pos_x":                 float(row["pos_x"]),
                "pos_y":                 float(row["pos_y"]),
                "sim_time":              float(row["sim_time"]),
            })

    df = pd.DataFrame(records)

    out_path = out_dir / "geometric_transitions.parquet"
    df.to_parquet(out_path, index=False)

    print(f"\nGeometric hindsight dataset:")
    print(f"  Valid states       : {len(valid_t):,}")
    print(f"  Goals/transition   : {num_goals}")
    print(f"  Total transitions  : {len(df):,}")
    print(f"  p (geometric)      : {p}  [NOT paper-specified]")
    print(f"  Reward=1 (done=1)  : {df['reward'].sum():,} ({100*df['reward'].mean():.2f}%)")
    print(f"  Avg goal offset k  : {df['offset_k'].mean():.1f} steps")
    print(f"  Saved -> {out_path}")

    return df
