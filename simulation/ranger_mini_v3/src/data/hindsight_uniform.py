"""
src/data/hindsight_uniform.py
============================================================
Phase 9: Global Uniform Goal Sampling.

PAPER SPECIFIES:
  g ~ Uniform(G)
  G = global valid goal set (SSD > 0.02)
  Goals may come from ANY segment.
  Goals may be temporally before OR after the current state.
  ONLY valid SSD-filtered goals may be sampled.

PAPER DOES NOT SPECIFY:
  Number of goals per transition.
  IMPLEMENTATION CHOICE: num_goals is configurable.

IMPLEMENTATION NOTES:
  - Uses the full global G regardless of segment.
  - Does NOT restrict goals to the same segment as current state.
  - Does NOT restrict goals to future frames.
  - Reward is computed by visual similarity, not temporal proximity.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from src.data.reward import batch_compute_reward_done


FRAME_STACK = 4


def build_uniform_dataset(
    phi_vectors:    np.ndarray,     # [N, 384]
    frame_index:    pd.DataFrame,   # N rows
    valid_goals_df: pd.DataFrame,   # rows from goal_set.parquet
    ssd_scores:     np.ndarray,     # [N]
    num_goals:      int,
    out_dir:        Path,
    max_frames:     int = None,
    seed:           int = 42,
) -> pd.DataFrame:
    """
    Generate uniform hindsight transitions.

    For each valid current state t:
      - goal ~ Uniform(G)  where G = all valid SSD-filtered goals (global pool)
      - goal may be from any segment, any time (past or future)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    N = len(phi_vectors)
    if max_frames is not None:
        N = min(N, max_frames)
        phi_vectors = phi_vectors[:N]
        frame_index = frame_index.iloc[:N].reset_index(drop=True)

    # Valid current states: t in [3, N-2] (same as geometric)
    valid_t = np.arange(FRAME_STACK - 1, N - 1)

    # Build goal pool: valid goals that fall within our N-frame window
    goal_mask = valid_goals_df["global_step"] < N
    goal_pool = valid_goals_df[goal_mask].reset_index(drop=True)
    goal_indices = goal_pool["embedding_index"].values    # indices into phi_vectors
    goal_ssd     = goal_pool["ssd_score"].values

    if len(goal_pool) == 0:
        raise ValueError("No valid goals in the goal pool after filtering to max_frames!")

    print(f"  Goal pool size: {len(goal_pool):,} valid goals")

    records = []

    for t in valid_t:
        row = frame_index.iloc[t]
        seg_t = row["segment_id"]
        
        # Enforce: State stacking must not cross segment boundaries
        seg_t3 = frame_index.iloc[t-3]["segment_id"]
        if seg_t3 != seg_t:
            continue
            
        state_phis      = phi_vectors[t-3 : t+1]    # [4, D]
        next_state_phis = phi_vectors[t-2 : t+2]    # [4, D]

        for _ in range(num_goals):
            # PAPER SPECIFIES: g ~ Uniform(G) from GLOBAL pool
            j        = rng.integers(0, len(goal_pool))
            goal_idx = int(goal_indices[j])
            goal_phi = phi_vectors[goal_idx]         # [D]
            g_ssd    = float(goal_ssd[j])

            sim    = float((state_phis @ goal_phi).mean())
            reward = int(sim >= 0.8)
            done   = reward

            records.append({
                "current_global_step":  int(t),
                "goal_global_step":     int(goal_idx),
                "sampling_method":      "uniform",
                "segment_id":           row["segment_id"],
                "goal_segment_id":      goal_pool.iloc[j]["segment_id"],
                "action_linear":        float(row["linear_vel_cmd"]),
                "action_lateral":       float(row["lateral_vel_cmd"]),
                "action_angular":       float(row["angular_vel_cmd"]),
                # Embedding indices
                "state_idx_t3":         int(t - 3),
                "state_idx_t2":         int(t - 2),
                "state_idx_t1":         int(t - 1),
                "state_idx_t0":         int(t),
                "next_state_idx_t2":    int(t - 2),
                "next_state_idx_t1":    int(t - 1),
                "next_state_idx_t0":    int(t),
                "next_state_idx_t1_fw": int(t + 1),
                "goal_embedding_idx":   int(goal_idx),
                "goal_ssd_score":       g_ssd,
                "goal_similarity":      float(sim),
                "reward":               reward,
                "done":                 done,
                # Metadata
                "pos_x":                float(row["pos_x"]),
                "pos_y":                float(row["pos_y"]),
                "sim_time":             float(row["sim_time"]),
            })

    df = pd.DataFrame(records)

    out_path = out_dir / "uniform_transitions.parquet"
    df.to_parquet(out_path, index=False)

    print(f"\nUniform hindsight dataset:")
    print(f"  Valid states       : {len(valid_t):,}")
    print(f"  Goal pool size     : {len(goal_pool):,}")
    print(f"  Goals/transition   : {num_goals}")
    print(f"  Total transitions  : {len(df):,}")
    print(f"  Reward=1 (done=1)  : {df['reward'].sum():,} ({100*df['reward'].mean():.2f}%)")
    print(f"  Saved -> {out_path}")

    return df
