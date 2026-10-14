"""
realworld/src/synthetic_augmentation.py — Synthetic Data Augmentation for MINav.
================================================================================
Provides dataset amplification and synthetic trajectory generation:
  1. Temporal Reversal:
     Generates backward trajectories (vx -> -vx, wz -> -wz) from existing forward
     exploration segments. Enables policy to learn reverse driving and backtracking.
  2. Action Noise Perturbation:
     Regularizes behavioral actions with mild Gaussian noise to smooth action distribution.
  3. Goal Proximity Curriculum Mining:
     Enriches uniform hindsight dataset with balanced near-goal (sim >= 0.70) and
     hard-negative (sim 0.3-0.5) transitions.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path


FRAME_STACK = 4


def generate_temporal_reversal_transitions(
    phi_vectors: np.ndarray,
    frame_index: pd.DataFrame,
    p_values: List[float] = [0.70, 0.90, 0.99],
    p_weights: Optional[List[int]] = [4, 3, 3],
    num_goals: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic reverse-motion transitions by reversing temporal sequence
    within each exploration segment.

    Actions are negated:
      - vx -> -vx (robot learns to reverse)
      - vy -> -vy
      - wz -> -wz (reverse steering geometry)

    Returns a DataFrame matching the schema of geometric_transitions.parquet.
    """
    rng = np.random.default_rng(seed)
    records = []

    # Build p_schedule
    if p_weights is not None and len(p_weights) == len(p_values):
        w_sum = sum(p_weights)
        if w_sum == num_goals:
            p_schedule = []
            for p_val, count in zip(p_values, p_weights):
                p_schedule.extend([p_val] * int(count))
        else:
            norm_w = [w / w_sum for w in p_weights]
            counts = [max(1, int(round(w * num_goals))) for w in norm_w]
            while sum(counts) > num_goals:
                counts[int(np.argmax(counts))] -= 1
            while sum(counts) < num_goals:
                counts[int(np.argmin(counts))] += 1
            p_schedule = []
            for p_val, count in zip(p_values, counts):
                p_schedule.extend([p_val] * count)
    else:
        p_schedule = []
        base_count = num_goals // len(p_values)
        rem = num_goals % len(p_values)
        for idx, p_val in enumerate(p_values):
            c = base_count + (1 if idx < rem else 0)
            p_schedule.extend([p_val] * c)

    # Process each segment independently
    unique_segments = frame_index["segment_id"].unique()

    for seg_id in unique_segments:
        seg_rows = frame_index[frame_index["segment_id"] == seg_id]
        seg_indices = seg_rows.index.tolist()
        seg_len = len(seg_indices)

        # Require at least FRAME_STACK + 2 frames to build transitions
        if seg_len < FRAME_STACK + 2:
            continue

        # In reversed segment:
        # Time runs from index seg_len - 1 down to 0
        # Let rev_seq be the list of global frame indices in reversed chronological order
        rev_seq = seg_indices[::-1]

        # For reversed time step i in [FRAME_STACK - 1, seg_len - 2]:
        # State: 4 frames preceding and including rev_seq[i] in the reversed order
        # Next state: shifts one step forward in reversed order
        for i in range(FRAME_STACK - 1, seg_len - 1):
            s_t3 = rev_seq[i - 3]
            s_t2 = rev_seq[i - 2]
            s_t1 = rev_seq[i - 1]
            s_t0 = rev_seq[i]

            ns_t2 = rev_seq[i - 2]
            ns_t1 = rev_seq[i - 1]
            ns_t0 = rev_seq[i]
            ns_t1_fw = rev_seq[i + 1]

            # The original forward step was moving from s_t0 (which was f_{t+1}) to ns_t1_fw (which was f_t)
            # Reversing this motion negates the forward linear velocity and angular velocity
            orig_row = frame_index.iloc[s_t0]
            action_linear = -float(orig_row["linear_vel_cmd"])
            action_lateral = -float(orig_row["lateral_vel_cmd"])
            action_angular = -float(orig_row["angular_vel_cmd"])

            state_phis = phi_vectors[[s_t3, s_t2, s_t1, s_t0]]
            next_state_phis = phi_vectors[[ns_t2, ns_t1, ns_t0, ns_t1_fw]]

            # Sample hindsight goals strictly in the reversed future (i.e. indices > i in rev_seq)
            remaining_steps = seg_len - 1 - i
            if remaining_steps <= 0:
                continue

            for p_val in p_schedule:
                k = int(rng.geometric(p=1.0 - p_val))
                offset_k = min(k, remaining_steps)
                goal_rev_idx = i + offset_k
                goal_global_idx = rev_seq[goal_rev_idx]

                goal_phi = phi_vectors[goal_global_idx]
                sim = float((state_phis @ goal_phi).mean())
                reward = int(sim >= 0.8)
                done = reward

                records.append({
                    "current_global_step": int(s_t0),
                    "goal_global_step": int(goal_global_idx),
                    "offset_k": int(offset_k),
                    "sampling_method": "geometric_reversed",
                    "p_geometric": float(p_val),
                    "segment_id": seg_id,
                    "action_linear": action_linear,
                    "action_lateral": action_lateral,
                    "action_angular": action_angular,
                    "state_idx_t3": int(s_t3),
                    "state_idx_t2": int(s_t2),
                    "state_idx_t1": int(s_t1),
                    "state_idx_t0": int(s_t0),
                    "next_state_idx_t2": int(ns_t2),
                    "next_state_idx_t1": int(ns_t1),
                    "next_state_idx_t0": int(ns_t0),
                    "next_state_idx_t1_fw": int(ns_t1_fw),
                    "goal_embedding_idx": int(goal_global_idx),
                    "goal_similarity": float(sim),
                    "reward": reward,
                    "done": done,
                    "pos_x": float(orig_row["pos_x"]) if "pos_x" in orig_row else 0.0,
                    "pos_y": float(orig_row["pos_y"]) if "pos_y" in orig_row else 0.0,
                    "sim_time": float(orig_row["sim_time"]) if "sim_time" in orig_row else 0.0,
                })

    df = pd.DataFrame(records)
    print(f"\nSynthetic Temporal Reversal Augmentation:")
    print(f"  Generated transitions : {len(df):,}")
    if len(df) > 0:
        print(f"  Linear vel range (vx) : [{df['action_linear'].min():.2f}, {df['action_linear'].max():.2f}] m/s")
        print(f"  Angular vel range (wz): [{df['action_angular'].min():.2f}, {df['action_angular'].max():.2f}] rad/s")
        print(f"  Reward=1 fraction     : {100 * df['reward'].mean():.2f}%")
    return df


def add_action_noise(
    df: pd.DataFrame,
    noise_std: Tuple[float, float, float] = (0.02, 0.02, 0.05),
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (-0.9, 1.8), (-1.0, 1.0), (-1.5, 1.5)
    ),
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate perturbed copies of transitions with small Gaussian action noise
    to regularize the behavioral cloning loss and improve policy robustness.
    """
    rng = np.random.default_rng(seed)
    df_noisy = df.copy()

    vx_noise = rng.normal(0, noise_std[0], size=len(df))
    vy_noise = rng.normal(0, noise_std[1], size=len(df))
    wz_noise = rng.normal(0, noise_std[2], size=len(df))

    df_noisy["action_linear"] = np.clip(df["action_linear"] + vx_noise, bounds[0][0], bounds[0][1])
    df_noisy["action_lateral"] = np.clip(df["action_lateral"] + vy_noise, bounds[1][0], bounds[1][1])
    df_noisy["action_angular"] = np.clip(df["action_angular"] + wz_noise, bounds[2][0], bounds[2][1])
    df_noisy["sampling_method"] = df["sampling_method"] + "_noisy"

    return df_noisy


def mine_curriculum_uniform_transitions(
    phi_vectors: np.ndarray,
    frame_index: pd.DataFrame,
    valid_goals_df: pd.DataFrame,
    num_near_goals: int = 3,
    num_hard_negatives: int = 3,
    num_random_goals: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Mine balanced goal curriculum for uniform hindsight:
      - Near goals (sim >= 0.70): teaches slowing down and arrival behavior
      - Hard negatives (sim 0.30 - 0.55): teaches fine-grained visual discrimination
      - Random uniform goals: global pool exploration
    """
    rng = np.random.default_rng(seed)
    N = len(phi_vectors)
    valid_t = np.arange(FRAME_STACK - 1, N - 1)

    goal_indices = valid_goals_df["embedding_index"].values
    goal_phis = phi_vectors[goal_indices]  # [M, 384]

    records = []

    for t in valid_t:
        row = frame_index.iloc[t]
        seg_t = row["segment_id"]

        seg_t3 = frame_index.iloc[t - 3]["segment_id"]
        if seg_t3 != seg_t:
            continue

        state_phis = phi_vectors[t - 3 : t + 1]  # [4, 384]
        next_state_phis = phi_vectors[t - 2 : t + 2]
        curr_phi = phi_vectors[t]

        # Compute cosine similarities to all valid goals
        sims = goal_phis @ curr_phi  # [M]

        # 1. Easy positive / near goals (sim >= 0.70)
        pos_mask = np.where(sims >= 0.70)[0]
        if len(pos_mask) > 0:
            chosen_pos = rng.choice(pos_mask, size=min(num_near_goals, len(pos_mask)), replace=True)
            for p_idx in chosen_pos:
                g_idx = int(goal_indices[p_idx])
                g_phi = phi_vectors[g_idx]
                sim = float((state_phis @ g_phi).mean())
                reward = int(sim >= 0.8)
                records.append({
                    "current_global_step": int(t),
                    "goal_global_step": int(g_idx),
                    "offset_k": int(abs(g_idx - t)),
                    "sampling_method": "uniform_near",
                    "segment_id": seg_t,
                    "action_linear": float(row["linear_vel_cmd"]),
                    "action_lateral": float(row["lateral_vel_cmd"]),
                    "action_angular": float(row["angular_vel_cmd"]),
                    "state_idx_t3": int(t - 3),
                    "state_idx_t2": int(t - 2),
                    "state_idx_t1": int(t - 1),
                    "state_idx_t0": int(t),
                    "next_state_idx_t2": int(t - 2),
                    "next_state_idx_t1": int(t - 1),
                    "next_state_idx_t0": int(t),
                    "next_state_idx_t1_fw": int(t + 1),
                    "goal_embedding_idx": int(g_idx),
                    "goal_similarity": float(sim),
                    "reward": reward,
                    "done": reward,
                    "pos_x": float(row["pos_x"]) if "pos_x" in row else 0.0,
                    "pos_y": float(row["pos_y"]) if "pos_y" in row else 0.0,
                    "sim_time": float(row["sim_time"]) if "sim_time" in row else 0.0,
                })

        # 2. Hard negatives (sim in [0.30, 0.55])
        hard_mask = np.where((sims >= 0.30) & (sims <= 0.55))[0]
        if len(hard_mask) > 0:
            chosen_hard = rng.choice(hard_mask, size=min(num_hard_negatives, len(hard_mask)), replace=True)
            for h_idx in chosen_hard:
                g_idx = int(goal_indices[h_idx])
                g_phi = phi_vectors[g_idx]
                sim = float((state_phis @ g_phi).mean())
                reward = int(sim >= 0.8)
                records.append({
                    "current_global_step": int(t),
                    "goal_global_step": int(g_idx),
                    "offset_k": int(abs(g_idx - t)),
                    "sampling_method": "uniform_hard_neg",
                    "segment_id": seg_t,
                    "action_linear": float(row["linear_vel_cmd"]),
                    "action_lateral": float(row["lateral_vel_cmd"]),
                    "action_angular": float(row["angular_vel_cmd"]),
                    "state_idx_t3": int(t - 3),
                    "state_idx_t2": int(t - 2),
                    "state_idx_t1": int(t - 1),
                    "state_idx_t0": int(t),
                    "next_state_idx_t2": int(t - 2),
                    "next_state_idx_t1": int(t - 1),
                    "next_state_idx_t0": int(t),
                    "next_state_idx_t1_fw": int(t + 1),
                    "goal_embedding_idx": int(g_idx),
                    "goal_similarity": float(sim),
                    "reward": reward,
                    "done": reward,
                    "pos_x": float(row["pos_x"]) if "pos_x" in row else 0.0,
                    "pos_y": float(row["pos_y"]) if "pos_y" in row else 0.0,
                    "sim_time": float(row["sim_time"]) if "sim_time" in row else 0.0,
                })

        # 3. Random uniform negatives
        chosen_rand = rng.choice(len(goal_indices), size=num_random_goals, replace=True)
        for r_idx in chosen_rand:
            g_idx = int(goal_indices[r_idx])
            g_phi = phi_vectors[g_idx]
            sim = float((state_phis @ g_phi).mean())
            reward = int(sim >= 0.8)
            records.append({
                "current_global_step": int(t),
                "goal_global_step": int(g_idx),
                "offset_k": int(abs(g_idx - t)),
                "sampling_method": "uniform_random",
                "segment_id": seg_t,
                "action_linear": float(row["linear_vel_cmd"]),
                "action_lateral": float(row["lateral_vel_cmd"]),
                "action_angular": float(row["angular_vel_cmd"]),
                "state_idx_t3": int(t - 3),
                "state_idx_t2": int(t - 2),
                "state_idx_t1": int(t - 1),
                "state_idx_t0": int(t),
                "next_state_idx_t2": int(t - 2),
                "next_state_idx_t1": int(t - 1),
                "next_state_idx_t0": int(t),
                "next_state_idx_t1_fw": int(t + 1),
                "goal_embedding_idx": int(g_idx),
                "goal_similarity": float(sim),
                "reward": reward,
                "done": reward,
                "pos_x": float(row["pos_x"]) if "pos_x" in row else 0.0,
                "pos_y": float(row["pos_y"]) if "pos_y" in row else 0.0,
                "sim_time": float(row["sim_time"]) if "sim_time" in row else 0.0,
            })

    df = pd.DataFrame(records)
    print(f"\nMined Curriculum Uniform Transitions:")
    print(f"  Total mined transitions: {len(df):,}")
    print(f"  Positive reward share  : {100 * df['reward'].mean():.2f}%")
    return df
