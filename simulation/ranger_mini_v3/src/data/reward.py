"""
src/data/reward.py
============================================================
Reward and done computation for hindsight goal relabeling.

PAPER SPECIFIES:
  s_t = [o'_{t-3}, o'_{t-2}, o'_{t-1}, o'_t]  (4-frame state)

  S(s_t, g) = (1/4) * sum_i cos(o'_{t-3+i}, g)
            = mean cosine similarity of 4 frames with goal

  delta_done = 0.8

  reward = 1 if S(s_t, g) >= 0.8  else 0
  done   = 1 if S(s_t, g) >= 0.8  else 0

PAPER DOES NOT SPECIFY:
  - Whether g is the full patch grid or a pooled vector.
  IMPLEMENTATION CHOICE: both state frames and goal use L2-normalized
  mean-pooled 384-D vectors (phi), so cosine similarity is
  dot product of unit vectors.

NOTE: done=1 ONLY from visual similarity.
  - Segment boundaries are NOT done signals.
  - Dataset exhaustion (t+1 doesn't exist) is NOT done=1.
"""

import numpy as np


SIMILARITY_THRESHOLD = 0.8    # PAPER SPECIFIES delta_done = 0.8


def compute_similarity(state_phis: np.ndarray, goal_phi: np.ndarray) -> float:
    """
    Compute S(s_t, g) = (1/4) * sum_i cos(o'_{t-3+i}, g).

    PAPER SPECIFIES this formula.

    Args:
        state_phis : [4, D] array of L2-normalized phi vectors for frames t-3..t
        goal_phi   : [D]    L2-normalized phi vector for goal

    Returns:
        float similarity in [-1, 1]
    """
    assert state_phis.shape[0] == 4, \
        f"Expected 4 state frames, got {state_phis.shape[0]}"
    # Both state_phis and goal_phi are already L2-normalized (unit vectors)
    # cosine similarity = dot product
    cos_sims = state_phis @ goal_phi  # [4]
    return float(cos_sims.mean())


def compute_reward_done(similarity: float) -> tuple[int, int]:
    """
    Compute reward and done from similarity score.

    PAPER SPECIFIES:
      reward = 1 if S >= 0.8 else 0
      done   = 1 if S >= 0.8 else 0
    """
    success = similarity >= SIMILARITY_THRESHOLD
    return int(success), int(success)


def batch_compute_reward_done(
    state_phis:  np.ndarray,   # [B, 4, D]
    goal_phis:   np.ndarray,   # [B, D]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized batch version.

    Returns:
        similarities: [B] float32
        rewards:      [B] int32
        dones:        [B] int32
    """
    # state_phis: [B, 4, D] @ goal_phis: [B, D, 1] -> [B, 4]
    goal_phis_3d = goal_phis[:, :, np.newaxis]           # [B, D, 1]
    cos_sims = np.matmul(state_phis, goal_phis_3d).squeeze(-1)  # [B, 4]
    similarities = cos_sims.mean(axis=1).astype(np.float32)     # [B]
    success = (similarities >= SIMILARITY_THRESHOLD)
    rewards = success.astype(np.int32)
    dones   = success.astype(np.int32)
    return similarities, rewards, dones
