import torch
import numpy as np
import pandas as pd
from pathlib import Path

class TransitionBuffer:
    """
    Holds one type of transitions (geometric or uniform).
    Uses pandas to load the parquet, then converts to numpy arrays for fast indexing.
    """
    def __init__(self, parquet_path, phi_cache):
        print(f"Loading {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        
        self.state_idx = df[['state_idx_t3', 'state_idx_t2', 'state_idx_t1', 'state_idx_t0']].values.astype(np.int64)
        self.next_state_idx = df[['next_state_idx_t2', 'next_state_idx_t1', 'next_state_idx_t0', 'next_state_idx_t1_fw']].values.astype(np.int64)
        self.goal_idx = df['goal_embedding_idx'].values.astype(np.int64)
        
        self.actions = df[['action_linear', 'action_lateral', 'action_angular']].values.astype(np.float32)
        self.rewards = df['reward'].values.astype(np.float32).reshape(-1, 1)
        self.dones = df['done'].values.astype(np.float32).reshape(-1, 1)
        
        self.phi_cache = phi_cache
        self.size = len(df)
        
    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        
        # State construction: 4 frames * 384 dim = 1536
        s_idx = self.state_idx[idx]
        ns_idx = self.next_state_idx[idx]
        g_idx = self.goal_idx[idx]
        
        # shape [B, 4, 384] -> [B, 1536]
        states = self.phi_cache[s_idx].reshape(batch_size, -1)
        next_states = self.phi_cache[ns_idx].reshape(batch_size, -1)
        goals = self.phi_cache[g_idx]
        
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        dones = self.dones[idx]
        
        return states, actions, next_states, rewards, dones, goals


class MinavHindsightDataset:
    def __init__(self, dataset_path, feature_loading='ram', device='cpu'):
        self.dataset_path = Path(dataset_path)
        self.device = device
        
        phi_path = self.dataset_path / "dinov3" / "phi_vectors.npy"
        print(f"Loading phi_cache from {phi_path} (mode: {feature_loading})...")
        if feature_loading == 'mmap':
            self.phi_cache = np.load(phi_path, mmap_mode='r')
        else:
            self.phi_cache = np.load(phi_path)
            
        geom_path = self.dataset_path / "hindsight" / "geometric_transitions.parquet"
        unif_path = self.dataset_path / "hindsight" / "uniform_transitions.parquet"
        
        self.geom_buffer = TransitionBuffer(geom_path, self.phi_cache)
        self.unif_buffer = TransitionBuffer(unif_path, self.phi_cache)
        
    def _to_tensor(self, arrays):
        return tuple(torch.tensor(a, dtype=torch.float32, device=self.device) for a in arrays)

    def sample_critic(self, batch_size):
        """
        Critic sampling: 50% geometric, 50% uniform (Paper specified).
        """
        half = batch_size // 2
        rest = batch_size - half
        
        g_states, g_actions, g_next, g_rewards, g_dones, g_goals = self.geom_buffer.sample(half)
        u_states, u_actions, u_next, u_rewards, u_dones, u_goals = self.unif_buffer.sample(rest)
        
        states = np.concatenate([g_states, u_states], axis=0)
        actions = np.concatenate([g_actions, u_actions], axis=0)
        next_states = np.concatenate([g_next, u_next], axis=0)
        rewards = np.concatenate([g_rewards, u_rewards], axis=0)
        dones = np.concatenate([g_dones, u_dones], axis=0)
        goals = np.concatenate([g_goals, u_goals], axis=0)
        
        return self._to_tensor((states, actions, next_states, rewards, dones, goals))
        
    def sample_actor(self, batch_size):
        """
        Actor sampling: 100% uniform (Paper specified).
        """
        arrays = self.unif_buffer.sample(batch_size)
        return self._to_tensor(arrays)

