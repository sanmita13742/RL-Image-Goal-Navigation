"""
shared/action_normalizer.py — Affine action normalization for Ranger Mini V3.
=============================================================================
Extracted from simulation/ranger_mini_v3/src/rl/utils.py.
Pure PyTorch — no MuJoCo dependency.

CRITICAL (from AGENTS.md):
  Actions must strictly be affine normalized between their physical limits
  ([-0.9, -1.0, -1.5] to [1.8, 1.0, 1.5]) and [-1, 1].
  Simple division distorts the asymmetric vx behavior policy.
"""

import torch


class ActionNormalizer:
    """Explicit affine normalization between the ACTUAL physical action bounds and [-1, 1].

    Physical action bounds (from AGENTS.md):
        action_min = [-0.9, -1.0, -1.5]  (vx, vy, omega)
        action_max = [ 1.8,  1.0,  1.5]

    Note the asymmetric vx range: the robot drives faster forward (1.8 m/s)
    than backward (0.9 m/s). Simple symmetric normalization (dividing by max)
    would distort this asymmetry.
    """

    def __init__(self, device='cpu'):
        self.device = device
        self.a_min = torch.tensor([-0.9, -1.0, -1.5], dtype=torch.float32, device=device)
        self.a_max = torch.tensor([ 1.8,  1.0,  1.5], dtype=torch.float32, device=device)

    def to(self, device):
        """Move normalizer tensors to the specified device."""
        self.device = device
        self.a_min = self.a_min.to(device)
        self.a_max = self.a_max.to(device)
        return self

    def normalize(self, a):
        """Physical action → normalized action in [-1, 1].

        a_norm = 2 * (a - a_min) / (a_max - a_min) - 1
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, dtype=torch.float32, device=self.device)
        return 2.0 * (a - self.a_min) / (self.a_max - self.a_min) - 1.0

    def denormalize(self, a_norm):
        """Normalized action in [-1, 1] → physical action.

        a_physical = a_min + (a_norm + 1)/2 * (a_max - a_min)
        """
        if not isinstance(a_norm, torch.Tensor):
            a_norm = torch.tensor(a_norm, dtype=torch.float32, device=self.device)
        return self.a_min + (a_norm + 1.0) / 2.0 * (self.a_max - self.a_min)
