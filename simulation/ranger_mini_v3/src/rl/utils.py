import random
import numpy as np
import torch
from pathlib import Path

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ActionNormalizer:
    """
    Explicit affine normalization between the ACTUAL physical action bounds and [-1, 1].
    action_min = [-0.9, -1.0, -1.5]  (vx, vy, omega)
    action_max = [ 1.8,  1.0,  1.5]
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.a_min = torch.tensor([-0.9, -1.0, -1.5], dtype=torch.float32, device=device)
        self.a_max = torch.tensor([ 1.8,  1.0,  1.5], dtype=torch.float32, device=device)
        
    def to(self, device):
        self.device = device
        self.a_min = self.a_min.to(device)
        self.a_max = self.a_max.to(device)
        return self

    def normalize(self, a):
        """
        physical action -> normalized action in [-1, 1]
        a_norm = 2 * (a - a_min) / (a_max - a_min) - 1
        """
        # Ensure a is on the right device and a tensor
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, dtype=torch.float32, device=self.device)
        return 2.0 * (a - self.a_min) / (self.a_max - self.a_min) - 1.0

    def denormalize(self, a_norm):
        """
        normalized action in [-1, 1] -> physical action
        a_physical = a_min + (a_norm + 1)/2 * (a_max - a_min)
        """
        if not isinstance(a_norm, torch.Tensor):
            a_norm = torch.tensor(a_norm, dtype=torch.float32, device=self.device)
        return self.a_min + (a_norm + 1.0) / 2.0 * (self.a_max - self.a_min)
