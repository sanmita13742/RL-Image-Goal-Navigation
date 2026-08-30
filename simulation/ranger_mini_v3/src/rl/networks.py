import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super().__init__()
        # State + Goal as input
        input_dim = state_dim + goal_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        
    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=1)
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim):
        super().__init__()
        input_dim = state_dim + action_dim + goal_dim
        
        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, state, action, goal):
        x = torch.cat([state, action, goal], dim=1)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2
        
    def q1_forward(self, state, action, goal):
        x = torch.cat([state, action, goal], dim=1)
        return self.q1(x)
