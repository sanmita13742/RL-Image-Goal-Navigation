import torch
import torch.nn.functional as F
import copy
from src.rl.networks import Critic

class FQE:
    def __init__(
        self,
        state_dim,
        action_dim,
        goal_dim,
        device,
        lr=3e-4,
        gamma=0.99,
        tau=0.005
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        
        self.critic = Critic(state_dim, action_dim, goal_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
    def train_step(self, state, next_state, reward, done, goal, actor):
        """
        Evaluate policy(s,g) rather than behavior action.
        """
        with torch.no_grad():
            # Get next action from the policy we are evaluating
            # The actor outputs normalized actions directly
            next_action_norm = actor(next_state, goal)
            
            target_Q1, target_Q2 = self.critic_target(next_state, next_action_norm, goal)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1.0 - done) * self.gamma * target_Q
            
            # For the current state, evaluate the policy's action
            curr_action_norm = actor(state, goal)
            
        current_Q1, current_Q2 = self.critic(state, curr_action_norm, goal)
        loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return loss.item(), current_Q1.mean().item()
