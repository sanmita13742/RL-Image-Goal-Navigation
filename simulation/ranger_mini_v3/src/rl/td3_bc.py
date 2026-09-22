import copy
import torch
import torch.nn.functional as F
from src.rl.networks import Actor, Critic
from src.rl.utils import ActionNormalizer

class TD3_BC:
    def __init__(
        self,
        state_dim,
        action_dim,
        goal_dim,
        device,
        lr_actor=3e-4,
        lr_critic=3e-4,
        gamma=0.99,
        tau=0.005,
        policy_delay=2,
        target_noise=0.2,
        target_noise_clip=0.5,
        lambda_bc=0.001
    ):
        self.device = device
        
        # Action Normalizer explicitly scales between real bounds and [-1, 1]
        self.normalizer = ActionNormalizer(device)
        
        self.actor = Actor(state_dim, goal_dim, action_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        self.critic = Critic(state_dim, action_dim, goal_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.target_noise = target_noise
        self.target_noise_clip = target_noise_clip
        self.lambda_bc = lambda_bc
        
        self.total_it = 0

    def train_critic(self, state, action, next_state, reward, done, goal):
        """
        Critic uses: 50% geometric, 50% uniform dataset sampled.
        Action is physical, we MUST normalize it for the critic input.
        """
        # 1. Normalize actions to [-1, 1]
        action_norm = self.normalizer.normalize(action)
        
        with torch.no_grad():
            # Target policy smoothing (on normalized actions)
            noise = (torch.randn_like(action_norm) * self.target_noise).clamp(
                -self.target_noise_clip, self.target_noise_clip
            )
            
            # Actor target outputs directly in [-1, 1]
            next_action_norm = (self.actor_target(next_state, goal) + noise).clamp(-1.0, 1.0)
            
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action_norm, goal)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1.0 - done) * self.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action_norm, goal)

        # Critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss.item(), current_Q1.mean().item(), current_Q2.mean().item(), target_Q.mean().item()

    def train_actor(self, state, action, goal):
        """
        Actor uses: 100% uniform dataset sampled.
        """
        # Normalize the behavior action from dataset
        action_norm = self.normalizer.normalize(action)
        
        pi_norm = self.actor(state, goal)
        Q = self.critic.q1_forward(state, pi_norm, goal)
        
        # BC loss in normalized action space
        # L = -lambda * Q + MSE(pi, a_behavior)
        # In TD3+BC paper, alpha = lambda_bc / (abs(Q).mean()).detach()
        # Wait, the prompt explicitly said:
        # L_actor = -Q1(s, actor(s,g), g) + lambda_bc * MSE(actor(s,g), behavior_action)
        # We will use exactly what the prompt specified.
        lmbda = self.lambda_bc
        # The paper (Fujimoto & Gu 2021) usually does alpha = 2.5 / abs(Q).mean(), 
        # but the prompt specifically says "lambda_bc = 0.001" and gives the exact formula.
        
        bc_loss_batch = torch.sum((pi_norm - action_norm) ** 2, dim=1)
        bc_loss_tensor = bc_loss_batch.mean()
        actor_loss = -Q.mean() + 0.001 * bc_loss_tensor
        bc_loss = bc_loss_tensor.item()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item(), bc_loss

    def train_step(self, critic_batch, actor_batch):
        self.total_it += 1
        
        # Critic Update
        state, action, next_state, reward, done, goal = critic_batch
        c_loss, q1_m, q2_m, tq_m = self.train_critic(state, action, next_state, reward, done, goal)
        
        a_loss = 0.0
        bc_loss = 0.0
        
        # Delayed Policy Updates
        if self.total_it % self.policy_delay == 0:
            a_state, a_action, _, _, _, a_goal = actor_batch
            a_loss, bc_loss = self.train_actor(a_state, a_action, a_goal)
            
            # Update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
        return {
            "critic_loss": c_loss,
            "q1_mean": q1_m,
            "q2_mean": q2_m,
            "target_q_mean": tq_m,
            "actor_loss": a_loss,
            "bc_loss": bc_loss
        }

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "total_it": self.total_it
        }

    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_target.load_state_dict(state_dict["actor_target"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic.load_state_dict(state_dict["critic"])
        self.critic_target.load_state_dict(state_dict["critic_target"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        self.total_it = state_dict["total_it"]
