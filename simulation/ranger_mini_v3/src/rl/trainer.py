import os
import yaml
import torch
import pandas as pd
from pathlib import Path
from collections import defaultdict
from src.rl.dataset import MinavHindsightDataset
from src.rl.td3_bc import TD3_BC
from src.rl.fqe import FQE
from src.rl.utils import set_seed

class Trainer:
    def __init__(self, config_dict, device_arg):
        self.config = config_dict
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device_arg != 'auto':
            self.device = torch.device(device_arg)
            
        set_seed(self.config['seed'])
        
        # Resolve output dir
        self.out_dir = Path(self.config['output_dir'])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.out_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.out_dir / "config.yaml", "w") as f:
            yaml.dump(self.config, f)
            
        # Dataset
        self.dataset = MinavHindsightDataset(
            self.config['dataset_path'], 
            feature_loading=self.config['feature_loading'],
            device=self.device
        )
        
        state_dim = 1536
        goal_dim = 384
        action_dim = 3
        
        self.agent = TD3_BC(
            state_dim=state_dim,
            action_dim=action_dim,
            goal_dim=goal_dim,
            device=self.device,
            lr_actor=self.config['learning_rate_actor'],
            lr_critic=self.config['learning_rate_critic'],
            gamma=self.config['gamma'],
            tau=self.config['tau'],
            policy_delay=self.config['policy_delay'],
            target_noise=self.config['target_noise'],
            target_noise_clip=self.config['target_noise_clip'],
            lambda_bc=self.config['lambda_bc']
        )
        
        self.fqe = FQE(
            state_dim=state_dim,
            action_dim=action_dim,
            goal_dim=goal_dim,
            device=self.device
        )
        
        self.metrics = defaultdict(list)
        self.best_fqe_q = -float('inf')
        self.best_fqe_step = -1
        
    def train(self):
        batch_size = self.config['batch_size']
        total_steps = self.config['total_gradient_steps']
        
        print(f"Starting training on {self.device} for {total_steps} steps...")
        
        for step in range(1, total_steps + 1):
            critic_batch = self.dataset.sample_critic(batch_size)
            actor_batch = self.dataset.sample_actor(batch_size)
            
            info = self.agent.train_step(critic_batch, actor_batch)
            
            if step % 100 == 0:
                self.metrics["step"].append(step)
                for k, v in info.items():
                    self.metrics[k].append(v)
                    
                df = pd.DataFrame(self.metrics)
                df.to_csv(self.out_dir / "metrics.csv", index=False)
                
                print(f"Step {step}/{total_steps} | c_loss: {info['critic_loss']:.4f} | a_loss: {info['actor_loss']:.4f} | bc_loss: {info['bc_loss']:.4f}")
                
            if step % self.config['checkpoint_frequency'] == 0 or step == total_steps:
                self.save_checkpoint(step)
                
                with torch.no_grad():
                    state, action, next_state, reward, done, goal = self.dataset.sample_actor(5000)
                    pi_norm = self.agent.actor(state, goal)
                    pi_phys = self.agent.normalizer.denormalize(pi_norm)
                    
                    diff = torch.abs(pi_phys - action)
                    diff_mean = diff.mean(dim=0).cpu().numpy()
                    l2_dist = torch.norm(pi_phys - action, dim=1).mean().item()
                    pi_phys_np = pi_phys.cpu().numpy()
                    
                    print(f"\n--- Checkpoint {step} Diagnostics ---")
                    print(f"Actor vs Dataset L2 Dist: {l2_dist:.4f}")
                    print(f"Mean Abs Diff |pi - a|  : vx={diff_mean[0]:.4f}, vy={diff_mean[1]:.4f}, wz={diff_mean[2]:.4f}")
                    print(f"Actor vx (mean/std/min/max): {pi_phys_np[:,0].mean():.4f}/{pi_phys_np[:,0].std():.4f}/{pi_phys_np[:,0].min():.4f}/{pi_phys_np[:,0].max():.4f}")
                    print(f"Actor vy (mean/std/min/max): {pi_phys_np[:,1].mean():.4f}/{pi_phys_np[:,1].std():.4f}/{pi_phys_np[:,1].min():.4f}/{pi_phys_np[:,1].max():.4f}")
                    print(f"Actor wz (mean/std/min/max): {pi_phys_np[:,2].mean():.4f}/{pi_phys_np[:,2].std():.4f}/{pi_phys_np[:,2].min():.4f}/{pi_phys_np[:,2].max():.4f}")
                    print(f"-----------------------------------\n")
                
            if step % self.config['fqe_frequency'] == 0 or step == total_steps:
                q_val = self.run_fqe(step)
                
                # Check if this is the best checkpoint according to FQE
                if q_val > self.best_fqe_q:
                    self.best_fqe_q = q_val
                    self.best_fqe_step = step
                    # Symlink or copy to best_checkpoint.pt
                    import shutil
                    src_path = self.ckpt_dir / f"checkpoint_{step}.pt"
                    dst_path = self.ckpt_dir / "best_checkpoint.pt"
                    if src_path.exists():
                        shutil.copy(src_path, dst_path)
                        print(f"New best checkpoint! (Q: {q_val:.4f})")
                        
        print(f"Training complete. Best checkpoint was step {self.best_fqe_step} with Q-value {self.best_fqe_q:.4f}")
                
    def save_checkpoint(self, step):
        path = self.ckpt_dir / f"checkpoint_{step}.pt"
        torch.save(self.agent.state_dict(), path)
        torch.save(self.agent.state_dict(), self.ckpt_dir / "latest.pt")
        print(f"Saved checkpoint to {path}")
        
    def run_fqe(self, step):
        print(f"Running FQE at step {step}...")
        fqe_steps = self.config.get('fqe_steps', 200)
        final_q = 0.0
        for i in range(fqe_steps):
            # FQE evaluates on uniform validation data
            state, action, next_state, reward, done, goal = self.dataset.sample_actor(self.config['batch_size'])
            fqe_loss, q_val = self.fqe.train_step(state, next_state, reward, done, goal, self.agent.actor)
            final_q = q_val
        print(f"FQE finished. Final loss: {fqe_loss:.4f} | Q_val: {final_q:.4f}")
        return final_q
