import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.rl.utils import ActionNormalizer
from src.rl.dataset import MinavHindsightDataset
from src.rl.td3_bc import TD3_BC
from src.rl.fqe import FQE

def test_action_normalization():
    print("\n--- Testing Action Normalization ---")
    norm = ActionNormalizer('cpu')
    a_min = norm.a_min
    a_max = norm.a_max
    
    # 1. action_min maps exactly to -1
    out_min = norm.normalize(a_min)
    assert torch.allclose(out_min, torch.tensor([-1.0, -1.0, -1.0])), f"Min map failed: {out_min}"
    print("[PASS] action_min maps to -1")
    
    # 2. action_max maps exactly to +1
    out_max = norm.normalize(a_max)
    assert torch.allclose(out_max, torch.tensor([1.0, 1.0, 1.0])), f"Max map failed: {out_max}"
    print("[PASS] action_max maps to +1")
    
    # 3. midpoint maps to 0
    mid = (a_min + a_max) / 2.0
    out_mid = norm.normalize(mid)
    assert torch.allclose(out_mid, torch.zeros(3)), f"Mid map failed: {out_mid}"
    print("[PASS] midpoint maps to 0")
    
    # 4. Round trip reconstruction
    test_act = torch.tensor([0.0, 0.5, -0.5])
    reconstructed = norm.denormalize(norm.normalize(test_act))
    assert torch.allclose(test_act, reconstructed, atol=1e-6), f"Reconstruction failed"
    print("[PASS] normalize -> denormalize reconstructs physical action")

def test_dataset_and_samplers():
    print("\n--- Testing Dataset & Samplers ---")
    dataset = MinavHindsightDataset('data/processed', feature_loading='ram', device='cpu')
    
    # Check 5: Every normalized dataset action lies within [-1, 1]
    norm = ActionNormalizer('cpu')
    geom_act = dataset.geom_buffer.actions
    unif_act = dataset.unif_buffer.actions
    all_act = torch.tensor(np.concatenate([geom_act, unif_act], axis=0))
    norm_act = norm.normalize(all_act)
    
    assert torch.all(norm_act >= -1.0) and torch.all(norm_act <= 1.0), "Normalized actions out of bounds!"
    print("[PASS] Every normalized dataset action is strictly in [-1, 1]")
    
    print("Physical action bounds:")
    print(f"    vx     [{all_act[:,0].min():.2f}, {all_act[:,0].max():.2f}]")
    print(f"    vy     [{all_act[:,1].min():.2f}, {all_act[:,1].max():.2f}]")
    print(f"    omega  [{all_act[:,2].min():.2f}, {all_act[:,2].max():.2f}]")
    print("Normalized action bounds:")
    print(f"    vx     [{norm_act[:,0].min():.2f}, {norm_act[:,0].max():.2f}]")
    print(f"    vy     [{norm_act[:,1].min():.2f}, {norm_act[:,1].max():.2f}]")
    print(f"    omega  [{norm_act[:,2].min():.2f}, {norm_act[:,2].max():.2f}]")
    
    critic_batch = dataset.sample_critic(64)
    actor_batch = dataset.sample_actor(64)
    
    assert critic_batch[0].shape == (64, 1536), "Critic state shape wrong"
    assert critic_batch[1].shape == (64, 3), "Critic action shape wrong"
    assert actor_batch[0].shape == (64, 1536), "Actor state shape wrong"
    
    print("[PASS] Dataset shapes correct")
    return dataset

def test_td3_bc_forward_backward(dataset):
    print("\n--- Testing TD3+BC Forward/Backward passes ---")
    agent = TD3_BC(
        state_dim=1536, action_dim=3, goal_dim=384, device='cpu'
    )
    
    critic_batch = dataset.sample_critic(32)
    actor_batch = dataset.sample_actor(32)
    
    info = agent.train_step(critic_batch, actor_batch)
    
    assert not np.isnan(info['critic_loss']), "Critic loss is NaN"
    assert not np.isnan(info['actor_loss']), "Actor loss is NaN"
    
    print("[PASS] Forward and Backward passes successful (no NaNs)")
    return agent

def test_fqe(dataset, agent):
    print("\n--- Testing FQE Smoke Test ---")
    fqe = FQE(state_dim=1536, action_dim=3, goal_dim=384, device='cpu')
    state, action, next_state, reward, done, goal = dataset.sample_actor(32)
    
    loss, q_val = fqe.train_step(state, next_state, reward, done, goal, agent.actor, agent.normalizer)
    
    assert not np.isnan(loss), "FQE loss is NaN"
    print("[PASS] FQE Smoke test successful")

def test_checkpointing(agent):
    print("\n--- Testing Checkpoint Save/Load ---")
    tmp_path = Path("test_checkpoint_tmp.pt")
    
    try:
        sd = agent.state_dict()
        torch.save(sd, tmp_path)
        
        agent_new = TD3_BC(state_dim=1536, action_dim=3, goal_dim=384, device='cpu')
        loaded_sd = torch.load(tmp_path, map_location='cpu')
        agent_new.load_state_dict(loaded_sd)
        print("[PASS] Checkpoint save and load successful")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

def main():
    try:
        test_action_normalization()
        ds = test_dataset_and_samplers()
        agent = test_td3_bc_forward_backward(ds)
        test_fqe(ds, agent)
        test_checkpointing(agent)
        
        print("\n===============================")
        print("ALL CHECKS PASSED")
        print("===============================\n")
    except Exception as e:
        print(f"\nVALIDATION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
