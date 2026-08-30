import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def get_stats(data, prefix=""):
    return {
        f"{prefix}mean": np.mean(data),
        f"{prefix}median": np.median(data),
        f"{prefix}p01": np.percentile(data, 1),
        f"{prefix}p05": np.percentile(data, 5),
        f"{prefix}p10": np.percentile(data, 10),
        f"{prefix}p25": np.percentile(data, 25),
        f"{prefix}p50": np.percentile(data, 50),
        f"{prefix}p75": np.percentile(data, 75),
        f"{prefix}p90": np.percentile(data, 90),
        f"{prefix}p95": np.percentile(data, 95),
        f"{prefix}p99": np.percentile(data, 99),
        f"{prefix}min": np.min(data),
        f"{prefix}max": np.max(data),
    }

def cosine_similarity(a, b):
    # A and B are expected to be unit normalized
    return np.sum(a * b, axis=-1)

def main():
    root = Path(__file__).parent.parent
    old_dir = root / "data" / "processed_PROVISIONAL_OLD_MEAN_POOLED" / "test_run" / "dinov3"
    new_dir = root / "data" / "processed" / "test_run" / "dinov3"
    out_dir = root / "diagnostics" / "cls_representation"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not old_dir.exists() or not new_dir.exists():
        print(f"Directories missing: \nOld: {old_dir}\nNew: {new_dir}")
        return

    phi_old = np.load(old_dir / "phi_vectors.npy")[:1000]
    phi_new = np.load(new_dir / "phi_vectors.npy")
    
    ssd_old = np.load(old_dir / "ssd_scores.npy")[:1000]
    ssd_new = np.load(new_dir / "ssd_scores.npy")
    
    # Optional: L2 normalize if not already
    phi_old_norm = phi_old / np.linalg.norm(phi_old, axis=-1, keepdims=True)
    phi_new_norm = phi_new / np.linalg.norm(phi_new, axis=-1, keepdims=True)
    
    results = {}
    
    for name, phi, ssd in [("Mean-Pool", phi_old_norm, ssd_old), ("DINOv3 Normalized CLS", phi_new_norm, ssd_new)]:
        n = len(phi)
        t_to_t1 = cosine_similarity(phi[:-1], phi[1:])
        t_to_t10 = cosine_similarity(phi[:-10], phi[10:])
        t_to_t100 = cosine_similarity(phi[:-100], phi[100:])
        t_to_t500 = cosine_similarity(phi[:-500], phi[500:])
        
        # Random distant
        idx_a = np.random.randint(0, n, 10000)
        idx_b = np.random.randint(0, n, 10000)
        dist = np.abs(idx_a - idx_b) > 100
        idx_a = idx_a[dist]
        idx_b = idx_b[dist]
        rand_dist = cosine_similarity(phi[idx_a], phi[idx_b])
        
        feature_norm = np.linalg.norm(np.mean(phi, axis=0))
        
        # State goal similarities from hindsight
        s_t = (phi[0:n-3] + phi[1:n-2] + phi[2:n-1] + phi[3:n]) / 4.0
        
        geom_goals = []
        for i in range(len(s_t)):
            k = np.random.geometric(0.01)
            g_idx = min(i+3+k, n-1)
            geom_goals.append(phi[g_idx])
        geom_goals = np.array(geom_goals)
        
        # Paper's SSD threshold is 0.02
        valid_indices = np.where(ssd > 0.02)[0]
        if len(valid_indices) == 0:
            unif_goals = phi[np.random.randint(0, n, len(s_t))]
        else:
            unif_goals = phi[np.random.choice(valid_indices, len(s_t))]
        
        geom_sim = cosine_similarity(s_t, geom_goals)
        unif_sim = cosine_similarity(s_t, unif_goals)
        
        geom_reward = np.mean(geom_sim >= 0.8) * 100
        unif_reward = np.mean(unif_sim >= 0.8) * 100
        
        results[name] = {
            "t -> t+1": np.mean(t_to_t1),
            "t -> t+10": np.mean(t_to_t10),
            "t -> t+100": np.mean(t_to_t100),
            "t -> t+500": np.mean(t_to_t500),
            "random distant": np.mean(rand_dist),
            "geometric state->goal": np.mean(geom_sim),
            "uniform state->goal": np.mean(unif_sim),
            "uniform reward=1 %": unif_reward,
            "geometric reward=1 %": geom_reward,
            "mean feature norm": feature_norm,
        }
        
    df = pd.DataFrame(results).T
    print(df.T.to_markdown())
    
    with open(out_dir / "comparison_report.md", "w") as f:
        f.write("# Representation Comparison\\n\\n")
        f.write(df.T.to_markdown())

if __name__ == "__main__":
    main()
