import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import json

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "processed" / "test_run"
OUT_DIR = ROOT / "diagnostics" / "cls_representation" / "boundary_examples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_image(path):
    p = Path(path)
    if not p.exists():
        p = ROOT / p.name
    return Image.open(path)

def calc_stats(x):
    if len(x) == 0:
        return {"mean": 0, "median": 0, "count": 0}
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "count": len(x)
    }

def generate_examples(df, df_idx, pos_x, pos_y, paths, name, ranges):
    stats = {}
    
    # 1. Similarity percentages
    sim = df["goal_similarity"].values
    stats["reward_1_pct"] = float((sim >= 0.8).mean() * 100)
    stats["reward_0_pct"] = float((sim < 0.8).mean() * 100)
    
    # 2. Physical distance stats
    t = df["current_global_step"].values
    g = df["goal_embedding_idx"].values
    dist = np.sqrt((pos_x[t] - pos_x[g])**2 + (pos_y[t] - pos_y[g])**2)
    
    stats["dist_lt_0.8"] = calc_stats(dist[sim < 0.8])
    stats["dist_0.8_to_0.9"] = calc_stats(dist[(sim >= 0.8) & (sim < 0.9)])
    stats["dist_ge_0.9"] = calc_stats(dist[sim >= 0.9])
    
    # 3. Generate examples
    for r_name, (low, high) in ranges.items():
        mask = (sim >= low) & (sim < high)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        
        # sample up to 20
        np.random.seed(42)
        sampled = np.random.choice(indices, size=min(20, len(indices)), replace=False)
        
        for i, idx in enumerate(sampled):
            row = df.iloc[idx]
            t_curr = row["current_global_step"]
            t_goal = row["goal_embedding_idx"]
            
            # state is t-3, t-2, t-1, t
            t3, t2, t1, t0 = row["state_idx_t3"], row["state_idx_t2"], row["state_idx_t1"], row["state_idx_t0"]
            
            fig, axes = plt.subplots(1, 5, figsize=(15, 3))
            
            # load images
            axes[0].imshow(load_image(paths[t3]))
            axes[0].set_title("t-3")
            axes[0].axis('off')
            
            axes[1].imshow(load_image(paths[t2]))
            axes[1].set_title("t-2")
            axes[1].axis('off')
            
            axes[2].imshow(load_image(paths[t1]))
            axes[2].set_title("t-1")
            axes[2].axis('off')
            
            axes[3].imshow(load_image(paths[t0]))
            axes[3].set_title(f"t0 (step {t0})")
            axes[3].axis('off')
            
            axes[4].imshow(load_image(paths[t_goal]))
            axes[4].set_title(f"GOAL (step {t_goal})")
            axes[4].axis('off')
            
            # Text info
            d = np.sqrt((pos_x[t_curr] - pos_x[t_goal])**2 + (pos_y[t_curr] - pos_y[t_goal])**2)
            title = (f"{name.upper()} | Range: {r_name} | Sim: {row['goal_similarity']:.4f} | R: {row['reward']}\n"
                     f"t: {t_curr} -> g: {t_goal} | Temp Offset: {abs(t_curr - t_goal)} | Phys Dist: {d:.2f}m")
            plt.suptitle(title, fontsize=10)
            
            out_name = f"{name}_{r_name}_{i:02d}.jpg"
            plt.tight_layout()
            plt.savefig(OUT_DIR / out_name, dpi=100)
            plt.close()
            
    return stats

def recalculate_similarities(df, phi):
    t_arr = df["current_global_step"].values
    g_arr = df["goal_embedding_idx"].values
    sims = np.zeros(len(df), dtype=np.float32)
    for i, (t, g) in enumerate(zip(t_arr, g_arr)):
        state_phis = phi[t-3:t+1]
        sims[i] = (state_phis @ phi[g]).mean()
    df["goal_similarity"] = sims
    df["reward"] = (sims >= 0.8).astype(int)
    return df

def main():
    print("Loading datasets and phi vectors...")
    phi = np.load(DATA_DIR / "dinov3" / "phi_vectors.npy")
    # L2 normalize phi since the CLS token output wasn't normalized in the encoder
    phi = phi / np.linalg.norm(phi, axis=-1, keepdims=True)
    
    df_idx = pd.read_parquet(DATA_DIR / "frame_index" / "frame_index.parquet")
    df_geom = pd.read_parquet(DATA_DIR / "hindsight_geometric" / "geometric_transitions.parquet")
    df_unif = pd.read_parquet(DATA_DIR / "hindsight_uniform" / "uniform_transitions.parquet")
    
    # Recalculate similarities using the loaded phi vectors
    print("Recalculating similarities with current phi(o)...")
    df_geom = recalculate_similarities(df_geom, phi)
    df_unif = recalculate_similarities(df_unif, phi)
    
    pos_x = df_idx["pos_x"].values
    pos_y = df_idx["pos_y"].values
    paths = df_idx["rgb_abs_path"].values
    
    ranges = {
        "0.75_0.80": (0.75, 0.80),
        "0.80_0.85": (0.80, 0.85),
        "0.90_0.95": (0.90, 0.95),
        "0.95_1.00": (0.95, 1.0001)
    }
    
    print("Processing Geometric...")
    geom_stats = generate_examples(df_geom, df_idx, pos_x, pos_y, paths, "geometric", ranges)
    
    print("Processing Uniform...")
    unif_stats = generate_examples(df_unif, df_idx, pos_x, pos_y, paths, "uniform", ranges)
    
    # Save stats
    all_stats = {
        "geometric": geom_stats,
        "uniform": unif_stats
    }
    
    with open(OUT_DIR.parent / "boundary_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)
        
    print("\n[GEOMETRIC]")
    print(f"  Sim >= 0.8: {geom_stats['reward_1_pct']:.1f}%")
    print(f"  Sim <  0.8: {geom_stats['reward_0_pct']:.1f}%")
    print("  Physical Distance Stats:")
    print(f"    < 0.8:       mean={geom_stats['dist_lt_0.8']['mean']:.2f}m (N={geom_stats['dist_lt_0.8']['count']})")
    print(f"    0.8 - 0.9:   mean={geom_stats['dist_0.8_to_0.9']['mean']:.2f}m (N={geom_stats['dist_0.8_to_0.9']['count']})")
    print(f"    >= 0.9:      mean={geom_stats['dist_ge_0.9']['mean']:.2f}m (N={geom_stats['dist_ge_0.9']['count']})")
    
    print("\n[UNIFORM]")
    print(f"  Sim >= 0.8: {unif_stats['reward_1_pct']:.1f}%")
    print(f"  Sim <  0.8: {unif_stats['reward_0_pct']:.1f}%")
    print("  Physical Distance Stats:")
    print(f"    < 0.8:       mean={unif_stats['dist_lt_0.8']['mean']:.2f}m (N={unif_stats['dist_lt_0.8']['count']})")
    print(f"    0.8 - 0.9:   mean={unif_stats['dist_0.8_to_0.9']['mean']:.2f}m (N={unif_stats['dist_0.8_to_0.9']['count']})")
    print(f"    >= 0.9:      mean={unif_stats['dist_ge_0.9']['mean']:.2f}m (N={unif_stats['dist_ge_0.9']['count']})")

    print("\nDone! Examples saved to diagnostics/cls_representation/boundary_examples/")

if __name__ == "__main__":
    main()
