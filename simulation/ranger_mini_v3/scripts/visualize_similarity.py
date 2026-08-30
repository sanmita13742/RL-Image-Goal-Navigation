import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "processed" / "test_run"
OUT_DIR = ROOT / "diagnostics" / "cls_representation" / "examples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_image(path):
    # Some paths might be absolute, some relative. If absolute but wrong root, fix it.
    p = Path(path)
    if not p.exists():
        # attempt to fix path relative to root
        p = ROOT / p.name  # simplistic fallback if needed, but the index should have absolute
    return Image.open(path)

def save_pair(img_a_path, img_b_path, title_info, out_filename):
    try:
        img_a = load_image(img_a_path)
        img_b = load_image(img_b_path)
    except Exception as e:
        print(f"Error loading images: {e}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_a)
    axes[0].axis('off')
    axes[0].set_title("Image A (State)")
    
    axes[1].imshow(img_b)
    axes[1].axis('off')
    axes[1].set_title("Image B (Goal / Compare)")
    
    plt.suptitle(title_info, fontsize=10, y=0.98)
    plt.tight_layout()
    plt.savefig(OUT_DIR / out_filename, bbox_inches='tight', dpi=150)
    plt.close()

def main():
    print("Loading data for visualization...")
    phi = np.load(DATA_DIR / "dinov3" / "phi_vectors.npy")
    df_idx = pd.read_parquet(DATA_DIR / "frame_index" / "frame_index.parquet")
    df_geom = pd.read_parquet(DATA_DIR / "hindsight_geometric" / "geometric_transitions.parquet")
    df_unif = pd.read_parquet(DATA_DIR / "hindsight_uniform" / "uniform_transitions.parquet")
    
    pos_x = df_idx["pos_x"].values
    pos_y = df_idx["pos_y"].values
    paths = df_idx["rgb_abs_path"].values
    
    N = len(phi)
    
    # A. very high similarity between physically distant frames
    print("Generating Group A (High Sim, Distant)...")
    np.random.seed(1)
    idx_a = np.random.randint(0, N, 10000)
    idx_b = np.random.randint(0, N, 10000)
    dist = np.sqrt((pos_x[idx_a] - pos_x[idx_b])**2 + (pos_y[idx_a] - pos_y[idx_b])**2)
    sim = (phi[idx_a] * phi[idx_b]).sum(axis=1)
    
    mask = (dist > 1.0) & (sim > 0.8)
    valid_a, valid_b, valid_sim, valid_dist = idx_a[mask], idx_b[mask], sim[mask], dist[mask]
    
    for i in range(min(10, len(valid_a))):
        a, b = valid_a[i], valid_b[i]
        t_info = (f"A: step {a} | B: step {b} | phys_dist: {valid_dist[i]:.2f}m\n"
                  f"temp_dist: {abs(a-b)} | sim: {valid_sim[i]:.4f}")
        save_pair(paths[a], paths[b], t_info, f"A_high_sim_distant_{i}.jpg")

    # B. very low similarity between temporally adjacent frames
    print("Generating Group B (Low Sim, Adjacent)...")
    sim_adj = (phi[:-1] * phi[1:]).sum(axis=1)
    # find lowest similarity adjacent frames
    lowest_idx = np.argsort(sim_adj)[:10]
    for i, a in enumerate(lowest_idx):
        b = a + 1
        d = np.sqrt((pos_x[a] - pos_x[b])**2 + (pos_y[a] - pos_y[b])**2)
        t_info = (f"A: step {a} | B: step {b} | phys_dist: {d:.2f}m\n"
                  f"temp_dist: 1 | sim: {sim_adj[a]:.4f}")
        save_pair(paths[a], paths[b], t_info, f"B_low_sim_adjacent_{i}.jpg")
        
    # C. geometric goal pairs
    print("Generating Group C (Geometric Goals)...")
    sample_geom = df_geom.sample(min(10, len(df_geom)), random_state=2)
    for i, row in enumerate(sample_geom.itertuples()):
        t = row.current_global_step
        g = row.goal_embedding_idx
        d = np.sqrt((pos_x[t] - pos_x[g])**2 + (pos_y[t] - pos_y[g])**2)
        t_info = (f"Geometric - A: step {t} | B: step {g} | phys_dist: {d:.2f}m\n"
                  f"temp_dist: {abs(t-g)} | S(s_t,g): {row.goal_similarity:.4f} | R: {row.reward} | Done: {row.done}")
        save_pair(paths[t], paths[g], t_info, f"C_geometric_goal_{i}.jpg")

    # D. uniform goal pairs
    print("Generating Group D (Uniform Goals)...")
    sample_unif = df_unif.sample(min(10, len(df_unif)), random_state=3)
    for i, row in enumerate(sample_unif.itertuples()):
        t = row.current_global_step
        g = row.goal_embedding_idx
        d = np.sqrt((pos_x[t] - pos_x[g])**2 + (pos_y[t] - pos_y[g])**2)
        t_info = (f"Uniform - A: step {t} | B: step {g} | phys_dist: {d:.2f}m\n"
                  f"temp_dist: {abs(t-g)} | S(s_t,g): {row.goal_similarity:.4f} | R: {row.reward} | Done: {row.done}")
        save_pair(paths[t], paths[g], t_info, f"D_uniform_goal_{i}.jpg")
        
    print("Visualizations complete.")

if __name__ == "__main__":
    main()
