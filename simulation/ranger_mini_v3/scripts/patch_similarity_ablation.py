import os
import sys
import numpy as np
import pandas as pd
import json
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).parent.parent
OLD_CACHE = ROOT / "data" / "processed_PROVISIONAL_OLD_MEAN_POOLED" / "test_run" / "dinov3" / "patch_grids.npy"
NEW_CACHE_DIR = ROOT / "data" / "processed" / "test_run"
OUT_DIR = ROOT / "diagnostics" / "patch_similarity"

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "examples").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "maps").mkdir(parents=True, exist_ok=True)

def check_cache():
    print("Verifying Cache...")
    grid = np.load(OLD_CACHE)
    print(f"  Path: {OLD_CACHE}")
    print(f"  Shape: {grid.shape}")
    print(f"  Dtype: {grid.dtype}")
    print(f"  Min/Max: {grid.min():.4f} / {grid.max():.4f}")
    print(f"  NaNs: {np.isnan(grid).sum()}")
    print(f"  Infs: {np.isinf(grid).sum()}")
    
    # Check if normalized
    norm = np.linalg.norm(grid[0, 0, 0].astype(np.float32))
    print(f"  Sample patch norm: {norm:.4f}")
    is_norm = abs(norm - 1.0) < 1e-4
    print(f"  Is normalized: {is_norm}")
    
    return grid, is_norm

def get_stats(data):
    if len(data) == 0:
        return {k: 0.0 for k in ['mean', 'median', 'std', 'min', 'max', 'p01', 'p05', 'p10', 'p25', 'p75', 'p90', 'p95', 'p99']}
    return {
        "count": len(data),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "p01": float(np.percentile(data, 1)),
        "p05": float(np.percentile(data, 5)),
        "p10": float(np.percentile(data, 10)),
        "p25": float(np.percentile(data, 25)),
        "median": float(np.median(data)),
        "p75": float(np.percentile(data, 75)),
        "p90": float(np.percentile(data, 90)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
    }

def compute_similarities(idx_A, idx_B, phi_cls, patches):
    """
    Compute CLS, corresponding, best-match, and symmetric best-match.
    patches is expected to be [N, 1372, 384] and L2 normalized.
    """
    n_pairs = len(idx_A)
    sim_cls = np.zeros(n_pairs, dtype=np.float32)
    sim_corr = np.zeros(n_pairs, dtype=np.float32)
    sim_best = np.zeros(n_pairs, dtype=np.float32)
    sim_symm = np.zeros(n_pairs, dtype=np.float32)
    
    bs = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i in range(0, n_pairs, bs):
        a = idx_A[i:i+bs]
        b = idx_B[i:i+bs]
        
        # CLS
        cls_a = torch.from_numpy(phi_cls[a]).to(device)
        cls_b = torch.from_numpy(phi_cls[b]).to(device)
        s_cls = (cls_a * cls_b).sum(dim=-1).cpu().numpy()
        sim_cls[i:i+bs] = s_cls
        
        # Patches
        p_a = torch.from_numpy(patches[a]).to(device) # [bs, 1372, 384]
        p_b = torch.from_numpy(patches[b]).to(device)
        
        # Corresponding
        s_corr = (p_a * p_b).sum(dim=-1).mean(dim=-1).cpu().numpy()
        sim_corr[i:i+bs] = s_corr
        
        # Best-match (A -> B)
        # sim_matrix = [bs, 1372, 1372]
        sim_mat = torch.bmm(p_a, p_b.transpose(1, 2))
        s_best_A = sim_mat.max(dim=-1)[0].mean(dim=-1)
        sim_best[i:i+bs] = s_best_A.cpu().numpy()
        
        # Symmetric (B -> A)
        s_best_B = sim_mat.max(dim=-2)[0].mean(dim=-1)
        s_symm = 0.5 * (s_best_A + s_best_B)
        sim_symm[i:i+bs] = s_symm.cpu().numpy()
        
    return sim_cls, sim_corr, sim_best, sim_symm

def compute_4frame_similarities(t_idx, g_idx, phi_cls, patches):
    """
    S(s_t, g) = 1/4 * sum_i=0..3 S(t-3+i, g)
    """
    n = len(t_idx)
    res_cls = np.zeros(n, dtype=np.float32)
    res_corr = np.zeros(n, dtype=np.float32)
    res_best = np.zeros(n, dtype=np.float32)
    res_symm = np.zeros(n, dtype=np.float32)
    
    # To reuse batching, we just unfold the 4-frame comparison
    A_idx = []
    B_idx = []
    for t, g in zip(t_idx, g_idx):
        A_idx.extend([t-3, t-2, t-1, t])
        B_idx.extend([g, g, g, g])
        
    s_cls, s_corr, s_best, s_symm = compute_similarities(A_idx, B_idx, phi_cls, patches)
    
    res_cls = s_cls.reshape(-1, 4).mean(axis=1)
    res_corr = s_corr.reshape(-1, 4).mean(axis=1)
    res_best = s_best.reshape(-1, 4).mean(axis=1)
    res_symm = s_symm.reshape(-1, 4).mean(axis=1)
    
    return res_cls, res_corr, res_best, res_symm

def generate_visual_example(idx_A, idx_B, phi_cls, patches, paths, pos_x, pos_y, name, s_cls, s_corr, s_best):
    # Calculate maps
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    p_a = torch.from_numpy(patches[idx_A:idx_A+1]).to(device)
    p_b = torch.from_numpy(patches[idx_B:idx_B+1]).to(device)
    
    corr_map = (p_a * p_b).sum(dim=-1).reshape(28, 49).cpu().numpy()
    sim_mat = torch.bmm(p_a, p_b.transpose(1, 2))
    best_map = sim_mat.max(dim=-1)[0].reshape(28, 49).cpu().numpy()
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    def get_img(p):
        path = Path(p)
        if not path.exists(): path = ROOT / path.name
        return Image.open(path)
        
    axes[0, 0].imshow(get_img(paths[idx_A]))
    axes[0, 0].set_title(f"Image A (t={idx_A})")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(get_img(paths[idx_B]))
    axes[0, 1].set_title(f"Image B (t={idx_B})")
    axes[0, 1].axis('off')
    
    im1 = axes[1, 0].imshow(corr_map, cmap='viridis', vmin=corr_map.min(), vmax=corr_map.max())
    axes[1, 0].set_title(f"Corr-Patch Map (mean={s_corr:.4f})")
    fig.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    axes[1, 0].axis('off')
    
    im2 = axes[1, 1].imshow(best_map, cmap='viridis', vmin=best_map.min(), vmax=best_map.max())
    axes[1, 1].set_title(f"Best-Match Map (mean={s_best:.4f})")
    fig.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    axes[1, 1].axis('off')
    
    dist = np.sqrt((pos_x[idx_A] - pos_x[idx_B])**2 + (pos_y[idx_A] - pos_y[idx_B])**2)
    plt.suptitle(f"{name}\nTemp dist: {abs(idx_A-idx_B)} | Phys dist: {dist:.2f}m\nCLS: {s_cls:.4f} (R={(s_cls>=0.8)*1}) | Corr: {s_corr:.4f} (R={(s_corr>=0.8)*1}) | Best: {s_best:.4f} (R={(s_best>=0.8)*1})", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "examples" / f"{name}.jpg", dpi=100)
    plt.close()

def main():
    grid, is_norm = check_cache()
    
    print("Normalizing patches...")
    grid = grid.astype(np.float32)
    grid = grid.reshape(len(grid), -1, 384)
    patches = grid / np.linalg.norm(grid, axis=-1, keepdims=True)
    
    print("Loading CLS features...")
    phi_cls = np.load(NEW_CACHE_DIR / "dinov3" / "phi_vectors.npy")
    phi_cls = phi_cls / np.linalg.norm(phi_cls, axis=-1, keepdims=True)
    
    print("Loading datasets...")
    df_idx = pd.read_parquet(NEW_CACHE_DIR / "frame_index" / "frame_index.parquet")
    df_geom = pd.read_parquet(NEW_CACHE_DIR / "hindsight_geometric" / "geometric_transitions.parquet")
    df_unif = pd.read_parquet(NEW_CACHE_DIR / "hindsight_uniform" / "uniform_transitions.parquet")
    
    pos_x = df_idx["pos_x"].values
    pos_y = df_idx["pos_y"].values
    paths = df_idx["rgb_abs_path"].values
    
    results = {}
    
    N = 1000
    all_indices = np.arange(N)
    
    def process_pairs(name, idx_A, idx_B):
        s_cls, s_corr, s_best, s_symm = compute_similarities(idx_A, idx_B, phi_cls, patches)
        results[f"{name}_cls"] = get_stats(s_cls)
        results[f"{name}_corr"] = get_stats(s_corr)
        results[f"{name}_best"] = get_stats(s_best)
        results[f"{name}_symm"] = get_stats(s_symm)
        return s_cls, s_corr, s_best, s_symm

    print("Temporal t -> t+1...")
    process_pairs("t1", all_indices[:-1], all_indices[1:])
    
    print("Temporal t -> t+10...")
    process_pairs("t10", all_indices[:-10], all_indices[10:])
    
    print("Temporal t -> t+100...")
    process_pairs("t100", all_indices[:-100], all_indices[100:])
    
    print("Temporal t -> t+500...")
    process_pairs("t500", all_indices[:-500], all_indices[500:])
    
    print("Random distant...")
    np.random.seed(42)
    idx_a = np.random.randint(0, N, 10000)
    idx_b = np.random.randint(0, N, 10000)
    mask = np.abs(idx_a - idx_b) > 100
    idx_a, idx_b = idx_a[mask], idx_b[mask]
    idx_a, idx_b = idx_a[:1000], idx_b[:1000]
    
    r_cls, r_corr, r_best, r_symm = process_pairs("random", idx_a, idx_b)
    
    # 4-frame
    print("Geometric state -> goal...")
    s_cls, s_corr, s_best, s_symm = compute_4frame_similarities(
        df_geom["current_global_step"].values, df_geom["goal_embedding_idx"].values, phi_cls, patches)
    results["geom_cls"] = get_stats(s_cls)
    results["geom_corr"] = get_stats(s_corr)
    results["geom_best"] = get_stats(s_best)
    results["geom_symm"] = get_stats(s_symm)
    
    print("Uniform state -> goal...")
    u_cls, u_corr, u_best, u_symm = compute_4frame_similarities(
        df_unif["current_global_step"].values, df_unif["goal_embedding_idx"].values, phi_cls, patches)
    results["unif_cls"] = get_stats(u_cls)
    results["unif_corr"] = get_stats(u_corr)
    results["unif_best"] = get_stats(u_best)
    results["unif_symm"] = get_stats(u_symm)
    
    # Reward threshold analysis
    thresh_stats = {}
    for prefix, arrs in [("random", (r_cls, r_corr, r_best, r_symm)),
                         ("geom", (s_cls, s_corr, s_best, s_symm)),
                         ("unif", (u_cls, u_corr, u_best, u_symm))]:
        for m_name, arr in zip(["cls", "corr", "best", "symm"], arrs):
            thresh_stats[f"{prefix}_{m_name}_ge_0.8"] = float(np.mean(arr >= 0.8) * 100)
            
    # Physical Distance Analysis
    # Let's compute all similarities for a large random sample and bin them
    np.random.seed(43)
    p_a = np.random.randint(0, N, 5000)
    p_b = np.random.randint(0, N, 5000)
    pd_dist = np.sqrt((pos_x[p_a] - pos_x[p_b])**2 + (pos_y[p_a] - pos_y[p_b])**2)
    pd_cls, pd_corr, pd_best, _ = process_pairs("phys_sample", p_a, p_b)
    
    bins = [(0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 999.0)]
    bin_names = ["0-0.25m", "0.25-0.5m", "0.5-1m", "1-2m", "2-5m", ">5m"]
    phys_stats = {b: {} for b in bin_names}
    
    for (low, high), b_name in zip(bins, bin_names):
        mask = (pd_dist >= low) & (pd_dist < high)
        phys_stats[b_name]["count"] = int(np.sum(mask))
        if np.sum(mask) > 0:
            phys_stats[b_name]["cls_mean"] = float(np.mean(pd_cls[mask]))
            phys_stats[b_name]["corr_mean"] = float(np.mean(pd_corr[mask]))
            phys_stats[b_name]["best_mean"] = float(np.mean(pd_best[mask]))
            
    # Save CSVs
    pd.DataFrame(results).T.to_csv(OUT_DIR / "pair_statistics.csv")
    pd.DataFrame(phys_stats).T.to_csv(OUT_DIR / "distance_statistics.csv")
    pd.Series(thresh_stats).to_csv(OUT_DIR / "threshold_statistics.csv")
    
    # Generate visual examples
    print("Generating visual examples...")
    # A. nearby frames with high similarity
    mask = (pd_dist < 0.25) & (pd_corr > 0.9)
    if np.any(mask):
        idx = np.where(mask)[0][0]
        generate_visual_example(p_a[idx], p_b[idx], phi_cls, patches, paths, pos_x, pos_y, "A_nearby_high_sim", pd_cls[idx], pd_corr[idx], pd_best[idx])
        
    # B. distant frames with high similarity (focusing on CLS false positives)
    mask = (pd_dist > 3.0) & (pd_cls > 0.85)
    if np.any(mask):
        idx = np.where(mask)[0][0]
        generate_visual_example(p_a[idx], p_b[idx], phi_cls, patches, paths, pos_x, pos_y, "B_distant_high_sim_cls_fp", pd_cls[idx], pd_corr[idx], pd_best[idx])
        
    # C. distant frames with low similarity
    mask = (pd_dist > 5.0) & (pd_best < 0.6)
    if np.any(mask):
        idx = np.where(mask)[0][0]
        generate_visual_example(p_a[idx], p_b[idx], phi_cls, patches, paths, pos_x, pos_y, "C_distant_low_sim", pd_cls[idx], pd_corr[idx], pd_best[idx])
        
    # D. Viewpoint shift (temporally close, but some rotation)
    mask = (np.abs(p_a - p_b) > 10) & (np.abs(p_a - p_b) < 30) & (pd_dist < 0.5)
    if np.any(mask):
        idx = np.where(mask)[0][0]
        generate_visual_example(p_a[idx], p_b[idx], phi_cls, patches, paths, pos_x, pos_y, "D_viewpoint_shift", pd_cls[idx], pd_corr[idx], pd_best[idx])
        
    print("Done generating examples.")
    
    # Generate Report
    report = f"""# MINav Patch-Wise Similarity Ablation Report

*Patch-wise similarity was evaluated as an ablation because the paper specifies DINOv3 final hidden-layer patch embeddings but does not explicitly specify the exact transformation used to obtain the image-level phi(o) used in the cosine reward.*

## 1. Required Comparison Table

| Metric | CLS | Corresponding Patch | Best-Match Patch | Symmetric Best-Match |
|---|---:|---:|---:|---:|
| t->t+1 | {results['t1_cls']['mean']:.4f} | {results['t1_corr']['mean']:.4f} | {results['t1_best']['mean']:.4f} | {results['t1_symm']['mean']:.4f} |
| t->t+10 | {results['t10_cls']['mean']:.4f} | {results['t10_corr']['mean']:.4f} | {results['t10_best']['mean']:.4f} | {results['t10_symm']['mean']:.4f} |
| t->t+100 | {results['t100_cls']['mean']:.4f} | {results['t100_corr']['mean']:.4f} | {results['t100_best']['mean']:.4f} | {results['t100_symm']['mean']:.4f} |
| t->t+500 | {results['t500_cls']['mean']:.4f} | {results['t500_corr']['mean']:.4f} | {results['t500_best']['mean']:.4f} | {results['t500_symm']['mean']:.4f} |
| random distant | {results['random_cls']['mean']:.4f} | {results['random_corr']['mean']:.4f} | {results['random_best']['mean']:.4f} | {results['random_symm']['mean']:.4f} |
| geometric state->goal | {results['geom_cls']['mean']:.4f} | {results['geom_corr']['mean']:.4f} | {results['geom_best']['mean']:.4f} | {results['geom_symm']['mean']:.4f} |
| uniform state->goal | {results['unif_cls']['mean']:.4f} | {results['unif_corr']['mean']:.4f} | {results['unif_best']['mean']:.4f} | {results['unif_symm']['mean']:.4f} |
| random reward=1 % | {thresh_stats['random_cls_ge_0.8']:.1f}% | {thresh_stats['random_corr_ge_0.8']:.1f}% | {thresh_stats['random_best_ge_0.8']:.1f}% | {thresh_stats['random_symm_ge_0.8']:.1f}% |
| geometric reward=1 % | {thresh_stats['geom_cls_ge_0.8']:.1f}% | {thresh_stats['geom_corr_ge_0.8']:.1f}% | {thresh_stats['geom_best_ge_0.8']:.1f}% | {thresh_stats['geom_symm_ge_0.8']:.1f}% |
| uniform reward=1 % | {thresh_stats['unif_cls_ge_0.8']:.1f}% | {thresh_stats['unif_corr_ge_0.8']:.1f}% | {thresh_stats['unif_best_ge_0.8']:.1f}% | {thresh_stats['unif_symm_ge_0.8']:.1f}% |

## 2. Spatial Discrimination Table

| Physical distance | CLS | Corresponding Patch | Best-Match Patch |
|---|---:|---:|---:|
"""
    for b_name in bin_names:
        stats = phys_stats[b_name]
        if stats['count'] > 0:
            report += f"| {b_name} | {stats['cls_mean']:.4f} | {stats['corr_mean']:.4f} | {stats['best_mean']:.4f} |\n"
        else:
            report += f"| {b_name} | N/A | N/A | N/A |\n"

    report += """
## 3. Final Verdict (Template - Please Edit After Review)
PATCH-WISE STATUS:
    Corresponding patch:
        BETTER / SAME / WORSE than CLS
    Best-match patch:
        BETTER / SAME / WORSE than CLS
    Symmetric best-match:
        BETTER / SAME / WORSE than CLS

1. Which method gives the strongest spatial discrimination?
2. Which method is most robust to viewpoint changes?
3. Which method produces the fewest obvious distant false positives?
4. Does patch information appear useful beyond CLS?
5. Is there evidence that the current CLS representation is sufficient?
6. Should we continue with CLS or investigate patch-aware phi further?
"""
    with open(OUT_DIR / "PATCH_SIMILARITY_REPORT.md", "w") as f:
        f.write(report)
        
if __name__ == "__main__":
    main()
