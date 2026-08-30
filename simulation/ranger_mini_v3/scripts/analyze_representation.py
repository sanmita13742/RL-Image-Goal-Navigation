import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "processed" / "test_run"
OUT_DIR = ROOT / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def calc_stats(x):
    if len(x) == 0:
        return {}
    return {
        "count": len(x),
        "min": float(np.min(x)),
        "p01": float(np.percentile(x, 1)),
        "p05": float(np.percentile(x, 5)),
        "p10": float(np.percentile(x, 10)),
        "p25": float(np.percentile(x, 25)),
        "median": float(np.median(x)),
        "p75": float(np.percentile(x, 75)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x))
    }

def print_stats(name, s):
    if not s: return
    print(f"\n[{name}] (N={s['count']})")
    print(f"  Mean: {s['mean']:.4f}  Median: {s['median']:.4f}  Std: {s['std']:.4f}")
    print(f"  Min: {s['min']:.4f}  Max: {s['max']:.4f}")
    print(f"  Percentiles: 1%={s['p01']:.4f}, 10%={s['p10']:.4f}, 25%={s['p25']:.4f}, 75%={s['p75']:.4f}, 90%={s['p90']:.4f}, 99%={s['p99']:.4f}")

def binned_counts(sims):
    bins = [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.00001]
    hist, _ = np.histogram(sims, bins=bins)
    pcts = hist / len(sims) * 100
    res = {}
    for i in range(len(bins)-1):
        label = f"[{bins[i]:.2f}, {bins[i+1]:.2f})" if i < len(bins)-2 else f"[{bins[i]:.2f}, 1.0]"
        res[label] = float(pcts[i])
    return res

def main():
    print("Loading data...")
    phi = np.load(DATA_DIR / "dinov3" / "phi_vectors.npy") # [N, 384]
    df_idx = pd.read_parquet(DATA_DIR / "frame_index" / "frame_index.parquet")
    df_geom = pd.read_parquet(DATA_DIR / "hindsight_geometric" / "geometric_transitions.parquet")
    df_unif = pd.read_parquet(DATA_DIR / "hindsight_uniform" / "uniform_transitions.parquet")
    
    N = len(phi)
    print(f"Dataset Size: {N} frames")
    
    report_data = {"dataset_size": N}
    
    # --- DIAGNOSTIC 1: TEMPORAL SIMILARITY ---
    temporal = {}
    for offset in [1, 10, 100, 500]:
        valid_t = np.arange(N - offset)
        sims = (phi[valid_t] * phi[valid_t + offset]).sum(axis=1) # dot product since L2 norm
        temporal[f"t+{offset}"] = calc_stats(sims)
        print_stats(f"Temporal t vs t+{offset}", temporal[f"t+{offset}"])
    
    report_data["temporal"] = temporal
    
    # --- DIAGNOSTIC 2: RANDOM GLOBAL SIMILARITY ---
    np.random.seed(42)
    rand_t = np.random.randint(0, N, 10000)
    rand_j = np.random.randint(0, N, 10000)
    valid_mask = np.abs(rand_j - rand_t) > 100
    rand_t = rand_t[valid_mask]
    rand_j = rand_j[valid_mask]
    
    rand_sims = (phi[rand_t] * phi[rand_j]).sum(axis=1)
    rand_stats = calc_stats(rand_sims)
    print_stats("Random Global Similarity (|j-t| > 100)", rand_stats)
    report_data["random_global"] = rand_stats
    
    # --- DIAGNOSTIC 3 & 4: HINDSIGHT GOALS ---
    hindsight = {}
    for df, name in [(df_geom, "geometric"), (df_unif, "uniform")]:
        t_arr = df["current_global_step"].values
        g_arr = df["goal_embedding_idx"].values
        
        # calculate independently S(s_t, g)
        s_g_sims = np.zeros(len(df))
        for i, (t, g) in enumerate(zip(t_arr, g_arr)):
            state_phis = phi[t-3:t+1] # 4 frames
            s_g_sims[i] = (state_phis @ phi[g]).mean()
            
        r_calc = (s_g_sims >= 0.8).astype(int)
        assert np.all(r_calc == df["reward"].values), "Reward logic mismatch!"
        
        stats = calc_stats(s_g_sims)
        offsets = g_arr - t_arr
        
        print_stats(f"{name.capitalize()} Goal Similarity S(s_t, g)", stats)
        print(f"  Reward=1: {(r_calc==1).mean()*100:.2f}% | Reward=0: {(r_calc==0).mean()*100:.2f}%")
        
        off_stats = calc_stats(np.abs(offsets))
        print(f"  Absolute Offset (frames) Mean: {off_stats['mean']:.1f}, Median: {off_stats['median']:.1f}")
        
        hindsight[name] = {
            "similarity_stats": stats,
            "reward_1_pct": float((r_calc==1).mean()*100),
            "reward_0_pct": float((r_calc==0).mean()*100),
            "offset_stats": off_stats,
            "threshold_bins": binned_counts(s_g_sims)
        }
    
    report_data["hindsight"] = hindsight
    
    # --- DIAGNOSTIC 6: REWARD THRESHOLD CHECK ---
    print("\n[Reward Threshold Check Bins]")
    for name in ["geometric", "uniform"]:
        print(f"  {name}:")
        for k, v in hindsight[name]["threshold_bins"].items():
            print(f"    {k}: {v:.2f}%")

    # --- DIAGNOSTIC 7: REPRESENTATION COLLAPSE ---
    mu = phi.mean(axis=0)
    mu_norm = np.linalg.norm(mu)
    print(f"\n[Representation Concentration] ||mu|| = {mu_norm:.4f}")
    report_data["mu_norm"] = float(mu_norm)
    
    # --- DIAGNOSTIC 8: POSITION-AWARE ANALYSIS ---
    pos_x = df_idx["pos_x"].values
    pos_y = df_idx["pos_y"].values
    
    np.random.seed(43)
    pt1 = np.random.randint(0, N, 10000)
    pt2 = np.random.randint(0, N, 10000)
    
    phys_dist = np.sqrt((pos_x[pt1] - pos_x[pt2])**2 + (pos_y[pt1] - pos_y[pt2])**2)
    vis_sim = (phi[pt1] * phi[pt2]).sum(axis=1)
    
    bins = [(0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 999.0)]
    pos_stats = {}
    for (low, high) in bins:
        mask = (phys_dist >= low) & (phys_dist < high)
        s = vis_sim[mask]
        if len(s) > 0:
            m = float(s.mean())
            pos_stats[f"{low}-{high}m"] = {"mean": m, "count": int(mask.sum())}
            print(f"Phys Dist {low}-{high}m -> Mean Vis Sim: {m:.4f} (N={mask.sum()})")
    
    report_data["position_aware"] = pos_stats

    # Dump JSON
    with open(OUT_DIR / "representation_statistics.json", "w") as f:
        json.dump(report_data, f, indent=2)
    
    print("\nDone. Wrote statistics to diagnostics/representation_statistics.json")

if __name__ == "__main__":
    main()
