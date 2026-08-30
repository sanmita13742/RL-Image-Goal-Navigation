import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "diagnostics" / "hindsight_visualization"
GOAL_DIR = ROOT / "diagnostics" / "goals"
SESSION_DIR = ROOT / "dataset" / "20260811_234812"

def get_img_path(df_row, frame_idx_df, step):
    """
    Kaggle absolute paths won't work locally. 
    Construct local path using segment_id and filename.
    """
    row = frame_idx_df.iloc[step]
    seg_id = row["segment_id"]
    filename = Path(row["rgb_abs_path"]).name
    path = SESSION_DIR / seg_id / "rgb" / filename
    return path

def load_img(path):
    if not path.exists():
        # Return a blank gray image if not found
        return np.ones((448, 784, 3), dtype=np.uint8) * 128
    return Image.open(path).convert("RGB")

def create_visualization(r, frame_idx_df, out_path, is_uniform=False):
    t = int(r["current_global_step"])
    g = int(r["goal_global_step"])
    
    # Text info
    sim = r["goal_similarity"]
    reward = int(r["reward"])
    done = int(r["done"])
    k = g - t
    action = f"({r['action_linear']:.2f}, {r['action_lateral']:.2f}, {r['action_angular']:.2f})"
    
    phys_dist_str = "N/A"
    if "meta_pos_x" in r:
        try:
            rt = frame_idx_df.iloc[t]
            rg = frame_idx_df.iloc[g]
            dx = float(rt.get("pos_x", r["meta_pos_x"])) - float(rg.get("pos_x", frame_idx_df.iloc[g].get("pos_x", 0)))
            dy = float(rt.get("pos_y", r["meta_pos_y"])) - float(rg.get("pos_y", frame_idx_df.iloc[g].get("pos_y", 0)))
            # If frame_index doesn't have pos_x, we can't reliably get the goal's pos_x.
            # We'll skip physical distance if it's not robustly available in frame_idx_df.
            phys_dist_str = "Available in DB, not mapped here perfectly"
        except:
            pass

    # Figure out intermediates
    intermediates = []
    if g > t:
        # 6 to 10 intermediate frames
        n_inter = min(8, g - t - 1)
        if n_inter > 0:
            step_size = max(1, (g - t) // (n_inter + 1))
            intermediates = list(range(t + step_size, g, step_size))[:8]
    
    n_cols = max(4, len(intermediates) + 1)
    
    fig = plt.figure(figsize=(4 * n_cols, 12))
    
    # ROW 1: State
    ax_state = [plt.subplot2grid((3, n_cols), (0, i)) for i in range(4)]
    state_steps = [t-3, t-2, t-1, t]
    for i, stp in enumerate(state_steps):
        img = load_img(get_img_path(r, frame_idx_df, stp))
        ax_state[i].imshow(img)
        ax_state[i].set_title(f"State: t={stp}", fontsize=14)
        ax_state[i].axis('off')
        
    for i in range(4, n_cols):
        ax_empty = plt.subplot2grid((3, n_cols), (0, i))
        ax_empty.axis('off')

    # ROW 2: Progression + Goal
    ax_prog = [plt.subplot2grid((3, n_cols), (1, i)) for i in range(len(intermediates) + 1)]
    for i, stp in enumerate(intermediates):
        img = load_img(get_img_path(r, frame_idx_df, stp))
        ax_prog[i].imshow(img)
        ax_prog[i].set_title(f"Inter: t={stp}", fontsize=14)
        ax_prog[i].axis('off')
        
    goal_ax = ax_prog[-1]
    goal_img = load_img(get_img_path(r, frame_idx_df, g))
    goal_ax.imshow(goal_img)
    if is_uniform and g < t:
        goal_ax.set_title(f"GOAL (t={g})\nTEMPORALLY PAST!", color='red', fontsize=16, fontweight='bold')
    else:
        goal_ax.set_title(f"GOAL (t={g})", color='green', fontsize=16, fontweight='bold')
    goal_ax.axis('off')
    
    for i in range(len(intermediates) + 1, n_cols):
        ax_empty = plt.subplot2grid((3, n_cols), (1, i))
        ax_empty.axis('off')

    # ROW 3: Text
    ax_text = plt.subplot2grid((3, n_cols), (2, 0), colspan=n_cols)
    ax_text.axis('off')
    
    info = (
        f"Sampling Type : {r['sampling_type'].upper()}\n"
        f"Current Step  : {t}\n"
        f"Goal Step     : {g}\n"
        f"Offset k      : {k}\n"
        f"Similarity S  : {sim:.4f}\n"
        f"Reward        : {reward}\n"
        f"Done          : {done}\n"
        f"Action        : {action}\n"
        f"Phys Dist     : diagnostic only\n"
        f"State steps   : {[t-3, t-2, t-1, t]}\n"
        f"Next steps    : {[t-2, t-1, t, t+1]}"
    )
    ax_text.text(0.5, 0.5, info, ha='center', va='center', fontsize=18, family='monospace', 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()

def plot_goal_gallery(df, frame_idx_df, out_prefix, n=20):
    df_sorted = df.sort_values("goal_similarity", ascending=False).head(n)
    fig, axes = plt.subplots(4, 5, figsize=(25, 16))
    fig.suptitle(f"Goal Gallery: {out_prefix.upper()}", fontsize=20)
    
    for ax, (_, r) in zip(axes.flatten(), df_sorted.iterrows()):
        g = int(r["goal_global_step"])
        img = load_img(get_img_path(r, frame_idx_df, g))
        ax.imshow(img)
        ax.axis('off')
        
        sim = r["goal_similarity"]
        rew = int(r["reward"])
        seg = frame_idx_df.iloc[g]["segment_id"]
        ssd = r.get("goal_ssd", 0.0) # Might not be in geometric
        
        title = f"Step: {g}\nSeg: {seg}\nS={sim:.4f} R={rew}"
        if ssd > 0: title += f"\nSSD={ssd:.4f}"
        ax.set_title(title, fontsize=10)
        
    plt.tight_layout()
    plt.savefig(GOAL_DIR / f"gallery_{out_prefix}.png", dpi=100)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["geometric", "uniform", "both"], default="both")
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    GOAL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    proc_dir = ROOT / "data" / "processed"
    if not proc_dir.exists():
        print(f"ERROR: {proc_dir} not found.")
        sys.exit(1)

    frame_idx_df = pd.read_parquet(proc_dir / "dinov3" / "embedding_index.parquet")
    geom_df = pd.read_parquet(proc_dir / "hindsight" / "geometric_transitions.parquet")
    unif_df = pd.read_parquet(proc_dir / "hindsight" / "uniform_transitions.parquet")

    # Filter rules
    def get_samples(df, condition, n):
        sub = df[condition]
        if len(sub) == 0: return pd.DataFrame()
        return sub.sample(min(n, len(sub)))

    report_data = []

    def process_category(df, cat_name, condition, prefix, is_uniform=False):
        samples = get_samples(df, condition, args.num)
        for i, (_, r) in enumerate(samples.iterrows()):
            name = f"{prefix}_{cat_name}_{i:03d}.png"
            create_visualization(r, frame_idx_df, OUT_DIR / name, is_uniform)
            report_data.append({
                "type": prefix,
                "category": cat_name,
                "current_step": r["current_global_step"],
                "goal_step": r["goal_global_step"],
                "k": r["goal_global_step"] - r["current_global_step"],
                "similarity": r["goal_similarity"],
                "reward": r["reward"],
                "done": r["done"],
            })

    # GEOMETRIC
    if args.type in ["geometric", "both"]:
        print("Generating Geometric Visualizations...")
        process_category(geom_df, "high_similarity", geom_df["goal_similarity"] >= 0.95, "geometric")
        process_category(geom_df, "medium_similarity", (geom_df["goal_similarity"] >= 0.85) & (geom_df["goal_similarity"] < 0.95), "geometric")
        process_category(geom_df, "boundary", (geom_df["goal_similarity"] >= 0.75) & (geom_df["goal_similarity"] < 0.85), "geometric")
        process_category(geom_df, "negative", geom_df["goal_similarity"] < 0.75, "geometric")
        
        process_category(geom_df, "dist_k_lt_25", geom_df["offset_k"] < 25, "geometric")
        process_category(geom_df, "dist_k_25_99", (geom_df["offset_k"] >= 25) & (geom_df["offset_k"] < 100), "geometric")
        process_category(geom_df, "dist_k_100_249", (geom_df["offset_k"] >= 100) & (geom_df["offset_k"] < 250), "geometric")
        process_category(geom_df, "dist_k_ge_250", geom_df["offset_k"] >= 250, "geometric")
        
        # Cross segment
        seg_changes = frame_idx_df["segment_id"].ne(frame_idx_df["segment_id"].shift()).values
        bdry_steps = frame_idx_df["global_step"].values[np.where(seg_changes)[0][1:]]
        
        cross_cond = geom_df["current_global_step"] < 0 # dummy
        for b in bdry_steps[:5]:
            cross_cond = cross_cond | ((geom_df["current_global_step"] < b) & (geom_df["goal_global_step"] >= b))
        process_category(geom_df, "cross_segment", cross_cond, "geometric")

        print("Generating Geometric Goal Gallery...")
        plot_goal_gallery(geom_df, frame_idx_df, "geometric")

    # UNIFORM
    if args.type in ["uniform", "both"]:
        print("Generating Uniform Visualizations...")
        process_category(unif_df, "high_similarity", unif_df["goal_similarity"] >= 0.95, "uniform", True)
        process_category(unif_df, "medium_similarity", (unif_df["goal_similarity"] >= 0.85) & (unif_df["goal_similarity"] < 0.95), "uniform", True)
        process_category(unif_df, "boundary", (unif_df["goal_similarity"] >= 0.75) & (unif_df["goal_similarity"] < 0.85), "uniform", True)
        process_category(unif_df, "negative", unif_df["goal_similarity"] < 0.75, "uniform", True)
        
        print("Generating Uniform Goal Gallery...")
        plot_goal_gallery(unif_df, frame_idx_df, "uniform")

    # REPORT
    df_rep = pd.DataFrame(report_data)
    
    geo_inspected = len(df_rep[df_rep['type'] == 'geometric'])
    uni_inspected = len(df_rep[df_rep['type'] == 'uniform'])
    geo_r1 = df_rep[(df_rep['type'] == 'geometric') & (df_rep['reward'] == 1)]
    uni_r1 = df_rep[(df_rep['type'] == 'uniform') & (df_rep['reward'] == 1)]
    geo_k = df_rep[df_rep['type'] == 'geometric']['k'].mean() if geo_inspected > 0 else 0
    
    with open(OUT_DIR / "INSPECTION_REPORT.md", "w") as f:
        f.write("# Hindsight Relabeling Visual Inspection Report\n\n")
        f.write(f"- Total inspected: {len(df_rep)}\n")
        f.write(f"- Geometric examples: {geo_inspected}\n")
        f.write(f"- Uniform examples: {uni_inspected}\n")
        f.write(f"- Reward=1 examples: {len(df_rep[df_rep['reward']==1])}\n")
        f.write(f"- Reward=0 examples: {len(df_rep[df_rep['reward']==0])}\n")
        f.write(f"- Boundary examples: {len(df_rep[df_rep['category']=='boundary'])}\n\n")
        
        f.write("## Sample Table\n\n")
        f.write(df_rep.to_markdown(index=False))

    print("\n" + "="*64)
    print("FINAL SUMMARY")
    print("="*64)
    print(f"Geometric:")
    print(f"    N inspected : {geo_inspected}")
    print(f"    reward=1 %  : {100*len(geo_r1)/geo_inspected if geo_inspected else 0:.1f}%")
    print(f"    reward=0 %  : {100*(geo_inspected-len(geo_r1))/geo_inspected if geo_inspected else 0:.1f}%")
    print(f"    average k   : {geo_k:.1f}")
    print(f"\nUniform:")
    print(f"    N inspected : {uni_inspected}")
    print(f"    reward=1 %  : {100*len(uni_r1)/uni_inspected if uni_inspected else 0:.1f}%")
    print(f"    reward=0 %  : {100*(uni_inspected-len(uni_r1))/uni_inspected if uni_inspected else 0:.1f}%")
    
    print(f"\nVisualizations saved to: {OUT_DIR}")
    print(f"Goal galleries saved to: {GOAL_DIR}")
    print(f"Inspection report saved to: {OUT_DIR / 'INSPECTION_REPORT.md'}")
    print("\nSTOP. (DO NOT implement TD3+BC yet)")

if __name__ == "__main__":
    main()
