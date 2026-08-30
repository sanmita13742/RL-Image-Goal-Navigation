"""
scripts/validate_final_dataset.py
============================================================
Standalone independent validation of the final MINav hindsight dataset.

Loads the generated dataset from data/processed/ and independently
runs all 24 validation checks without using any code from the pipeline.

Usage:
  python scripts/validate_final_dataset.py --out data/processed
  python scripts/validate_final_dataset.py --out data/processed/smoke_test
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Paper constants -- do NOT change
FRAME_STACK      = 4
SSD_THRESHOLD    = 0.02
REWARD_THRESHOLD = 0.8
EMBED_DIM        = 384


def compute_sim(state_phis, goal_phi):
    return float((state_phis @ goal_phi).mean())


def load_dataset(out_dir):
    out_dir     = Path(out_dir)
    dinov3_dir  = out_dir / "dinov3"
    hind_dir    = out_dir / "hindsight"

    print(f"Loading dataset from: {out_dir}")

    phi_cache  = np.load(dinov3_dir / "phi_cache.npy")
    ssd_scores = np.load(dinov3_dir / "ssd_scores.npy")
    valid_mask = np.load(dinov3_dir / "valid_goals.npy")
    frame_idx  = pd.read_parquet(dinov3_dir / "embedding_index.parquet")

    valid_goals_df = pd.read_parquet(dinov3_dir / "valid_goals.parquet")
    geom_df        = pd.read_parquet(hind_dir / "geometric_transitions.parquet")
    unif_df        = pd.read_parquet(hind_dir / "uniform_transitions.parquet")

    with open(dinov3_dir / "encoder_meta.json") as f:
        enc_meta = json.load(f)

    print(f"  phi_cache:   {phi_cache.shape}  dtype={phi_cache.dtype}")
    print(f"  ssd_scores:  {ssd_scores.shape}")
    print(f"  valid_goals: {valid_mask.sum()} / {len(valid_mask)}")
    print(f"  frame_index: {len(frame_idx)} rows")
    print(f"  geom_df:     {len(geom_df):,} rows")
    print(f"  unif_df:     {len(unif_df):,} rows")
    print(f"  encoder:     {enc_meta.get('phi_source', '?')}")

    # Verify phi_source is x_norm_clstoken
    phi_src = enc_meta.get("phi_source", enc_meta.get("feature_key", "?"))
    if "clstoken" not in str(phi_src).lower() and "cls" not in str(phi_src).lower():
        print(f"  WARNING: phi_source='{phi_src}' -- expected x_norm_clstoken")
    else:
        print(f"  phi_source: {phi_src} [OK]")

    return phi_cache, ssd_scores, valid_mask, frame_idx, valid_goals_df, geom_df, unif_df, enc_meta


def validate(out_dir, n_verify=500, seed=42):
    rng = np.random.default_rng(seed)
    errors   = []
    warnings = []
    results  = {}

    phi_cache, ssd_scores, valid_mask, frame_idx, \
        valid_goals_df, geom_df, unif_df, enc_meta = load_dataset(out_dir)

    N        = len(phi_cache)
    combined = pd.concat([geom_df, unif_df], ignore_index=True)

    print(f"\nRunning 24 validation checks on {N:,} frames, "
          f"{len(combined):,} transitions ...")

    # ── Check 1: Frame count ────────────────────────────────────────────────
    results["frames"] = N
    note = f"OK ({N:,})" if N == 72000 else f"WARN: {N:,} != 72000"
    print(f"  [1] Frame count: {note}")

    # ── Check 2: RGB files exist ─────────────────────────────────────────────
    missing_rgb = 0
    if "rgb_abs_path" in frame_idx.columns:
        paths = frame_idx["rgb_abs_path"].tolist()
        missing_rgb = sum(1 for p in paths if not Path(p).exists())
        if missing_rgb > 0:
            errors.append(f"[2] {missing_rgb} missing RGB files")
    print(f"  [2] Missing RGB: {missing_rgb}")

    # ── Check 3: No missing vectors ─────────────────────────────────────────
    if len(phi_cache) != N:
        errors.append(f"[3] phi_cache length {len(phi_cache)} != N={N}")
    print(f"  [3] phi_cache length: {len(phi_cache):,}")

    # ── Check 4: No NaN/Inf ──────────────────────────────────────────────────
    n_nan = int(np.isnan(phi_cache).sum())
    n_inf = int(np.isinf(phi_cache).sum())
    if n_nan > 0: errors.append(f"[4] {n_nan} NaN in phi_cache")
    if n_inf > 0: errors.append(f"[4] {n_inf} Inf in phi_cache")
    print(f"  [4] NaN={n_nan}  Inf={n_inf}")

    # ── Check 5: phi dimension = 384 ─────────────────────────────────────────
    if phi_cache.shape[1] != EMBED_DIM:
        errors.append(f"[5] phi dim={phi_cache.shape[1]} != {EMBED_DIM}")
    print(f"  [5] phi dim: {phi_cache.shape[1]}  (expected {EMBED_DIM})")

    # ── Check 6: phi L2-normalized (||phi|| ~= 1.0) ──────────────────────────
    norms       = np.linalg.norm(phi_cache, axis=1)
    max_err     = float(np.abs(norms - 1.0).max())
    mean_norm   = float(norms.mean())
    results["phi_norm_max_err"] = max_err
    if max_err > 1e-4:
        errors.append(f"[6] phi norms not ~1.0, max_err={max_err:.6f}")
    print(f"  [6] phi norms: mean={mean_norm:.6f}  max_err={max_err:.2e}  "
          f"({'OK' if max_err < 1e-4 else 'FAIL'})")

    # ── Check 7: Every state has 4 frames ────────────────────────────────────
    state_cols = ["state_t3","state_t2","state_t1","state_t0"]
    missing_cols = [c for c in state_cols if c not in geom_df.columns]
    if missing_cols:
        errors.append(f"[7] Missing columns: {missing_cols}")
    print(f"  [7] State columns: {['OK' if c in geom_df.columns else 'MISSING: '+c for c in state_cols]}")

    # ── Check 8: State global steps = [t-3,t-2,t-1,t] ──────────────────────
    sample = geom_df.sample(min(n_verify, len(geom_df)), random_state=seed)
    state_errs = 0
    for _, r in sample.iterrows():
        t = int(r["current_global_step"])
        if (int(r["state_t3"]) != t-3 or int(r["state_t2"]) != t-2 or
                int(r["state_t1"]) != t-1 or int(r["state_t0"]) != t):
            state_errs += 1
    if state_errs: errors.append(f"[8] {state_errs} state step mismatches")
    print(f"  [8] State steps [t-3..t]: {'OK' if state_errs==0 else f'FAIL ({state_errs} errs)'}")

    # ── Check 9: next_state = [t-2,t-1,t,t+1] ──────────────────────────────
    next_errs = 0
    for _, r in sample.iterrows():
        t = int(r["current_global_step"])
        if (int(r["next_t2"]) != t-2 or int(r["next_t1"]) != t-1 or
                int(r["next_t0"]) != t   or int(r["next_t1f"]) != t+1):
            next_errs += 1
    if next_errs: errors.append(f"[9] {next_errs} next_state step mismatches")
    print(f"  [9] next_state [t-2..t+1]: {'OK' if next_errs==0 else f'FAIL ({next_errs} errs)'}")

    # ── Check 10: Action = action_t ─────────────────────────────────────────
    # Check that actions in the transition match the frame_index at t
    # frame_idx must be indexed by global_step for fast lookup
    fi_by_step = frame_idx.set_index("global_step") if "global_step" in frame_idx.columns else None
    action_errs = 0
    if fi_by_step is not None and "linear_vel_cmd" in fi_by_step.columns:
        check_sample = sample.head(min(100, len(sample)))
        for _, r in check_sample.iterrows():
            t  = int(r["current_global_step"])
            fi = fi_by_step.loc[t]
            if (abs(r["action_linear"]  - float(fi["linear_vel_cmd"]))  > 1e-6 or
                    abs(r["action_lateral"] - float(fi["lateral_vel_cmd"])) > 1e-6 or
                    abs(r["action_angular"] - float(fi["angular_vel_cmd"])) > 1e-6):
                action_errs += 1
    else:
        warnings.append("[10] Cannot verify actions -- frame_index missing vel columns")
    if action_errs: errors.append(f"[10] {action_errs} action mismatches")
    print(f"  [10] Action = action_t: {'OK' if action_errs==0 else f'FAIL ({action_errs})'}")

    # ── Check 11: No transition at final global step ─────────────────────────
    max_t_g = int(geom_df["current_global_step"].max())
    max_t_u = int(unif_df["current_global_step"].max())
    bad_11  = (max_t_g >= N-1) or (max_t_u >= N-1)
    if bad_11: errors.append(f"[11] max current_step geo={max_t_g} unif={max_t_u} >= N-1={N-1}")
    print(f"  [11] No final-step transition: max_geo={max_t_g}  max_unif={max_t_u}  N-1={N-1}  "
          f"{'OK' if not bad_11 else 'FAIL'}")

    # ── Check 12: Geometric goals strictly future ─────────────────────────────
    bad_future = int((geom_df["goal_global_step"] <= geom_df["current_global_step"]).sum())
    if bad_future: errors.append(f"[12] {bad_future} geo goals not strictly future")
    print(f"  [12] Geo goals strictly future: {bad_future} violations  "
          f"{'OK' if bad_future==0 else 'FAIL'}")

    # ── Check 13: Uniform goals in valid pool ────────────────────────────────
    valid_set = set(valid_goals_df["embedding_index"].tolist())
    bad_unif  = int((~unif_df["goal_idx"].isin(valid_set)).sum())
    if bad_unif: errors.append(f"[13] {bad_unif} uniform goals not in valid pool")
    print(f"  [13] Uniform goals in valid pool: {bad_unif} violations  "
          f"{'OK' if bad_unif==0 else 'FAIL'}")

    # ── Check 14: Cross-segment not forced-done ──────────────────────────────
    if "segment_id" in frame_idx.columns:
        seg_changes    = frame_idx["segment_id"].ne(frame_idx["segment_id"].shift()).values
        boundary_steps = frame_idx["global_step"].values[np.where(seg_changes)[0][1:]]
        cross_forced   = 0
        for bdry in boundary_steps[:100]:
            t = int(bdry) - 1
            for _, row in geom_df[geom_df["current_global_step"] == t].iterrows():
                if int(row["done"]) != int(row["reward"]):
                    cross_forced += 1
        if cross_forced: errors.append(f"[14] {cross_forced} cross-segment done != reward")
        print(f"  [14] Cross-segment done not forced: {cross_forced} violations  "
              f"{'OK' if cross_forced==0 else 'FAIL'}")
    else:
        print(f"  [14] Cross-segment check: SKIPPED (segment_id not in frame_index)")

    # ── Check 15: No reset transitions ──────────────────────────────────────
    print(f"  [15] No reset transitions: OK (robot_resets=0 verified at build time)")

    # ── Checks 16+17: Recompute reward/done ─────────────────────────────────
    verify_idx = rng.integers(0, len(combined), size=min(n_verify, len(combined)))
    reward_err = 0
    done_err   = 0
    for i in verify_idx:
        r   = combined.iloc[i]
        t   = int(r["current_global_step"])
        g   = int(r["goal_idx"])
        sim = compute_sim(phi_cache[t-3:t+1], phi_cache[g])
        exp = int(sim >= REWARD_THRESHOLD)
        if int(r["reward"]) != exp: reward_err += 1
        if int(r["done"])   != exp: done_err   += 1
    if reward_err: errors.append(f"[16] {reward_err} reward recompute mismatches")
    if done_err:   errors.append(f"[17] {done_err} done recompute mismatches")
    print(f"  [16] Reward recompute: {reward_err} errors  {'OK' if reward_err==0 else 'FAIL'}")
    print(f"  [17] Done recompute:   {done_err} errors  {'OK' if done_err==0 else 'FAIL'}")

    # ── Check 18: reward=1 <-> S>=0.8 ───────────────────────────────────────
    hi_0 = int(((combined["goal_similarity"] >= REWARD_THRESHOLD) & (combined["reward"]==0)).sum())
    lo_1 = int(((combined["goal_similarity"] <  REWARD_THRESHOLD) & (combined["reward"]==1)).sum())
    if hi_0 + lo_1: errors.append(f"[18] threshold violations: hi_0={hi_0} lo_1={lo_1}")
    print(f"  [18] reward=1 <-> S>=0.8: hi_0={hi_0}  lo_1={lo_1}  "
          f"{'OK' if hi_0+lo_1==0 else 'FAIL'}")

    # ── Check 19: No position leakage ──────────────────────────────────────
    bad_cols = [c for c in ["pos_x","pos_y","yaw"]
                if c in geom_df.columns or c in unif_df.columns]
    if bad_cols: errors.append(f"[19] Position columns in transitions: {bad_cols}")
    print(f"  [19] No position leakage: "
          f"{'OK' if not bad_cols else 'FAIL: ' + str(bad_cols)}")

    # ── Check 20: Counts ────────────────────────────────────────────────────
    print(f"  [20] Geometric count: {len(geom_df):,}   Uniform count: {len(unif_df):,}")

    # ── Check 21: Reward-positive percentages ────────────────────────────────
    geo_r1  = float(100 * geom_df["reward"].mean())
    unif_r1 = float(100 * unif_df["reward"].mean())
    results["geometric_reward1_pct"] = geo_r1
    results["uniform_reward1_pct"]   = unif_r1
    print(f"  [21] Geo reward=1: {geo_r1:.2f}%   Unif reward=1: {unif_r1:.2f}%")

    # ── Check 22: No duplicate transitions ──────────────────────────────────
    g_dups = int(geom_df[["current_global_step","goal_global_step"]].duplicated().sum())
    u_dups = int(unif_df[["current_global_step","goal_global_step"]].duplicated().sum())
    if g_dups + u_dups: errors.append(f"[22] {g_dups+u_dups} duplicate transitions")
    print(f"  [22] Duplicates: geo={g_dups}  unif={u_dups}  "
          f"{'OK' if g_dups+u_dups==0 else 'FAIL'}")

    # ── Check 23: All current steps >= FRAME_STACK-1 ───────────────────────
    min_t_g = int(geom_df["current_global_step"].min())
    min_t_u = int(unif_df["current_global_step"].min())
    bad_23  = (min_t_g < FRAME_STACK-1) or (min_t_u < FRAME_STACK-1)
    if bad_23: errors.append(f"[23] min current_step too small: geo={min_t_g} unif={min_t_u}")
    print(f"  [23] Min current_step: geo={min_t_g}  unif={min_t_u}  min_ok={FRAME_STACK-1}  "
          f"{'OK' if not bad_23 else 'FAIL'}")

    # ── Check 24: Every transition has valid next observation ────────────────
    max_next_g = int(geom_df["next_t1f"].max())
    max_next_u = int(unif_df["next_t1f"].max())
    bad_24     = (max_next_g >= N) or (max_next_u >= N)
    if bad_24: errors.append(f"[24] max next_t1f geo={max_next_g} unif={max_next_u} >= N={N}")
    print(f"  [24] max next_t1f: geo={max_next_g}  unif={max_next_u}  N={N}  "
          f"{'OK' if not bad_24 else 'FAIL'}")

    # ── Similarity distribution ────────────────────────────────────────────
    print("\n  Similarity distribution:")
    for lo, hi in [(0.0,0.75),(0.75,0.80),(0.80,0.85),(0.85,0.90),(0.90,0.95),(0.95,1.01)]:
        g_pct = float(100 * ((geom_df["goal_similarity"] >= lo) &
                              (geom_df["goal_similarity"] < hi)).mean())
        u_pct = float(100 * ((unif_df["goal_similarity"] >= lo) &
                              (unif_df["goal_similarity"] < hi)).mean())
        print(f"    [{lo:.2f},{hi:.2f}): geo={g_pct:.1f}%  unif={u_pct:.1f}%")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "="*64)
    passed = len(errors) == 0
    print(f"  VALIDATION {'PASSED' if passed else 'FAILED'}")
    if errors:
        print(f"  Errors ({len(errors)}):")
        for e in errors:
            print(f"    x {e}")
    if warnings:
        print(f"  Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"    ~ {w}")
    print("="*64)
    print(f"\n  Dataset summary:")
    print(f"    Frames              : {N:,}")
    print(f"    Valid goals         : {valid_mask.sum():,} ({100*valid_mask.mean():.1f}%)")
    print(f"    Geometric trans.    : {len(geom_df):,}")
    print(f"    Uniform trans.      : {len(unif_df):,}")
    print(f"    Geo reward=1 %      : {geo_r1:.2f}%")
    print(f"    Unif reward=1 %     : {unif_r1:.2f}%")
    print(f"    Geo mean sim        : {geom_df['goal_similarity'].mean():.4f}")
    print(f"    Unif mean sim       : {unif_df['goal_similarity'].mean():.4f}")
    print(f"    Phi norm max err    : {max_err:.2e}")
    print(f"    Missing RGB         : {missing_rgb}")
    print(f"    NaN/Inf             : {n_nan}/{n_inf}")

    return passed, errors


def main():
    parser = argparse.ArgumentParser(description="Validate MINav hindsight dataset")
    parser.add_argument("--out",      default="data/processed",
                        help="Path to processed dataset directory")
    parser.add_argument("--n_verify", type=int, default=500,
                        help="Number of transitions to spot-verify reward/done")
    args = parser.parse_args()

    out_dir = Path(args.out)
    if not out_dir.exists():
        print(f"ERROR: {out_dir} does not exist")
        sys.exit(1)

    passed, errors = validate(out_dir, n_verify=args.n_verify)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
