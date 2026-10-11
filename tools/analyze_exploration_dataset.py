#!/usr/bin/env python3
"""
tools/analyze_exploration_dataset.py — Exploration dataset audit & analysis.
=============================================================================
Reads segment CSVs from an exploration session and generates a comprehensive
report distinguishing target/requested/executed action distributions.

Usage:
    python tools/analyze_exploration_dataset.py \
        --session output-realworld/exploration/ \
        --outdir tools/analysis_output/

If the CSV contains the extended diagnostic columns (target_vx, target_vy,
target_wz, obstacle_distance, relevant_point_count, cloud_age), this script
uses them. Otherwise, it falls back to the legacy columns.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────
# CSV Loading
# ─────────────────────────────────────────────────────────────

def load_session_csvs(session_dir: Path) -> list[dict]:
    """Load all segment CSVs from an exploration session directory.

    Handles both legacy (observations.csv) and new (segment.csv) names.
    Returns a list of row dicts with unified column names.
    """
    rows = []
    segment_dirs = sorted(session_dir.glob("segment_*"))

    if not segment_dirs:
        # Maybe the CSV is directly in the session dir
        for csv_name in ["observations.csv", "segment.csv"]:
            csv_path = session_dir / csv_name
            if csv_path.exists():
                rows.extend(_read_csv(csv_path))
        if not rows:
            print(f"ERROR: No segment CSVs found in {session_dir}")
            sys.exit(1)
        return rows

    for seg_dir in segment_dirs:
        for csv_name in ["observations.csv", "segment.csv"]:
            csv_path = seg_dir / csv_name
            if csv_path.exists():
                rows.extend(_read_csv(csv_path))
                break

    return rows


def _read_csv(path: Path) -> list[dict]:
    """Read a single CSV file and return list of row dicts."""
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────
# Column extraction with fallback
# ─────────────────────────────────────────────────────────────

def _float_or_nan(val: str) -> float:
    """Convert string to float, returning NaN for empty/invalid values."""
    if val is None or val.strip() == "":
        return float("nan")
    try:
        return float(val)
    except ValueError:
        return float("nan")


def extract_arrays(rows: list[dict]) -> dict:
    """Extract typed arrays from row dicts. Returns dict of numpy arrays.

    Handles both legacy and extended CSV schemas.
    """
    cols = rows[0].keys() if rows else set()

    # Detect schema (legacy vs new minimal real-world schema)
    has_target = "target_vx" in cols
    has_extended_safety = "obstacle_distance" in cols

    n = len(rows)
    data = {}

    # Timestamps
    data["sim_time"] = np.array([_float_or_nan(r.get("timestamp", r.get("sim_time", ""))) for r in rows])
    data["global_step"] = np.array([_float_or_nan(r.get("global_step", "0")) for r in rows])

    # ── Requested action (always available in legacy and new schemas) ──
    data["requested_vx"] = np.array([_float_or_nan(r.get("cmd_vx", r.get("linear_vel_cmd", ""))) for r in rows])
    data["requested_vy"] = np.array([_float_or_nan(r.get("cmd_vy", r.get("lateral_vel_cmd", ""))) for r in rows])
    data["requested_wz"] = np.array([_float_or_nan(r.get("cmd_wz", r.get("angular_vel_cmd", ""))) for r in rows])

    # ── Executed action ──
    data["executed_vx"] = np.array([_float_or_nan(r.get("executed_vx", r.get("executed_linear_vel", ""))) for r in rows])
    data["executed_vy"] = np.array([_float_or_nan(r.get("executed_vy", r.get("executed_lateral_vel", ""))) for r in rows])
    data["executed_wz"] = np.array([_float_or_nan(r.get("executed_wz", r.get("executed_angular_vel", ""))) for r in rows])

    # ── Target action (extended schema only) ──
    if has_target:
        data["target_vx"] = np.array([_float_or_nan(r.get("target_vx", "")) for r in rows])
        data["target_vy"] = np.array([_float_or_nan(r.get("target_vy", "")) for r in rows])
        data["target_wz"] = np.array([_float_or_nan(r.get("target_wz", "")) for r in rows])
    else:
        data["target_vx"] = None
        data["target_vy"] = None
        data["target_wz"] = None

    # ── Safety ──
    data["safety_blocked"] = np.array([
        int(_float_or_nan(r.get("safety_blocked", r.get("safety_intervention", "0")))) for r in rows
    ])

    if has_extended_safety:
        data["obstacle_distance"] = np.array([
            _float_or_nan(r.get("obstacle_distance", "")) for r in rows
        ])
        data["relevant_point_count"] = np.array([
            _float_or_nan(r.get("relevant_point_count", "")) for r in rows
        ])
        data["cloud_age"] = np.array([
            _float_or_nan(r.get("cloud_age", "")) for r in rows
        ])
    else:
        data["obstacle_distance"] = None
        data["relevant_point_count"] = None
        data["cloud_age"] = None

    # ── Position ──
    data["pos_x"] = np.array([_float_or_nan(r.get("pos_x", "")) for r in rows])
    data["pos_y"] = np.array([_float_or_nan(r.get("pos_y", "")) for r in rows])
    data["yaw"] = np.array([_float_or_nan(r.get("yaw", "")) for r in rows])

    data["has_target"] = has_target
    data["has_extended_safety"] = has_extended_safety

    return data


# ─────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────

def axis_stats(arr: np.ndarray, name: str, lo: float, hi: float) -> dict:
    """Compute comprehensive statistics for a single action dimension."""
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return {"name": name, "count": 0, "error": "all NaN"}

    near_zero_thresh = 0.01
    near_bound_thresh = 0.05 * (hi - lo)  # 5% of range

    stats = {
        "name": name,
        "count": len(valid),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "median": float(np.median(valid)),
        "pct_near_zero": float(np.mean(np.abs(valid) < near_zero_thresh) * 100),
        "pct_near_lower": float(np.mean(np.abs(valid - lo) < near_bound_thresh) * 100),
        "pct_near_upper": float(np.mean(np.abs(valid - hi) < near_bound_thresh) * 100),
    }

    # Histogram (10 bins across the configured range)
    bins = np.linspace(lo, hi, 11)
    counts, edges = np.histogram(valid, bins=bins)
    stats["histogram_counts"] = counts.tolist()
    stats["histogram_edges"] = edges.tolist()

    return stats


def lag_autocorrelation(arr: np.ndarray, lag: int) -> float:
    """Compute normalized autocorrelation at a given lag."""
    if lag == 0:
        return 1.0
    valid = arr[np.isfinite(arr)]
    if len(valid) <= lag:
        return float("nan")
    mean = np.mean(valid)
    n = len(valid)
    c0 = np.sum((valid - mean) ** 2) / n
    if c0 < 1e-15:
        return float("nan")
    ck = np.sum((valid[:-lag] - mean) * (valid[lag:] - mean)) / n
    return float(ck / c0)


def action_entropy_1d(arr: np.ndarray, n_bins: int = 50) -> float:
    """Compute differential entropy estimate using histogram binning."""
    valid = arr[np.isfinite(arr)]
    if len(valid) < 10:
        return float("nan")
    counts, _ = np.histogram(valid, bins=n_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs + 1e-15)))


def blocking_episodes(blocked: np.ndarray, dt_array: np.ndarray) -> dict:
    """Compute statistics about continuous blocking episodes."""
    episodes = []
    in_block = False
    start_idx = 0

    for i in range(len(blocked)):
        if blocked[i] == 1 and not in_block:
            in_block = True
            start_idx = i
        elif blocked[i] == 0 and in_block:
            in_block = False
            episodes.append((start_idx, i))

    if in_block:
        episodes.append((start_idx, len(blocked)))

    if not episodes:
        return {
            "num_episodes": 0,
            "mean_duration_s": 0.0,
            "median_duration_s": 0.0,
            "max_duration_s": 0.0,
            "total_blocked_s": 0.0,
        }

    durations = []
    for s, e in episodes:
        if np.all(np.isfinite(dt_array[s:e])):
            dur = dt_array[e - 1] - dt_array[s] if e > s else 0.0
            durations.append(max(dur, 0.0))
        else:
            durations.append((e - s) * 0.1)  # fallback: assume 10Hz

    durations = np.array(durations)
    return {
        "num_episodes": len(episodes),
        "mean_duration_s": float(np.mean(durations)),
        "median_duration_s": float(np.median(durations)),
        "max_duration_s": float(np.max(durations)),
        "total_blocked_s": float(np.sum(durations)),
    }


# ─────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────

def run_analysis(data: dict) -> dict:
    """Run the full analysis and return a structured report dict."""
    report = {}
    n = len(data["requested_vx"])
    report["total_samples"] = n

    # Duration & sampling rate
    times = data["sim_time"]
    valid_times = times[np.isfinite(times)]
    if len(valid_times) >= 2:
        duration_s = float(valid_times[-1] - valid_times[0])
        report["duration_s"] = duration_s
        report["duration_min"] = duration_s / 60.0
        report["sampling_rate_hz"] = n / duration_s if duration_s > 0 else 0.0
    else:
        report["duration_s"] = 0.0
        report["duration_min"] = 0.0
        report["sampling_rate_hz"] = 0.0

    # Ranges for stats
    vx_range = (0.0, 0.6)
    vy_range = (-0.3, 0.3)
    wz_range = (-1.0, 1.0)

    # ── 4. Target action statistics ──
    if data["has_target"]:
        report["target_action_stats"] = {
            "vx": axis_stats(data["target_vx"], "target_vx", *vx_range),
            "vy": axis_stats(data["target_vy"], "target_vy", *vy_range),
            "wz": axis_stats(data["target_wz"], "target_wz", *wz_range),
        }
    else:
        report["target_action_stats"] = "NOT AVAILABLE — target columns missing from CSV"

    # ── 5. Requested action statistics ──
    report["requested_action_stats"] = {
        "vx": axis_stats(data["requested_vx"], "requested_vx", *vx_range),
        "vy": axis_stats(data["requested_vy"], "requested_vy", *vy_range),
        "wz": axis_stats(data["requested_wz"], "requested_wz", *wz_range),
    }

    # ── 6. Executed action statistics ──
    report["executed_action_stats"] = {
        "vx": axis_stats(data["executed_vx"], "executed_vx", *vx_range),
        "vy": axis_stats(data["executed_vy"], "executed_vy", *vy_range),
        "wz": axis_stats(data["executed_wz"], "executed_wz", *wz_range),
    }

    # ── 7. Safety intervention rate ──
    blocked = data["safety_blocked"]
    total_blocked = int(np.sum(blocked == 1))
    report["safety_intervention_rate"] = {
        "blocked_samples": total_blocked,
        "total_samples": n,
        "rate": total_blocked / n if n > 0 else 0.0,
    }

    # ── 8. Translational clamp rate ──
    req_vx = data["requested_vx"]
    req_vy = data["requested_vy"]
    exe_vx = data["executed_vx"]
    exe_vy = data["executed_vy"]

    req_trans_nonzero = (np.abs(req_vx) > 1e-6) | (np.abs(req_vy) > 1e-6)
    exe_trans_zero = (np.abs(exe_vx) < 1e-6) & (np.abs(exe_vy) < 1e-6)
    clamped = req_trans_nonzero & exe_trans_zero
    report["translational_clamp_rate"] = {
        "clamped_samples": int(np.sum(clamped)),
        "total_samples": n,
        "rate": float(np.mean(clamped)) if n > 0 else 0.0,
    }

    # ── 9. Per-axis modification rate ──
    req_wz = data["requested_wz"]
    exe_wz = data["executed_wz"]

    report["per_axis_modification_rate"] = {
        "P(req_vx != exe_vx)": float(np.mean(np.abs(req_vx - exe_vx) > 1e-6)),
        "P(req_vy != exe_vy)": float(np.mean(np.abs(req_vy - exe_vy) > 1e-6)),
        "P(req_wz != exe_wz)": float(np.mean(np.abs(req_wz - exe_wz) > 1e-6)),
    }

    # ── 10. Action modification magnitude ──
    diff = np.sqrt(
        (req_vx - exe_vx)**2 +
        (req_vy - exe_vy)**2 +
        (req_wz - exe_wz)**2
    )
    report["action_modification_magnitude"] = {
        "mean": float(np.nanmean(diff)),
        "std": float(np.nanstd(diff)),
        "max": float(np.nanmax(diff)),
        "median": float(np.nanmedian(diff)),
    }

    # ── 11. Continuous blocking episodes ──
    report["blocking_episodes"] = blocking_episodes(blocked, times)

    # ── 12. Correlation: requested vs executed ──
    def _corr(a, b):
        valid = np.isfinite(a) & np.isfinite(b)
        if valid.sum() < 2:
            return float("nan")
        a_v, b_v = a[valid], b[valid]
        if np.std(a_v) < 1e-12 or np.std(b_v) < 1e-12:
            return float("nan")
        return float(np.corrcoef(a_v, b_v)[0, 1])

    report["requested_vs_executed_correlation"] = {
        "vx": _corr(req_vx, exe_vx),
        "vy": _corr(req_vy, exe_vy),
        "wz": _corr(req_wz, exe_wz),
    }

    # ── 13. Temporal autocorrelation ──
    lags = [1, 2, 5, 10, 20]

    def _lag_table(arr, name):
        return {f"lag_{l}": lag_autocorrelation(arr, l) for l in lags}

    temporal = {}
    for prefix, vx_arr, vy_arr, wz_arr in [
        ("requested", req_vx, req_vy, req_wz),
        ("executed", exe_vx, exe_vy, exe_wz),
    ]:
        temporal[f"{prefix}_vx"] = _lag_table(vx_arr, f"{prefix}_vx")
        temporal[f"{prefix}_vy"] = _lag_table(vy_arr, f"{prefix}_vy")
        temporal[f"{prefix}_wz"] = _lag_table(wz_arr, f"{prefix}_wz")

    if data["has_target"]:
        for axis, arr in [("vx", data["target_vx"]), ("vy", data["target_vy"]), ("wz", data["target_wz"])]:
            temporal[f"target_{axis}"] = _lag_table(arr, f"target_{axis}")

    report["temporal_autocorrelation"] = temporal

    # ── Action entropy ──
    entropy = {}
    for prefix, vx_arr, vy_arr, wz_arr in [
        ("requested", req_vx, req_vy, req_wz),
        ("executed", exe_vx, exe_vy, exe_wz),
    ]:
        entropy[f"{prefix}_vx"] = action_entropy_1d(vx_arr)
        entropy[f"{prefix}_vy"] = action_entropy_1d(vy_arr)
        entropy[f"{prefix}_wz"] = action_entropy_1d(wz_arr)

    if data["has_target"]:
        for axis, arr in [("vx", data["target_vx"]), ("vy", data["target_vy"]), ("wz", data["target_wz"])]:
            entropy[f"target_{axis}"] = action_entropy_1d(arr)

    report["action_entropy"] = entropy

    # ── Spatial coverage ──
    pos_x = data["pos_x"]
    pos_y = data["pos_y"]
    valid_pos = np.isfinite(pos_x) & np.isfinite(pos_y)
    if valid_pos.sum() > 10:
        x_v = pos_x[valid_pos]
        y_v = pos_y[valid_pos]
        report["spatial_coverage"] = {
            "x_range": [float(np.min(x_v)), float(np.max(x_v))],
            "y_range": [float(np.min(y_v)), float(np.max(y_v))],
            "unique_1m_cells": int(len(set(
                (int(np.floor(x)), int(np.floor(y)))
                for x, y in zip(x_v, y_v)
            ))),
            "total_distance_m": float(np.sum(np.sqrt(
                np.diff(x_v)**2 + np.diff(y_v)**2
            ))),
        }
    else:
        report["spatial_coverage"] = "NOT AVAILABLE — pos_x/pos_y missing or all NaN"

    return report


# ─────────────────────────────────────────────────────────────
# Report formatting
# ─────────────────────────────────────────────────────────────

def format_stats(s: dict, indent: str = "    ") -> str:
    """Format axis_stats dict as readable text."""
    if "error" in s:
        return f"{indent}{s['name']}: {s['error']}"
    lines = [
        f"{indent}{s['name']}:",
        f"{indent}  count={s['count']}  min={s['min']:.4f}  max={s['max']:.4f}",
        f"{indent}  mean={s['mean']:.4f}  std={s['std']:.4f}  median={s['median']:.4f}",
        f"{indent}  %near_zero={s['pct_near_zero']:.1f}%  "
        f"%near_lower={s['pct_near_lower']:.1f}%  "
        f"%near_upper={s['pct_near_upper']:.1f}%",
        f"{indent}  histogram: {s['histogram_counts']}",
    ]
    return "\n".join(lines)


def format_report(report: dict) -> str:
    """Format the full report as human-readable text."""
    lines = []
    lines.append("=" * 70)
    lines.append("EXPLORATION DATASET AUDIT REPORT")
    lines.append("=" * 70)
    lines.append("")

    lines.append(f"Total samples:     {report['total_samples']}")
    lines.append(f"Duration:          {report['duration_s']:.1f}s ({report['duration_min']:.1f} min)")
    lines.append(f"Sampling rate:     {report['sampling_rate_hz']:.1f} Hz")
    lines.append("")

    # Target
    lines.append("-" * 50)
    lines.append("TARGET ACTION STATISTICS (pre-smoothing)")
    lines.append("-" * 50)
    t = report["target_action_stats"]
    if isinstance(t, str):
        lines.append(f"  {t}")
    else:
        for axis in ["vx", "vy", "wz"]:
            lines.append(format_stats(t[axis]))
    lines.append("")

    # Requested
    lines.append("-" * 50)
    lines.append("REQUESTED ACTION STATISTICS (post-smoothing, pre-safety)")
    lines.append("-" * 50)
    for axis in ["vx", "vy", "wz"]:
        lines.append(format_stats(report["requested_action_stats"][axis]))
    lines.append("")

    # Executed
    lines.append("-" * 50)
    lines.append("EXECUTED ACTION STATISTICS (post-safety)")
    lines.append("-" * 50)
    for axis in ["vx", "vy", "wz"]:
        lines.append(format_stats(report["executed_action_stats"][axis]))
    lines.append("")

    # Safety
    lines.append("-" * 50)
    lines.append("SAFETY GATE ANALYSIS")
    lines.append("-" * 50)
    sir = report["safety_intervention_rate"]
    lines.append(f"  Blocked samples: {sir['blocked_samples']} / {sir['total_samples']} "
                 f"({sir['rate']*100:.1f}%)")

    tcr = report["translational_clamp_rate"]
    lines.append(f"  Translational clamp: {tcr['clamped_samples']} / {tcr['total_samples']} "
                 f"({tcr['rate']*100:.1f}%)")

    pam = report["per_axis_modification_rate"]
    lines.append(f"  P(req_vx != exe_vx) = {pam['P(req_vx != exe_vx)']*100:.1f}%")
    lines.append(f"  P(req_vy != exe_vy) = {pam['P(req_vy != exe_vy)']*100:.1f}%")
    lines.append(f"  P(req_wz != exe_wz) = {pam['P(req_wz != exe_wz)']*100:.1f}%")

    amm = report["action_modification_magnitude"]
    lines.append(f"  ||req - exe|| mean={amm['mean']:.4f}  std={amm['std']:.4f}  "
                 f"max={amm['max']:.4f}  median={amm['median']:.4f}")
    lines.append("")

    # Blocking episodes
    be = report["blocking_episodes"]
    lines.append("  Blocking episodes:")
    lines.append(f"    count:           {be['num_episodes']}")
    lines.append(f"    mean duration:   {be['mean_duration_s']:.2f}s")
    lines.append(f"    median duration: {be['median_duration_s']:.2f}s")
    lines.append(f"    max duration:    {be['max_duration_s']:.2f}s")
    lines.append(f"    total blocked:   {be['total_blocked_s']:.2f}s")
    lines.append("")

    # Correlation
    lines.append("-" * 50)
    lines.append("REQUESTED vs EXECUTED CORRELATION")
    lines.append("-" * 50)
    rve = report["requested_vs_executed_correlation"]
    for axis in ["vx", "vy", "wz"]:
        val = rve[axis]
        if isinstance(val, float) and np.isnan(val):
            lines.append(f"  corr({axis}): NaN (constant or insufficient data)")
        else:
            lines.append(f"  corr({axis}): {val:.4f}")
    lines.append("")

    # Temporal autocorrelation
    lines.append("-" * 50)
    lines.append("TEMPORAL AUTOCORRELATION")
    lines.append("-" * 50)
    tc = report["temporal_autocorrelation"]
    for key in sorted(tc.keys()):
        vals = tc[key]
        lag_str = "  ".join(f"lag{k.split('_')[1]}={v:.3f}" for k, v in vals.items())
        lines.append(f"  {key}: {lag_str}")
    lines.append("")

    # Action entropy
    lines.append("-" * 50)
    lines.append("ACTION ENTROPY (histogram, 50 bins)")
    lines.append("-" * 50)
    ae = report["action_entropy"]
    for key in sorted(ae.keys()):
        lines.append(f"  {key}: {ae[key]:.4f}")
    lines.append("")

    # Spatial coverage
    lines.append("-" * 50)
    lines.append("SPATIAL COVERAGE")
    lines.append("-" * 50)
    sc = report["spatial_coverage"]
    if isinstance(sc, str):
        lines.append(f"  {sc}")
    else:
        lines.append(f"  X range: [{sc['x_range'][0]:.2f}, {sc['x_range'][1]:.2f}]")
        lines.append(f"  Y range: [{sc['y_range'][0]:.2f}, {sc['y_range'][1]:.2f}]")
        lines.append(f"  Unique 1m cells: {sc['unique_1m_cells']}")
        lines.append(f"  Total distance: {sc['total_distance_m']:.2f}m")
    lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Matplotlib visualizations (optional)
# ─────────────────────────────────────────────────────────────

def generate_plots(data: dict, report: dict, outdir: Path):
    """Generate all diagnostic plots. Requires matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available, skipping plots.")
        return

    outdir.mkdir(parents=True, exist_ok=True)

    vx_range = (0.0, 0.6)
    vy_range = (-0.3, 0.3)
    wz_range = (-1.0, 1.0)

    # ── A/B/C: Action histograms ──
    for prefix, suffix in [("requested", "req"), ("executed", "exe")]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"{prefix.capitalize()} Action Distributions")
        for ax, (axis, rng) in zip(axes, [("vx", vx_range), ("vy", vy_range), ("wz", wz_range)]):
            arr = data[f"{prefix}_{axis}"]
            valid = arr[np.isfinite(arr)]
            ax.hist(valid, bins=50, range=rng, alpha=0.7, edgecolor="black", linewidth=0.5)
            ax.set_title(f"{prefix}_{axis}")
            ax.set_xlabel(axis)
            ax.set_ylabel("count")
            ax.axvline(0, color="red", linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / f"hist_{suffix}_actions.png", dpi=150)
        plt.close()

    if data["has_target"]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("Target Action Distributions (pre-smoothing)")
        for ax, (axis, rng) in zip(axes, [("vx", vx_range), ("vy", vy_range), ("wz", wz_range)]):
            arr = data[f"target_{axis}"]
            valid = arr[np.isfinite(arr)]
            ax.hist(valid, bins=50, range=rng, alpha=0.7, color="green", edgecolor="black", linewidth=0.5)
            ax.set_title(f"target_{axis}")
            ax.set_xlabel(axis)
            ax.set_ylabel("count")
        plt.tight_layout()
        plt.savefig(outdir / "hist_target_actions.png", dpi=150)
        plt.close()

    # ── D: Requested vs executed scatter ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Requested vs Executed Actions")
    for ax, axis in zip(axes, ["vx", "vy", "wz"]):
        req = data[f"requested_{axis}"]
        exe = data[f"executed_{axis}"]
        valid = np.isfinite(req) & np.isfinite(exe)
        if valid.sum() > 0:
            ax.scatter(req[valid], exe[valid], alpha=0.1, s=2)
            lo = min(req[valid].min(), exe[valid].min())
            hi = max(req[valid].max(), exe[valid].max())
            ax.plot([lo, hi], [lo, hi], "r--", alpha=0.5, label="y=x")
        ax.set_xlabel(f"requested_{axis}")
        ax.set_ylabel(f"executed_{axis}")
        ax.set_title(axis)
        ax.legend()
    plt.tight_layout()
    plt.savefig(outdir / "scatter_req_vs_exe.png", dpi=150)
    plt.close()

    # ── E: Blocked timeline ──
    fig, ax = plt.subplots(figsize=(15, 3))
    times = data["sim_time"]
    blocked = data["safety_blocked"]
    valid_t = np.isfinite(times)
    if valid_t.sum() > 0:
        ax.fill_between(times[valid_t], 0, blocked[valid_t], alpha=0.5, color="red", step="post")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Blocked")
        ax.set_title("Safety Gate Blocked Timeline")
        ax.set_ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.savefig(outdir / "timeline_blocked.png", dpi=150)
    plt.close()

    # ── F: Robot XY trajectory ──
    pos_x = data["pos_x"]
    pos_y = data["pos_y"]
    valid_pos = np.isfinite(pos_x) & np.isfinite(pos_y)
    if valid_pos.sum() > 10:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(pos_x[valid_pos], pos_y[valid_pos], linewidth=0.5, alpha=0.7)
        ax.scatter(pos_x[valid_pos][0], pos_y[valid_pos][0], c="green", s=50, zorder=5, label="start")
        ax.scatter(pos_x[valid_pos][-1], pos_y[valid_pos][-1], c="red", s=50, zorder=5, label="end")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Robot XY Trajectory")
        ax.set_aspect("equal")
        ax.legend()
        plt.tight_layout()
        plt.savefig(outdir / "trajectory_xy.png", dpi=150)
        plt.close()

    # ── G: Action entropy comparison bar chart ──
    ae = report["action_entropy"]
    fig, ax = plt.subplots(figsize=(10, 5))
    keys = sorted(ae.keys())
    vals = [ae[k] for k in keys]
    ax.bar(range(len(keys)), vals, tick_label=keys)
    ax.set_ylabel("Entropy (nats)")
    ax.set_title("Action Entropy Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "entropy_comparison.png", dpi=150)
    plt.close()

    print(f"Plots saved to {outdir}/")


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────

def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Analyze exploration dataset: target/requested/executed action coverage."
    )
    parser.add_argument(
        "--session", type=str, required=True,
        help="Path to exploration session directory (contains segment_* subdirs)"
    )
    parser.add_argument(
        "--outdir", type=str, default=None,
        help="Output directory for report and plots (default: <session>/analysis_output)"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip matplotlib plot generation"
    )
    args = parser.parse_args()

    session_dir = Path(args.session).resolve()
    if not session_dir.exists():
        print(f"ERROR: Session directory not found: {session_dir}")
        sys.exit(1)

    outdir = Path(args.outdir).resolve() if args.outdir else session_dir / "analysis_output"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {session_dir}")
    rows = load_session_csvs(session_dir)
    print(f"Loaded {len(rows)} rows")

    if not rows:
        print("ERROR: No data loaded.")
        sys.exit(1)

    data = extract_arrays(rows)
    print(f"Schema: has_target={data['has_target']}, "
          f"has_extended_safety={data['has_extended_safety']}")

    report = run_analysis(data)
    text = format_report(report)

    # Print to stdout
    print(text)

    # Save text report
    report_path = outdir / "exploration_audit_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nText report saved to: {report_path}")

    # Save JSON report
    json_path = outdir / "exploration_audit_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"JSON report saved to: {json_path}")

    # Generate plots
    if not args.no_plots:
        generate_plots(data, report, outdir)


if __name__ == "__main__":
    main()
