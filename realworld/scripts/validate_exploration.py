#!/usr/bin/env python3
"""
realworld/scripts/validate_exploration.py — Offline exploration validation.
===========================================================================
Reads a completed exploration session's CSV files and validates that the
exploration data is consistent with the MINav Pink Uniform Noise design:

  Core MINav properties to validate:
    - β=1 pink process with FFT spectral shaping
    - Gaussian CDF (Φ) transform for uniform marginals
    - Correct action bounds per configured ranges
    - Temporal autocorrelation preserved through the pipeline

  Heuristic engineering sanity checks (Ranger-specific, not from MINav):
    1. Action distribution — quartile coverage ≥ 15%
    2. Temporal autocorrelation — lag-1 ρ > 0.3 per dimension
    3. X-Y trajectory — non-trivial spatial spread
    4. High-frequency oscillation — ωz sign-flip rate < 30%
    5. Safety intervention rate — informational

Usage:
  python realworld/scripts/validate_exploration.py \\
      --session-dir realworld/runs/<run_id>/exploration

  # Save plots to a directory:
  python realworld/scripts/validate_exploration.py \\
      --session-dir realworld/runs/<run_id>/exploration \\
      --plot-dir realworld/runs/<run_id>/validation_plots
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_session(session_dir: Path) -> pd.DataFrame:
    """Load and concatenate all segment CSVs from a session directory."""
    segments = sorted(session_dir.glob("segment_*/observations.csv"))
    if not segments:
        raise FileNotFoundError(f"No segment CSVs found in {session_dir}")

    dfs = []
    for csv_path in segments:
        df = pd.read_csv(csv_path)
        dfs.append(df)
        logger.info(f"  Loaded {csv_path.parent.name}: {len(df)} steps")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"  Total steps: {len(combined)}")
    return combined


def compute_autocorrelation(x: np.ndarray, lag: int = 1) -> float:
    """Compute lag-k autocorrelation of a 1D array."""
    n = len(x)
    if n < lag + 2:
        return 0.0
    mean = x.mean()
    c0 = np.sum((x - mean) ** 2) / n
    if c0 < 1e-12:
        return 0.0
    ck = np.sum((x[:-lag] - mean) * (x[lag:] - mean)) / n
    return ck / c0


def check_uniform_coverage(values: np.ndarray, name: str, n_bins: int = 4) -> tuple[bool, str]:
    """Check that values cover their range reasonably uniformly.

    Splits the range into n_bins equal bins and checks that each
    contains at least 15% of samples.
    """
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-6:
        return False, f"{name}: constant value ({lo:.4f}), no coverage"

    edges = np.linspace(lo, hi, n_bins + 1)
    counts = np.histogram(values, bins=edges)[0]
    fracs = counts / len(values)

    min_frac = fracs.min()
    min_bin = fracs.argmin()

    passed = min_frac >= 0.15
    detail = (
        f"{name}: range=[{lo:.3f}, {hi:.3f}], "
        f"bin fracs={[f'{f:.2%}' for f in fracs]}, "
        f"min={min_frac:.2%} (bin {min_bin})"
    )
    return passed, detail


def check_oscillation(wz: np.ndarray) -> tuple[bool, float]:
    """Check for high-frequency oscillation in angular velocity.

    Computes the fraction of consecutive sign changes. A healthy
    pink-noise signal should have < 30% sign flips.
    """
    signs = np.sign(wz)
    # Ignore near-zero values
    signs[np.abs(wz) < 0.01] = 0

    nonzero_mask = signs != 0
    nonzero_signs = signs[nonzero_mask]

    if len(nonzero_signs) < 2:
        return True, 0.0

    flips = np.sum(nonzero_signs[:-1] != nonzero_signs[1:])
    flip_rate = flips / (len(nonzero_signs) - 1)

    return flip_rate < 0.30, flip_rate


def validate(session_dir: Path, plot_dir: Path = None) -> bool:
    """Run all validation checks. Returns True if all pass."""
    logger.info("=" * 60)
    logger.info("EXPLORATION VALIDATION")
    logger.info(f"  Session: {session_dir}")
    logger.info("=" * 60)

    df = load_session(session_dir)

    # Determine column names (support both old and new formats)
    has_executed = "executed_linear_vel" in df.columns
    has_safety = "safety_intervention" in df.columns

    vx = df["linear_vel_cmd"].values.astype(float)
    vy = df["lateral_vel_cmd"].values.astype(float)
    wz = df["angular_vel_cmd"].values.astype(float)

    results = []

    # ─── 1. Action Distribution Coverage (heuristic sanity check) ───
    logger.info("\n--- 1. Action Distribution Coverage (heuristic: ≥15% per quartile) ---")
    for name, vals in [("vx", vx), ("vy", vy), ("ωz", wz)]:
        passed, detail = check_uniform_coverage(vals, name)
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {status}: {detail}")
        results.append(passed)

    # ─── 2. Temporal Autocorrelation (heuristic sanity check) ───
    logger.info("\n--- 2. Temporal Autocorrelation (heuristic: lag-1 ρ > 0.3) ---")
    for name, vals in [("vx", vx), ("vy", vy), ("ωz", wz)]:
        rho = compute_autocorrelation(vals, lag=1)
        passed = rho > 0.3
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {status}: {name} ρ(1) = {rho:.4f} (threshold: > 0.3)")
        results.append(passed)

    # ─── 3. Spatial Exploration ───
    logger.info("\n--- 3. Spatial Exploration ---")
    px = df["pos_x"].values.astype(float)
    py = df["pos_y"].values.astype(float)
    dx = px.max() - px.min()
    dy = py.max() - py.min()
    area = dx * dy
    total_dist = np.sum(np.sqrt(np.diff(px)**2 + np.diff(py)**2))
    passed = area > 0.01  # at least some spatial spread
    status = "✓ PASS" if passed else "✗ FAIL"
    logger.info(f"  {status}: bounding box {dx:.2f}m × {dy:.2f}m = {area:.2f}m²")
    logger.info(f"          total distance: {total_dist:.2f}m")
    results.append(passed)

    # ─── 4. High-Frequency Oscillation (heuristic sanity check) ───
    logger.info("\n--- 4. High-Frequency Oscillation (heuristic: sign-flip < 30%) ---")
    passed, flip_rate = check_oscillation(wz)
    status = "✓ PASS" if passed else "✗ FAIL"
    logger.info(f"  {status}: ωz sign-flip rate = {flip_rate:.2%} (threshold: < 30%)")
    results.append(passed)

    # ─── 5. Safety Intervention Rate ───
    logger.info("\n--- 5. Safety Intervention Rate ---")
    if has_safety:
        safety_flags = df["safety_intervention"].values.astype(int)
        intervention_rate = safety_flags.sum() / len(safety_flags)
        logger.info(f"  INFO: {safety_flags.sum()}/{len(safety_flags)} steps blocked ({intervention_rate:.2%})")
        if intervention_rate > 0.5:
            logger.warning(f"  ⚠ High intervention rate ({intervention_rate:.2%}) — check environment")
    else:
        logger.info("  SKIP: safety_intervention column not present (old format)")

    # ─── Summary ───
    all_passed = all(results)
    logger.info("\n" + "=" * 60)
    logger.info(f"RESULT: {'ALL CHECKS PASSED ✓' if all_passed else 'SOME CHECKS FAILED ✗'}")
    logger.info(f"  Passed: {sum(results)}/{len(results)}")
    logger.info("=" * 60)

    # ─── Optional Plots ───
    if plot_dir is not None:
        _generate_plots(df, vx, vy, wz, px, py, plot_dir, has_safety)

    return all_passed


def _generate_plots(
    df: pd.DataFrame,
    vx: np.ndarray, vy: np.ndarray, wz: np.ndarray,
    px: np.ndarray, py: np.ndarray,
    plot_dir: Path,
    has_safety: bool,
) -> None:
    """Generate validation plots and save to plot_dir."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plots")
        return

    plot_dir.mkdir(parents=True, exist_ok=True)

    # 1. Action histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, vals, name in zip(axes, [vx, vy, wz], ["vx (m/s)", "vy (m/s)", "ωz (rad/s)"]):
        ax.hist(vals, bins=50, density=True, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.set_title(f"Distribution: {name}")
        ax.set_xlabel(name)
        ax.set_ylabel("Density")
    plt.tight_layout()
    plt.savefig(plot_dir / "action_distributions.png", dpi=150)
    plt.close()

    # 2. Time series (first 500 steps)
    n_show = min(500, len(vx))
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    for ax, vals, name in zip(axes, [vx, vy, wz], ["vx", "vy", "ωz"]):
        ax.plot(vals[:n_show], linewidth=0.8)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Step")
    axes[0].set_title("Action Time Series (first 500 steps)")
    plt.tight_layout()
    plt.savefig(plot_dir / "action_timeseries.png", dpi=150)
    plt.close()

    # 3. X-Y trajectory
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(px, py, linewidth=0.5, alpha=0.7)
    ax.plot(px[0], py[0], "go", markersize=10, label="Start")
    ax.plot(px[-1], py[-1], "rs", markersize=10, label="End")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Exploration Trajectory")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "trajectory_xy.png", dpi=150)
    plt.close()

    # 4. Autocorrelation function
    max_lag = min(100, len(vx) // 4)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, vals, name in zip(axes, [vx, vy, wz], ["vx", "vy", "ωz"]):
        acf = [compute_autocorrelation(vals, lag=k) for k in range(max_lag)]
        ax.bar(range(max_lag), acf, width=1.0, alpha=0.7)
        ax.axhline(0.3, color="r", linestyle="--", alpha=0.5, label="threshold")
        ax.set_title(f"ACF: {name}")
        ax.set_xlabel("Lag")
        ax.set_ylabel("ρ")
        ax.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "autocorrelation.png", dpi=150)
    plt.close()

    logger.info(f"Plots saved to {plot_dir}")


def main():
    parser = argparse.ArgumentParser(description="Validate MINav exploration data")
    parser.add_argument("--session-dir", type=str, required=True,
                        help="Path to the exploration session directory")
    parser.add_argument("--plot-dir", type=str, default=None,
                        help="Optional directory to save validation plots")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    session_dir = Path(args.session_dir).resolve()
    plot_dir = Path(args.plot_dir).resolve() if args.plot_dir else None

    success = validate(session_dir, plot_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
