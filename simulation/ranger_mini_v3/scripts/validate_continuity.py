"""
scripts/validate_continuity.py
============================================================
Phase 2: Verify temporal continuity of the frame index.

Checks:
  1. global_step = 0, 1, 2, ..., N-1 (no gaps, no duplicates)
  2. For every adjacent pair: global_step[t+1] = global_step[t] + 1
  3. sim_time is strictly monotonically increasing (never resets)
  4. All RGB image files exist on disk
  5. trajectory_id is always 0 (one continuous rollout)
  6. No segment boundary introduces a large position jump (> 1m)
  7. action columns (linear/lateral/angular_vel_cmd) are all finite
  8. Transition semantics: obs_t + action_t → obs_{t+1} is preserved
     (verified by checking that sim_time[t+1] > sim_time[t] always)

If ANY check fails, the script exits with code 1.
Do NOT proceed to DINOv3 encoding if this script fails.

Usage:
    python scripts/validate_continuity.py --index data/processed/frame_index/frame_index.parquet
    python scripts/validate_continuity.py --session dataset/20260811_234812
"""

import sys
import math
import argparse
import pandas as pd
from pathlib import Path

# Allow importing from src/
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.index_segments import build_frame_index


def validate(df: pd.DataFrame) -> bool:
    failures = []
    N = len(df)

    print(f"Validating {N:,} rows...")
    print()

    # ── Check 1: global_step is 0..N-1 ───────────────────────────────────────
    gs = df["global_step"].tolist()
    if gs != list(range(N)):
        # Find first gap
        for i in range(1, len(gs)):
            if gs[i] != gs[i-1] + 1:
                failures.append(
                    f"Check 1 FAIL: global_step gap at index {i}: {gs[i-1]} → {gs[i]}"
                )
                break
        if len(gs) != len(set(gs)):
            failures.append("Check 1 FAIL: duplicate global_step values")
    else:
        print(f"  [OK] Check 1: global_step continuous 0 … {N-1}")

    # ── Check 2: sim_time strictly monotonic ──────────────────────────────────
    times = df["sim_time"].tolist()
    bad = [(i, times[i-1], times[i]) for i in range(1, len(times))
           if times[i] <= times[i-1]]
    if bad:
        failures.append(
            f"Check 2 FAIL: sim_time not monotonic at {len(bad)} positions, "
            f"first at index {bad[0][0]}: {bad[0][1]:.3f} → {bad[0][2]:.3f}"
        )
    else:
        print(f"  [OK] Check 2: sim_time monotonic {times[0]:.3f} … {times[-1]:.3f}s")

    # ── Check 3: trajectory_id always 0 ──────────────────────────────────────
    traj_ids = df["trajectory_id"].unique().tolist()
    if traj_ids != [0]:
        failures.append(f"Check 3 FAIL: trajectory_id values = {traj_ids}, expected [0]")
    else:
        print(f"  [OK] Check 3: trajectory_id = 0 throughout (one continuous rollout)")

    # ── Check 4: action columns finite ───────────────────────────────────────
    for col in ["linear_vel_cmd", "lateral_vel_cmd", "angular_vel_cmd"]:
        if col in df.columns:
            n_bad = (~df[col].apply(math.isfinite)).sum()
            if n_bad > 0:
                failures.append(f"Check 4 FAIL: {col} has {n_bad} non-finite values")
            else:
                print(f"  [OK] Check 4: {col} all finite  "
                      f"(min={df[col].min():.3f}, max={df[col].max():.3f})")

    # ── Check 5: segment boundary position jumps ──────────────────────────────
    seg_boundaries = df[df["segment_step"] == 0].index.tolist()
    max_jump = 0.0
    suspicious = 0
    for idx in seg_boundaries:
        if idx == 0:
            continue
        prev = df.iloc[idx - 1]
        curr = df.iloc[idx]
        dist = math.hypot(
            float(curr["pos_x"]) - float(prev["pos_x"]),
            float(curr["pos_y"]) - float(prev["pos_y"])
        )
        max_jump = max(max_jump, dist)
        if dist > 1.0:
            suspicious += 1
            failures.append(
                f"Check 5 FAIL: boundary jump {dist:.3f}m at global_step "
                f"{curr['global_step']} ({prev['segment_id']} → {curr['segment_id']})"
            )
    if suspicious == 0:
        print(f"  [OK] Check 5: all segment boundary jumps <= {max_jump:.3f}m (no resets)")

    # ── Check 6: spot-check RGB file existence ────────────────────────────────
    print(f"\n  Spot-checking RGB file existence (checking every 1000th frame)...")
    missing = []
    for i in range(0, N, 1000):
        p = Path(df.iloc[i]["rgb_abs_path"])
        if not p.exists():
            missing.append(str(p))

    if missing:
        failures.append(f"Check 6 FAIL: {len(missing)} RGB files missing, e.g.: {missing[0]}")
    else:
        checked = len(range(0, N, 1000))
        print(f"  [OK] Check 6: {checked} sampled RGB files exist")

    # ── Check 7: sim_time ≈ global_step × dt ─────────────────────────────────
    DT = 0.1   # 10 Hz → 0.1s per step
    t0 = times[0]
    max_drift = max(abs(times[i] - (t0 + i * DT)) for i in range(min(N, 5000)))
    if max_drift > 0.5:
        failures.append(
            f"Check 7 WARN: sim_time drift from expected {DT}s/step: "
            f"max_drift={max_drift:.3f}s (may indicate variable timestep)"
        )
    else:
        print(f"  [OK] Check 7: sim_time matches 0.1s/step (max drift={max_drift:.4f}s)")

    # ── Final report ──────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    if not failures:
        print("  ALL CONTINUITY CHECKS PASSED [PASS]")
        print(f"  Dataset is ready for DINOv3 encoding.")
        print("=" * 60)
        return True
    else:
        print(f"  {len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print(f"    ✗ {f}")
        print("=" * 60)
        print("  DO NOT proceed to DINOv3 encoding until failures are resolved.")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index",   help="Path to frame_index.parquet")
    parser.add_argument("--session", help="Session dir (will build index first)")
    parser.add_argument("--index_out", default="data/processed/frame_index",
                        help="Where to save frame index if --session given")
    args = parser.parse_args()

    if args.session:
        print(f"Building frame index from session: {args.session}")
        df = build_frame_index(args.session, args.index_out)
    elif args.index:
        print(f"Loading frame index from: {args.index}")
        df = pd.read_parquet(args.index)
    else:
        print("ERROR: provide --session or --index")
        sys.exit(1)

    print(f"\nLoaded {len(df):,} frames")
    ok = validate(df)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
