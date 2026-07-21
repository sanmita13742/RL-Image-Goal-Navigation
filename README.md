# Ranger Mini V3 RL Project

## What was done
- Added and configured the Ranger Mini V3 robot model for MuJoCo simulation.
- Configured Ackermann steering (4WS) kinematics for the robot.
- Created random exploration script (`random_explore.py`) to gather dataset logs.
- Wrote trajectory plotting and heatmap visualization scripts (`plot_trajectory.py`).

## Issues & Fixes
- **Issue**: MuJoCo raised `mjERR_BADGEOM` related to capsule geometries.
  **Fix**: Adjusted the wheel geometries to use correct cylinders instead of capsules with invalid dimensions.
- **Issue**: Missing Pandas and Matplotlib dependencies for analysis.
  **Fix**: Installed missing packages.
- **Issue**: `pandas.errors.ParserError` during CSV parsing because of lines with inconsistent field counts.
  **Fix**: Modified the script to use `on_bad_lines='skip'` in `pd.read_csv()` to gracefully handle corrupted rows.
- **Issue**: Path configurations and dimension mismatches (like wheel radius) across the robot classes.
  **Fix**: Corrected the constants in the robot subclasses (`WHEEL_RADIUS`, `TRACK_WIDTH`, `WHEELBASE`).

## Docs
The specification document for the Ranger Mini is available at `docs/raner-mini-specification.md`.
