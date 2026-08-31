# New Navigation Map for MINav

## Overview
This is a custom indoor navigation maze environment created for the MINav offline RL reproduction. It is designed to rigorously test the DINOv3 visual representation encoder and the hindsight relabeling geometric properties.

## Map Properties
- **Dimensions**: 16m x 16m
- **Robot Starting Position**: Bottom Left (`pos="-7 -7 1.0"`)
- **Corridor Structure**: The maze is split into interconnected sections containing dead-ends, U-turns, long corridors, and central junctions.

## Visual Aliasing & Perception
The map intentionally introduces perceptual aliasing to prevent simple color-tracking models from trivially memorizing positions. 
- **Aliased U-Structures**: Two identical sequences of walls (Green -> Yellow -> Orange) exist at diametrically opposite corners of the map (Bottom-Left and Top-Right).
- **Unique Landmarks**: A Cyan cylinder (Top-Left), Red cone (Center-Left), and Purple box (Bottom-Right) provide distinct localization targets.

## Expected Navigation Behavior
- The `PrimitiveExplorationPolicy` (10 Hz pink noise) will continuously navigate these corridors without resetting.
- Due to the large scale (16x16m), the robot will experience deep, multi-turn trajectories, requiring the DINOv3 representation to successfully encode both distant object visibility and immediate corridor safety boundaries.
