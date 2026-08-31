# NEW_MAP_DESIGN_REPORT

## 1. Motivation for the New Environment
The original MINav environment map (`_test_world.xml`) was relatively small (12x8m) and offered limited opportunities for deep corridor exploration, making it difficult to fully evaluate how the DINOv3 visual encoder discriminates between diverse, structurally deep scenes and handles perceptual aliasing. This new environment was explicitly designed as a 16x16m indoor maze to rigorous stress-test visual goal-conditioned reinforcement learning pipelines.

## 2. Geometry
- **Scale**: 16x16m
- **Structure**: The maze features a central horizontal divider, a top-left vertical barrier, and a bottom-right blocker.
- **Complexity**: It comprises 3 primary interconnected corridors, 2 potential dead-ends (depending on approach), and a central multi-directional intersection. 
- **Clearance**: Minimum corridor width is deliberately maintained at ~2.0m, ensuring safe traversal for the 0.5x0.6m Ranger Mini robot without introducing artificial physical clamping or traps.

## 3. Visual Structure & Materials
The environment employs distinct solid materials to define visual boundaries, similar to the original reference concept:
- **Green, Blue, Yellow, Orange, Red walls** are distributed to create unique scene layouts.
- **Soft Lighting**: Multi-directional soft lighting prevents extreme shadows and ensures DINOv3 can accurately process geometry rather than being misled by sharp lighting artifacts.

## 4. Repeated/Aliased Regions (DESIGN CHOICE)
To evaluate the *geometric* vs *purely visual* robustness of the downstream TD3+BC pipeline, intentional perceptual aliasing is integrated:
- **Aliased Structure 1 (Bottom-Left)**: A West-facing U-shaped enclave featuring Green -> Yellow -> Orange walls.
- **Aliased Structure 2 (Top-Right)**: An identically composed Green -> Yellow -> Orange U-shape, placed on the opposite side of the map.
If the agent merely learns color templates, it will conflate these two physically distant regions. This explicitly challenges the `hindsight` DINOv3 SSD mechanism.

## 5. Landmarks (DESIGN CHOICE)
While parts of the map are aliased, key spatial anchors are provided to allow global localization:
- **Cyan Cylinder**: Top-Left intersection.
- **Red Cone**: Center-Left cross-section.
- **Purple Box**: Bottom-Right corridor.

## 6. Expected Exploration Behavior
The existing 10 Hz Pink Noise Primitive Policy continuously samples linear and angular velocity curves based on depth clearance. We expect the robot to organically snake through the corridors. Because there are no immediate dead-ends near the start, the robot will naturally penetrate deep into the maze within 5-10 minutes of simulated time, providing rich visual diversity.

## 7. Why the environment is useful for MINav
MINav fundamentally relies on the hypothesis that pre-trained foundational models (DINOv3) produce spatial semantic representations that gracefully map to physical distance. A simple room fails to test this. A maze with aliased corridors and non-linear paths is the precise topology required to validate whether Euclidean physical distance cleanly correlates with DINOv3 cosine similarity.

## 8. Differences from the previous map
- **Original**: 12x8m, single central obstacle, mostly open-plan room.
- **New**: 16x16m, hard maze barriers, strict corridors, intentionally repeated material sequences, asymmetric start position.

## 9. Validation Results
Automated validation via `validate_new_map.py` confirmed:
- Zero XML parsing or geometry overlap errors.
- Successful rendering of 320x240 RGB and 640x60 Depth tensors.
- 1,000 steps of physical simulation executed with zero NaN states.
- The robot safely translated > 5 meters radially from the start position without collision lockups.

## 10. 5-Minute Exploration Results
A 5-minute (3000 step) live run generated a continuous stream of structurally rich images covering multiple corridors, capturing views of the Red Cone landmark and both Aliased U-structures from varied approach angles. The pink-noise policy successfully prevented getting trapped in the maze corners.
