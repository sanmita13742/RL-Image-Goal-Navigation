# MINav Hindsight Relabeling Pipeline: Reproduction Report

This report documents the finalized visual representation and goal-relabeling pipeline constructed for the reproduction of MINav ("120 Minutes and a Laptop: Minimalist Image-goal Navigation via Unsupervised Exploration and Offline RL", arXiv:2603.26441).

To strictly adhere to the rule that we must distinguish between details specified in the original paper and our own reasonable implementation choices, we categorize all configuration and logic into these two sections.

## 1. PAPER-SPECIFIED
The following details are explicitly stated in the paper and exactly reproduced in this pipeline:
- frozen DINOv3
- final hidden-layer patch embeddings
- normalized visual representation
- 4-frame state
- cosine similarity
- threshold 0.8

## 2. IMPLEMENTATION CHOICES
The following details were **not** specified in the paper, and represent our necessary technical decisions:
- use DINOv3 x_norm_clstoken as phi(o)
- use x_norm_patchtokens for SSD
- SSD crop 14x25
- p=0.99
- ToTensor-only preprocessing

## 3. Dataset Integrity
To ensure the pipeline generates valid RL experiences without violating temporal boundaries or physics:
* **Storage Segments Are Not Reset Trajectories**: Segment directories (`segment_000`, `segment_001`, etc.) merely represent file storage limits (500 frames). The physical robot was never reset. We index using `global_step` across the entire 72,000-frame session and allow 4-frame states and future goals to seamlessly span segment boundaries.
* **Valid Transitions Only**: We verify that for any valid transition $(s_t, a_t, s_{t+1}, g, r)$, the observation $o_{t+1}$ exists in the dataset. The final index $N-1$ is never used as a current state since there is no forward timestep for it.
* **State Excludes Physics Ground Truth**: We meticulously verify that X/Y coordinates do not leak into the state representation or the visual similarity reward. The RL agent receives strictly 384-dimensional $\phi(o)$ visual features.
