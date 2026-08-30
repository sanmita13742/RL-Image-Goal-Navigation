# Hindsight Relabeling & Final Dataset Documentation

## 1. Hindsight Relabeling Final Implementation
The hindsight goal-relabeling pipeline is designed to reproduce the MINav architecture ("120 Minutes and a Laptop: Minimalist Image-goal Navigation"). Relabeling allows us to retrospectively assign future states as "goals" during offline RL training, maximizing the utility of unguided exploration data.

### 1.1 Papers Implementation vs. Our Choice (with Reasons)
When adapting the paper to our physical and simulated Ranger Mini architecture, we strictly delineated between the original specs and our necessary implementation choices:

**Paper-Specified Details (Strictly Reproduced):**
- **Frozen DINOv3** as the visual encoder.
- **Final hidden-layer patch embeddings** used for feature extraction.
- Normalized visual representation.
- **4-frame state history** for observation input.
- **Cosine Similarity** reward metric with a threshold of **0.8**.

**Our Implementation Choices & Reasoning:**
- **$\phi(o)$ Representation:** We used DINOv3's `x_norm_clstoken` as the visual state representation $\phi(o)$ to ensure compact yet globally context-aware embeddings.
- **SSD (Sum of Squared Differences):** We used `x_norm_patchtokens` for SSD computations with a crop of 14x25 to focus strictly on navigable space, removing sky/ceiling noise.
- **Preprocessing:** We relied exclusively on `ToTensor` preprocessing to minimize latency during the live loop.
- **Discount Factor:** We chose $p = 0.99$ to balance long-horizon pathfinding.

## 2. Final Dataset & Validation Document
The data extracted from ROS 2 bags (using [`scripts/extract_dataset.py`](file:///c:/Users/sanmi/Desktop/projects/RL/scripts/extract_dataset.py)) undergoes strict validation.

### Dataset Integrity & Segment Structure
- **Storage Segments Are Not Reset Trajectories:** Data is stored in chunks (`segment_000`, `segment_001`, etc.) capped at 500 frames simply to avoid filesystem bottlenecks. The real robot was **never reset**. We use a continuous `global_step` index, allowing 4-frame histories and future hindsight goals to seamlessly cross segment boundaries.
- **Valid Transitions:** For any transition $(s_t, a_t, s_{t+1}, g, r)$, the observation $o_{t+1}$ is guaranteed to exist.
- **State Segregation:** Physics ground-truth data (X/Y coordinates) is strictly excluded from the observation space. The RL agent receives exactly 384-dimensional $\phi(o)$ visual features, guaranteeing it learns from vision rather than cheating via coordinate injection.
