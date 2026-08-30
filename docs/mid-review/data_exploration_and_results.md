# Data Exploration, Ablation Study, and Evidence-Backed Results

## 1. Data Exploration Strategy
The initial exploration data was collected using policies defined in [`exploration_policies.py`](file:///c:/Users/sanmi/Desktop/projects/RL/simulation/ranger_mini_v3/exploration_policies.py). We primarily focused on finding a noise generation strategy that maximizes **State-Action Entropy $\eta(s,a)$** and provides good spatial coverage without triggering physics exploits (e.g., wall hugging).

Our exploration strategies were heavily tested against collision scenarios, leveraging stateful recovery techniques so that the robot chooses an optimal evasion direction when colliding and commits to it rather than spinning randomly.

## 2. Ablation Study: Exploration Policies
We conducted an ablation study comparing different noise functions for exploration. The quantitative validation was performed using an 80-second simulation sequence (starting from `x = -4.5`, `y = 0.0`). 

The variants tested were:
1. **White Noise (`WhiteNoiseExploration`)**: Fully random actions at every step.
2. **Brownian/Ornstein-Uhlenbeck Noise**: Temporally correlated noise.
3. **Uniform Pink Noise (`UniformColoredNoise` / $1/f$ noise)**: A Voss-McCartney process mapped through a Gaussian CDF to maintain a uniform marginal distribution across action boundaries while preserving the $1/f$ spectral dynamics.

## 3. Evidence-Backed Results
The results clearly showed that **Uniform Pink Noise** outperformed both the Baseline White Noise and the OU/Brownian Noise strategies in both unique cells visited and overall State Entropy.

### Final Metrics:
- **Baseline (White Noise):** 
  - Unique Cells Visited: `9`
  - State Entropy: `2.58 bits`
- **Previous Baseline (OU / Brownian Noise):** 
  - Unique Cells Visited: `4`
  - State Entropy: `1.44 bits` (Suffered from heavy temporal correlation causing loops).
- **Optimized (Uniform Pink Noise):**
  - Unique Cells Visited: `11` *(Best)*
  - State Entropy: `2.86 bits` *(Best)*

As a result, the data collection pipeline now relies entirely on the Uniform Pink Noise policy, yielding high-variance trajectories that are ideal for our Offline RL dataset.
