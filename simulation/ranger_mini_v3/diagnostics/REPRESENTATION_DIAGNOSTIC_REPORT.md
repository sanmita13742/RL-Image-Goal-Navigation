# MINav DINOv3 Representation Diagnostic Report

## 1. Dataset Size & Cache
- **Frames Analyzed:** 1,000 (from Phase 17 dry-run cache)
- **DINOv3 Representation Shape:** `[1000, 384]` (L2-normalized, mean-pooled patch embeddings)

## 2. Temporal Similarity Statistics
Cosine similarity decays slightly over time but remains extremely high across all temporal offsets:
- **t vs t+1 (0.1s):** Mean=0.9932 | Median=0.9960 | 1%=0.9638 | Min=0.9386
- **t vs t+10 (1.0s):** Mean=0.9638 | Median=0.9754 | 1%=0.8043 | Min=0.7776
- **t vs t+100 (10.0s):** Mean=0.9259 | Median=0.9379 | 1%=0.7953 | Min=0.7546
- **t vs t+500 (50.0s):** Mean=0.9217 | Median=0.9291 | 1%=0.8040 | Min=0.7766

## 3. Random Global Similarity Statistics
Cosine similarity for randomly paired frames strictly separated by `|j - t| > 100` frames:
- **Mean:** 0.9260
- **Median:** 0.9369
- **Percentiles:** 1%=0.7977 | 10%=0.8643 | 90%=0.9641
- **Min:** 0.7369

## 4. Geometric Goal Similarity
Evaluating the 4-frame state $S(s_t, g)$ against geometrically sampled future goals ($p=0.99$):
- **Average Offset:** 91.1 frames (9.1 seconds)
- **Similarity Mean:** 0.9426 | **Median:** 0.9529
- **Reward=1 (S >= 0.8):** 99.50%
- **Reward=0 (S < 0.8):** 0.50%

## 5. Uniform Goal Similarity
Evaluating $S(s_t, g)$ against randomly sampled valid global goals:
- **Average Absolute Offset:** 339.9 frames (33.9 seconds)
- **Similarity Mean:** 0.9356 | **Median:** 0.9413
- **Reward=1 (S >= 0.8):** 99.60%
- **Reward=0 (S < 0.8):** 0.40%

## 6. State-to-Goal Summary Table
| Comparison | Mean | Median | P10 | P90 |
| :--- | :--- | :--- | :--- | :--- |
| t → t+1 | 0.9932 | 0.9960 | 0.9840 | 0.9986 |
| t → t+10 | 0.9638 | 0.9754 | 0.9264 | 0.9923 |
| t → t+100 | 0.9259 | 0.9379 | 0.8480 | 0.9726 |
| t → t+500 | 0.9217 | 0.9291 | 0.8811 | 0.9581 |
| t → random distant | 0.9260 | 0.9369 | 0.8643 | 0.9641 |
| geometric state → goal | 0.9426 | 0.9529 | 0.8708 | 0.9899 |
| uniform state → goal | 0.9356 | 0.9413 | 0.8898 | 0.9712 |

## 7. Reward Threshold Distribution
Distribution of $S(s_t, g)$ around the 0.8 decision boundary:
| Bin | Geometric | Uniform |
| :--- | :--- | :--- |
| **[0.0, 0.75)** | 0.00% | 0.00% |
| **[0.75, 0.8)** | 0.50% | 0.40% |
| **[0.8, 0.85)** | 5.72% | 3.82% |
| **[0.85, 0.9)** | 9.04% | 8.94% |
| **[0.9, 0.95)** | 31.33% | 47.69% |
| **[0.95, 1.0]** | 53.41% | 39.16% |

## 8. Global Representation Concentration (Collapse Check)
- Mean vector norm: **$||\mu|| = 0.9641$**
Because all $\phi$ vectors are L2-normalized unit vectors, an average vector norm of 0.964 means that nearly all embeddings are pointing in the exact same direction in the 384-dimensional hypersphere. The feature space is highly concentrated.

## 9. Position-vs-Feature Similarity
Measuring how visual similarity degrades as ground-truth physical distance increases:
- **0.00 - 0.25m:** Mean Sim = 0.9589
- **0.25 - 0.50m:** Mean Sim = 0.9398
- **0.50 - 1.00m:** Mean Sim = 0.9366
- **> 1.00m:** Mean Sim = 0.9267
*Observation: Similarity drops by only ~0.03 between adjacent observations and observations on the opposite side of the room.*

## 10. Visual Examples
Visual examples mapping numerical similarities to image pairs have been saved in `diagnostics/examples/`. 
*(E.g. A_high_sim_distant, B_low_sim_adjacent, C_geometric, D_uniform)*

## Interpretation & Verdict

**REPRESENTATION STATUS: POTENTIALLY COLLAPSED / INSUFFICIENTLY DISCRIMINATIVE**

**Evidence:**
1. **Zero Discriminative Power at 0.8:** The paper's threshold of 0.8 completely fails to separate successful goals from random frames. The 1st percentile of completely random, temporally distant pairs ($|j-t|>100$) is 0.797. Consequently, 99.6% of entirely random uniform goals are incorrectly labeled as "successful" (reward=1).
2. **Feature Space Concentration:** The mean vector norm $||\mu||$ is 0.964. The theoretical maximum is 1.0 (if all images produced the exact same vector). The DINOv3 mean-pooled representations occupy a minuscule cone of the 384-D space.
3. **Positional Invariance:** Frames separated by >1.0 meters in the room still register a mean similarity of 0.926, which is barely lower than the 0.958 for frames <0.25m apart. 

**Conclusion:** 
The decision to globally mean-pool all 1,372 spatial patch tokens into a single vector destroys the specific geometric/viewpoint information required for navigation. The background environment dominates the average, causing all frames in the room to look numerically identical (similarity > 0.9). 
While the pipeline correctly executes the mathematical reward formula, the current $\phi(o)$ encoding is insufficiently discriminative to serve as a visual similarity signal for RL.
