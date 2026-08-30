# FINAL SPATIAL SIMILARITY REPORT

This report evaluates how well the DINOv3 CLS visual similarity correlates with physical distance across the complete 72,000-frame MINav hindsight dataset.

### Dataset
- Total frames: 72,000
- Total sampled pairs: 15,000
- Distant pairs (temp_dist > 100): 14,963
- Random seed: 123

### Representation
- DINOv3 model: vit_small_patch16_dinov3
- Feature: x_norm_clstoken / CLS
- Dimension: 384
- Normalization: L2-normalized

### Spatial relationship (All Sampled Pairs)

| Physical Distance | Count | Mean S | Median S | % S>=0.8 |
|---|---:|---:|---:|---:|
| 0-0.25m | 228 | 0.9489 | 0.9863 | 93.0% |
| 0.25-0.5m | 148 | 0.9033 | 0.9251 | 85.8% |
| 0.5-1m | 357 | 0.8407 | 0.8584 | 74.8% |
| 1-2m | 1432 | 0.8189 | 0.8382 | 68.3% |
| 2-5m | 5956 | 0.7894 | 0.8051 | 51.9% |
| >5m | 6879 | 0.8097 | 0.8289 | 62.8% |

### Spatial relationship (Distant Pairs, temporal > 100)

| Physical Distance | Count | Mean S | Median S | % S>=0.8 |
|---|---:|---:|---:|---:|
| 0-0.25m | 203 | 0.9447 | 0.9848 | 92.1% |
| 0.25-0.5m | 144 | 0.9017 | 0.9194 | 85.4% |
| 0.5-1m | 354 | 0.8402 | 0.8582 | 74.6% |
| 1-2m | 1430 | 0.8188 | 0.8381 | 68.3% |
| 2-5m | 5953 | 0.7894 | 0.8051 | 51.9% |
| >5m | 6879 | 0.8097 | 0.8289 | 62.8% |

### Temporal relationship

| Temporal Step   |     Mean |   Median |      Min |      Max |      p10 |      p90 |
|:----------------|---------:|---------:|---------:|---------:|---------:|---------:|
| t -> t+1        | 0.991671 | 0.997183 | 0.632391 | 1        | 0.976985 | 0.999772 |
| t -> t+10       | 0.967503 | 0.991943 | 0.237886 | 0.999998 | 0.907538 | 0.99909  |
| t -> t+100      | 0.916441 | 0.944882 | 0.239191 | 0.999988 | 0.795886 | 0.997114 |
| t -> t+500      | 0.853454 | 0.861178 | 0.246842 | 0.999548 | 0.713659 | 0.990921 |

### Correlation

**All Sampled Pairs:**
- Pearson: 0.0009
- Spearman: 0.0105

**Distant Pairs (>100 frames apart):**
- Pearson: 0.0082
- Spearman: 0.0169

### Interpretation

1. Visual similarity shows useful local spatial discrimination: similarity decreases substantially from 0–0.25m through 1–2m.

2. At larger distances, the representation exhibits substantial visual aliasing. The >5m group has higher mean similarity than the 2–5m group.

3. Pearson/Spearman correlations near zero indicate that physical distance is NOT strongly correlated with CLS cosine similarity over the full sampled range.

4. The fixed S>=0.8 threshold does NOT cleanly identify physical proximity: >5m pairs still have 62.8% reward-positive.

5. The MINav reward is defined in visual representation space rather than physical distance. DINOv3 CLS provides useful local visual discrimination but exhibits substantial spatial aliasing at larger distances. The representation should therefore be interpreted as a visual goal-similarity signal, not a calibrated physical-distance metric.

**Note:** Physical distance was used ONLY as an offline diagnostic and never enters state, goal, reward, done, or hindsight sampling.

### FINAL DECISION

**REPRESENTATION STATUS:**
    ACCEPTABLE FOR REPRODUCTION / RL EXPERIMENTATION

**TD3+BC:**
    PROCEED
