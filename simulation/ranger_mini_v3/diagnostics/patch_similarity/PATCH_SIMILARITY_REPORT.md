# MINav Patch-Wise Similarity Ablation Report

*Patch-wise similarity was evaluated as an ablation because the paper specifies DINOv3 final hidden-layer patch embeddings but does not explicitly specify the exact transformation used to obtain the image-level phi(o) used in the cosine reward.*

## 1. Required Comparison Table

| Metric | CLS | Corresponding Patch | Best-Match Patch | Symmetric Best-Match |
|---|---:|---:|---:|---:|
| t->t+1 | 0.9800 | 0.9789 | 0.9876 | 0.9876 |
| t->t+10 | 0.9094 | 0.8942 | 0.9317 | 0.9320 |
| t->t+100 | 0.8158 | 0.8162 | 0.8647 | 0.8646 |
| t->t+500 | 0.8049 | 0.7970 | 0.8504 | 0.8487 |
| random distant | 0.8152 | 0.8039 | 0.8555 | 0.8554 |
| geometric state->goal | 0.8568 | 0.8476 | 0.8920 | 0.8921 |
| uniform state->goal | 0.8344 | 0.8188 | 0.8690 | 0.8679 |
| random reward=1 % | 66.2% | 56.7% | 91.8% | 91.1% |
| geometric reward=1 % | 75.5% | 73.3% | 93.9% | 93.8% |
| uniform reward=1 % | 73.0% | 64.3% | 96.2% | 94.8% |

## 2. Spatial Discrimination Table

| Physical distance | CLS | Corresponding Patch | Best-Match Patch |
|---|---:|---:|---:|
| 0-0.25m | 0.8881 | 0.8883 | 0.9207 |
| 0.25-0.5m | 0.8633 | 0.8568 | 0.8975 |
| 0.5-1m | 0.8453 | 0.8367 | 0.8841 |
| 1-2m | 0.8339 | 0.8170 | 0.8671 |
| 2-5m | 0.8144 | 0.7998 | 0.8527 |
| >5m | 0.8158 | 0.8064 | 0.8572 |

## 3. Final Verdict
PATCH-WISE STATUS:
    Corresponding patch:
        SAME / SLIGHTLY BETTER (Discriminative but brittle) than CLS
    Best-match patch:
        WORSE (Too flexible, loses layout) than CLS
    Symmetric best-match:
        WORSE (Too flexible, loses layout) than CLS

1. **Which method gives the strongest spatial discrimination?**
   Corresponding Patch. It has the lowest similarity at >5m (0.8064) and the lowest false-positive rate on random distant frames (56.7% vs CLS's 66.2%).
2. **Which method is most robust to viewpoint changes?**
   Best-Match Patch (t->t+10 similarity is 0.9317). However, this extreme flexibility comes at the cost of breaking spatial layout constraint.
3. **Which method produces the fewest obvious distant false positives?**
   Corresponding Patch (56.7%), followed closely by CLS (66.2%). Best-Match methods fail completely here (91%+ false positives).
4. **Does patch information appear useful beyond CLS?**
   Not significantly for a simple cosine-similarity reward. While strictly enforcing patch-to-patch correspondence improves rejection of distant frames slightly, it is brittle to viewpoint shifts.
5. **Is there evidence that the current CLS representation is sufficient?**
   Yes. The CLS token natively balances spatial layout awareness (rejecting >5m distant frames decently well) with viewpoint robustness, matching the corresponding-patch performance very closely without requiring 1372x more features per frame.
6. **Should we continue with CLS or investigate patch-aware phi further?**
   Continue with CLS. The marginal discrimination gains from corresponding-patch evaluation do not justify the massive memory and computational overhead required to store and compute dot products over 1,372 tokens per frame in an RL training loop. Best-match (Chamfer) destroys necessary spatial constraints.
