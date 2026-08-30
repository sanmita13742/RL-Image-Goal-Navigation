# MINav Representation & Phase 17 Test Report

Before running the full 72,000-frame processing, we conducted a live probe of the DINOv3 model and executed the Phase 17 test (1,000 frames).

> [!IMPORTANT]
> Please review this report and answer the open questions at the bottom. The full dataset processing is paused until these decisions are finalized.

---

## 1. DINOv3 Live Verification

We passed a real 448×784 RGB image from `segment_000` through `vit_small_patch16_dinov3`. 

**Findings:**
- **Input:** 448 × 784, `torch.float32`, range `[0.000, 0.655]`
- **Tokens:** 1,377 total tokens
  - 5 Special Tokens (1 CLS + 4 register tokens)
  - 1,372 Patch Tokens
- **Patch Grid:** Exactly matches spatial dims: 28 × 49 = 1,372
- **Embedding Dim (D):** 384
- **Shapes:**
  - `model(x)` (classification head): `[1, 384]`
  - `model.forward_features(x)`: `[1, 1377, 384]`
- **Frozen Status:** `timm.create_model(..., pretrained=True)` does **not** freeze weights by default. Our `FrozenDINOv3` class explicitly sets `requires_grad=False` for all parameters.

---

## 2. Image Preprocessing

The implementation choice in the provided DINOv3 codebase uses only:
```python
transforms.Resize((448, 784))
transforms.ToTensor()
```
**Observation:** Standard ImageNet normalization (subtracting mean, dividing by std) is **not** applied. The pixel values remain in `[0, 1]`.

---

## 3. Geometric $p$

We updated `configs/minav_hindsight.yaml` to explicitly document the implementation choice for $p$. 
- **Value:** $p = 0.99$
- **Expected Offset:** $E[K] = \frac{1}{1 - 0.99} = 100$ frames.
- **Physical Time:** At 10 Hz, goals are sampled on average 10 seconds into the future. 
- This value is marked as an **Implementation Choice**; it is not specified in the paper.

---

## 4. Phase 17 Test Output (1,000 frames)

We ran the pipeline on the first 1,000 frames of the dataset, pausing after Phase 5 (Valid Goal Construction).

**Dataset Continuity**
- `global_step` is strictly continuous (0..999).
- `sim_time` is strictly monotonic (0.00s .. 99.9s).
- Segment boundary checks passed (max positional jump at boundaries is 0.069m).

**SSD Filtering**
- **Threshold:** $\delta_{SSD} = 0.02$ (Paper Specified)
- **Crop:** 14 × 25 center crop (Implementation Choice)
- **SSD Range:** `[0.0122, 0.0274]` (Mean: `0.0234` ± `0.0027`)
- **Valid Goals:** 885 / 1,000 (88.5%)

**Patch Grid Storage**
- Stored the full `[1000, 28, 49, 384]` grid in `float16`.
- **Disk Size:** 1.05 GB.
- *(Note: Extrapolating this to 72,000 frames would yield ~75 GB for the final dataset.)*

---

## 5. $\phi(o)$ / Visual Representation Candidates

The paper states: *"$\phi$ uses normalized visual feature extraction / final hidden-layer patch embeddings."* 

Since we have the full `[28, 49, 384]` grid, we must decide how to represent $\phi(o)$ for the **state representation**, the **goal**, and the **cosine similarity reward** $S(s_t, g)$.

Here are the three defensible candidates:

### Candidate A: L2-Normalized Mean-Pooled Patch Tokens (Recommended)
- **Shape:** `[384]`
- **Method:** L2-normalize each patch token, mean-pool across all 1,372 tokens, and L2-normalize the resulting vector.
- **Reward Computation:** $S(s_t, g)$ becomes a simple dot product between two 384-D unit vectors.
- **Pros:** Extremely fast (state is a `[4, 384]` tensor, goal is `[384]`), caches easily (108 MB for 72k frames).
- **Cons:** Loses explicit spatial geometry, though DINOv3's self-attention means global vectors retain strong spatial information.

### Candidate B: Full Patch Grid
- **Shape:** `[28, 49, 384]` or flattened `[1372, 384]`
- **Method:** Retain all patch embeddings.
- **Reward Computation:** $S(s_t, g)$ requires computing dense similarity (e.g., mean of patch-wise cosine similarities, or flattened vector cosine similarity).
- **Pros:** Retains exact spatial mapping.
- **Cons:** Very large memory footprint for replay buffer (`[4, 1372, 384]` per state), 75 GB dataset cache, slower reward computation.

### Candidate C: CLS Token
- **Shape:** `[384]`
- **Method:** Extract the special CLS token `feats[:, 0]`.
- **Pros:** Standard for ViT global representations.
- **Cons:** DINOv2/v3 papers explicitly state that mean-pooled patch tokens are superior to the CLS token for dense visual retrieval tasks.

*(Note: SSD computation will always use the unpooled spatial crop, regardless of which candidate is chosen for $\phi(o)$).*

---

## User Review Required

Before generating the final datasets, please confirm the following:

1. **Representation:** Which candidate (A, B, or C) should be used for $\phi(o)$ in the state, goal, and reward computation?
2. **Normalization:** Should we stick with the implementation choice of `ToTensor()` (no ImageNet normalization)?
3. **Geometric $p$:** Do you approve using $p=0.99$ (expected offset 10 seconds)?
4. **SSD Crop:** Do you approve the 14×25 center crop for SSD filtering?

Once confirmed, we will finalize the encoder and run the full 72,000-frame pipeline.
