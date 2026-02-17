# Latent-space geometry losses for object inpainting (Flux Kontext LoRA)

This module provides **drop-in PyTorch loss functions** that help **stabilize object scale and boundaries** when you train a LoRA to inpaint/insert **a single product/object** into a **predefined mask region**.

The core idea is simple:

- **Reconstruction in latent space** (MSE) is necessary but not sufficient: the model can still “cheat” by shrinking/expanding the object while keeping the overall reconstruction plausible.
- Adding **mask-geometry losses** forces the model to **fill exactly the target mask** (area + boundary) and to **avoid unintended edits outside** the insertion region.

These losses are designed for pipelines that train on `(context → target)` pairs with a known insertion mask (exactly your setting with Flux Kontext).

---

## What’s included

### Loss components (latent space)

| Component | Function | Purpose |
|---|---|---|
| Weighted Masked MSE | `weighted_masked_mse_latent` | Main reconstruction loss, with strong emphasis on the **mask boundary band** to prevent shrink/expand. |
| Background preservation | `background_preservation_mse_latent` | Penalizes changes **outside** (dilated) mask: prevents “compensating” scale errors by altering background. |
| Soft change mask (helper) | `soft_change_mask_from_latents` | Builds a differentiable **“where did the model change the image”** mask from latent deltas. |
| Dice loss | `dice_loss` | Forces the **change mask** to match the **target mask** → very strong scale stabilizer. |
| Area ratio loss | `area_ratio_loss` | Direct penalty on **area mismatch** (shrink/expand). |
| Signed distance map (SDM) | `signed_distance_map_from_mask` | Computes signed distances used by boundary loss (requires SciPy). |
| Boundary loss | `boundary_loss` | Encourages correct **contour alignment** using SDM (Boundary Loss style). |

### Mask utilities

- `downsample_mask_area` — area-preserving downsample from pixel mask to latent mask
- `dilate`, `erode`, `boundary_band` — simple morphology on masks using pooling

---

## Installation / dependencies

Minimal:

```bash
pip install torch
```

For `signed_distance_map_from_mask` (boundary loss SDM) you also need SciPy:

```bash
pip install scipy
```

Optional: if you want a GPU-accelerated EDT alternative, CuPy provides `cupyx.scipy.ndimage.distance_transform_edt` (API mirrors SciPy).

---

## Conventions & shapes

All losses are implemented for **latent tensors** shaped:

- `z_*`: `[B, C, H, W]` (batch, channels, latent height, latent width)
- `mask_latent`: `[B, 1, H, W]` in **[0, 1]**

Pixel-space masks `M` (e.g. `[H_img, W_img]`) should be converted to latent resolution via:

```python
M_lat = downsample_mask_area(M_px, (z_pred.shape[-2], z_pred.shape[-1]))
```

This uses `interpolate(mode="area")`, which behaves like average pooling when downsampling.

---

## Why these losses stabilize object size

In your setting:

- A LoRA is trained **per object**.
- Insertion is constrained to a **known mask region**.
- Target object mask **must match** insertion mask (context→target supervision).

So the best “size anchor” is not a generic segmenter, but **the target mask itself**.

Key mechanisms:

1. **Soft change mask**: if the model inserts a too-small object, it will not change enough pixels inside the target mask → Dice/Area penalties rise.
2. **Boundary emphasis**: errors near the mask boundary are punished more strongly than interior errors → shrink/expand becomes expensive.
3. **Background preservation**: stops the model from reducing loss by editing outside the intended region.

---

## API reference

### `weighted_masked_mse_latent(z_pred, z_tgt, mask_latent, ...)`

**Goal:** reconstruction MSE with spatial weighting:
- inside mask: normal weight
- around boundary: high weight
- outside (far): small weight

Key params:
- `band_r` *(int)*: boundary band radius at latent resolution (start with `1`)
- `w_in, w_band, w_out` *(float)*: weights (good start: `1.0, 6.0, 0.2`)
- `channel_reduce`: `"mean"` (default) or `"sum"`

Returns: scalar loss (`torch.Tensor`).

---

### `background_preservation_mse_latent(z_pred, z_ctx, mask_latent, ...)`

**Goal:** discourage edits outside the insertion area:

- builds `outside = 1 - dilate(mask_bin, band_r)`
- penalizes `||z_pred - z_ctx||^2` on `outside`

Key params:
- `band_r`: same radius as above (start `1`)

Returns: scalar loss.

---

### `soft_change_mask_from_latents(z_pred, z_ctx, ...)`

**Goal:** produce a differentiable mask `M_hat` showing where latents changed:

1. `D(x) = ||z_pred - z_ctx||_2` (over channels)
2. `M_hat = sigmoid((D - tau) / s)`

If `tau` and `s` are not provided, they’re estimated from **sure-background** pixels:

- `tau = mean_bg + k_sigma * std_bg`
- `s = std_bg`

Returns: `(M_hat, stats, D)` where:
- `M_hat`: `[B,1,H,W]` in `[0,1]`
- `stats.tau`, `stats.s`: tensors shaped `[B,1,1,1]` (per-sample)
- `D`: `[B,1,H,W]`

Recommended:
- `k_sigma = 2.0`
- pass `mask_latent` so the background statistics are robust

---

### `dice_loss(pred_mask, target_mask)`

Standard **soft Dice**:

- `pred_mask`, `target_mask` are `[B,1,H,W]` floats.
- returns batch mean Dice loss.

Use case:
- `dice_loss(M_hat, M_lat)`

---

### `area_ratio_loss(pred_mask, target_mask)`

Penalizes area mismatch with a log-ratio:

\[
(\log(A_{pred} / A_{tgt}))^2
\]

Use case:
- `area_ratio_loss(M_hat, M_lat)`

---

### `signed_distance_map_from_mask(mask_bin, normalize=False)`

Computes a **signed distance map** (SDM):

- positive outside object
- negative inside object
- ~0 on boundary

Input:
- `mask_bin`: `[B,1,H,W]` bool or 0/1 float
- `normalize=True` optionally scales SDM by its max absolute value (often helpful to keep loss magnitudes stable)

This function uses **SciPy** `distance_transform_edt` under the hood.

---

### `boundary_loss(pred_soft_mask, signed_distance_map)`

Boundary Loss core:

\[
\text{mean}( p(x)\,\phi(x) )
\]

- `p` is your soft mask (`M_hat`)
- `phi` is the SDM of the **ground-truth** mask

Note:
- This loss can become negative near optimum; that’s expected for this formulation. In practice you combine it with region losses (Dice/MSE) and use a moderate weight.

---

## Recommended “starter” total loss

A practical default (tuned for latent-space training where MSE is the backbone):

```python
L = (
    1.00 * L_mse +
    0.60 * L_bg +
    0.90 * L_dice +
    0.20 * L_area +
    0.40 * L_bnd
)
```

Notes:
- If the object still occasionally “shrinks”: increase `w_band` to `8–10`.
- If you see background drift: increase `L_bg` weight.
- If boundaries are unstable: raise `L_bnd` slightly **or** increase `band_r`/`w_band`.

---

## End-to-end usage example

```python
import torch

# z_ctx: [B,C,H,W] context latents
# z_tgt: [B,C,H,W] target latents
# z_pred: [B,C,H,W] predicted latents from your model
# M_px:  [B,1,H_img,W_img] or [B,H_img,W_img] or [H_img,W_img] target mask in pixel space

from losses import (
    downsample_mask_area,
    weighted_masked_mse_latent,
    background_preservation_mse_latent,
    soft_change_mask_from_latents,
    dice_loss,
    area_ratio_loss,
    signed_distance_map_from_mask,
    boundary_loss,
)

# 1) mask -> latent resolution
M_lat = downsample_mask_area(M_px, (z_pred.shape[-2], z_pred.shape[-1]))  # [B,1,H,W]

# 2) reconstruction / preservation
L_mse = weighted_masked_mse_latent(z_pred, z_tgt, M_lat, band_r=1, w_in=1.0, w_band=6.0, w_out=0.2)
L_bg  = background_preservation_mse_latent(z_pred, z_ctx, M_lat, band_r=1)

# 3) soft change mask from latents
M_hat, stats, D = soft_change_mask_from_latents(z_pred, z_ctx, mask_latent=M_lat, band_r=1, k_sigma=2.0)

# 4) geometry losses
L_dice = dice_loss(M_hat, M_lat)
L_area = area_ratio_loss(M_hat, M_lat)

phi = signed_distance_map_from_mask(M_lat > 0.5, normalize=True)
L_bnd = boundary_loss(M_hat, phi)

# 5) total
L = 1.00*L_mse + 0.60*L_bg + 0.90*L_dice + 0.20*L_area + 0.40*L_bnd
L.backward()
```

---

## Practical tips & troubleshooting

### 1) Mask downsampling matters
Use `mode="area"` (average/area pooling). Nearest-neighbor can distort area and make `L_area` noisy.

### 2) Choose `band_r` at latent resolution
At latent sizes (e.g., 64×64), `band_r=1` is already a meaningful strip. Increasing to `2` can help if objects still “float”, but may overconstrain very small masks.

### 3) Stable `tau/s` for change masks
If `tau` is too low → everything becomes “changed” → Dice becomes trivial.
If too high → nothing is “changed” → Dice saturates.
The background-statistics estimation in `soft_change_mask_from_latents` usually avoids hand-tuning.

### 4) Boundary loss scaling
SDM magnitudes grow with spatial size. If you train on multiple latent resolutions, set `normalize=True` in `signed_distance_map_from_mask` to keep scales comparable.

### 5) Performance
- `signed_distance_map_from_mask` runs on CPU via SciPy. For efficiency:
  - precompute SDMs for masks (if masks repeat), or
  - compute SDM less frequently, or
  - use a GPU EDT implementation (e.g., CuPy’s `cupyx.scipy.ndimage.distance_transform_edt`).

---

## References

- **Boundary Loss** (Kervadec et al.): “Boundary loss for highly unbalanced segmentation” (arXiv:1812.07032).  
- **Latent diffusion** background: “High-Resolution Image Synthesis with Latent Diffusion Models” (Rombach et al., CVPR 2022).  
- **SciPy EDT**: `scipy.ndimage.distance_transform_edt` documentation.  
- **MultipleNegativesRankingLoss / InfoNCE**: Sentence-Transformers loss docs (useful if you add an embedding/contrastive term).

(Links are included in the docs and can be clicked on GitHub.)
