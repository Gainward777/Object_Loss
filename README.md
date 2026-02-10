# Embedding-only losses for Flux Kontext LoRA object replacement

A small, dependency-light collection of **PyTorch loss functions** designed for **image editing / object replacement** training setups (e.g., **Flux Kontext LoRA** trained on **before/after pairs**).

The key idea: **all losses operate on precomputed embeddings** (global vectors, patch tokens, or spatial feature maps).  
You can plug in **any encoder** you like (VAE latents, decoded-image encoders like CLIP/DINOv2, diffusion U-Net features, etc.) and freely combine/weight losses in your training loop.

---

## Why this repo exists

When training a before/after editing LoRA for object replacement (e.g., *replace a chair with a learned stool*), common failure modes are:

1. **No edit happens** (model outputs the original input).
2. **Geometry is distorted** (stool shape collapses or warps).
3. **Overlay / double object** (new object is painted on top of the old one).

This repo provides small, composable losses to address each:

- `loss_no_edit_identity(...)` — penalize staying too close to the **source**.
- `loss_geometry_match(...)` — pull predicted features toward **ground truth** (choose a shape-sensitive encoder).
- `loss_overlay_avoidance(...)` — penalize being close to an **overlay negative**.

> You decide how to compute embeddings and how to mix the losses.

---

## What you get

- Generic metric-learning building blocks:
  - Contrastive (pair) loss (Hadsell et al., 2006)
  - Triplet margin / hinge ranking (FaceNet-style)
  - Multi-negative ranking (hinge vs many negatives)
  - InfoNCE (in-batch or explicit negatives)
- Task-focused wrappers aligned with the three editing failure modes above
- Support for embeddings shaped as:
  - `[B, D]` global embeddings
  - `[B, P, D]` patch/token embeddings
  - `[B, H, W, D]` spatial feature grids (or any `[B, ..., D]`)

---

## Installation

Copy `losses.py` into your project, or install as a local package if you wrap it.

Minimal requirements:

- Python 3.9+
- PyTorch 2.x

---

## Quick start

Below is the typical training-time setup:

- `z_pred`   = embedding of your model output (predicted edit)
- `z_gt`     = embedding of your target (after / ground truth)
- `z_src`    = embedding of the source (before)
- `z_overlay`= embedding of a *synthetic negative* representing an undesirable "overlay" result

```python
import torch
from losses import (
    loss_no_edit_identity,
    loss_geometry_match,
    loss_overlay_avoidance,
)

# Example shapes:
# z_* could be [B, D] or [B, P, D] or [B, H, W, D]
z_pred    = torch.randn(8, 256)
z_gt      = torch.randn(8, 256)
z_src     = torch.randn(8, 256)
z_overlay = torch.randn(8, 256)

w_id, w_geo, w_ov = 1.0, 0.5, 0.7

L = (
    w_id  * loss_no_edit_identity(z_pred, z_gt, z_src, margin=0.25, dist="cos") +
    w_geo * loss_geometry_match(z_pred, z_gt, kind="l2") +
    w_ov  * loss_overlay_avoidance(z_pred, z_gt, z_overlay, margin=0.25, dist="cos")
)

L.backward()
```

### Where do I get `z_overlay`?

You create it in *your* pipeline. Common patterns:

- **Mask-based overlay**: keep the original object and paste in the new object (or vice-versa).
- **Composable negatives**: blend source & target in the edit region.
- **Hard negatives from a buffer**: previously bad generations (optional).

This repo intentionally stays out of image/latent processing; it only consumes embeddings.

---

## API overview

All public functions are pure-PyTorch and differentiable.

### Conventions

- `B` = batch size
- `D` = feature dimension
- Any additional dims `...` (patch tokens, spatial grids) are reduced using an aggregation strategy:
  - `agg="mean"` (default): average over non-batch dims
  - `agg="max"` or `agg="sum"`

Most losses return a scalar by default (`reduction="mean"`), but can return per-sample losses with `reduction="none"`.

---

## Reference: utilities

### `pair_distance(a, b, dist="cos", agg="mean") -> Tensor[B]`

Computes per-sample distance between embeddings:

- `dist="cos"`: cosine distance `1 - cos_sim` (inputs are normalized)
- `dist="l2"`: squared L2 distance summed over the last dimension

Accepts `a, b` with shape `[B, ..., D]` and returns `[B]` (reduces `...` using `agg`).

---

## Reference: generic losses

### `feature_match_loss(pred, target, kind="l2", agg="mean", reduction="mean", weights=None)`

Pull embeddings together (reconstruction in feature space).

- `kind="l2"` uses squared L2
- `kind="l1"` uses L1 over last dim
- `kind="cos"` uses cosine distance

Useful for: **geometry / structure preservation** (choose a structure-aware encoder).

---

### `contrastive_pair_loss(z1, z2, y, margin=1.0, dist="l2", ...)`

Classic contrastive loss (pairwise):

- `y=1`: positive (pull together)
- `y=0`: negative (push apart by `margin`)

Recommended if your data naturally comes as positive/negative pairs.

Reference: *Dimensionality Reduction by Learning an Invariant Mapping* (Hadsell et al., 2006).  
https://lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

---

### `triplet_margin_loss(anchor, positive, negative, margin=0.2, dist="cos", ...)`

Triplet hinge loss:

\[
L = \max(0, d(a,p) - d(a,n) + \text{margin})
\]

Reference: FaceNet (Schroff et al., 2015).  
https://arxiv.org/abs/1503.03832

---

### `margin_ranking_on_distances(d_pos, d_neg, margin=0.2, ...)`

Same hinge ranking, but expects **precomputed distances**:

\[
L = \max(0, d_{pos} - d_{neg} + \text{margin})
\]

This is conceptually aligned with PyTorch margin ranking loss; see:
- https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.margin_ranking_loss.html
- https://docs.pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html

---

### `multi_negative_ranking_loss(z_anchor, z_positive, z_negs, margin=0.2, ...)`

Ranking loss vs **multiple negatives**:

\[
L_i = \text{Agg}_j \max(0, d(a,p) - d(a,n_j) + m)
\]

- `z_negs` can be `[B, K, ..., D]` or a list of `K` tensors shaped like anchor.
- `neg_agg` controls aggregation over negatives: `"mean"`, `"max"`, `"sum"`.

Great when you have multiple negative constructions: identity, overlay, other hard negatives.

---

### `info_nce_loss(anchor, positive, negatives=None, temperature=0.07, ...)`

InfoNCE classification-style contrastive loss.

- If `negatives=None`, it uses **in-batch negatives** (anchor vs all positives).
- If `negatives` are provided, it uses them explicitly.

Helpful if you want a stronger "instance discrimination" objective with many negatives.

Background reading:
- https://lilianweng.github.io/posts/2021-05-31-contrastive/
- (More formal analysis) https://www.ijcai.org/proceedings/2022/0348.pdf

---

## Reference: task-focused wrappers (Flux Kontext object replacement)

These wrap the generic losses but keep your training code clean.

### 1) Identity avoidance — **fix “no edit happened”**

```python
loss_no_edit_identity(z_pred, z_gt, z_src, margin=0.2, dist="cos")
```

Implements:

- Anchor: `z_pred`
- Positive: `z_gt`
- Negative: `z_src`

So the model is penalized if its output is closer to **before** than to **after**.

---

### 2) Geometry matching — **fix “distorted stool”**

```python
loss_geometry_match(z_pred, z_gt, kind="l2")
```

This is a pure feature reconstruction / perceptual term.  
Best results usually come from encoders whose features preserve local structure (patch/spatial features).

---

### 3) Overlay avoidance — **fix “new object is painted on top”**

```python
loss_overlay_avoidance(z_pred, z_gt, z_overlay, margin=0.2, dist="cos")
```

Same idea as (1), but the negative is a **bad overlay** example.

---

## Practical recipes

### Recommended loss mix (starting point)

```python
L = (
    1.0 * loss_no_edit_identity(z_pred, z_gt, z_src, margin=0.25, dist="cos")
  + 0.5 * loss_geometry_match(z_pred, z_gt, kind="l2")
  + 0.7 * loss_overlay_avoidance(z_pred, z_gt, z_overlay, margin=0.25, dist="cos")
)
```

Then tune:

- Increase identity term if edits are too weak.
- Increase geometry term if the new object shape collapses.
- Increase overlay term if double-object artifacts persist.

### Encoder choices (rule of thumb)

Because these functions accept **only embeddings**, you can experiment quickly:

- **VAE latents**: good for color/texture consistency; may be weak on semantics.
- **CLIP image encoder**: good for semantic presence of the desired object.
- **DINO/DINOv2**: often good for geometry/structure (patch tokens).
- **Diffusion U-Net features**: can preserve spatial structure naturally; good for patch-wise losses.

You can also combine multiple encoders by computing multiple embedding sets and summing corresponding losses.

---

## Notes on shapes and aggregation

If your encoder outputs patch tokens, e.g. `[B, P, D]`, this repo:

- computes per-token distances
- reduces tokens using `agg` (default `"mean"`)

For spatial grids `[B, H, W, D]` it reduces over `H, W` similarly.

If you need more control (e.g., separate foreground/background weighting), compute separate embeddings upstream and pass them as separate loss terms.

---

## Testing

A minimal sanity check:

```python
import torch
from losses import loss_no_edit_identity

B, D = 4, 16
z_pred = torch.randn(B, D, requires_grad=True)
z_gt   = torch.randn(B, D)
z_src  = torch.randn(B, D)

L = loss_no_edit_identity(z_pred, z_gt, z_src)
L.backward()
assert z_pred.grad is not None
```

---

## References

- Hadsell, Chopra, LeCun (2006): Contrastive loss
  - https://lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
- Schroff et al. (2015): Triplet loss (FaceNet)
  - https://arxiv.org/abs/1503.03832
- PyTorch margin ranking loss docs:
  - https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.margin_ranking_loss.html
  - https://docs.pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html
- Contrastive learning / InfoNCE background:
  - https://lilianweng.github.io/posts/2021-05-31-contrastive/

---

## License

MIT (recommended). Replace with your preferred license.

---

## Contributing

PRs welcome. If you add new losses, please:

- keep functions embedding-only (no image decoding/cropping inside the loss)
- document expected shapes and reduction behavior
- include a minimal gradient sanity test
