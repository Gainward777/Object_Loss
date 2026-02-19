from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------
# shape helpers + morphology
# ---------------------------

def _as_4d(x: torch.Tensor, like: torch.Tensor, *, name: str) -> torch.Tensor:
    """
    Convert x to [B,1,H,W] float on like.device.
    Accepts [H,W], [B,H,W], [B,1,H,W].
    """
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x = x.unsqueeze(1)
    elif x.dim() == 4:
        pass
    else:
        raise ValueError(f"{name} must have 2..4 dims, got {x.shape}")

    if x.shape[0] != like.shape[0]:
        if x.shape[0] == 1:
            x = x.expand(like.shape[0], -1, -1, -1)
        else:
            raise ValueError(f"{name} batch {x.shape[0]} != like batch {like.shape[0]}")

    return x.to(device=like.device, dtype=torch.float32)


def dilate(mask01: torch.Tensor, r: int) -> torch.Tensor:
    """Dilation via max_pool2d (mask01: [B,1,H,W])."""
    if r <= 0:
        return mask01
    k = 2 * r + 1
    return F.max_pool2d(mask01, kernel_size=k, stride=1, padding=r)


def erode(mask01: torch.Tensor, r: int) -> torch.Tensor:
    """Erosion via duality: erode(m)=1-dilate(1-m)."""
    if r <= 0:
        return mask01
    k = 2 * r + 1
    return 1.0 - F.max_pool2d(1.0 - mask01, kernel_size=k, stride=1, padding=r)


def boundary_band(mask_bin: torch.Tensor, r: int) -> torch.Tensor:
    """Boundary band: dilate XOR erode, approximated as clamp(dilate - erode)."""
    if r <= 0:
        return torch.zeros_like(mask_bin)
    d = dilate(mask_bin, r)
    e = erode(mask_bin, r)
    return (d - e).clamp_(0.0, 1.0)


# ---------------------------
# depth gradients (finite differences)
# ---------------------------

def depth_gradients(depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    depth: [B,1,H,W]
    returns: (dx, dy) each [B,1,H,W] with same shape
    """
    dx = depth[..., :, 1:] - depth[..., :, :-1]
    dy = depth[..., 1:, :] - depth[..., :-1, :]

    dx = F.pad(dx, (0, 1, 0, 0))  # pad last column
    dy = F.pad(dy, (0, 0, 0, 1))  # pad last row
    return dx, dy


# ---------------------------
# weighted robust loss
# ---------------------------

def weighted_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    beta: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Weighted SmoothL1 (Huber-like) loss, reduction = weighted mean.

    pred/target/weight: [B,1,H,W]
    """
    per = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    wsum = weight.sum().clamp_min(eps)
    return (per * weight).sum() / wsum


@dataclass
class DepthConsistencyTerms:
    total: torch.Tensor
    inside: torch.Tensor
    outside: torch.Tensor
    edge: torch.Tensor


def depth_consistency_loss(
    d_pred: torch.Tensor,
    d_tgt: torch.Tensor,
    mask: torch.Tensor,
    *,
    conf_pred: Optional[torch.Tensor] = None,
    # region shaping
    outside_dilate_r: int = 1,   # exclude a small neighborhood around the mask from "outside"
    edge_band_r: int = 1,        # width of boundary band for gradient term
    # robust loss params
    beta: float = 0.1,
    # weights of terms
    w_inside: float = 1.0,
    w_outside: float = 0.5,
    w_edge: float = 0.5,
    # numerical
    eps: float = 1e-8,
) -> DepthConsistencyTerms:
    """
    Depth consistency loss when outside(mask) in target == context, so only d_tgt is needed.

    Inputs (already prepared to same convention/scale):
      d_pred: predicted depth/disparity for generated image   [B,1,H,W]
      d_tgt:  GT depth/disparity for target render            [B,1,H,W]
      mask:   insertion mask (1 inside insertion region)      [B,1,H,W] or [B,H,W] or [H,W]
      conf_pred (optional): confidence map for d_pred in [0,1], same shape as d_pred (or broadcastable)

    Terms:
      - Inside mask:   d_pred ≈ d_tgt  (strong)
      - Outside mask:  d_pred ≈ d_tgt  (weaker; outside == context)
      - Boundary band: match gradients ∇d_pred ≈ ∇d_tgt to enforce correct depth edges
    """
    like = d_pred if d_pred.dim() == 4 else d_pred.unsqueeze(1)
    d_pred = _as_4d(d_pred, like, name="d_pred")
    d_tgt  = _as_4d(d_tgt,  like, name="d_tgt")
    m      = _as_4d(mask,   like, name="mask").clamp(0.0, 1.0)
    m_bin  = (m > 0.5).to(m.dtype)

    # validity (ignore NaN/Inf)
    valid = (torch.isfinite(d_pred) & torch.isfinite(d_tgt)).to(d_pred.dtype)

    # optional confidence weighting
    if conf_pred is not None:
        c = _as_4d(conf_pred, like, name="conf_pred").clamp(0.0, 1.0)
        valid = valid * c

    # regions
    outside_region = 1.0 - dilate(m_bin, outside_dilate_r)  # "sure outside"
    inside_region  = m                                      # soft inside is ok
    band_region    = boundary_band(m_bin, edge_band_r)

    # weights per term
    w_in   = inside_region * valid
    w_out  = outside_region * valid
    w_band = band_region * valid

    # inside / outside depth matching
    L_in  = weighted_smooth_l1(d_pred, d_tgt, w_in,  beta=beta, eps=eps)
    L_out = weighted_smooth_l1(d_pred, d_tgt, w_out, beta=beta, eps=eps)

    # boundary gradient matching
    dx_p, dy_p = depth_gradients(d_pred)
    dx_t, dy_t = depth_gradients(d_tgt)

    L_edge_x = weighted_smooth_l1(dx_p, dx_t, w_band, beta=beta, eps=eps)
    L_edge_y = weighted_smooth_l1(dy_p, dy_t, w_band, beta=beta, eps=eps)
    L_edge = 0.5 * (L_edge_x + L_edge_y)

    total = w_inside * L_in + w_outside * L_out + w_edge * L_edge
    return DepthConsistencyTerms(total=total, inside=L_in, outside=L_out, edge=L_edge)
