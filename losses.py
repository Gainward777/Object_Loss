from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from utils import *




# ---------------------------
# 1) Weighted masked MSE (latent)
# ---------------------------

def weighted_masked_mse_latent(
    z_pred: torch.Tensor,
    z_tgt: torch.Tensor,
    mask_latent: torch.Tensor,
    *,
    band_r: int = 1,
    w_in: float = 1.0,
    w_band: float = 6.0,
    w_out: float = 0.2,
    channel_reduce: Literal["mean", "sum"] = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Weighted masked MSE in latent space.

    z_pred, z_tgt: [B,C,H,W]
    mask_latent:   [B,1,H,W] (or broadcastable) float in [0,1]

    Weights:
      inside object: w_in
      boundary band: w_band
      outside (far): w_out (outside dilated mask)
    """
    if z_pred.shape != z_tgt.shape:
        raise ValueError(f"z_pred {z_pred.shape} != z_tgt {z_tgt.shape}")

    m = _as_4d_mask(mask_latent, z_pred)
    m_bin = (m > 0.5).to(m.dtype)

    dil = dilate(m_bin, band_r)
    band = boundary_band(m_bin, band_r)

    W = w_in * m + w_band * band + w_out * (1.0 - dil)

    diff2 = (z_pred - z_tgt).pow(2)
    if channel_reduce == "sum":
        per_pix = diff2.sum(dim=1, keepdim=True)
    elif channel_reduce == "mean":
        per_pix = diff2.mean(dim=1, keepdim=True)
    else:
        raise ValueError("channel_reduce must be 'mean' or 'sum'")

    return (W * per_pix).sum() / (W.sum() + eps)


# ---------------------------
# 2) Background preservation MSE (latent)
# ---------------------------

def background_preservation_mse_latent(
    z_pred: torch.Tensor,
    z_ctx: torch.Tensor,
    mask_latent: torch.Tensor,
    *,
    band_r: int = 1,
    channel_reduce: Literal["mean", "sum"] = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Penalize changes OUTSIDE the (dilated) insertion area:
      || z_pred - z_ctx ||^2 on outside region.

    z_pred, z_ctx: [B,C,H,W]
    mask_latent:   [B,1,H,W] float in [0,1]
    """
    if z_pred.shape != z_ctx.shape:
        raise ValueError(f"z_pred {z_pred.shape} != z_ctx {z_ctx.shape}")

    m = _as_4d_mask(mask_latent, z_pred)
    m_bin = (m > 0.5).to(m.dtype)
    outside = 1.0 - dilate(m_bin, band_r)  # [B,1,H,W]

    diff2 = (z_pred - z_ctx).pow(2)
    if channel_reduce == "sum":
        per_pix = diff2.sum(dim=1, keepdim=True)
    elif channel_reduce == "mean":
        per_pix = diff2.mean(dim=1, keepdim=True)
    else:
        raise ValueError("channel_reduce must be 'mean' or 'sum'")

    return (outside * per_pix).sum() / (outside.sum() + eps)


# ---------------------------
# 3) Soft change mask + Dice (latent)
# ---------------------------

@dataclass
class ChangeMaskStats:
    tau: torch.Tensor  # [B,1,1,1] or scalar
    s: torch.Tensor    # [B,1,1,1] or scalar


def soft_change_mask_from_latents(
    z_pred: torch.Tensor,
    z_ctx: torch.Tensor,
    *,
    mask_latent: Optional[torch.Tensor] = None,
    band_r: int = 1,
    k_sigma: float = 2.0,
    tau: Optional[torch.Tensor] = None,
    s: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, ChangeMaskStats, torch.Tensor]:
    """
    Build soft change mask \hat M from latent delta magnitude:
      D = || z_pred - z_ctx ||_2  (over channels)
      \hat M = sigmoid((D - tau) / s)

    If tau/s not provided, estimate from background pixels:
      tau = mean_bg + k_sigma * std_bg
      s   = std_bg

    Returns:
      M_hat: [B,1,H,W]
      stats: tau/s
      D:     [B,1,H,W]
    """
    if z_pred.shape != z_ctx.shape:
        raise ValueError(f"z_pred {z_pred.shape} != z_ctx {z_ctx.shape}")

    # D: [B,1,H,W]
    D = torch.linalg.vector_norm(z_pred - z_ctx, ord=2, dim=1, keepdim=True)

    if tau is None or s is None:
        if mask_latent is not None:
            m = _as_4d_mask(mask_latent, z_pred)
            m_bin = (m > 0.5).to(m.dtype)
            bg = 1.0 - dilate(m_bin, band_r)  # sure-background region

            bg_sum = bg.sum(dim=(2, 3), keepdim=True)  # [B,1,1,1]
            # fallback if bg is empty
            bg_sum_safe = torch.where(bg_sum < 1.0, torch.ones_like(bg_sum), bg_sum)
            bg_safe = torch.where(bg_sum < 1.0, torch.ones_like(bg), bg)

            mu = (D * bg_safe).sum(dim=(2, 3), keepdim=True) / (bg_sum_safe + eps)
            var = ((D - mu).pow(2) * bg_safe).sum(dim=(2, 3), keepdim=True) / (bg_sum_safe + eps)
            sigma = torch.sqrt(var + eps)
        else:
            mu = D.mean(dim=(2, 3), keepdim=True)
            sigma = D.std(dim=(2, 3), keepdim=True) + eps

        if tau is None:
            tau = mu + k_sigma * sigma
        if s is None:
            s = sigma

    M_hat = torch.sigmoid((D - tau) / (s + eps))
    return M_hat, ChangeMaskStats(tau=tau, s=s), D


def dice_loss(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Soft Dice loss over batch.
    pred_mask, target_mask: [B,1,H,W] floats
    """
    # ensure same shape
    pred = pred_mask.float()
    tgt = target_mask.float()
    if pred.shape != tgt.shape:
        raise ValueError(f"pred {pred.shape} != tgt {tgt.shape}")

    # per-sample
    num = 2.0 * (pred * tgt).sum(dim=(1, 2, 3)) + eps
    den = pred.sum(dim=(1, 2, 3)) + tgt.sum(dim=(1, 2, 3)) + eps
    return (1.0 - num / den).mean()


# ---------------------------
# 4) Area ratio loss
# ---------------------------

def area_ratio_loss(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Penalize mismatch of areas:
      (log(A_pred / A_tgt))^2

    pred_mask, target_mask: [B,1,H,W] floats
    """
    pred = pred_mask.float()
    tgt = target_mask.float()
    if pred.shape != tgt.shape:
        raise ValueError(f"pred {pred.shape} != tgt {tgt.shape}")

    A_pred = pred.sum(dim=(1, 2, 3))
    A_tgt = tgt.sum(dim=(1, 2, 3))
    return torch.log((A_pred + eps) / (A_tgt + eps)).pow(2).mean()


# ---------------------------
# 5) Boundary loss (signed distance map)
# ---------------------------

def signed_distance_map_from_mask(
    mask_bin: torch.Tensor,
    *,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Compute signed distance map (SDM) from binary mask:
      positive outside, negative inside, ~0 on boundary.

    This follows the common construction used with Boundary Loss:
      sdm = edt(negmask)*negmask - (edt(posmask)-1)*posmask
    (where posmask is foreground). :contentReference[oaicite:1]{index=1}

    mask_bin: [B,1,H,W] bool/0-1 float (foreground=1)
    returns:  [B,1,H,W] float32 on same device
    """
    try:
        import numpy as np
        from scipy.ndimage import distance_transform_edt as edt
    except Exception as e:
        raise ImportError(
            "signed_distance_map_from_mask requires scipy. "
            "Install with: pip install scipy"
        ) from e

    m = mask_bin
    if m.dim() != 4 or m.shape[1] != 1:
        # bring to [B,1,H,W]
        dummy = torch.zeros((m.shape[0] if m.dim() > 2 else 1, 1, m.shape[-2], m.shape[-1]),
                            device=m.device, dtype=torch.float32)
        m = _as_4d_mask(m, dummy)
    m = (m > 0.5)

    B, _, H, W = m.shape
    m_np = m.detach().cpu().numpy().astype(bool)

    out = torch.empty((B, 1, H, W), dtype=torch.float32)

    for b in range(B):
        pos = m_np[b, 0]          # foreground
        neg = ~pos                # background

        # Handle degenerate cases
        if pos.any() and neg.any():
            dist_neg = edt(neg)               # distance for background pixels to nearest foreground
            dist_pos = edt(pos)               # distance for foreground pixels to nearest background
            sdm = dist_neg * neg - (dist_pos - 1.0) * pos
        elif pos.any():  # all foreground
            dist_pos = edt(pos)
            sdm = -(dist_pos - 1.0) * pos
        else:            # all background
            dist_neg = edt(neg)
            sdm = dist_neg * neg

        if normalize:
            mx = float(abs(sdm).max()) if sdm.size else 0.0
            if mx > 0:
                sdm = sdm / mx

        out[b, 0] = torch.from_numpy(sdm.astype("float32"))

    return out.to(device=mask_bin.device)


def boundary_loss(
    pred_soft_mask: torch.Tensor,
    signed_distance_map: torch.Tensor,
) -> torch.Tensor:
    """
    Boundary loss core:
      mean( pred_soft_mask * SDM )
    where SDM is signed distance map of the GT mask. :contentReference[oaicite:2]{index=2}

    pred_soft_mask: [B,1,H,W] float in [0,1]
    signed_distance_map: [B,1,H,W] float
    """
    p = pred_soft_mask.float()
    phi = signed_distance_map.float()
    if p.shape != phi.shape:
        raise ValueError(f"pred {p.shape} != sdm {phi.shape}")
    # note: this loss can be negative at optimum; that's expected for this formulation.
    return (p * phi).mean()
