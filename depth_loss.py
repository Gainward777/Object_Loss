from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from depth_utils import *

# -----------------------
# Depth-aware scale loss
# -----------------------

@dataclass
class DepthScaleLossInfo:
    s_pred: torch.Tensor      # [B,1,1,1]
    s_exp: torch.Tensor       # [B,1,1,1]
    depth_stat: torch.Tensor  # [B,1,1,1]


def depth_aware_scale_loss(
    pred_mask01: torch.Tensor,
    depth_map: torch.Tensor,
    f_px: torch.Tensor,
    k_obj: torch.Tensor,
    *,
    depth_mode: Literal["inv_depth", "depth"] = "inv_depth",
    depth_stat: Literal["median", "mean"] = "median",
    scale_measure: Literal["sqrt_area"] = "sqrt_area",
    detach_depth: bool = True,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, DepthScaleLossInfo]:
    """
    Depth-aware scale loss that ties predicted object scale to scene depth + focal length.

    Model:
      pinhole: image scale ∝ f / Z :contentReference[oaicite:5]{index=5}
      if depth_mode == "inv_depth": use d ≈ 1/Z (MiDaS provides relative inverse depth) :contentReference[oaicite:6]{index=6}
        => s_expected = k_obj * f_px * d_roi
      if depth_mode == "depth": use Z directly (metric depth)
        => s_expected = k_obj * f_px / Z_roi

    Where:
      - s_pred is a scale proxy from pred_mask (default sqrt(area))
      - d_roi / Z_roi is a robust statistic inside the mask region
      - k_obj is a per-object constant (calibrated offline from training set)

    Args:
        pred_mask01: [B,1,H,W] soft mask of "inserted object" (e.g., your change mask M_hat)
        depth_map:   [B,1,H,W] depth or inverse-depth at same resolution as pred_mask01
        f_px:        [B] or scalar focal length in pixels
        k_obj:       scalar or [B] per-object calibration constant
        depth_mode:  "inv_depth" (default, MiDaS-style) or "depth" (metric)
        depth_stat:  "median" (robust) or "mean"
        detach_depth: if True, stops gradients through depth (recommended; depth model is frozen)
    Returns:
        (loss, info)
    """
    if pred_mask01.dim() == 3:
        pred_mask01 = pred_mask01.unsqueeze(1)
    if depth_map.dim() == 3:
        depth_map = depth_map.unsqueeze(1)

    if pred_mask01.shape != depth_map.shape:
        raise ValueError(f"pred_mask {pred_mask01.shape} != depth_map {depth_map.shape}")

    B = pred_mask01.shape[0]
    f_px = f_px.view(-1, 1, 1, 1) if f_px.numel() == B else f_px.view(1, 1, 1, 1)
    k_obj = k_obj.view(-1, 1, 1, 1) if k_obj.numel() == B else k_obj.view(1, 1, 1, 1)

    d = depth_map
    if detach_depth:
        d = d.detach()

    # depth summary inside object region
    if depth_stat == "median":
        d_roi = masked_nanmedian_2d(d, pred_mask01, eps=eps)
    elif depth_stat == "mean":
        d_roi = masked_mean_2d(d, pred_mask01, eps=eps)
    else:
        raise ValueError("depth_stat must be 'median' or 'mean'")

    # predicted scale
    if scale_measure == "sqrt_area":
        s_pred = mask_area_scale(pred_mask01, eps=eps)
    else:
        raise ValueError("scale_measure must be 'sqrt_area'")

    # expected scale from depth + focal
    if depth_mode == "inv_depth":
        s_exp = k_obj * f_px * d_roi
    elif depth_mode == "depth":
        s_exp = k_obj * f_px / (d_roi + eps)
    else:
        raise ValueError("depth_mode must be 'inv_depth' or 'depth'")

    # log-ratio penalty (scale-invariant, stable)
    loss = torch.log((s_pred + eps) / (s_exp + eps)).pow(2).mean()

    return loss, DepthScaleLossInfo(s_pred=s_pred, s_exp=s_exp, depth_stat=d_roi)


# -----------------------
# k_obj calibration (offline)
# -----------------------

def estimate_k_obj_from_dataset(
    gt_mask01: torch.Tensor,
    depth_map: torch.Tensor,
    f_px: torch.Tensor,
    *,
    depth_mode: Literal["inv_depth", "depth"] = "inv_depth",
    depth_stat: Literal["median", "mean"] = "median",
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Estimate a per-object constant k_obj from training data using GT mask scale.

    For inv_depth:
        sqrt(area_gt) ≈ k_obj * f_px * d_roi   => k_obj ≈ sqrt(area_gt) / (f_px * d_roi)
    For depth:
        sqrt(area_gt) ≈ k_obj * f_px / Z_roi   => k_obj ≈ sqrt(area_gt) * Z_roi / f_px

    Returns:
        scalar k_obj (median over dataset for robustness)
    """
    if gt_mask01.dim() == 3:
        gt_mask01 = gt_mask01.unsqueeze(1)
    if depth_map.dim() == 3:
        depth_map = depth_map.unsqueeze(1)

    if gt_mask01.shape != depth_map.shape:
        raise ValueError(f"gt_mask {gt_mask01.shape} != depth_map {depth_map.shape}")

    # dataset can be batched here; we just compute per-sample k and take median
    if depth_stat == "median":
        d_roi = masked_nanmedian_2d(depth_map, gt_mask01, eps=eps)  # [B,1,1,1]
    elif depth_stat == "mean":
        d_roi = masked_mean_2d(depth_map, gt_mask01, eps=eps)
    else:
        raise ValueError("depth_stat must be 'median' or 'mean'")

    s_gt = mask_area_scale(gt_mask01, eps=eps)  # [B,1,1,1]

    f_px4 = f_px.view(-1, 1, 1, 1) if f_px.numel() == gt_mask01.shape[0] else f_px.view(1, 1, 1, 1)

    if depth_mode == "inv_depth":
        k = s_gt / (f_px4 * d_roi + eps)
    elif depth_mode == "depth":
        k = (s_gt * d_roi) / (f_px4 + eps)
    else:
        raise ValueError("depth_mode must be 'inv_depth' or 'depth'")

    k_flat = k.flatten()
    k_obj = torch.median(k_flat)  # robust global scalar
    return k_obj
