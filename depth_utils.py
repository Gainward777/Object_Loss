from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F


# -----------------------
# Focal length utilities
# -----------------------

def focal_mm_to_px(
    focal_mm: torch.Tensor,
    image_width_px: torch.Tensor,
    sensor_width_mm: torch.Tensor,
) -> torch.Tensor:
    """
    Convert focal length from millimeters to pixels (horizontal focal):
        f_px = (f_mm / sensor_width_mm) * image_width_px

    This is the standard conversion used in many CV pipelines when sensor width is known. :contentReference[oaicite:4]{index=4}

    Args:
        focal_mm:        [B] or scalar
        image_width_px:  [B] or scalar
        sensor_width_mm: [B] or scalar

    Returns:
        f_px: [B]
    """
    return (focal_mm / sensor_width_mm) * image_width_px


# -----------------------
# Masked statistics
# -----------------------

def masked_nanmedian_2d(x: torch.Tensor, mask01: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Robust median over spatial dims using NaN masking.

    x:      [B,1,H,W] (or [B,H,W] -> will be expanded)
    mask01: [B,1,H,W] float in [0,1]

    Returns: [B,1,1,1]
    """
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if mask01.dim() == 3:
        mask01 = mask01.unsqueeze(1)

    if x.shape != mask01.shape:
        raise ValueError(f"x {x.shape} != mask {mask01.shape}")

    nan = torch.tensor(float("nan"), device=x.device, dtype=x.dtype)
    xm = torch.where(mask01 > 0.5, x, nan)
    flat = xm.flatten(2)  # [B,1,H*W]
    # torch.nanmedian exists in modern torch; if your build lacks it, replace with quantile on filtered values.
    med = torch.nanmedian(flat, dim=-1).values  # [B,1]
    return med.view(x.shape[0], 1, 1, 1)


def masked_mean_2d(x: torch.Tensor, mask01: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Mean over spatial dims within mask.
    Returns [B,1,1,1].
    """
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if mask01.dim() == 3:
        mask01 = mask01.unsqueeze(1)
    w = mask01.float()
    num = (x * w).sum(dim=(2, 3), keepdim=True)
    den = w.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
    return num / den


# -----------------------
# Scale measures from mask
# -----------------------

def mask_area_scale(mask01: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Scale proxy from mask area: s = sqrt(area)
    mask01: [B,1,H,W]
    returns: [B,1,1,1]
    """
    if mask01.dim() == 3:
        mask01 = mask01.unsqueeze(1)
    area = mask01.float().sum(dim=(2, 3), keepdim=True)
    return torch.sqrt(area + eps)
