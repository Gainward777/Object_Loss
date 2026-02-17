from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------
# helpers: shape + morphology
# ---------------------------

def _as_4d_mask(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """
    Convert mask to shape [B, 1, H, W] and float32 on like.device.
    Accepts:
      - [H, W]
      - [B, H, W]
      - [B, 1, H, W]
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
    elif mask.dim() == 4:
        pass
    else:
        raise ValueError(f"mask must have 2..4 dims, got {mask.shape}")

    if mask.shape[0] != like.shape[0]:
        # allow broadcasting batch=1
        if mask.shape[0] == 1:
            mask = mask.expand(like.shape[0], -1, -1, -1)
        else:
            raise ValueError(f"mask batch {mask.shape[0]} != like batch {like.shape[0]}")

    return mask.to(device=like.device, dtype=torch.float32)


def downsample_mask_area(mask: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    """
    Area-preserving downsample (good for masks) to latent resolution.
    mask: [B,1,H,W] or [B,H,W] or [H,W]
    returns: [B,1,h,w] float in [0,1]
    """
    # We use interpolate(mode="area") which is effectively average pooling for downsampling.
    dummy = torch.zeros((mask.shape[0], 1, hw[0], hw[1]), device=mask.device, dtype=torch.float32) \
        if mask.dim() == 4 else torch.zeros((1, 1, hw[0], hw[1]), device=mask.device, dtype=torch.float32)
    m4 = _as_4d_mask(mask, dummy)
    return F.interpolate(m4, size=hw, mode="area")


def dilate(mask01: torch.Tensor, r: int) -> torch.Tensor:
    """
    Binary/soft dilation via max_pool2d.
    mask01: [B,1,H,W] float in [0,1]
    r: radius in pixels at the mask resolution
    """
    if r <= 0:
        return mask01
    k = 2 * r + 1
    return F.max_pool2d(mask01, kernel_size=k, stride=1, padding=r)


def erode(mask01: torch.Tensor, r: int) -> torch.Tensor:
    """
    Binary/soft erosion via duality: erode(m) = 1 - dilate(1-m).
    """
    if r <= 0:
        return mask01
    k = 2 * r + 1
    return 1.0 - F.max_pool2d(1.0 - mask01, kernel_size=k, stride=1, padding=r)


def boundary_band(mask_bin: torch.Tensor, r: int) -> torch.Tensor:
    """
    Band around boundary: dilate XOR erode (implemented as clamp(dilate - erode)).
    mask_bin: [B,1,H,W] float 0/1
    """
    if r <= 0:
        return torch.zeros_like(mask_bin)
    d = dilate(mask_bin, r)
    e = erode(mask_bin, r)
    return (d - e).clamp_(0.0, 1.0)
