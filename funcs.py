from __future__ import annotations

from typing import Literal, Optional, Sequence, Union
import torch
import torch.nn.functional as F

Reduction = Literal["mean", "sum", "none"]
DistKind = Literal["l2", "cos"]
AggKind = Literal["mean", "max", "sum"]


# ---------------------------
# utils: reductions & distances
# ---------------------------

def _reduce_over_nonbatch(x: torch.Tensor, agg: AggKind = "mean") -> torch.Tensor:
    """
    Reduce tensor over all dims except batch dim 0.
    If x is [B], returns x unchanged.
    If x is [B, ...], reduces over dims 1..end.
    """
    if x.ndim <= 1:
        return x
    dims = tuple(range(1, x.ndim))
    if agg == "mean":
        return x.mean(dim=dims)
    if agg == "sum":
        return x.sum(dim=dims)
    if agg == "max":
        # max over multiple dims: fold sequentially
        y = x
        for d in reversed(dims):
            y = y.max(dim=d).values
        return y
    raise ValueError(f"Unknown agg={agg}")


def _apply_reduction(
    per_sample: torch.Tensor,
    reduction: Reduction = "mean",
    weights: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    per_sample: [B]
    weights: [B] (optional)
    """
    if weights is not None:
        weights = weights.to(dtype=per_sample.dtype, device=per_sample.device)
        if reduction == "none":
            return per_sample * weights
        if reduction == "sum":
            return (per_sample * weights).sum()
        # weighted mean
        return (per_sample * weights).sum() / (weights.sum() + eps)

    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "mean":
        return per_sample.mean()
    raise ValueError(f"Unknown reduction={reduction}")


def l2_distance(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    squared: bool = True,
    agg: AggKind = "mean",
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Returns per-sample distance [B] for a,b shaped [B,...,D] (same shape).
    """
    diff = a - b
    d = (diff * diff).sum(dim=-1)  # [B,...]
    if not squared:
        d = torch.sqrt(d + eps)
    return _reduce_over_nonbatch(d, agg=agg)  # [B]


def cosine_distance(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    normalize: bool = True,
    agg: AggKind = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Returns per-sample cosine distance [B] for a,b shaped [B,...,D].
    distance = 1 - cos_sim
    """
    if normalize:
        a = F.normalize(a, dim=-1, eps=eps)
        b = F.normalize(b, dim=-1, eps=eps)
    sim = (a * b).sum(dim=-1)  # [B,...]
    d = 1.0 - sim
    return _reduce_over_nonbatch(d, agg=agg)  # [B]


def pair_distance(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    dist: DistKind = "cos",
    agg: AggKind = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    if dist == "cos":
        return cosine_distance(a, b, normalize=True, agg=agg, eps=eps)
    if dist == "l2":
        return l2_distance(a, b, squared=True, agg=agg, eps=eps)
    raise ValueError(f"Unknown dist={dist}")


# ---------------------------
# building blocks (generic losses)
# ---------------------------

def feature_match_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    kind: Literal["l2", "l1", "cos"] = "l2",
    agg: AggKind = "mean",
    reduction: Reduction = "mean",
    weights: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Pull pred embeddings towards target embeddings (perceptual/feature reconstruction).
    Supports [B,D] or [B,P,D] etc.
    """
    if kind == "l2":
        per = l2_distance(pred, target, squared=True, agg=agg, eps=eps)
    elif kind == "l1":
        per = _reduce_over_nonbatch((pred - target).abs().sum(dim=-1), agg=agg)
    elif kind == "cos":
        per = cosine_distance(pred, target, normalize=True, agg=agg, eps=eps)
    else:
        raise ValueError(f"Unknown kind={kind}")
    return _apply_reduction(per, reduction=reduction, weights=weights)


def contrastive_pair_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    y: torch.Tensor,
    *,
    margin: float = 1.0,
    dist: DistKind = "l2",
    agg: AggKind = "mean",
    reduction: Reduction = "mean",
    weights: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Classic contrastive loss:
      L = y * D^2 + (1-y) * max(0, margin - D)^2
    y=1 positive, y=0 negative.  (Hadsell et al., 2006) :contentReference[oaicite:4]{index=4}
    """
    y = y.to(dtype=z1.dtype, device=z1.device)
    if dist == "l2":
        D = l2_distance(z1, z2, squared=False, agg=agg, eps=eps)  # [B]
    elif dist == "cos":
        D = cosine_distance(z1, z2, normalize=True, agg=agg, eps=eps)  # [B]
    else:
        raise ValueError(f"Unknown dist={dist}")

    per = y * (D ** 2) + (1.0 - y) * (F.relu(torch.tensor(margin, device=D.device, dtype=D.dtype) - D) ** 2)
    return _apply_reduction(per, reduction=reduction, weights=weights)


def triplet_margin_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    margin: float = 0.2,
    dist: DistKind = "cos",
    agg: AggKind = "mean",
    reduction: Reduction = "mean",
    weights: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Triplet loss:
      L = max(0, d(a,p) - d(a,n) + margin)
    (FaceNet) :contentReference[oaicite:5]{index=5}
    """
    d_ap = pair_distance(anchor, positive, dist=dist, agg=agg, eps=eps)  # [B]
    d_an = pair_distance(anchor, negative, dist=dist, agg=agg, eps=eps)  # [B]
    per = F.relu(d_ap - d_an + margin)
    return _apply_reduction(per, reduction=reduction, weights=weights)


def margin_ranking_on_distances(
    d_pos: torch.Tensor,
    d_neg: torch.Tensor,
    *,
    margin: float = 0.2,
    reduction: Reduction = "mean",
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Hinge ranking on distances:
      want d_pos + margin <= d_neg
      L = max(0, d_pos - d_neg + margin)
    Equivalent to triplet with precomputed distances.
    """
    per = F.relu(d_pos - d_neg + margin)
    return _apply_reduction(per, reduction=reduction, weights=weights)


def multi_negative_ranking_loss(
    z_anchor: torch.Tensor,
    z_positive: torch.Tensor,
    z_negs: Union[torch.Tensor, Sequence[torch.Tensor]],
    *,
    margin: float = 0.2,
    dist: DistKind = "cos",
    agg: AggKind = "mean",
    neg_agg: AggKind = "mean",
    reduction: Reduction = "mean",
    weights: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Ranking vs multiple negatives:
      L_i = agg_j max(0, d(a,p) - d(a,n_j) + margin)
    z_negs: [B,K,D] or list of [B,D] (also supports [B,K,P,D]).
    """
    if isinstance(z_negs, (list, tuple)):
        z_negs = torch.stack(list(z_negs), dim=1)  # [B,K,...,D]

    # Compute d(a,p): [B]
    d_pos = pair_distance(z_anchor, z_positive, dist=dist, agg=agg, eps=eps)  # [B]

    # Compute d(a,n_j): do broadcast by expanding anchor along K
    B = z_anchor.shape[0]
    K = z_negs.shape[1]
    # expand anchor to [B,K,...,D]
    expand_shape = [B, K] + list(z_anchor.shape[1:])
    z_a = z_anchor.unsqueeze(1).expand(*expand_shape)
    d_neg = pair_distance(z_a, z_negs, dist=dist, agg=agg, eps=eps)  # [B] if agg reduces all, but we need [B,K]

    # The above reduces over nonbatch dims including K if we pass [B,K,...,D].
    # So compute per-negative distances manually:
    if dist == "cos":
        # cosine per element -> [B,K,...]
        a_n = F.normalize(z_a, dim=-1, eps=eps)
        n_n = F.normalize(z_negs, dim=-1, eps=eps)
        sim = (a_n * n_n).sum(dim=-1)  # [B,K,...]
        d = 1.0 - sim                  # [B,K,...]
        d_neg = _reduce_over_nonbatch(d, agg=agg)  # [B,K]
    else:
        diff = z_a - z_negs
        d = (diff * diff).sum(dim=-1)             # [B,K,...]
        d_neg = _reduce_over_nonbatch(d, agg=agg)  # [B,K]

    per = F.relu(d_pos.unsqueeze(1) - d_neg + margin)  # [B,K]

    if neg_agg == "mean":
        per = per.mean(dim=1)
    elif neg_agg == "max":
        per = per.max(dim=1).values
    elif neg_agg == "sum":
        per = per.sum(dim=1)
    else:
        raise ValueError(f"Unknown neg_agg={neg_agg}")

    return _apply_reduction(per, reduction=reduction, weights=weights)


def info_nce_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
    *,
    temperature: float = 0.07,
    normalize: bool = True,
    reduction: Reduction = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    InfoNCE:
      L = -log exp(sim(a,p)/t) / (exp(sim(a,p)/t) + sum_j exp(sim(a,n_j)/t))
    If negatives=None: uses in-batch negatives (anchor vs all positives).
    (common formulation) :contentReference[oaicite:6]{index=6}
    """
    if normalize:
        anchor = F.normalize(anchor, dim=-1, eps=eps)
        positive = F.normalize(positive, dim=-1, eps=eps)

    B = anchor.shape[0]

    if negatives is None:
        logits = (anchor @ positive.t()) / temperature  # [B,B]
        targets = torch.arange(B, device=anchor.device)
        return F.cross_entropy(logits, targets, reduction=reduction)

    if isinstance(negatives, (list, tuple)):
        negatives = torch.stack(list(negatives), dim=1)  # [B,K,D]
    if normalize:
        negatives = F.normalize(negatives, dim=-1, eps=eps)

    pos_logit = (anchor * positive).sum(dim=-1, keepdim=True) / temperature  # [B,1]
    neg_logits = torch.einsum("bd,bkd->bk", anchor, negatives) / temperature  # [B,K]
    logits = torch.cat([pos_logit, neg_logits], dim=1)  # [B,1+K]
    targets = torch.zeros(B, dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits, targets, reduction=reduction)


# ---------------------------
# your 3 task-specific "points" as separate functions
# ---------------------------

def loss_no_edit_identity(
    z_pred: torch.Tensor,
    z_gt: torch.Tensor,
    z_src: torch.Tensor,
    *,
    margin: float = 0.2,
    dist: DistKind = "cos",
    agg: AggKind = "mean",
    reduction: Reduction = "mean",
    weights: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    (1) Борьба с "не заменяет / оставляет исходник":
    хотим pred ближе к GT, чем к исходному (identity) на margin.

    Это triplet: anchor=pred, positive=gt, negative=src.
    """
    return triplet_margin_loss(
        anchor=z_pred, positive=z_gt, negative=z_src,
        margin=margin, dist=dist, agg=agg, reduction=reduction, weights=weights, eps=eps
    )


def loss_geometry_match(
    z_pred: torch.Tensor,
    z_gt: torch.Tensor,
    *,
    kind: Literal["l2", "l1", "cos"] = "l2",
    agg: AggKind = "mean",
    reduction: Reduction = "mean",
    weights: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    (2) Борьба с "искажена геометрия табуретки":
    тянем pred-эмбеддинги к gt-эмбеддингам (лучше всего, если эмбеддер чувствителен к форме:
    patch-фичи DINO/ViT, или spatial-фичи U-Net и т.п.).
    """
    return feature_match_loss(
        pred=z_pred, target=z_gt,
        kind=kind, agg=agg, reduction=reduction, weights=weights, eps=eps
    )


def loss_overlay_avoidance(
    z_pred: torch.Tensor,
    z_gt: torch.Tensor,
    z_overlay: torch.Tensor,
    *,
    margin: float = 0.2,
    dist: DistKind = "cos",
    agg: AggKind = "mean",
    reduction: Reduction = "mean",
    weights: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    (3) Борьба с "новый объект наложен на исходный":
    хотим pred ближе к GT, чем к "оверлею" на margin.

    Это тоже triplet: anchor=pred, positive=gt, negative=overlay.
    """
    return triplet_margin_loss(
        anchor=z_pred, positive=z_gt, negative=z_overlay,
        margin=margin, dist=dist, agg=agg, reduction=reduction, weights=weights, eps=eps
    )
