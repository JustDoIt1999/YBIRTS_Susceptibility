from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .constants import (
    RANK_LOSS_MODE_DISABLED,
    RANK_LOSS_MODE_NDCG_AWARE_LAMBDARANK,
)


def _to_numpy_vector(values: Any) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().reshape(-1)
    return np.asarray(values, dtype=float).reshape(-1)


def build_rank_group_indices(
    rank_targets: Any,
    years: Any,
    clusters: Any | None = None,
    *,
    use_cluster_in_rank_loss: bool = False,
) -> list[np.ndarray]:
    rank_values = _to_numpy_vector(rank_targets)
    year_values = _to_numpy_vector(years).astype(int, copy=False)
    if rank_values.size <= 1:
        return []

    if not use_cluster_in_rank_loss:
        return _build_year_only_groups(year_values)

    if clusters is None:
        cluster_values = np.zeros_like(year_values, dtype=int)
    else:
        cluster_values = _to_numpy_vector(clusters).astype(int, copy=False)
    return _build_cluster_aware_groups(rank_values, year_values, cluster_values)


def _build_year_only_groups(year_values: np.ndarray) -> list[np.ndarray]:
    groups: list[np.ndarray] = []
    for year in np.unique(year_values):
        group_indices = np.flatnonzero(year_values == year).astype(int)
        if group_indices.size > 1:
            groups.append(group_indices)
    return groups


def _build_cluster_aware_groups(
    rank_values: np.ndarray,
    year_values: np.ndarray,
    cluster_values: np.ndarray,
) -> list[np.ndarray]:
    groups: list[np.ndarray] = []
    for year in np.unique(year_values):
        year_indices = np.flatnonzero(year_values == year).astype(int)
        if year_indices.size <= 1:
            continue

        year_ranks = rank_values[year_indices]
        year_clusters = cluster_values[year_indices]

        negative_indices = year_indices[(year_ranks == 0) & (year_clusters == 0)]
        valid_positive_mask = (year_ranks != 0) & (year_clusters != 0)
        valid_positive_indices = year_indices[valid_positive_mask]
        if valid_positive_indices.size == 0:
            continue

        valid_positive_clusters = year_clusters[valid_positive_mask]
        for cluster in np.unique(valid_positive_clusters):
            cluster_positive_indices = valid_positive_indices[valid_positive_clusters == cluster]
            group_indices = np.concatenate([cluster_positive_indices, negative_indices]).astype(int)
            if group_indices.size > 1:
                groups.append(group_indices)
    return groups


def masked_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    mask = (targets != -1).float()
    if mask.sum() <= 0:
        return logits.new_tensor(0.0)
    loss = F.binary_cross_entropy_with_logits(logits, targets.clamp(min=0), reduction="none")
    return (loss * mask).sum() / mask.sum()


def _ideal_dcg(gains: torch.Tensor) -> torch.Tensor:
    if gains.numel() == 0:
        return gains.new_tensor(0.0)
    sorted_gains, _ = torch.sort(gains, descending=True)
    positions = torch.arange(sorted_gains.numel(), device=gains.device, dtype=torch.float32)
    discounts = 1.0 / torch.log2(positions + 2.0)
    return torch.sum(sorted_gains * discounts)


def ndcg_aware_lambdarank_loss(
    logits: torch.Tensor,
    rank_targets: torch.Tensor,
    years: torch.Tensor | None = None,
    clusters: torch.Tensor | None = None,
    *,
    use_cluster_in_rank_loss: bool = False,
) -> tuple[torch.Tensor, int]:
    s_values = logits.reshape(-1)
    r_values = rank_targets.reshape(-1)
    if s_values.numel() <= 1:
        return logits.new_tensor(0.0), 0

    year_values = torch.zeros_like(r_values) if years is None else years.reshape(-1)
    cluster_values = torch.zeros_like(r_values) if clusters is None else clusters.reshape(-1)
    total_weighted_loss = logits.new_tensor(0.0)
    total_weight = logits.new_tensor(0.0)
    total_pairs = 0
    two = logits.new_tensor(2.0)

    for group_indices in build_rank_group_indices(
        r_values,
        year_values,
        cluster_values,
        use_cluster_in_rank_loss=use_cluster_in_rank_loss,
    ):
        group_scores = s_values[group_indices.tolist()]
        group_rels = r_values[group_indices.tolist()]
        if group_scores.numel() <= 1:
            continue

        order = torch.argsort(group_scores, descending=True)
        positions = torch.empty_like(order)
        positions[order] = torch.arange(order.numel(), device=logits.device, dtype=order.dtype)
        discounts = 1.0 / torch.log2(positions.float() + 2.0)
        gains = torch.pow(two, group_rels.float()) - 1.0
        idcg = _ideal_dcg(gains)
        if not torch.isfinite(idcg) or idcg <= 0:
            continue

        pair_mask = group_rels.unsqueeze(1) > group_rels.unsqueeze(0)
        pair_count = int(pair_mask.sum().item())
        if pair_count == 0:
            continue

        score_diff = group_scores.unsqueeze(1) - group_scores.unsqueeze(0)
        pair_losses = F.binary_cross_entropy_with_logits(
            score_diff,
            torch.ones_like(score_diff),
            reduction="none",
        )
        delta_dcg = (gains.unsqueeze(1) - gains.unsqueeze(0)).abs() * (
            discounts.unsqueeze(1) - discounts.unsqueeze(0)
        ).abs()
        pair_weights = (delta_dcg / idcg) * pair_mask.float()
        weight_sum = pair_weights.sum()
        if weight_sum <= 0:
            continue

        total_weighted_loss = total_weighted_loss + (pair_weights * pair_losses).sum()
        total_weight = total_weight + weight_sum
        total_pairs += pair_count

    if total_weight <= 0:
        return logits.new_tensor(0.0), 0
    return total_weighted_loss / total_weight, total_pairs


def compute_rank_loss(
    rank_loss_mode: str,
    logits: torch.Tensor,
    rank_targets: torch.Tensor,
    years: torch.Tensor | None = None,
    clusters: torch.Tensor | None = None,
    *,
    use_cluster_in_rank_loss: bool = False,
) -> tuple[torch.Tensor, int]:
    if rank_loss_mode == RANK_LOSS_MODE_NDCG_AWARE_LAMBDARANK:
        return ndcg_aware_lambdarank_loss(
            logits,
            rank_targets,
            years,
            clusters,
            use_cluster_in_rank_loss=use_cluster_in_rank_loss,
        )
    if rank_loss_mode == RANK_LOSS_MODE_DISABLED:
        return logits.new_tensor(0.0), 0
    raise ValueError(f"Unsupported rank loss mode: {rank_loss_mode}")
