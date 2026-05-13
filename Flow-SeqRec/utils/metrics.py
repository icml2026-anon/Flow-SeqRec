"""Evaluation metrics for sequential recommendation.

All metrics operate under the full-rank evaluation protocol:
scores are computed against ALL items rather than a sampled subset.
"""

import numpy as np
import torch


def recall_at_k(
    scores: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10,
) -> float:
    """Recall@K under full-rank evaluation.

    Args:
        scores: (batch, num_items) predicted scores for all items
        targets: (batch,) ground-truth item ids
        k: cutoff

    Returns:
        Average Recall@K across the batch.
    """
    _, topk_indices = torch.topk(scores, k, dim=-1)  # (B, K)
    targets_expanded = targets.unsqueeze(-1)  # (B, 1)
    hits = (topk_indices == targets_expanded).any(dim=-1).float()  # (B,)
    return hits.mean().item()


def ndcg_at_k(
    scores: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10,
) -> float:
    """NDCG@K under full-rank evaluation.

    Since there is exactly one relevant item per query, NDCG simplifies to
    1/log2(rank+1) if the item is in the top-K, else 0.

    Args:
        scores: (batch, num_items) predicted scores
        targets: (batch,) ground-truth item ids
        k: cutoff

    Returns:
        Average NDCG@K across the batch.
    """
    _, topk_indices = torch.topk(scores, k, dim=-1)  # (B, K)
    targets_expanded = targets.unsqueeze(-1)  # (B, 1)

    # Find the rank position (0-indexed) within top-K
    match_mask = (topk_indices == targets_expanded)  # (B, K)
    # Position: 0 to K-1
    positions = torch.arange(k, device=scores.device).unsqueeze(0).expand_as(match_mask)
    # DCG contribution: 1/log2(pos+2) (since pos is 0-indexed, rank = pos+1)
    dcg = (match_mask.float() / torch.log2(positions.float() + 2)).sum(dim=-1)  # (B,)

    return dcg.mean().item()


def compute_metrics(
    scores: torch.Tensor,
    targets: torch.Tensor,
    ks: list = None,
) -> dict:
    """Compute Recall@K and NDCG@K for multiple K values.

    Args:
        scores: (batch, num_items)
        targets: (batch,)
        ks: list of cutoff values

    Returns:
        Dictionary of metric_name -> value
    """
    if ks is None:
        ks = [5, 10, 20]

    results = {}
    for k in ks:
        results[f"Recall@{k}"] = recall_at_k(scores, targets, k)
        results[f"NDCG@{k}"] = ndcg_at_k(scores, targets, k)

    return results
