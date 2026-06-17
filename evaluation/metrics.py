"""Ranking metrics: Recall@K, NDCG@K, MRR@K, and batch evaluation utilities."""

from math import log2
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F


def recall_at_k(recommended: List[int], ground_truth: Iterable[int], k: int) -> float:
    """Compute Recall@K for a single user.

    Args:
        recommended: ranked list of item_ids.
        ground_truth: iterable of relevant item_ids (e.g., validation/test items).
        k: cutoff.
    """
    if k <= 0:
        return 0.0
    gt = set(ground_truth)
    if not gt:
        return 0.0
    rec_k = recommended[:k]
    hits = len(gt.intersection(rec_k))
    return hits / float(len(gt))


def ndcg_at_k(recommended: List[int], ground_truth: Iterable[int], k: int) -> float:
    """Compute NDCG@K for a single user.

    Uses binary relevance (item is either relevant or not).
    """
    if k <= 0:
        return 0.0
    gt = set(ground_truth)
    if not gt:
        return 0.0

    rec_k = recommended[:k]
    dcg = 0.0
    for idx, item_id in enumerate(rec_k):
        if item_id in gt:
            dcg += 1.0 / log2(idx + 2.0)  # log2(rank+1)

    # Ideal DCG: all relevant items at the top
    ideal_hits = min(len(gt), k)
    idcg = sum(1.0 / log2(i + 2.0) for i in range(ideal_hits))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def hit_at_k(recommended: List[int], ground_truth: Iterable[int], k: int) -> float:
    """Compute Hit Rate@K for a single user.
    
    Hit Rate@K = 1 if at least one relevant item is in top-K, else 0.
    Also known as Precision@K > 0 or Recall@K > 0.
    
    Args:
        recommended: ranked list of item_ids.
        ground_truth: iterable of relevant item_ids.
        k: cutoff.
        
    Returns:
        Hit rate: 1.0 if hit, 0.0 otherwise.
    """
    if k <= 0:
        return 0.0
    gt = set(ground_truth)
    if not gt:
        return 0.0
    rec_k = recommended[:k]
    return 1.0 if len(gt.intersection(rec_k)) > 0 else 0.0


def absolute_recall_mrr_ndcg_for_ks(
    scores: torch.Tensor,
    labels: torch.Tensor,
    ks: List[int]
) -> Dict[str, float]:
    """Compute Recall@K, MRR@K, and NDCG@K for multiple K values from score tensors.
    
    This function is useful for batch evaluation when you have score matrices
    and ground truth labels as tensors.
    
    Args:
        scores: Tensor of shape [batch_size, num_items] - prediction scores
        labels: Tensor of shape [batch_size] - ground truth item IDs
        ks: List of K values to compute metrics for
        
    Returns:
        Dict with keys like "Recall@1", "Recall@5", "NDCG@10", "MRR@10", etc.
    """
    metrics: Dict[str, float] = {}
    one_hot = F.one_hot(labels, num_classes=scores.size(1))
    answer_count = one_hot.sum(1)
    
    labels_float = one_hot.float()
    rank = (-scores).argsort(dim=1)
    
    cut = rank
    device = scores.device
    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]
        hits = labels_float.gather(1, cut)
        
        # Recall@K
        denom = torch.min(torch.tensor([k], device=device), labels_float.sum(1).float())
        metrics[f"Recall@{k}"] = (hits.sum(1) / denom).mean().cpu().item()
        
        # MRR@K
        positions = torch.arange(1, k + 1, device=device).unsqueeze(0)
        metrics[f"MRR@{k}"] = (hits / positions).sum(1).mean().cpu().item()
        
        # NDCG@K
        position = torch.arange(2, 2 + k, device=device)
        weights = 1.0 / torch.log2(position.float())
        dcg = (hits * weights.to(device)).sum(1)
        idcg = torch.tensor([
            weights[: int(min(int(n.item()), k))].sum().item() for n in answer_count
        ], device=device)
        ndcg = (dcg / idcg).mean()
        metrics[f"NDCG@{k}"] = ndcg.cpu().item()
    
    return metrics
