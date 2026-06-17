"""Evaluation utilities for offline Recall@K / NDCG@K experiments.

Includes evaluation for:
- Stage 1 only (retrieval-only)
- Full pipeline (Stage 1 + Stage 2)
- Stage 2 only (rerank-only with Stage 1 candidates + ground truth)
"""
