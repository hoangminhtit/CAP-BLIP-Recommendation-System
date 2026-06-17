"""Retrieval package (Stage 1 - candidate generation).

Expose registry helpers để tạo retriever từ tên string.
"""

from .base import BaseRetriever
from .registry import get_retriever_class, RETRIEVER_REGISTRY

__all__ = [
    "BaseRetriever",
    "get_retriever_class",
    "RETRIEVER_REGISTRY",
]
