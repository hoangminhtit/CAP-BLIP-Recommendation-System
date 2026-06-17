"""Registry cho các rerankers (Stage 2)."""

from typing import Dict, Type

from rerank.base import BaseReranker
from rerank.methods.qwen_reranker_unified import QwenReranker
from rerank.methods.vip5_reranker import VIP5Reranker


RERANKER_REGISTRY: Dict[str, Type[BaseReranker]] = {
    "qwen": QwenReranker,  # Unified reranker (supports text_only, caption, semantic_summary)
    "qwen3vl": QwenReranker,  # Backward compatibility: use unified reranker
    "vip5": VIP5Reranker,
}


def get_reranker_class(name: str) -> Type[BaseReranker]:
    name = name.lower()
    if name not in RERANKER_REGISTRY:
        raise KeyError(f"Unknown reranker: {name}. Available: {list(RERANKER_REGISTRY.keys())}")
    return RERANKER_REGISTRY[name]
