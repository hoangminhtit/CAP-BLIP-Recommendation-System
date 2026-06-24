"""Registry cho các rerankers (Stage 2)."""

from typing import Dict, Type

from rerank.base import BaseReranker
from rerank.methods.qwen_reranker_unified import QwenReranker


RERANKER_REGISTRY: Dict[str, Type[BaseReranker]] = {
    "qwen": QwenReranker, 
}


def get_reranker_class(name: str) -> Type[BaseReranker]:
    name = name.lower()
    if name not in RERANKER_REGISTRY:
        raise KeyError(f"Unknown reranker: {name}. Available: {list(RERANKER_REGISTRY.keys())}")
    return RERANKER_REGISTRY[name]
