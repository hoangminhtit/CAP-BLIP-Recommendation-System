"""Two-stage recommendation pipeline module.

This module provides the TwoStagePipeline class that combines
retrieval (Stage 1) and reranking (Stage 2) methods.
"""

from .base import (
    PipelineConfig,
    RetrievalConfig,
    RerankConfig,
    TwoStagePipeline,
)

__all__ = [
    "PipelineConfig",
    "RetrievalConfig",
    "RerankConfig",
    "TwoStagePipeline",
]

