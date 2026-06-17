"""Model components for retrieval stage (independent from LlamaRec).

Currently includes:
- NeuralLRURec: lightweight neural LRU-style sequential recommender.
- VBPR: Visual Bayesian Personalized Ranking model.
- BM3: Bootstrap Latent Representations for Multi-modal Recommendation model.
"""

from .neural_lru import NeuralLRUConfig, NeuralLRURec
from .vbpr import VBPR
from .bm3 import BM3

__all__ = [
    "NeuralLRUConfig",
    "NeuralLRURec",
    "VBPR",
    "BM3",
]
