"""Base class cho Stage 2 rerankers."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any


class BaseReranker(ABC):
    """Abstract base class cho các rerank model (VIP4, BERT4Rec, GPT4Rec, ...)."""

    def __init__(self, top_k: int = 50):
        self.top_k = top_k
        self.is_fitted = False

    @abstractmethod
    def fit(self, train_data: Dict[int, List[int]], **kwargs: Any) -> None:
        """Train reranker trên dữ liệu.

        Args:
            train_data: Dict {user_id: [item_ids]}
        """

    @abstractmethod
    def rerank(self, user_id: int, candidates: List[int], **kwargs: Any) -> List[Tuple[int, float]]:
        """Rerank danh sách candidates và trả về (item_id, score) đã sort giảm dần.
        """

    def get_name(self) -> str:
        return self.__class__.__name__

    def _validate_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(f"{self.get_name()} chưa được fit. Gọi fit() trước!")
