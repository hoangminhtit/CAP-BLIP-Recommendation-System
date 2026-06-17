"""
Base Retriever - Abstract class cho tất cả retrieval methods
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Set
import numpy as np


class BaseRetriever(ABC):
    """
    Abstract base class cho retrieval methods.
    
    Mỗi retriever phải implement:
    - fit(): Train model trên training data
    - retrieve(): Trả về top-K candidates cho user(s)
    """
    
    def __init__(self, top_k: int = 50):
        """
        Args:
            top_k: Số lượng candidates trả về (default: 50)
        """
        self.top_k = top_k
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, train_data: Dict[int, List[int]], **kwargs):
        """
        Train retriever trên training data.
        
        Args:
            train_data: Dict {user_id: [item_ids]}
            **kwargs: Additional arguments
        """
        pass
    
    @abstractmethod
    def retrieve(self, user_id: int, exclude_items: Set[int] = None) -> List[int]:
        """
        Retrieve top-K candidates cho một user.
        
        Args:
            user_id: ID của user
            exclude_items: Set các items cần loại trừ (đã tương tác)
            
        Returns:
            List[int]: Top-K item IDs (sorted by score descending)
        """
        pass
    
    def retrieve_batch(self, user_ids: List[int], 
                      exclude_items: Dict[int, Set[int]] = None) -> Dict[int, List[int]]:
        """
        Retrieve top-K candidates cho nhiều users.
        
        Args:
            user_ids: List các user IDs
            exclude_items: Dict {user_id: set(item_ids)} cần loại trừ
            
        Returns:
            Dict[int, List[int]]: {user_id: [top-K item_ids]}
        """
        if exclude_items is None:
            exclude_items = {}
            
        results = {}
        for user_id in user_ids:
            exclude = exclude_items.get(user_id, set())
            results[user_id] = self.retrieve(user_id, exclude)
            
        return results
    
    def get_name(self) -> str:
        """Trả về tên của retriever"""
        return self.__class__.__name__
    
    def _validate_fitted(self):
        """Kiểm tra model đã được fit chưa"""
        if not self.is_fitted:
            raise RuntimeError(f"{self.get_name()} chưa được fit. Gọi fit() trước!")
