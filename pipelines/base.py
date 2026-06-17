"""Base classes for two-stage recommendation pipeline.

This module defines the configuration dataclasses and TwoStagePipeline class
that combines retrieval (Stage 1) and reranking (Stage 2) methods.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class RetrievalConfig:
    """Configuration for Stage 1 retrieval.
    
    Available methods:
        - "lrurec": Neural LRU-based sequential recommender
        - "mmgcn": Multimodal Graph Convolutional Network (requires CLIP embeddings)
        - "vbpr": Visual Bayesian Personalized Ranking (requires CLIP image embeddings)
        - "bm3": Bootstrap Latent Representations for Multi-modal Recommendation (requires CLIP embeddings)
    
    Note:
        - `top_k` determines how many candidates are passed to Stage 2
        - For Qwen reranker: candidates are automatically adjusted (can use all from retrieval)
        - Default 200 is suitable for all rerankers
        - MMGCN and VBPR require visual features from CLIP embeddings
    """
    method: str = "lrurec"
    top_k: int = 200


@dataclass
class RerankConfig:
    """Configuration for Stage 2 reranking.
    
    Args:
        method: Rerank method name (qwen, qwen3vl, vip5, bert4rec)
        top_k: Final number of recommendations returned
        mode: Rerank mode
            - "retrieval": Use candidates from Stage 1 (default)
            - "ground_truth": Use ground truth + 19 random negatives (for rerank quality evaluation)
        num_negatives: Number of random negatives for ground_truth mode (default: 19)
        qwen_mode: Qwen reranker mode (only used if method=qwen or qwen3vl)
            - "text_only": Use description only (default)
            - "caption": Use image captions
            - "semantic_summary": Use semantic summaries
        qwen_model: Qwen model (only used if method=qwen or qwen3vl)
            - "qwen3-0.6b": Text model (for text_only mode)
            - "qwen3-2bvl": Vision-Language model (for caption/semantic_summary modes)
            - "qwen3-1.6b": Text model (for text_only or semantic_summary mode)
        qwen3vl_mode: [DEPRECATED] Legacy Qwen3-VL mode (use qwen_mode instead)
    
    Note:
        - `top_k` is the final number of recommendations returned
        - For Qwen reranker: candidates are automatically adjusted from retrieval stage
        - Max candidates for Qwen can be configured via `--qwen_max_candidates` in config.py
        - ground_truth mode: Creates candidates = [gt_item] + 19 random negatives
          This mode is useful for evaluating rerank quality independently from retrieval
    """
    method: str = "qwen"
    top_k: int = 50
    mode: str = "retrieval"  # "retrieval" or "ground_truth"
    num_negatives: int = 19  # For ground_truth mode
    qwen_mode: Optional[str] = None  # For Qwen: "text_only", "caption", "semantic_summary"
    qwen_model: Optional[str] = None  # For Qwen: "qwen3-0.6b", "qwen3-2bvl", "qwen3-1.6b"
    qwen3vl_mode: Optional[str] = None  # [DEPRECATED] Legacy: use qwen_mode instead


@dataclass
class PipelineConfig:
    """Configuration for two-stage pipeline."""
    retrieval: RetrievalConfig
    rerank: RerankConfig


class TwoStagePipeline:
    """Two-stage recommendation pipeline.
    
    Combines retrieval (Stage 1) and reranking (Stage 2) methods.
    Stage 2 is optional - can run retrieval-only mode.
    """
    
    def __init__(self, cfg: PipelineConfig):
        """
        Args:
            cfg: Pipeline configuration
        """
        # Stage 1: Always required
        from retrieval.registry import get_retriever_class
        
        RetrieverCls = get_retriever_class(cfg.retrieval.method)
        self.retriever = RetrieverCls(top_k=cfg.retrieval.top_k)
        
        # Stage 2: Optional (can be disabled for retrieval-only)
        rerank_method = (cfg.rerank.method or "").lower()
        if rerank_method in ("", "none"):
            self.reranker = None
        else:
            from rerank.registry import get_reranker_class
            from config import arg
            
            RerankerCls = get_reranker_class(cfg.rerank.method)
            
            # For Qwen rerankers, pass max_candidates from config
            reranker_kwargs = {"top_k": cfg.rerank.top_k}
            if rerank_method in ["qwen", "qwen3vl"]:
                # Use unified reranker options
                qwen_mode = cfg.rerank.qwen_mode or getattr(arg, 'qwen_mode', None)
                qwen_model = cfg.rerank.qwen_model or getattr(arg, 'qwen_model', None)
                
                # Backward compatibility: if qwen3vl_mode is set, use it
                if qwen_mode is None and cfg.rerank.qwen3vl_mode:
                    qwen3vl_mode_legacy = cfg.rerank.qwen3vl_mode
                    mode_mapping = {
                        'caption': 'caption',
                        'semantic_summary': 'semantic_summary',
                        'semantic_summary_small': 'semantic_summary'
                    }
                    qwen_mode = mode_mapping.get(qwen3vl_mode_legacy, 'text_only')
                    if qwen3vl_mode_legacy == 'semantic_summary_small' and qwen_model is None:
                        qwen_model = 'qwen3-0.6b'
                
                # Default to text_only if no mode specified
                if qwen_mode is None:
                    qwen_mode = 'text_only'
                if qwen_model is None:
                    qwen_model = 'qwen3-0.6b' if qwen_mode == 'text_only' else 'qwen3-2bvl'
                
                reranker_kwargs["mode"] = qwen_mode
                reranker_kwargs["model"] = qwen_model
                
                # Use qwen_max_candidates from config, or default to retrieval_top_k
                max_candidates = arg.qwen_max_candidates if hasattr(arg, 'qwen_max_candidates') and arg.qwen_max_candidates is not None else cfg.retrieval.top_k
                reranker_kwargs["max_candidates"] = max_candidates
                
                # Use qwen_max_history from config
                if hasattr(arg, 'qwen_max_history'):
                    reranker_kwargs["max_history"] = arg.qwen_max_history
            
            self.reranker = RerankerCls(**reranker_kwargs)
        
        self.cfg = cfg
    
    def fit(self, train_data: Dict[int, List[int]], **kwargs) -> None:
        """Train both stages on training data.
        
        Args:
            train_data: Dict {user_id: [item_ids]} - training interactions
            **kwargs: Additional arguments passed to fit methods:
                - retriever_kwargs: Dict of kwargs for retriever.fit()
                - reranker_kwargs: Dict of kwargs for reranker.fit()
                - skip_retrieval: If True, skip retrieval training (only train reranker)
                - skip_rerank: If True, skip rerank training (only train retriever)
        """
        skip_retrieval = kwargs.get("skip_retrieval", False)
        skip_rerank = kwargs.get("skip_rerank", False)
        
        # Fit Stage 1 (unless skipped)
        if not skip_retrieval:
            retriever_kwargs = kwargs.get("retriever_kwargs", {})
            self.retriever.fit(train_data, **retriever_kwargs)
        
        # Fit Stage 2 (if enabled and not skipped)
        if self.reranker is not None and not skip_rerank:
            reranker_kwargs = kwargs.get("reranker_kwargs", {})
            self.reranker.fit(train_data, **reranker_kwargs)
    
    def recommend(
        self,
        user_id: int,
        exclude_items: Optional[List[int]] = None,
        ground_truth: Optional[List[int]] = None
    ) -> List[int]:
        """Generate recommendations for a user.
        
        Args:
            user_id: ID of the user
            exclude_items: Items to exclude from recommendations
            ground_truth: Ground truth items (for ground_truth rerank mode)
            
        Returns:
            List[int]: Recommended item IDs (sorted by score descending)
        """
        # Stage 2: Rerank (if enabled)
        if self.reranker is None:
            # No reranker: just return Stage 1 candidates
            exclude_set = set(exclude_items) if exclude_items else set()
            candidates = self.retriever.retrieve(user_id, exclude_items=exclude_set)
            return candidates
        
        # Determine rerank mode
        rerank_mode = self.cfg.rerank.mode.lower()
        
        if rerank_mode == "ground_truth":
            # Mode 2: Ground truth + random negatives
            if ground_truth is None or len(ground_truth) == 0:
                # Fallback to retrieval mode if no ground truth
                exclude_set = set(exclude_items) if exclude_items else set()
                candidates = self.retriever.retrieve(user_id, exclude_items=exclude_set)
            else:
                candidates = self._build_ground_truth_candidates(
                    user_id, ground_truth, exclude_items
                )
        else:
            # Mode 1: Use candidates from Stage 1 (default)
            exclude_set = set(exclude_items) if exclude_items else set()
            candidates = self.retriever.retrieve(user_id, exclude_items=exclude_set)
        
        if not candidates:
            return []
        
        scored = self.reranker.rerank(user_id, candidates)
        return [item_id for item_id, _ in scored]
    
    def _build_ground_truth_candidates(
        self,
        user_id: int,
        ground_truth: List[int],
        exclude_items: Optional[List[int]] = None
    ) -> List[int]:
        """Build candidates for ground_truth mode: gt + random negatives.
        
        Args:
            user_id: ID of the user
            ground_truth: Ground truth items (usually 1 item)
            exclude_items: Items to exclude (user's history)
            
        Returns:
            List[int]: Candidates = [gt_item] + num_negatives random negatives
        """
        import random
        
        # Get user's interaction history to exclude
        exclude_set = set(exclude_items) if exclude_items else set()
        if hasattr(self.retriever, 'user_history'):
            user_history = self.retriever.user_history.get(user_id, [])
            exclude_set.update(user_history)
        
        # Get all items from retriever
        # Try different attributes that different retrievers might have
        all_items = None
        if hasattr(self.retriever, 'item_count') and self.retriever.item_count:
            all_items = set(range(1, self.retriever.item_count + 1))
        elif hasattr(self.retriever, 'num_item') and self.retriever.num_item:
            all_items = set(range(1, self.retriever.num_item + 1))
        elif hasattr(self.retriever, 'num_items') and self.retriever.num_items:
            all_items = set(range(1, self.retriever.num_items + 1))
        
        if all_items is None:
            # Fallback: retrieve all candidates and use them as pool
            # This is not ideal but works as fallback
            try:
                all_candidates = self.retriever.retrieve(user_id, exclude_items=set())
                # Estimate item_count from max candidate
                if all_candidates:
                    max_item = max(all_candidates)
                    all_items = set(range(1, max_item + 1))
                else:
                    raise ValueError("Cannot determine item_count for ground_truth mode")
            except Exception:
                raise ValueError(
                    "Cannot determine item_count for ground_truth mode. "
                    "Please ensure retriever has item_count, num_item, or num_items attribute."
                )
        
        # Remove excluded items
        candidate_pool = all_items - exclude_set - set(ground_truth)
        
        # Sample random negatives
        num_negatives = self.cfg.rerank.num_negatives
        num_negatives = min(num_negatives, len(candidate_pool))
        
        if num_negatives > 0:
            negatives = random.sample(list(candidate_pool), num_negatives)
        else:
            negatives = []
        
        # Combine: ground truth + negatives
        candidates = list(ground_truth) + negatives
        
        # Shuffle to avoid position bias
        random.shuffle(candidates)
        
        return candidates
    
    def recommend_batch(
        self,
        user_ids: List[int],
        exclude_items: Optional[Dict[int, List[int]]] = None,
        ground_truth: Optional[Dict[int, List[int]]] = None
    ) -> Dict[int, List[int]]:
        """Generate recommendations for multiple users.
        
        Args:
            user_ids: List of user IDs
            exclude_items: Dict {user_id: [item_ids]} to exclude
            ground_truth: Dict {user_id: [item_ids]} - ground truth items (for ground_truth rerank mode)
            
        Returns:
            Dict[int, List[int]]: {user_id: [recommended_item_ids]}
        """
        if exclude_items is None:
            exclude_items = {}
        if ground_truth is None:
            ground_truth = {}
        
        results = {}
        for user_id in user_ids:
            exclude = exclude_items.get(user_id, [])
            gt = ground_truth.get(user_id)
            results[user_id] = self.recommend(user_id, exclude_items=exclude, ground_truth=gt)
        
        return results

