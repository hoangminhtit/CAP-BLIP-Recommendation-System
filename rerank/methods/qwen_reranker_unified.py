"""Unified Qwen reranker supporting text-only and multimodal modes."""

from typing import Dict, List, Tuple, Any, Optional
import random
import numpy as np
import torch
from copy import deepcopy

from rerank.base import BaseReranker
from rerank.models.llm import LLMModel, build_prompt_from_candidates, rank_candidates
# ✅ REFACTORED: Qwen3VLModel is no longer used. All modes (text_only, caption, VIU) use LLMModel.
# from rerank.models.qwen3vl import Qwen3VLModel  # Deprecated
from evaluation.metrics import recall_at_k


def _get_max_seq_length() -> int:
    """Get max_seq_length from config.
    
    Returns:
        max_seq_length from config (default: 2048)
    """
    try:
        from config import arg
        return getattr(arg, 'qwen_max_seq_length', 2048)
    except ImportError:
        return 2048  # Default fallback


def _truncate_item_text(text: str, max_chars: int = 200) -> str:
    """Truncate item text metadata to prevent it from being too long.
    
    Note: Text may have already been truncated during data preparation (max_text_length).
    This function only truncates if text is longer than max_chars.
    
    To avoid double truncation, use max_chars >= max_text_length from config.
    """
    if not text:
        return text
    # Text may have already been truncated during data preparation
    # Only truncate if it's still longer than max_chars
    # If text is already <= max_chars, return as-is (no double truncation)
    if len(text) <= max_chars:
        return text
    # Only truncate if text is longer than max_chars
    return text[:max_chars - 3] + "..."


def _count_prompt_tokens(
    prompts: List[str],
    tokenizer,
    include_target: bool = False,
    targets: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Count tokens in prompts and return statistics.
    
    Args:
        prompts: List of prompt strings
        tokenizer: Tokenizer to use for counting
        include_target: Whether to include target tokens in count
        targets: List of target strings (required if include_target=True)
    
    Returns:
        Dict with statistics: min, max, mean, median, percentiles, token_counts list, etc.
    """
    token_counts = []
    
    for i, prompt in enumerate(prompts):
        # Tokenize prompt
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_count = len(prompt_tokens)
        
        # Add target tokens if requested
        if include_target and targets and i < len(targets):
            target = targets[i]
            target_tokens = tokenizer.encode(target, add_special_tokens=False)
            prompt_count += len(target_tokens)
        
        token_counts.append(prompt_count)
    
    if not token_counts:
        return {
            "min": 0,
            "max": 0,
            "mean": 0,
            "median": 0,
            "p50": 0,
            "p75": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "total": 0,
            "count": 0,
            "token_counts": [],
        }
    
    token_counts_arr = np.array(token_counts)
    
    return {
        "min": int(np.min(token_counts_arr)),
        "max": int(np.max(token_counts_arr)),
        "mean": float(np.mean(token_counts_arr)),
        "median": float(np.median(token_counts_arr)),
        "p50": float(np.percentile(token_counts_arr, 50)),
        "p75": float(np.percentile(token_counts_arr, 75)),
        "p90": float(np.percentile(token_counts_arr, 90)),
        "p95": float(np.percentile(token_counts_arr, 95)),
        "p99": float(np.percentile(token_counts_arr, 99)),
        "total": int(np.sum(token_counts_arr)),
        "count": len(token_counts_arr),
        "token_counts": token_counts,  # Keep original list for checking exceeding
    }


# Model name mappings
MODEL_MAPPING = {
    "qwen3-0.6b": "unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
    "qwen3-2bvl": "unsloth/Qwen3-VL-2B-Instruct",
    "qwen3-1.7b": "unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
    "qwen3-4b": "unsloth/Qwen3-4B-Instruct-2507"  # Note: Adjust model path if different
}


class QwenReranker(BaseReranker):
    """Unified Qwen reranker supporting multiple prompt modes.
    
    **IMPORTANT**: All 3 modes (`text_only`, `caption`, `VIU`) use the SAME training
    and evaluation process. They ONLY differ in how the prompt is built:
    - `text_only`: Uses only item text: "{text}"
    - `caption`: Adds caption info: "{text} (Image: {caption})"
    - `VIU`: Adds VIU: "{text} (VIU: {viu})"
    
    All modes use the same LLMModel (for all models including qwen3-2bvl),
    with identical training loops, loss functions, and evaluation metrics.
    
    **3 Prompt Modes**:
    1. `text_only`: Chỉ sử dụng description (text)
    2. `caption`: Sử dụng image caption (thêm thông tin caption vào prompt)
    3. `VIU`: Sử dụng image VIU (thêm thông tin VIU vào prompt)
    
    **3 Model Options**:
    1. `qwen3-0.6b`: Text model (for text_only, caption, VIU modes)
    2. `qwen3-2bvl`: Can be used for all modes (text_only, caption, VIU) - uses LLMModel
    3. `qwen3-1.6b`: Text model (for text_only, caption, VIU modes)
    
    Note: Even `qwen3-2bvl` (VL model) uses LLMModel for caption/VIU modes
    because captions/VIU are pre-generated text, not images.
    
    **Requirements**:
    - Mode `text_only`: Requires text in item_meta or item_id2text
    - Mode `caption`: Requires captions in item_meta (run data_prepare.py with --generate_caption)
    - Mode `VIU`: Requires VIU in item_meta (run data_prepare.py with --generate_viu)
    """
    
    def __init__(
        self,
        top_k: int = 50,
        mode: str = "text_only",
        model: str = "qwen3-0.6b",
        max_history: Optional[int] = None,
        max_candidates: Optional[int] = None,
        batch_size: int = 32,
        num_epochs: int = 10,
        lr: float = 1e-4,
        patience: Optional[int] = None,
    ) -> None:
        """
        Args:
            top_k: Số lượng items trả về sau rerank
            mode: Prompt mode - "text_only", "caption", "VIU"
            model: Model name - "qwen3-0.6b", "qwen3-2bvl", "qwen3-1.6b"
            max_history: Số lượng items trong history để dùng cho prompt (None = lấy từ config, default: 5)
            max_candidates: Maximum number of candidates to process (None = no limit)
            batch_size: Batch size cho training (default: 32)
            num_epochs: Số epochs (default: 10)
            lr: Learning rate (default: 1e-4)
            patience: Early stopping patience (None = no early stopping)
        """
        super().__init__(top_k=top_k)
        self.mode = mode
        self.model_name = model.lower()
        
        # Get max_history from config if not provided
        if max_history is None:
            try:
                from config import arg
                self.max_history = getattr(arg, 'qwen_max_history', 5)
            except ImportError:
                self.max_history = 5  # Default fallback
        else:
            self.max_history = max_history
        
        # Get max_candidates from config if not provided
        if max_candidates is None:
            try:
                from config import arg
                self.max_candidates = getattr(arg, 'qwen_max_candidates', None)
            except ImportError:
                self.max_candidates = None  # Default fallback (no limit)
        else:
            self.max_candidates = max_candidates
        
        # Get batch_size from config if not provided
        if batch_size is None:
            try:
                from config import arg
                self.batch_size = getattr(arg, 'rerank_batch_size', 16)
            except ImportError:
                self.batch_size = 16  # Default fallback
        else:
            self.batch_size = batch_size
        
        self.num_epochs = num_epochs
        self.lr = lr
        self.patience = patience
        
        # Validate mode
        if self.mode not in ["text_only", "caption", "VIU"]:
            raise ValueError(
                f"Invalid mode: {self.mode}. Must be one of: text_only, caption, VIU"
            )
        
        # Validate model
        # ✅ Allow direct HuggingFace model names if not in mapping
        if self.model_name not in MODEL_MAPPING:
            print(f"[QwenReranker] Model '{self.model_name}' not in MODEL_MAPPING, using as direct HuggingFace model name")
            print(f"[QwenReranker] Available mappings: {list(MODEL_MAPPING.keys())}")
        
        # Validate mode-model compatibility
        if self.mode == "text_only" and self.model_name == "qwen3-2bvl":
            raise ValueError(
                "text_only mode requires text model (qwen3-0.6b or qwen3-1.6b), not qwen3-2bvl"
            )
        # Note: caption and VIU modes can use text-only models (qwen3-0.6b, qwen3-1.6b)
        # because captions and VIU are pre-generated, no need for VL model
        # Only qwen3-2bvl is not allowed for text_only mode (already checked above)
        
        # Model instances
        self.llm_model: Optional[LLMModel] = None
        # ✅ REFACTORED: Qwen3VLModel is no longer used. All modes use LLMModel.
        # self.qwen3vl_model: Optional[Qwen3VLModel] = None  # Deprecated
        
        # Data structures
        self.user_history: Dict[int, List[str]] = {}  # user_id -> [item_texts]
        self.item_id2text: Dict[int, str] = {}  # item_id -> item_text (for text_only mode)
        self.item_meta: Dict[int, Dict[str, Any]] = {}  # item_id -> {caption, viu, text} (for multimodal modes)
        self.train_user_history: Dict[int, List[int]] = {}  # user_id -> [item_ids] for training
        
        # Debug flag: print sample test prompt only once
        self._debug_test_prompt_printed = False
        
        # Track eval prompts for token analysis (collect all prompts)
        self._eval_prompts = []
        self._eval_prompts_analyzed = False  # Track if we've done initial analysis
        self._eval_prompts_count_at_analysis = 0  # Track count when we did initial analysis
        self._eval_prompts_prebuilt = False  # Track if prompts have been pre-built
    
    def fit(
        self,
        train_data: Dict[int, List[int]],
        **kwargs: Any
    ) -> None:
        """Fit reranker with training.
        
        Args:
            train_data: Dict {user_id: [item_ids]} - training interactions
            **kwargs: Additional arguments:
                - item_id2text: Dict[int, str] - mapping item_id -> text (for text_only mode)
                - item_meta: Dict[int, Dict] - mapping item_id -> {caption, viu, text} (for multimodal modes)
                - user_history: Dict[int, List[str]] - user history texts
                - val_data: Dict[int, List[int]] - validation data for early stopping
                - train_data_for_llm: List[dict] - training data cho LLM (for text_only mode, optional)
                - num_epochs: int - override default num_epochs (optional)
                - batch_size: int - override default batch_size (optional)
                - lr: float - override default lr (optional)
                - patience: int - override default patience (optional)
        """
        # Override hyperparameters from kwargs if provided
        if "num_epochs" in kwargs:
            self.num_epochs = kwargs["num_epochs"]
        if "batch_size" in kwargs:
            self.batch_size = kwargs["batch_size"]
        if "lr" in kwargs:
            self.lr = kwargs["lr"]
        if "patience" in kwargs:
            self.patience = kwargs["patience"]
        
        # Load data structures
        self.item_id2text = kwargs.get("item_id2text", {})
        self.item_meta = kwargs.get("item_meta", {})
        self.user_history = kwargs.get("user_history", {})
        val_data = kwargs.get("val_data")
        
        # Store training data
        self.train_user_history = train_data
        
        # Get optimization settings
        from config import arg
        use_torch_compile = getattr(arg, 'use_torch_compile', False)
        
        # Load model based on mode and model type
        # ✅ IMPORTANT: All 3 modes (text_only, caption, VIU) use the SAME model and training process.
        #    They ONLY differ in prompt format (caption/VIU add extra info to prompt).
        # ✅ REFACTORED: All caption/VIU modes use LLMModel because captions/VIU 
        #    are pre-generated text (no need for VL model, even if using qwen3-2bvl).
        #    Qwen3VLModel is only needed for raw_image mode (loading images directly), which is not supported here.
        use_text_model = self.mode in ["text_only", "caption", "VIU"]
        
        if use_text_model:
            # ✅ Use LLMModel for ALL modes (text_only, caption, VIU)
            #    Training and evaluation are identical, only prompt format differs
            #    Note: Even if model is qwen3-2bvl (VL model), we use LLMModel because
            #    caption/VIU modes only use pre-generated text, not images.
            # ✅ Use mapping if available, otherwise use model_name directly as HuggingFace path
            model_path = MODEL_MAPPING.get(self.model_name, self.model_name)
            train_data_for_llm = kwargs.get("train_data_for_llm")
            
            # For caption/VIU modes, prepare training data from item_meta
            if self.mode in ["caption", "VIU"] and train_data_for_llm is None:
                train_samples = self._prepare_training_samples(train_data)
                if len(train_samples) > 0:
                    # Convert to LLM training format
                    from rerank.models.llm import build_prompt_from_candidates
                    train_data_for_llm = []
                    for sample in train_samples:
                        history = sample["history"]
                        candidates = sample["candidates"]
                        target_idx = sample["target_idx"]
                        
                        # Build prompt using item_meta (with caption/VIU)
                        history_texts = []
                        for item_id in history[-self.max_history:]:
                            meta = self.item_meta.get(item_id, {})
                            text = meta.get("text", f"item_{item_id}")
                            if self.mode == "caption":
                                caption = meta.get("caption", "")
                                if caption:
                                    history_texts.append(f"{text} (Image: {caption})")
                                else:
                                    history_texts.append(text)
                            elif self.mode == "VIU":
                                viu = meta.get("viu", "")
                                if viu:
                                    history_texts.append(f"{text} (VIU: {viu})")
                                else:
                                    history_texts.append(text)
                        
                        candidate_texts = []
                        for item_id in candidates:
                            meta = self.item_meta.get(item_id, {})
                            text = meta.get("text", f"item_{item_id}")
                            if self.mode == "caption":
                                caption = meta.get("caption", "")
                                if caption:
                                    candidate_texts.append(f"{text} (Image: {caption})")
                                else:
                                    candidate_texts.append(text)
                            elif self.mode == "VIU":
                                viu = meta.get("viu", "")
                                if viu:
                                    candidate_texts.append(f"{text} (VIU: {viu})")
                                else:
                                    candidate_texts.append(text)
                        
                        history_str = "\n".join([f"- {h}" for h in history_texts]) if history_texts else "No previous interactions."
                        # Use letters (LlamaRec style) instead of numbers
                        from rerank.models.llm import LETTERS
                        num_candidates = len(candidates)
                        if num_candidates > len(LETTERS):
                            raise ValueError(
                                f"Too many candidates ({num_candidates}). Maximum supported: {len(LETTERS)} candidates "
                                f"(using letters A-Z, a-z). Consider reducing max_candidates."
                            )
                        
                        cand_str = "\n".join([f"{LETTERS[i]}. {c}" for i, c in enumerate(candidate_texts)])
                        
                        # Answer format with letters
                        if num_candidates <= 26:
                            answer_format = f"Answer with only one letter (A-{LETTERS[num_candidates-1]})."
                        else:
                            answer_format = f"Answer with only one letter (A-Z, a-{LETTERS[num_candidates-1]})."
                        
                        prompt = f"""You are a recommendation ranking assistant.

Choose exactly ONE item the user is most likely to interact with next.

User history:
{history_str}

Candidate items:
{cand_str}

{answer_format}
""".strip()
                        
                        # Use letter index (LlamaRec style) instead of number
                        from rerank.models.llm import LETTERS
                        if target_idx >= len(LETTERS):
                            raise ValueError(
                                f"Target index {target_idx} exceeds max letters ({len(LETTERS)}). "
                                f"Reduce num_candidates or use number labels."
                            )
                        target = LETTERS[target_idx]  # Letter index (A, B, C, ...)
                        train_data_for_llm.append({
                            "messages": [
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": target}
                            ]
                        })
            
            if train_data_for_llm is not None:
                self.llm_model = LLMModel(
                    train_data=train_data_for_llm,
                    model_name=model_path
                )
                self.llm_model.load_model(use_torch_compile=False)
                # ✅ Check if eval mode - skip training if so
                try:
                    from config import arg
                    rerank_action = getattr(arg, 'rerank_action', 'train') or 'train'
                except ImportError:
                    rerank_action = 'train'
                
                if rerank_action != "eval":
                    self.llm_model.train(batch_size=self.batch_size)
                else:
                    print("  [Eval mode] Skipping training (trainer.train() skipped)")
            else:
                self.llm_model = LLMModel(
                    train_data=None,
                    model_name=model_path
                )
                self.llm_model.load_model(use_torch_compile=use_torch_compile)
        
        # ✅ REFACTORED: All modes (text_only, caption, VIU) now use LLMModel.
        #    Qwen3VLModel is only needed for raw_image mode (not supported in unified reranker).
        self.is_fitted = True
    
    def rerank(
        self,
        user_id: int,
        candidates: List[int],
        **kwargs: Any
    ) -> List[Tuple[int, float]]:
        """Rerank candidates cho một user.
        
        Args:
            user_id: ID của user
            candidates: List các item IDs cần rerank
            **kwargs: Additional arguments:
                - user_history: List[int] - user's interaction history (optional)
            
        Returns:
            List[Tuple[int, float]]: [(item_id, score)] đã sort giảm dần
        """
        self._validate_fitted()
        
        if not candidates:
            return []
        
        # Apply max_candidates limit if set
        if self.max_candidates is not None and len(candidates) > self.max_candidates:
            candidates = candidates[:self.max_candidates]
        
        # Get user history
        user_history = kwargs.get("user_history")
        if user_history is None:
            history = self.user_history.get(user_id, [])
        else:
            history = user_history
        history = history[-self.max_history:]  # Truncate to max_history
        
        # Predict probabilities based on mode and model type
        # ✅ IMPORTANT: All 3 modes use the SAME evaluation process. They only differ in prompt format.
        # ✅ REFACTORED: All caption/VIU modes use LLMModel because captions/VIU 
        #    are pre-generated text (no need for VL model, even if using qwen3-2bvl).
        num_candidates = len(candidates)
        use_text_model = self.mode in ["text_only", "caption", "VIU"]
        
        if use_text_model:
            # ✅ Use LLMModel for ALL modes (text_only, caption, VIU)
            #    Evaluation is identical, only prompt format differs
            #    Note: Even if model is qwen3-2bvl (VL model), we use LLMModel because
            #    caption/VIU modes only use pre-generated text, not images.
            if self.llm_model is None:
                raise RuntimeError("LLM model chưa được load. Gọi fit() trước!")
            
            # Build prompt based on mode
            if self.mode == "text_only":
                # Use helper function for text_only mode
                prompt = build_prompt_from_candidates(
                    history,
                    candidates,
                    self.item_id2text,
                    max_candidates=self.max_candidates
                )
            else:
                # Build prompt with caption/VIU for caption/VIU modes
                history_texts = []
                for item_id in history[-self.max_history:]:
                    meta = self.item_meta.get(item_id, {})
                    text = meta.get("text", f"item_{item_id}")
                    if self.mode == "caption":
                        caption = meta.get("caption", "")
                        if caption:
                            history_texts.append(f"{text} (Image: {caption})")
                        else:
                            history_texts.append(text)
                    elif self.mode == "VIU":
                        viu = meta.get("viu", "")
                        if viu:
                            history_texts.append(f"{text} (VIU: {viu})")
                        else:
                            history_texts.append(text)
                
                candidate_texts = []
                for item_id in candidates:
                    meta = self.item_meta.get(item_id, {})
                    text = meta.get("text", f"item_{item_id}")
                    if self.mode == "caption":
                        caption = meta.get("caption", "")
                        if caption:
                            candidate_texts.append(f"{text} (Image: {caption})")
                        else:
                            candidate_texts.append(text)
                    elif self.mode == "VIU":
                        viu = meta.get("viu", "")
                        if viu:
                            candidate_texts.append(f"{text} (VIU: {viu})")
                        else:
                            candidate_texts.append(text)
                
                # Use letters (LlamaRec style) instead of numbers
                from rerank.models.llm import LETTERS
                history_str = "\n".join([f"- {h}" for h in history_texts]) if history_texts else "No previous interactions."
                num_candidates = len(candidates)
                
                if num_candidates > len(LETTERS):
                    raise ValueError(
                        f"Too many candidates ({num_candidates}). Maximum supported: {len(LETTERS)} candidates "
                        f"(using letters A-Z, a-z). Consider reducing num_candidates."
                    )
                
                cand_str = "\n".join([f"{LETTERS[i]}. {c}" for i, c in enumerate(candidate_texts)])
                
                # Answer format with letters
                if num_candidates <= 26:
                    answer_format = f"Answer with only one letter (A-{LETTERS[num_candidates-1]})."
                else:
                    answer_format = f"Answer with only one letter (A-Z, a-{LETTERS[num_candidates-1]})."
                
                prompt = f"""You are a recommendation ranking assistant.

Choose exactly ONE item the user is most likely to interact with next.

User history:
{history_str}

Candidate items:
{cand_str}

{answer_format}
""".strip()
            
            # ✅ Collect eval prompts for token analysis (only if not pre-built)
            if not self._eval_prompts_prebuilt:
                self._eval_prompts.append(prompt)
                
                # Analyze after first prompt is printed and we have at least 10 prompts (early analysis)
                if not self._eval_prompts_analyzed and self._debug_test_prompt_printed and len(self._eval_prompts) >= 10:
                    print("\n[QwenReranker] Early eval prompt token analysis (first 10 prompts):")
                    self._analyze_eval_prompt_tokens()
                    self._eval_prompts_analyzed = True
                    self._eval_prompts_count_at_analysis = len(self._eval_prompts)
                # Final analysis if we've collected significantly more prompts (e.g., 10x more)
                elif self._eval_prompts_analyzed and len(self._eval_prompts) >= self._eval_prompts_count_at_analysis * 10:
                    print(f"\n[QwenReranker] Final eval prompt token analysis (all {len(self._eval_prompts)} prompts):")
                    self._analyze_eval_prompt_tokens()
                    self._eval_prompts_count_at_analysis = len(self._eval_prompts)  # Update to avoid repeated prints
            
            # ✅ Print sample test prompt for debugging (only once, controlled by verbose)
            if not self._debug_test_prompt_printed:
                try:
                    from config import arg
                    verbose = getattr(arg, 'qwen_verbose', 1)
                except ImportError:
                    verbose = 1
                
                if verbose >= 1:
                    # Minimal info for verbose >= 1
                    if hasattr(self.llm_model, 'tokenizer') and self.llm_model.tokenizer:
                        try:
                            tokens = self.llm_model.tokenizer.encode(prompt, add_special_tokens=False)
                            print(f"\n[QwenReranker] Sample eval prompt: {len(tokens)} tokens, {len(candidates)} candidates")
                        except:
                            print(f"\n[QwenReranker] Sample eval prompt: {len(candidates)} candidates")
                
                if verbose >= 2:
                    # Full details only for verbose >= 2
                    print("\n[QwenReranker] Sample Test Prompt (for debugging):")
                    print("-" * 80)
                    print(f"Mode: {self.mode}")
                    print(f"User ID: {user_id}")
                    print(f"History items: {history}")
                    print(f"Candidates: {candidates[:10]}..." if len(candidates) > 10 else f"Candidates: {candidates}")
                    # Debug: Check item_meta for VIU
                    if self.mode == "VIU":
                        print(f"\n[DEBUG] Checking VIU in item_meta:")
                        sample_item_id = candidates[0] if candidates else None
                        if sample_item_id:
                            sample_meta = self.item_meta.get(sample_item_id, {})
                            has_viu = "viu" in sample_meta and sample_meta.get("viu")
                            print(f"  Sample item {sample_item_id}: has VIU = {has_viu}")
                            if has_viu:
                                viu_preview = sample_meta["viu"][:100] if len(sample_meta["viu"]) > 100 else sample_meta["viu"]
                                print(f"  VIU preview: {viu_preview}...")
                            else:
                                print(f"  Available keys in item_meta: {list(sample_meta.keys())}")
                    print(f"\nPrompt:\n{prompt}")
                    # Count tokens if tokenizer is available
                    if hasattr(self.llm_model, 'tokenizer') and self.llm_model.tokenizer:
                        try:
                            tokens = self.llm_model.tokenizer.encode(prompt, add_special_tokens=False)
                            print(f"\nPrompt tokens: {len(tokens)}")
                        except:
                            pass
                    print("-" * 80)
                
                self._debug_test_prompt_printed = True
            
            probs = self.llm_model.predict(prompt, num_candidates=num_candidates)
        else:
            # ✅ REFACTORED: This branch should no longer be reached.
            #    All supported modes (text_only, caption, VIU) use LLMModel.
            raise ValueError(
                f"Invalid mode: {self.mode}. All supported modes (text_only, caption, VIU) "
                f"use LLMModel. Qwen3VLModel is only needed for raw_image mode, which is not supported here."
            )
        
        # Validate probabilities
        if len(probs) != len(candidates):
            raise ValueError(
                f"Mismatch: {len(candidates)} candidates but {len(probs)} probabilities."
            )
        
        # Rank candidates
        ranked_items = rank_candidates(probs, candidates)
        
        # Convert to (item_id, score) tuples
        item_to_score = {item_id: float(probs[i]) for i, item_id in enumerate(candidates)}
        scored = []
        for item_id in ranked_items[:self.top_k]:
            score = item_to_score.get(item_id, 0.0)
            scored.append((item_id, score))
        
        return scored
    
    def _prepare_training_samples(
        self,
        train_data: Dict[int, List[int]]
    ) -> List[Dict]:
        """Prepare training samples for multimodal modes."""
        samples = []
        all_items = set()
        for items in train_data.values():
            all_items.update(items)
        
        for user_id, items in train_data.items():
            if len(items) < 2:
                continue
            
            if len(items) > 3:
                end_pos = random.randint(1, len(items) - 1)
            else:
                end_pos = len(items) - 1
            
            history = items[:end_pos]
            target_item = items[end_pos]
            
            user_items_set = set(items)
            negative_candidates = [item for item in all_items if item not in user_items_set]
            
            # Get num_negatives from config (default: 19 for 20 total candidates)
            try:
                from config import arg
                # Use rerank_eval_candidates - 1 to get num_negatives (1 GT + N negatives = total)
                total_candidates = getattr(arg, 'rerank_eval_candidates', 20)
                num_negatives = total_candidates - 1  # 1 for ground truth
            except ImportError:
                num_negatives = 19  # Default fallback (1 GT + 19 negatives = 20 total)
            
            num_negatives = min(num_negatives, len(negative_candidates))
            if num_negatives > 0:
                negatives = random.sample(negative_candidates, num_negatives)
            else:
                negatives = []
            
            candidates = [target_item] + negatives
            random.shuffle(candidates)
            target_idx = candidates.index(target_item)
            
            samples.append({
                "user_id": user_id,
                "history": history,
                "candidates": candidates,
                "target_item": target_item,
                "target_idx": target_idx,
            })
        
        return samples
    
    def _train_model(
        self,
        train_samples: List[Dict],
        val_data: Optional[Dict[int, List[int]]] = None
    ) -> None:
        raise RuntimeError(
            f"_train_model() is deprecated. All modes (text_only, caption, VIU) "
            f"use LLMModel and training is handled by LLMModel.train() in fit() method."
        )

        from datasets import Dataset
        from transformers import TrainingArguments, Trainer
        
        training_data = []
        prompts = []
        targets = []
        for sample in train_samples:
            prompt = self._build_training_prompt(sample)
            # Use letter index (LlamaRec style) instead of number
            from rerank.models.llm import LETTERS
            target_idx = sample["target_idx"]
            if target_idx >= len(LETTERS):
                raise ValueError(
                    f"Target index {target_idx} exceeds max letters ({len(LETTERS)}). "
                    f"Reduce num_candidates or use number labels."
                )
            target = LETTERS[target_idx]  # Letter index (A, B, C, ...)
            
            prompts.append(prompt)
            targets.append(target)
            
            training_data.append({
                "instruction": prompt,
                "input": "",
                "output": target,
            })
        
        # ✅ Count tokens before training
        print("\n[QwenReranker] Analyzing prompt token counts...")
        stats = _count_prompt_tokens(
            prompts,
            self.qwen3vl_model.tokenizer,
            include_target=True,
            targets=targets
        )
        print(f"  Total samples: {stats['count']}")
        print(f"  Token statistics (prompt + target):")
        print(f"    Min: {stats['min']} tokens")
        print(f"    Max: {stats['max']} tokens")
        print(f"    Mean: {stats['mean']:.1f} tokens")
        print(f"    Median: {stats['median']:.1f} tokens")
        print(f"    Percentiles:")
        print(f"      P50: {stats['p50']:.1f} tokens")
        print(f"      P75: {stats['p75']:.1f} tokens")
        print(f"      P90: {stats['p90']:.1f} tokens")
        print(f"      P95: {stats['p95']:.1f} tokens")
        print(f"      P99: {stats['p99']:.1f} tokens")
        print(f"    Total tokens: {stats['total']:,}")
        
        # Check if any prompts exceed max_length
        max_length = _get_max_seq_length()  # From config
        num_exceeding = sum(1 for count in stats.get('token_counts', []) if count > max_length)
        if num_exceeding > 0:
            print(f"  ⚠️  WARNING: {num_exceeding}/{stats['count']} prompts exceed max_length={max_length} and will be truncated!")
        else:
            print(f"  ✅ All prompts fit within max_length={max_length}")
        
        # ✅ Print sample training prompt for debugging
        if prompts:
            print("\n[QwenReranker] Sample Training Prompt (for debugging):")
            print("-" * 80)
            sample_prompt = prompts[0]
            sample_target = targets[0] if targets else "N/A"
            print(f"Prompt:\n{sample_prompt}")
            print(f"\nTarget: {sample_target}")
            print(f"Prompt tokens: {stats.get('token_counts', [])[0] if stats.get('token_counts') else 'N/A'}")
            print("-" * 80)
        print()
        
        hf_train_dataset = Dataset.from_list(training_data)
        
        def tokenize_function(examples):
            texts = []
            for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
                text = f"{inst}\n{out}" if not inp else f"{inst}\n{inp}\n{out}"
                texts.append(text)
            
            max_length = _get_max_seq_length()  # From config
            tokenized = self.qwen3vl_model.tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        hf_train_dataset = hf_train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=hf_train_dataset.column_names,
        )
        
        training_args = TrainingArguments(
            output_dir="./qwen_rerank",
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=1,
            learning_rate=self.lr,
            num_train_epochs=1,
            logging_steps=50,
            save_steps=500,
            report_to="none",
            fp16=True,
            optim="adamw_torch",
        )
        
        trainer = Trainer(
            model=self.qwen3vl_model.model,
            args=training_args,
            train_dataset=hf_train_dataset,
            tokenizer=self.qwen3vl_model.tokenizer,
        )
        
        self._run_training_loop(trainer, val_data)
    
    def _train_vl_model(
        self,
        train_samples: List[Dict],
        val_data: Optional[Dict[int, List[int]]] = None
    ) -> None:
        """Train VL model (caption, VIU modes)."""
        if not hasattr(self.qwen3vl_model, 'model') or not hasattr(self.qwen3vl_model, 'processor'):
            print("Warning: VL model does not support training. Using pretrained model only.")
            return
        
        from datasets import Dataset
        from transformers import TrainingArguments, Trainer
        
        # Prepare training data
        training_data = []
        prompts = []
        targets = []
        for sample in train_samples:
            # Use letter index (LlamaRec style) instead of number
            from rerank.models.llm import LETTERS
            target_idx = sample["target_idx"]
            if target_idx >= len(LETTERS):
                raise ValueError(
                    f"Target index {target_idx} exceeds max letters ({len(LETTERS)}). "
                    f"Reduce num_candidates or use number labels."
                )
            target = LETTERS[target_idx]  # Letter index (A, B, C, ...)
            prompt = self._build_training_prompt(sample)
            
            prompts.append(prompt)
            targets.append(target)
            
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target}
            ]
            training_data.append({
                "messages": messages,
                "prompt": prompt,
                "target": target,
            })
        
        # ✅ Count tokens before training
        print("\n[QwenReranker] Analyzing prompt token counts...")
        stats = _count_prompt_tokens(
            prompts,
            self.qwen3vl_model.processor.tokenizer,
            include_target=True,
            targets=targets
        )
        print(f"  Total samples: {stats['count']}")
        print(f"  Token statistics (prompt + target):")
        print(f"    Min: {stats['min']} tokens")
        print(f"    Max: {stats['max']} tokens")
        print(f"    Mean: {stats['mean']:.1f} tokens")
        print(f"    Median: {stats['median']:.1f} tokens")
        print(f"    Percentiles:")
        print(f"      P50: {stats['p50']:.1f} tokens")
        print(f"      P75: {stats['p75']:.1f} tokens")
        print(f"      P90: {stats['p90']:.1f} tokens")
        print(f"      P95: {stats['p95']:.1f} tokens")
        print(f"      P99: {stats['p99']:.1f} tokens")
        print(f"    Total tokens: {stats['total']:,}")
        
        # Check if any prompts exceed max_length
        max_length = _get_max_seq_length()  # From config
        num_exceeding = sum(1 for count in stats.get('token_counts', []) if count > max_length)
        if num_exceeding > 0:
            print(f"  ⚠️  WARNING: {num_exceeding}/{stats['count']} prompts exceed max_length={max_length} and will be truncated!")
        else:
            print(f"  ✅ All prompts fit within max_length={max_length}")
        
        # ✅ Print sample training prompt for debugging
        if prompts:
            print("\n[QwenReranker] Sample Training Prompt (for debugging):")
            print("-" * 80)
            sample_prompt = prompts[0]
            sample_target = targets[0] if targets else "N/A"
            print(f"Prompt:\n{sample_prompt}")
            print(f"\nTarget: {sample_target}")
            print(f"Prompt tokens: {stats.get('token_counts', [])[0] if stats.get('token_counts') else 'N/A'}")
            print("-" * 80)
        print()
        
        hf_train_dataset = Dataset.from_list(training_data)
        
        # Custom data collator for VL model
        def collate_fn(batch):
            """Custom collate function for Qwen3-VL training."""
            processor = self.qwen3vl_model.processor
            
            batch_input_ids = []
            batch_attention_mask = []
            batch_labels = []
            
            for item in batch:
                target = item["target"]
                prompt = item["prompt"]
                
                inputs = processor.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=_get_max_seq_length(),  # From config
                )
                
                def ensure_cpu(obj):
                    if isinstance(obj, torch.Tensor):
                        return obj.cpu()
                    elif isinstance(obj, dict):
                        return {k: ensure_cpu(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return type(obj)(ensure_cpu(item) for item in obj)
                    else:
                        return obj
                
                inputs = ensure_cpu(inputs)
                
                input_ids = inputs["input_ids"].squeeze(0)
                attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
                if attention_mask.dim() > 1:
                    attention_mask = attention_mask.squeeze(0)
                
                target_tokens = processor.tokenizer.encode(
                    target,
                    add_special_tokens=False,
                    return_tensors="pt"
                ).squeeze(0).cpu()
                
                full_input_ids = torch.cat([input_ids, target_tokens])
                full_attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(len(target_tokens), dtype=attention_mask.dtype)
                ])
                
                labels = torch.cat([
                    torch.full_like(input_ids, -100),
                    target_tokens
                ])
                
                batch_input_ids.append(full_input_ids)
                batch_attention_mask.append(full_attention_mask)
                batch_labels.append(labels)
            
            # Pad to max length
            max_len = max(ids.size(0) for ids in batch_input_ids)
            pad_token_id = processor.tokenizer.pad_token_id
            
            padded_input_ids = []
            padded_attention_mask = []
            padded_labels = []
            
            for input_ids, attn_mask, labels in zip(batch_input_ids, batch_attention_mask, batch_labels):
                pad_len = max_len - input_ids.size(0)
                if pad_len > 0:
                    input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype)])
                    attn_mask = torch.cat([attn_mask, torch.zeros((pad_len,), dtype=attn_mask.dtype)])
                    labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=labels.dtype)])
                
                padded_input_ids.append(input_ids)
                padded_attention_mask.append(attn_mask)
                padded_labels.append(labels)
            
            return {
                "input_ids": torch.stack(padded_input_ids),
                "attention_mask": torch.stack(padded_attention_mask),
                "labels": torch.stack(padded_labels),
            }
        
        # Prepare dataset
        def prepare_dataset(examples):
            result = {"target": examples.get("target", [])}
            if "prompt" in examples:
                result["prompt"] = examples["prompt"]
            return result
        
        hf_train_dataset = hf_train_dataset.map(
            prepare_dataset,
            batched=True,
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./qwen_rerank_vl",
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=1,
            learning_rate=self.lr,
            num_train_epochs=1,
            logging_steps=50,
            save_steps=500,
            report_to="none",
            fp16=True,
            optim="adamw_torch",
            dataloader_pin_memory=True,
        )
        
        # Custom trainer with collate function
        class CustomTrainer(Trainer):
            def get_train_dataloader(self):
                from torch.utils.data import DataLoader
                return DataLoader(
                    self.train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    collate_fn=collate_fn,
                    shuffle=True,
                )
        
        trainer = CustomTrainer(
            model=self.qwen3vl_model.model,
            args=training_args,
            train_dataset=hf_train_dataset,
            tokenizer=self.qwen3vl_model.processor.tokenizer,
        )
        
        # Set model to train mode
        self.qwen3vl_model.model.train()
        
        # Training loop
        self._run_training_loop(trainer, val_data)
    
    def _build_training_prompt(self, sample: Dict) -> str:
        """Build training prompt from sample.
        
        ✅ IMPORTANT: This is the ONLY place where modes differ. All 3 modes (text_only, caption, VIU)
           use the same training process, loss function, and evaluation. They only differ in prompt format:
           - text_only: "{text}"
           - caption: "{text} (Image: {caption})"
           - VIU: "{text} (VIU: {viu})"
        """
        history = sample["history"]
        candidates = sample["candidates"]
        
        # Get history texts based on mode
        history_texts = []
        for item_id in history[-self.max_history:]:
            meta = self.item_meta.get(item_id, {})
            if self.mode == "caption":
                caption = meta.get("caption", "")
                text = meta.get("text", f"item_{item_id}")
                # Avoid double truncation: text was already truncated to max_text_length during data preparation
                # Only truncate if text is longer than max_text_length (shouldn't happen, but safety check)
                # Use max(200, max_text_length) to ensure we don't truncate unnecessarily
                truncate_limit = max(200, self.max_text_length) if hasattr(self, 'max_text_length') else 200
                text = _truncate_item_text(text, max_chars=truncate_limit)
                if caption:
                    history_texts.append(f"{text} (Image: {caption})")
                else:
                    history_texts.append(text)
            elif self.mode == "VIU":
                viu = meta.get("viu", "")
                text = meta.get("text", f"item_{item_id}")
                # Avoid double truncation: text was already truncated to max_text_length during data preparation
                # Only truncate if text is longer than max_text_length (shouldn't happen, but safety check)
                # Use max(200, max_text_length) to ensure we don't truncate unnecessarily
                truncate_limit = max(200, self.max_text_length) if hasattr(self, 'max_text_length') else 200
                text = _truncate_item_text(text, max_chars=truncate_limit)
                if viu:
                    history_texts.append(f"{text} (VIU: {viu})")
                else:
                    history_texts.append(text)
            else:  # text_only mode
                text = meta.get("text", f"item_{item_id}")
                # Avoid double truncation: text was already truncated to max_text_length during data preparation
                # Only truncate if text is longer than max_text_length (shouldn't happen, but safety check)
                # Use max(200, max_text_length) to ensure we don't truncate unnecessarily
                truncate_limit = max(200, self.max_text_length) if hasattr(self, 'max_text_length') else 200
                text = _truncate_item_text(text, max_chars=truncate_limit)
                history_texts.append(text)
        
        # Get candidate texts
        candidate_texts = []
        for item_id in candidates:
            meta = self.item_meta.get(item_id, {})
            if self.mode == "caption":
                caption = meta.get("caption", "")
                text = meta.get("text", f"item_{item_id}")
                # Avoid double truncation: text was already truncated to max_text_length during data preparation
                # Only truncate if text is longer than max_text_length (shouldn't happen, but safety check)
                # Use max(200, max_text_length) to ensure we don't truncate unnecessarily
                truncate_limit = max(200, self.max_text_length) if hasattr(self, 'max_text_length') else 200
                text = _truncate_item_text(text, max_chars=truncate_limit)
                if caption:
                    candidate_texts.append(f"{text} (Image: {caption})")
                else:
                    candidate_texts.append(text)
            elif self.mode == "VIU":
                viu = meta.get("viu", "")
                text = meta.get("text", f"item_{item_id}")
                # Avoid double truncation: text was already truncated to max_text_length during data preparation
                # Only truncate if text is longer than max_text_length (shouldn't happen, but safety check)
                # Use max(200, max_text_length) to ensure we don't truncate unnecessarily
                truncate_limit = max(200, self.max_text_length) if hasattr(self, 'max_text_length') else 200
                text = _truncate_item_text(text, max_chars=truncate_limit)
                if viu:
                    candidate_texts.append(f"{text} (VIU: {viu})")
                else:
                    candidate_texts.append(text)
            else:  # text_only mode
                text = meta.get("text", f"item_{item_id}")
                # Avoid double truncation: text was already truncated to max_text_length during data preparation
                # Only truncate if text is longer than max_text_length (shouldn't happen, but safety check)
                # Use max(200, max_text_length) to ensure we don't truncate unnecessarily
                truncate_limit = max(200, self.max_text_length) if hasattr(self, 'max_text_length') else 200
                text = _truncate_item_text(text, max_chars=truncate_limit)
                candidate_texts.append(text)
        
        # Build prompt
        history_str = "\n".join([f"- {h}" for h in history_texts]) if history_texts else "No previous interactions."
        # Use letters (LlamaRec style) instead of numbers
        from rerank.models.llm import LETTERS
        num_candidates = len(candidates)
        if num_candidates > len(LETTERS):
            raise ValueError(
                f"Too many candidates ({num_candidates}). Maximum supported: {len(LETTERS)} candidates "
                f"(using letters A-Z, a-z). Consider reducing num_candidates."
            )
        
        cand_str = "\n".join([f"{LETTERS[i]}. {c}" for i, c in enumerate(candidate_texts)])
        
        # Answer format with letters
        if num_candidates <= 26:
            answer_format = f"Answer with only one letter (A-{LETTERS[num_candidates-1]})."
        else:
            answer_format = f"Answer with only one letter (A-Z, a-{LETTERS[num_candidates-1]})."
        
        prompt = f"""You are a recommendation ranking assistant.

Choose exactly ONE item the user is most likely to interact with next.

User history:
{history_str}

Candidate items:
{cand_str}

{answer_format}
""".strip()
        
        return prompt
    
    def _build_test_prompt_sample(
        self,
        user_id: int,
        history: List[int],
        candidates: List[int]
    ) -> str:
        """Build a sample test prompt for debugging (similar format to training prompt)."""
        # Get history texts based on mode
        history_texts = []
        for item_id in history[-self.max_history:]:
            meta = self.item_meta.get(item_id, {})
            if self.mode == "caption":
                caption = meta.get("caption", "")
                text = meta.get("text", f"item_{item_id}")
                # Avoid double truncation: text was already truncated to max_text_length during data preparation
                # Only truncate if text is longer than max_text_length (shouldn't happen, but safety check)
                # Use max(200, max_text_length) to ensure we don't truncate unnecessarily
                truncate_limit = max(200, self.max_text_length) if hasattr(self, 'max_text_length') else 200
                text = _truncate_item_text(text, max_chars=truncate_limit)
                if caption:
                    history_texts.append(f"{text} (Image: {caption})")
                else:
                    history_texts.append(text)
            elif self.mode == "VIU":
                viu = meta.get("viu", "")
                text = meta.get("text", f"item_{item_id}")
                # Avoid double truncation: text was already truncated to max_text_length during data preparation
                # Only truncate if text is longer than max_text_length (shouldn't happen, but safety check)
                # Use max(200, max_text_length) to ensure we don't truncate unnecessarily
                truncate_limit = max(200, self.max_text_length) if hasattr(self, 'max_text_length') else 200
                text = _truncate_item_text(text, max_chars=truncate_limit)
                if viu:
                    history_texts.append(f"{text} (VIU: {viu})")
                else:
                    history_texts.append(text)
            else:  # text_only mode
                text = meta.get("text", f"item_{item_id}")
                # Avoid double truncation: text was already truncated to max_text_length during data preparation
                # Only truncate if text is longer than max_text_length (shouldn't happen, but safety check)
                # Use max(200, max_text_length) to ensure we don't truncate unnecessarily
                truncate_limit = max(200, self.max_text_length) if hasattr(self, 'max_text_length') else 200
                text = _truncate_item_text(text, max_chars=truncate_limit)
                history_texts.append(text)
        
        # Get candidate texts (limit to first 10 for display)
        candidate_texts = []
        for item_id in candidates[:10]:  # Show first 10 candidates
            meta = self.item_meta.get(item_id, {})
            if self.mode == "caption":
                caption = meta.get("caption", "")
                text = meta.get("text", f"item_{item_id}")
                # Avoid double truncation: text was already truncated to max_text_length during data preparation
                # Only truncate if text is longer than max_text_length (shouldn't happen, but safety check)
                # Use max(200, max_text_length) to ensure we don't truncate unnecessarily
                truncate_limit = max(200, self.max_text_length) if hasattr(self, 'max_text_length') else 200
                text = _truncate_item_text(text, max_chars=truncate_limit)
                if caption:
                    candidate_texts.append(f"{text} (Image: {caption})")
                else:
                    candidate_texts.append(text)
            elif self.mode == "VIU":
                viu = meta.get("viu", "")
                text = meta.get("text", f"item_{item_id}")
                # Avoid double truncation: text was already truncated to max_text_length during data preparation
                # Only truncate if text is longer than max_text_length (shouldn't happen, but safety check)
                # Use max(200, max_text_length) to ensure we don't truncate unnecessarily
                truncate_limit = max(200, self.max_text_length) if hasattr(self, 'max_text_length') else 200
                text = _truncate_item_text(text, max_chars=truncate_limit)
                if viu:
                    candidate_texts.append(f"{text} (VIU: {viu})")
                else:
                    candidate_texts.append(text)
            else:  # text_only mode
                text = meta.get("text", f"item_{item_id}")
                # Avoid double truncation: text was already truncated to max_text_length during data preparation
                # Only truncate if text is longer than max_text_length (shouldn't happen, but safety check)
                # Use max(200, max_text_length) to ensure we don't truncate unnecessarily
                truncate_limit = max(200, self.max_text_length) if hasattr(self, 'max_text_length') else 200
                text = _truncate_item_text(text, max_chars=truncate_limit)
                candidate_texts.append(text)
        
        # Build prompt with letters (LlamaRec style)
        from rerank.models.llm import LETTERS
        history_str = "\n".join([f"- {h}" for h in history_texts]) if history_texts else "No previous interactions."
        num_candidates = len(candidates)
        
        if num_candidates > len(LETTERS):
            raise ValueError(
                f"Too many candidates ({num_candidates}). Maximum supported: {len(LETTERS)} candidates "
                f"(using letters A-Z, a-z). Consider reducing num_candidates."
            )
        
        cand_str = "\n".join([f"{LETTERS[i]}. {c}" for i, c in enumerate(candidate_texts)])
        
        # Answer format with letters
        if num_candidates <= 26:
            answer_format = f"Answer with only one letter (A-{LETTERS[num_candidates-1]})."
        else:
            answer_format = f"Answer with only one letter (A-Z, a-{LETTERS[num_candidates-1]})."
        
        prompt = f"""You are a recommendation ranking assistant.

Choose exactly ONE item the user is most likely to interact with next.

User history:
{history_str}

Candidate items:
{cand_str}
{f'... (and {num_candidates - 10} more candidates)' if num_candidates > 10 else ''}

{answer_format}
""".strip()
        
        return prompt
    
    def _run_training_loop(
        self,
        trainer,
        val_data: Optional[Dict[int, List[int]]] = None
    ) -> None:
        """Run training loop with validation."""
        best_val_recall = -1.0
        epochs_no_improve = 0
        best_model_state = None
        
        for epoch in range(self.num_epochs):
            print(f"[QwenReranker] Training epoch {epoch+1}/{self.num_epochs}...")
            trainer.train()
            
            if val_data is not None:
                val_recall = self._evaluate_split(val_data, k=min(10, self.top_k))
                
                if val_recall > best_val_recall:
                    best_val_recall = val_recall
                    best_model_state = deepcopy(trainer.model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                
                print(f"[QwenReranker] Epoch {epoch+1}/{self.num_epochs} - "
                      f"val_Recall@{min(10, self.top_k)}: {val_recall:.4f}")
                
                if self.patience and epochs_no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        if best_model_state is not None:
            trainer.model.load_state_dict(best_model_state)
            print(f"Loaded best model with val_Recall@{min(10, self.top_k)}: {best_val_recall:.4f}")
        
        trainer.model.eval()
    
    def _evaluate_split(self, split: Dict[int, List[int]], k: int) -> float:
        """Compute average Recall@K for validation."""
        recalls = []
        
        # ✅ Debug: Track first few samples
        debug_samples = []
        
        # Get progress bar setting
        try:
            from config import arg
            disable_progress = getattr(arg, 'qwen_disable_progress_bar', False)
            verbose = getattr(arg, 'qwen_verbose', 1)
        except ImportError:
            disable_progress = False
            verbose = 1
        
        # Disable progress bars if requested
        if disable_progress:
            import os
            os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
            # Also disable tqdm if available
            try:
                from tqdm import tqdm
                tqdm._instances.clear()  # Clear existing instances
            except ImportError:
                pass
        
        total_users = len(split)
        processed = 0
        progress_interval = max(1, total_users // 20) if verbose >= 1 else total_users  # Show progress every 5% if verbose >= 1
        
        for user_id, gt_items in split.items():
            if user_id not in self.train_user_history:
                continue
            
            history = self.train_user_history[user_id]
            if len(history) == 0:
                continue
            
            # Load pre-generated candidates
            try:
                from evaluation.utils import load_rerank_candidates
                from config import arg
                
                all_candidates = load_rerank_candidates(
                    dataset_code=getattr(arg, 'dataset', 'beauty'),
                    min_rating=getattr(arg, 'min_rating', 0),
                    min_uc=getattr(arg, 'min_uc', 5),
                    min_sc=getattr(arg, 'min_sc', 5),
                )
                
                if user_id in all_candidates.get("val", {}):
                    candidates = all_candidates["val"][user_id]
                elif user_id in all_candidates.get("test", {}):
                    candidates = all_candidates["test"][user_id]
                else:
                    continue
            except Exception:
                continue
            
            # ✅ Debug: Check GT in candidates
            gt_in_candidates = any(item in candidates for item in gt_items)
            if not gt_in_candidates:
                print(f"[WARNING] User {user_id}: GT items {gt_items} not in candidates! Skipping.")
                continue
            
            random.shuffle(candidates)
            reranked = self.rerank(user_id, candidates, user_history=history)
            top_k_items = [item_id for item_id, _ in reranked[:k]]
            
            hits = len(set(top_k_items) & set(gt_items))
            if len(gt_items) > 0:
                recall = hits / len(gt_items)
                recalls.append(recall)
                
                # ✅ Debug: Track first 3 samples
                if len(debug_samples) < 3:
                    debug_samples.append({
                        "user_id": user_id,
                        "gt_items": gt_items,
                        "top_k": top_k_items[:5],
                        "hits": hits,
                        "recall": recall,
                        "reranked_scores": [score for _, score in reranked[:5]]
                    })
            
            # Progress update (only if verbose >= 1 and not disabled)
            processed += 1
            if not disable_progress and verbose >= 1 and processed % progress_interval == 0:
                print(f"  Processed {processed}/{total_users} users ({100*processed//total_users}%)...", end='\r')
        
        # Print final progress if needed
        if not disable_progress and verbose >= 1 and processed > 0 and processed % progress_interval != 0:
            print(f"  Processed {processed}/{total_users} users (100%)")
        
        # ✅ Debug: Print first few samples (only if verbose >= 2)
        if debug_samples and verbose >= 2:
            print(f"\n[DEBUG] Evaluation samples (first {len(debug_samples)}):")
            for i, sample in enumerate(debug_samples):
                print(f"  Sample {i+1}:")
                print(f"    User: {sample['user_id']}, GT: {sample['gt_items']}")
                print(f"    Top-5: {sample['top_k']}")
                print(f"    Hits: {sample['hits']}, Recall@{k}: {sample['recall']:.4f}")
                print(f"    Scores: {[f'{s:.4f}' for s in sample['reranked_scores'][:5]]}")
        
        return float(np.mean(recalls)) if recalls else 0.0
    
    def prepare_eval_prompts(
        self,
        users: List[int],
        candidates_by_user: Dict[int, List[int]],
        user_histories: Optional[Dict[int, List[int]]] = None
    ) -> None:
        """Build all eval prompts before reranking for token analysis.
        
        This method builds all prompts upfront, collects them, and analyzes token counts
        before any reranking happens. This is more efficient than collecting prompts
        one-by-one during reranking.
        
        Args:
            users: List of user IDs to build prompts for
            candidates_by_user: Dict mapping user_id to list of candidate item IDs
            user_histories: Optional dict mapping user_id to history. If None, uses self.user_history
        """
        if self._eval_prompts_prebuilt:
            return  # Already built
        
        # Only print if verbose >= 1
        try:
            from config import arg
            verbose = getattr(arg, 'qwen_verbose', 1)
        except ImportError:
            verbose = 1
        
        if verbose >= 1:
            print(f"\n[QwenReranker] Building all eval prompts ({len(users)} users)...")
        
        if user_histories is None:
            user_histories = self.user_history
        
        use_text_model = self.mode == "text_only" or (self.mode in ["caption", "VIU"] and self.model_name in ["qwen3-0.6b", "qwen3-1.6b"])
        
        for user_id in users:
            candidates = candidates_by_user.get(user_id, [])
            if not candidates:
                continue
            
            # Apply max_candidates limit if set
            if self.max_candidates is not None and len(candidates) > self.max_candidates:
                candidates = candidates[:self.max_candidates]
            
            # Get user history
            history = user_histories.get(user_id, [])
            history = history[-self.max_history:]  # Truncate to max_history
            
            # Build prompt based on mode
            if use_text_model:
                if self.mode == "text_only":
                    prompt = build_prompt_from_candidates(
                        history,
                        candidates,
                        self.item_id2text,
                        max_candidates=self.max_candidates
                    )
                else:
                    # Build prompt with caption/VIU
                    history_texts = []
                    for item_id in history:
                        meta = self.item_meta.get(item_id, {})
                        text = meta.get("text", f"item_{item_id}")
                        if self.mode == "caption":
                            caption = meta.get("caption", "")
                            if caption:
                                history_texts.append(f"{text} (Image: {caption})")
                            else:
                                history_texts.append(text)
                        elif self.mode == "VIU":
                            viu = meta.get("viu", "")
                            if viu:
                                history_texts.append(f"{text} (VIU: {viu})")
                            else:
                                history_texts.append(text)
                    
                    candidate_texts = []
                    for item_id in candidates:
                        meta = self.item_meta.get(item_id, {})
                        text = meta.get("text", f"item_{item_id}")
                        if self.mode == "caption":
                            caption = meta.get("caption", "")
                            if caption:
                                candidate_texts.append(f"{text} (Image: {caption})")
                            else:
                                candidate_texts.append(text)
                        elif self.mode == "VIU":
                            viu = meta.get("viu", "")
                            if viu:
                                candidate_texts.append(f"{text} (VIU: {viu})")
                            else:
                                candidate_texts.append(text)
                    
                    # Use letters (LlamaRec style) instead of numbers
                    from rerank.models.llm import LETTERS
                    history_str = "\n".join([f"- {h}" for h in history_texts]) if history_texts else "No previous interactions."
                    num_candidates = len(candidates)
                    
                    if num_candidates > len(LETTERS):
                        raise ValueError(
                            f"Too many candidates ({num_candidates}). Maximum supported: {len(LETTERS)} candidates "
                            f"(using letters A-Z, a-z). Consider reducing num_candidates."
                        )
                    
                    cand_str = "\n".join([f"{LETTERS[i]}. {c}" for i, c in enumerate(candidate_texts)])
                    
                    # Answer format with letters
                    if num_candidates <= 26:
                        answer_format = f"Answer with only one letter (A-{LETTERS[num_candidates-1]})."
                    else:
                        answer_format = f"Answer with only one letter (A-Z, a-{LETTERS[num_candidates-1]})."
                    
                    prompt = f"""You are a recommendation ranking assistant.

Choose exactly ONE item the user is most likely to interact with next.

User history:
{history_str}

Candidate items:
{cand_str}

{answer_format}
""".strip()
            else:
                # For VL model, build sample prompt
                prompt = self._build_test_prompt_sample(user_id, history, candidates)
            
            self._eval_prompts.append(prompt)
        
        self._eval_prompts_prebuilt = True
        
        # Get verbose level
        try:
            from config import arg
            verbose = getattr(arg, 'qwen_verbose', 1)
        except ImportError:
            verbose = 1
        
        if verbose >= 1:
            print(f"  Built {len(self._eval_prompts)} prompts")
        
        # Analyze all prompts (only if verbose >= 2)
        if len(self._eval_prompts) > 0 and verbose >= 2:
            print("\n[QwenReranker] Analyzing all eval prompt token counts...")
            self._analyze_eval_prompt_tokens()
            self._eval_prompts_analyzed = True
        elif len(self._eval_prompts) > 0:
            # Still analyze but don't print
            self._analyze_eval_prompt_tokens()
            self._eval_prompts_analyzed = True
    
    def _analyze_eval_prompt_tokens(self) -> None:
        """Analyze token counts for eval/test prompts."""
        if not self._eval_prompts:
            return
        
        # Get verbose level
        try:
            from config import arg
            verbose = getattr(arg, 'qwen_verbose', 1)
        except ImportError:
            verbose = 1
        
        # Determine which tokenizer to use
        # ✅ REFACTORED: All modes now use LLMModel, so only check llm_model.tokenizer
        tokenizer = None
        if self.llm_model and hasattr(self.llm_model, 'tokenizer'):
            tokenizer = self.llm_model.tokenizer
        
        if tokenizer is None:
            return
        
        # Count tokens for eval prompts
        stats = _count_prompt_tokens(
            self._eval_prompts,
            tokenizer,
            include_target=False,
            targets=None
        )
        
        # ✅ Print summary based on verbose level
        if verbose >= 1:
            print(f"  Total samples: {stats['count']}, Mean tokens: {stats['mean']:.1f}, Max: {stats['max']}")
        
        if verbose >= 2:
            # Full statistics only for verbose >= 2
            print(f"  Total samples analyzed: {stats['count']}")
            print(f"  Token statistics (prompt only):")
            print(f"    Min: {stats['min']} tokens")
            print(f"    Max: {stats['max']} tokens")
            print(f"    Mean: {stats['mean']:.1f} tokens")
            print(f"    Median: {stats['median']:.1f} tokens")
            print(f"    Percentiles:")
            print(f"      P50: {stats['p50']:.1f} tokens")
            print(f"      P75: {stats['p75']:.1f} tokens")
            print(f"      P90: {stats['p90']:.1f} tokens")
            print(f"      P95: {stats['p95']:.1f} tokens")
            print(f"      P99: {stats['p99']:.1f} tokens")
            print(f"    Total tokens: {stats['total']:,}")
        
        # Check if any prompts exceed max_length
        max_length = _get_max_seq_length()  # From config
        num_exceeding = sum(1 for count in stats.get('token_counts', []) if count > max_length)
        if num_exceeding > 0:
            if verbose >= 1:
                print(f"  ⚠️  WARNING: {num_exceeding}/{stats['count']} eval prompts exceed max_length={max_length} and will be truncated!")
        else:
            if verbose >= 2:
                print(f"  ✅ All eval prompts fit within max_length={max_length}")
        
        if verbose >= 1:
            print()  # Empty line for readability

