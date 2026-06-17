"""Utility functions for VIP5 reranking.

These functions help prepare inputs for VIP5 model following the original implementation.
"""

from typing import Dict, List, Tuple, Optional
import torch
import numpy as np
from pathlib import Path

# Try to import tokenizer from VIP5
try:
    import sys
    vip5_src_path = Path(__file__).parent.parent.parent / "retrieval" / "vip5_temp" / "src"
    if vip5_src_path.exists() and str(vip5_src_path) not in sys.path:
        sys.path.insert(0, str(vip5_src_path))
    from tokenization import P5Tokenizer
except ImportError:
    # Fallback to T5Tokenizer if P5Tokenizer not available
    from transformers import T5Tokenizer
    P5Tokenizer = T5Tokenizer


def prepare_vip5_input(
    text: str,
    visual_features: torch.Tensor,  # [num_items, feat_dim]
    tokenizer,
    max_length: int = 128,
    image_feature_size_ratio: int = 2,
) -> Dict[str, torch.Tensor]:
    """Prepare input for VIP5 model following original format.
    
    Args:
        text: Input text (e.g., "Given the following purchase history of user_1: item_1, item_2, ...")
        visual_features: Visual features [num_items, feat_dim] for items in text
        tokenizer: VIP5 tokenizer (P5Tokenizer or T5Tokenizer)
        max_length: Maximum sequence length
        image_feature_size_ratio: Number of visual tokens per item (default: 2)
        
    Returns:
        Dict with keys: input_ids, whole_word_ids, category_ids, vis_feats, attention_mask
    """
    # Tokenize text
    input_ids = tokenizer.encode(text, padding='max_length', truncation=True, max_length=max_length)
    input_ids = torch.LongTensor(input_ids)
    
    # Calculate whole_word_ids (for whole word embeddings)
    # In VIP5, whole_word_ids track word boundaries
    tokenized_text = tokenizer.tokenize(text)
    whole_word_ids = calculate_whole_word_ids(tokenized_text, input_ids.tolist(), tokenizer)
    whole_word_ids = torch.LongTensor(whole_word_ids)
    
    # Calculate category_ids: 1 for visual tokens (<extra_id_0>), 0 for text tokens
    # In VIP5, <extra_id_0> is used as placeholder for visual tokens
    if hasattr(tokenizer, 'convert_tokens_to_ids'):
        extra_id_0_token_id = tokenizer.convert_tokens_to_ids('<extra_id_0>')
    else:
        # Fallback: assume <extra_id_0> is in vocab
        extra_id_0_token_id = tokenizer.encode('<extra_id_0>', add_special_tokens=False)[0]
    
    category_ids = torch.LongTensor([
        1 if token_id == extra_id_0_token_id else 0 
        for token_id in input_ids.tolist()
    ])
    
    # Prepare visual features
    # Count number of visual tokens (category_ids == 1)
    num_visual_tokens = category_ids.sum().item()
    num_items = visual_features.size(0) if visual_features is not None else 0
    
    if num_visual_tokens > 0 and num_items > 0:
        # ✅ Get device from visual_features to ensure all tensors are on the same device
        device = visual_features.device
        
        # Reshape visual features to match visual tokens
        # Each item gets image_feature_size_ratio visual tokens
        expected_visual_tokens = num_items * image_feature_size_ratio
        if num_visual_tokens != expected_visual_tokens:
            # Pad or truncate
            if num_visual_tokens < expected_visual_tokens:
                # Pad with zeros (on same device as visual_features)
                padding = torch.zeros(
                    expected_visual_tokens - num_visual_tokens, 
                    visual_features.size(1),
                    device=device,
                    dtype=visual_features.dtype
                )
                vis_feats = torch.cat([visual_features.repeat_interleave(image_feature_size_ratio, dim=0), padding])
            else:
                # Truncate
                vis_feats = visual_features.repeat_interleave(image_feature_size_ratio, dim=0)[:num_visual_tokens]
        else:
            vis_feats = visual_features.repeat_interleave(image_feature_size_ratio, dim=0)
        
        # Reshape to [1, V_W_L, feat_dim] for VIP5
        vis_feats = vis_feats.unsqueeze(0)  # [1, num_visual_tokens, feat_dim]
    else:
        # No visual features - determine device and dtype from visual_features if available
        if visual_features is not None:
            device = visual_features.device
            dtype = visual_features.dtype
            feat_dim = visual_features.size(1)
        else:
            device = torch.device('cpu')
            dtype = torch.float32
            feat_dim = 512
        vis_feats = torch.zeros(1, num_visual_tokens, feat_dim, device=device, dtype=dtype)
    
    # Attention mask
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    return {
        "input_ids": input_ids.unsqueeze(0),  # [1, seq_len]
        "whole_word_ids": whole_word_ids.unsqueeze(0),  # [1, seq_len]
        "category_ids": category_ids.unsqueeze(0),  # [1, seq_len]
        "vis_feats": vis_feats,  # [1, V_W_L, feat_dim]
        "attention_mask": attention_mask.unsqueeze(0),  # [1, seq_len]
    }


def calculate_whole_word_ids(
    tokenized_text: List[str],
    input_ids: List[int],
    tokenizer
) -> List[int]:
    """Calculate whole_word_ids for VIP5.
    
    In VIP5, whole_word_ids track word boundaries for whole word embeddings.
    Tokens starting with '▁' (SentencePiece) or special tokens like '<extra_id_0>' 
    mark the start of a new word.
    
    Note: input_ids may be padded to max_length, so we need to pad whole_word_ids accordingly.
    """
    whole_word_ids = []
    curr = 0
    
    # Calculate whole_word_ids for each token in tokenized_text
    for token in tokenized_text:
        if token.startswith('▁') or token == '<extra_id_0>':
            curr += 1
            whole_word_ids.append(curr)
        else:
            whole_word_ids.append(curr)
    
    # Pad or truncate to match input_ids length exactly
    # input_ids may have been padded/truncated by tokenizer
    target_length = len(input_ids)
    
    if len(whole_word_ids) < target_length:
        # Pad with 0 (for padding tokens)
        pad_token_id = getattr(tokenizer, 'pad_token_id', 0)
        # Check if input_ids has padding tokens at the end
        num_padding = target_length - len(whole_word_ids)
        whole_word_ids = whole_word_ids + [0] * num_padding
    elif len(whole_word_ids) > target_length:
        # Truncate to match input_ids
        whole_word_ids = whole_word_ids[:target_length]
    
    # Ensure exact match
    assert len(whole_word_ids) == len(input_ids), \
        f"whole_word_ids length ({len(whole_word_ids)}) must match input_ids length ({len(input_ids)})"
    
    return whole_word_ids


def build_rerank_prompt(
    user_id: int,
    user_history: List[int],
    candidates: List[int],
    template: str = "Given the following purchase history of user_{} : \n {} \n predict next possible item to be purchased by the user ?"
) -> str:
    """Build prompt for VIP5 reranking following original templates.
    
    Args:
        user_id: User ID
        user_history: List of item IDs in user history
        candidates: List of candidate item IDs to rerank
        template: Prompt template (default: A-1 template from VIP5)
        
    Returns:
        Formatted prompt string
    """
    # Format history: item_1, item_2, ...
    history_str = ", ".join([f"item_{item_id}" for item_id in user_history])
    
    # Format candidates: item_1, item_2, ...
    candidates_str = ", ".join([f"item_{item_id}" for item_id in candidates])
    
    # Combine history and candidates
    full_list = history_str + ", " + candidates_str
    
    # Format prompt
    prompt = template.format(user_id, full_list)
    
    return prompt

