"""
BERT4Rec model implementation for sequential recommendation reranking.

Based on the paper "BERT4Rec: Sequential Recommendation with Bidirectional Encoder 
Representations from Transformer" and the implementation at https://github.com/FeiSun/BERT4Rec

Reference:
    @inproceedings{Sun:2019:BSR:3357384.3357895,
        author = {Sun, Fei and Liu, Jun and Wu, Jian and Pei, Changhua and Lin, Xiao and Ou, Wenwu and Jiang, Peng},
        title = {BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer},
        booktitle = {Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
        year = {2019},
        pages = {1441--1450}
    }
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class BertEmbedding(nn.Module):
    """BERT embedding layer: item embedding + position embedding + segment embedding."""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int = 200,
        dropout: float = 0.1,
    ):
        super(BertEmbedding, self).__init__()
        self.item_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embedding = nn.Embedding(2, hidden_size)  # 0: history, 1: candidate
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize embeddings
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.token_type_embedding.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len] (0 for history, 1 for candidate)
        
        Returns:
            Embeddings [batch_size, seq_len, hidden_size]
        """
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        item_emb = self.item_embedding(input_ids)
        position_emb = self.position_embedding(position_ids)
        token_type_emb = self.token_type_embedding(token_type_ids)
        
        embeddings = item_emb + position_emb + token_type_emb
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class BertSelfAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float = 0.1,
    ):
        super(BertSelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
            )
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(attention_probs_dropout_prob)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose for multi-head attention."""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # [batch, heads, seq_len, head_size]
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] (1 for valid, 0 for mask)
        
        Returns:
            context_layer: [batch_size, seq_len, hidden_size]
            attention_probs: [batch_size, num_heads, seq_len, seq_len]
        """
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            attention_mask = (1.0 - attention_mask.float()) * -10000.0
            attention_scores = attention_scores + attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer, attention_probs


class BertSelfOutput(nn.Module):
    """Output layer for self-attention."""
    
    def __init__(self, hidden_size: int, hidden_dropout_prob: float = 0.1):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
    
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    """BERT attention block: self-attention + output."""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
    ):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(
            hidden_size, num_attention_heads, attention_probs_dropout_prob
        )
        self.output = BertSelfOutput(hidden_size, hidden_dropout_prob)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self_output, attention_probs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_output, hidden_states)
        return attention_output, attention_probs


class BertIntermediate(nn.Module):
    """Intermediate layer (feed-forward)."""
    
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = "gelu"):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        if hidden_act == "gelu":
            self.intermediate_act_fn = F.gelu
        elif hidden_act == "relu":
            self.intermediate_act_fn = F.relu
        else:
            raise ValueError(f"Unknown activation: {hidden_act}")
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    """Output layer for feed-forward."""
    
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_dropout_prob: float = 0.1):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
    
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    """Single BERT layer: attention + feed-forward."""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        attention_probs_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        hidden_act: str = "gelu",
    ):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(
            hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob
        )
        self.intermediate = BertIntermediate(hidden_size, intermediate_size, hidden_act)
        self.output = BertOutput(hidden_size, intermediate_size, hidden_dropout_prob)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_output, attention_probs = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs


class BertEncoder(nn.Module):
    """BERT encoder: stack of BertLayer."""
    
    def __init__(
        self,
        num_hidden_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        attention_probs_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        hidden_act: str = "gelu",
    ):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([
            BertLayer(
                hidden_size, num_attention_heads, intermediate_size,
                attention_probs_dropout_prob, hidden_dropout_prob, hidden_act
            )
            for _ in range(num_hidden_layers)
        ])
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            [batch_size, seq_len, hidden_size]
        """
        for layer_module in self.layer:
            hidden_states, _ = layer_module(hidden_states, attention_mask)
        return hidden_states


class BERT4Rec(nn.Module):
    """
    BERT4Rec model for sequential recommendation reranking.
    
    Architecture:
        - BERT encoder với bidirectional attention
        - Input: user history sequence + candidate items
        - Output: scores for each candidate
    
    Based on the implementation at https://github.com/FeiSun/BERT4Rec
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 64,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 2,
        intermediate_size: int = 256,
        max_position_embeddings: int = 200,
        attention_probs_dropout_prob: float = 0.2,
        hidden_dropout_prob: float = 0.2,
        hidden_act: str = "gelu",
    ):
        """
        Args:
            vocab_size: Number of items (vocabulary size)
            hidden_size: Hidden dimension (default: 64)
            num_hidden_layers: Number of transformer layers (default: 2)
            num_attention_heads: Number of attention heads (default: 2)
            intermediate_size: Feed-forward intermediate size (default: 256)
            max_position_embeddings: Maximum sequence length (default: 200)
            attention_probs_dropout_prob: Attention dropout (default: 0.2)
            hidden_dropout_prob: Hidden dropout (default: 0.2)
            hidden_act: Activation function ("gelu" or "relu", default: "gelu")
        """
        super(BERT4Rec, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Embedding layer
        self.embeddings = BertEmbedding(
            vocab_size, hidden_size, max_position_embeddings, hidden_dropout_prob
        )
        
        # Encoder
        self.encoder = BertEncoder(
            num_hidden_layers, hidden_size, num_attention_heads, intermediate_size,
            attention_probs_dropout_prob, hidden_dropout_prob, hidden_act
        )
        
        # Output layer for prediction
        self.predictor = nn.Linear(hidden_size, vocab_size)
        
        # Initialize predictor
        nn.init.normal_(self.predictor.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.predictor.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: [batch_size, seq_len] - item IDs
            attention_mask: [batch_size, seq_len] - 1 for valid, 0 for mask
            token_type_ids: [batch_size, seq_len] - 0 for history, 1 for candidate
        
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        # Embeddings
        embedding_output = self.embeddings(input_ids, token_type_ids)
        
        # Encoder
        encoder_outputs = self.encoder(embedding_output, attention_mask)
        
        # Prediction
        prediction_scores = self.predictor(encoder_outputs)
        
        return prediction_scores
    
    def predict_scores(
        self,
        history: torch.Tensor,
        candidates: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict scores for candidates given history.
        
        For each candidate, create sequence [history, candidate] and predict score.
        Uses batch processing for efficiency.
        
        Args:
            history: [batch_size, history_len] - user history item IDs
            candidates: [batch_size, num_candidates] - candidate item IDs
            attention_mask: [batch_size, history_len] - attention mask for history
        
        Returns:
            Scores [batch_size, num_candidates]
        """
        batch_size = history.size(0)
        history_len = history.size(1)
        num_candidates = candidates.size(1)
        
        # Validate all IDs are within valid range [0, vocab_size-1]
        # Note: 0 is padding token, 1..vocab_size-1 are valid item IDs
        vocab_size = self.embeddings.item_embedding.num_embeddings
        if torch.any(history < 0) or torch.any(history >= vocab_size):
            invalid_history = torch.logical_or(history < 0, history >= vocab_size)
            print(f"[BERT4Rec.predict_scores] Warning: Found invalid history IDs. "
                  f"Valid range: [0, {vocab_size-1}], but found values outside this range.")
            # Clamp to valid range
            history = torch.clamp(history, 0, vocab_size - 1)
        
        if torch.any(candidates < 0) or torch.any(candidates >= vocab_size):
            invalid_candidates = torch.logical_or(candidates < 0, candidates >= vocab_size)
            print(f"[BERT4Rec.predict_scores] Warning: Found invalid candidate IDs. "
                  f"Valid range: [0, {vocab_size-1}], but found values outside this range.")
            # Clamp to valid range
            candidates = torch.clamp(candidates, 0, vocab_size - 1)
        
        # Expand history for each candidate: [batch, num_cand, history_len]
        history_expanded = history.unsqueeze(1).expand(batch_size, num_candidates, history_len)
        history_flat = history_expanded.reshape(batch_size * num_candidates, history_len)
        
        # Flatten candidates: [batch*num_cand, 1]
        candidates_flat = candidates.reshape(batch_size * num_candidates, 1)
        
        # Build input: [history, candidate] for each candidate
        input_ids = torch.cat([history_flat, candidates_flat], dim=1)  # [batch*num_cand, history_len+1]
        
        # Token type: 0 for history, 1 for candidate
        token_type_ids = torch.zeros_like(input_ids)
        token_type_ids[:, -1] = 1  # Last token is candidate
        
        # Attention mask
        if attention_mask is None:
            seq_attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask_expanded = attention_mask.unsqueeze(1).expand(
                batch_size, num_candidates, history_len
            )
            attention_mask_flat = attention_mask_expanded.reshape(
                batch_size * num_candidates, history_len
            )
            candidate_mask = torch.ones(batch_size * num_candidates, 1, device=input_ids.device)
            seq_attention_mask = torch.cat([attention_mask_flat, candidate_mask], dim=1)
        
        # Forward pass (batch all candidates together)
        logits = self.forward(input_ids, seq_attention_mask, token_type_ids)  # [batch*num_cand, seq_len, vocab_size]
        
        # Get logits for candidate position (last position)
        candidate_logits = logits[:, -1, :]  # [batch*num_cand, vocab_size]
        
        # Extract score for each candidate
        candidate_scores = candidate_logits.gather(1, candidates_flat)  # [batch*num_cand, 1]
        
        # Reshape back to [batch_size, num_candidates]
        scores = candidate_scores.reshape(batch_size, num_candidates)
        
        return scores

