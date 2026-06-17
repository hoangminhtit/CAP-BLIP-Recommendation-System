"""BERT4Rec-based retriever using bidirectional encoder for sequential recommendation."""

from typing import Dict, List, Set, Any, Optional
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

from retrieval.base import BaseRetriever
from rerank.models.bert4rec import BERT4Rec
from evaluation.metrics import recall_at_k, ndcg_at_k


class BERT4RecRetriever(BaseRetriever):
    """Retriever sử dụng BERT4Rec (Bidirectional Encoder Representations for Sequential Recommendation).
    
    BERT4Rec uses bidirectional BERT encoder to learn sequential patterns from user history.
    Based on the implementation at https://github.com/FeiSun/BERT4Rec
    
    Reference:
        Sun, F., et al. (2019). BERT4Rec: Sequential Recommendation with Bidirectional Encoder 
        Representations from Transformer. CIKM.
    """

    def __init__(
        self,
        top_k: int = 50,
        hidden_size: int = 64,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 2,
        intermediate_size: int = 256,
        max_seq_length: int = 200,
        attention_probs_dropout_prob: float = 0.2,
        hidden_dropout_prob: float = 0.2,
        hidden_act: str = "gelu",
        batch_size: int = 32,
        num_epochs: int = 10,
        lr: float = 1e-4,
        num_warmup_steps: int = 100,
        patience: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            top_k: Số lượng items trả về sau retrieval
            hidden_size: Hidden dimension (default: 64)
            num_hidden_layers: Number of transformer layers (default: 2)
            num_attention_heads: Number of attention heads (default: 2)
            intermediate_size: Feed-forward intermediate size (default: 256)
            max_seq_length: Maximum sequence length (default: 200)
            attention_probs_dropout_prob: Attention dropout (default: 0.2)
            hidden_dropout_prob: Hidden dropout (default: 0.2)
            hidden_act: Activation function ("gelu" or "relu", default: "gelu")
            batch_size: Batch size cho training (default: 32)
            num_epochs: Số epochs (default: 10)
            lr: Learning rate (default: 1e-4)
            num_warmup_steps: Warmup steps for learning rate (default: 100)
            patience: Early stopping patience (None = no early stopping)
            device: Device to use ("cuda" or "cpu", auto-detect if None)
        """
        super().__init__(top_k=top_k)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_seq_length = max_seq_length
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_act = hidden_act
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.num_warmup_steps = num_warmup_steps
        self.patience = patience
        
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: Optional[BERT4Rec] = None
        self.vocab_size: Optional[int] = None
        self.user_history: Dict[int, List[int]] = {}  # user_id -> [item_ids] (sequential)

    def fit(
        self,
        train_data: Dict[int, List[int]],
        **kwargs: Any
    ) -> None:
        """Train BERT4Rec model.
        
        Args:
            train_data: Dict {user_id: [item_ids]} - training interactions (sequential)
            **kwargs: Additional arguments:
                - vocab_size: int - vocabulary size (number of items)
                - val_data: Dict[int, List[int]] - validation data for early stopping
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
        
        # Get vocab_size or item_count
        # vocab_size = item_count + 1 (for padding token 0)
        if "vocab_size" in kwargs:
            self.vocab_size = kwargs["vocab_size"]
        elif "item_count" in kwargs:
            self.vocab_size = kwargs["item_count"] + 1  # +1 for padding token (0)
        else:
            # Infer from train_data
            all_items = set()
            for items in train_data.values():
                all_items.update(items)
            max_item = max(all_items) if all_items else 0
            if max_item == 0:
                raise ValueError("Cannot infer vocab_size from train_data")
            # vocab_size = max_item + 1 (max_item is the highest item_id, +1 for padding)
            self.vocab_size = max_item + 1
        
        # Validate and filter items to ensure all are within valid range [1, vocab_size-1]
        self.user_history = {}
        invalid_items_count = 0
        for uid, items in train_data.items():
            valid_items = [item for item in items if 1 <= item < self.vocab_size]
            if len(valid_items) > 0:
                self.user_history[uid] = valid_items
            if len(valid_items) < len(items):
                invalid_items_count += len(items) - len(valid_items)
        
        if invalid_items_count > 0:
            print(f"[BERT4RecRetriever] Warning: Filtered {invalid_items_count} invalid items "
                  f"(outside range [1, {self.vocab_size-1}]) from training data")
        
        # Initialize model
        self.model = BERT4Rec(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            max_position_embeddings=self.max_seq_length,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            hidden_dropout_prob=self.hidden_dropout_prob,
            hidden_act=self.hidden_act,
        ).to(self.device)
        
        # Prepare training samples (masked sequences)
        train_samples = self._prepare_training_samples(train_data)
        
        # Optimizer with warmup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Learning rate scheduler with warmup
        def lr_lambda(step):
            if step < self.num_warmup_steps:
                return float(step) / float(max(1, self.num_warmup_steps))
            return 1.0
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        best_state = None
        best_val_recall = -1.0
        epochs_no_improve = 0
        val_data = kwargs.get("val_data")
        
        global_step = 0
        
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            num_batches = 0
            
            # Shuffle training samples
            np.random.shuffle(train_samples)
            
            # Training batches
            for i in range(0, len(train_samples), self.batch_size):
                batch = train_samples[i:i + self.batch_size]
                if len(batch) == 0:
                    continue
                
                # Prepare batch
                input_ids, masked_lm_labels, attention_mask = self._prepare_batch(batch)
                
                input_ids = input_ids.to(self.device)
                masked_lm_labels = masked_lm_labels.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                prediction_scores = self.model(input_ids, attention_mask)
                
                # Compute masked LM loss
                loss_fct = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
                masked_lm_loss = loss_fct(
                    prediction_scores.view(-1, self.vocab_size),
                    masked_lm_labels.view(-1)
                )
                
                masked_lm_loss.backward()
                optimizer.step()
                scheduler.step()
                global_step += 1
                
                total_loss += masked_lm_loss.item()
                num_batches += 1
            
            avg_loss = total_loss / max(1, num_batches)
            
            # Validation
            if val_data is not None:
                val_metrics = self._evaluate_split(val_data, k=min(10, self.top_k))
                val_recall = val_metrics["recall"]
                
                if val_recall > best_val_recall:
                    best_val_recall = val_recall
                    best_state = self.model.state_dict().copy()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                
                print(f"[BERT4RecRetriever] Epoch {epoch+1}/{self.num_epochs} - "
                      f"loss: {avg_loss:.4f}, val_Recall@{min(10, self.top_k)}: {val_metrics['recall']:.4f}, "
                      f"val_NDCG@{min(10, self.top_k)}: {val_metrics['ndcg']:.4f}")
                
                if self.patience and epochs_no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                print(f"[BERT4RecRetriever] Epoch {epoch+1}/{self.num_epochs} - loss: {avg_loss:.4f}")
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        self.is_fitted = True

    def _prepare_training_samples(
        self,
        train_data: Dict[int, List[int]]
    ) -> List[Dict]:
        """Prepare masked training samples for BERT4Rec.
        
        BERT4Rec uses masked language modeling: mask some items in sequence and predict them.
        
        Args:
            train_data: Dict {user_id: [item_ids]}
            
        Returns:
            List of training samples with masked sequences
        """
        samples = []
        mask_prob = 0.15  # Mask 15% of items (following BERT)
        
        for user_id, items in train_data.items():
            if len(items) < 2:
                continue
            
            # Truncate to max_seq_length
            seq = items[-self.max_seq_length:]
            
            # Filter to valid range [1, vocab_size-1] (exclude 0 and >= vocab_size)
            seq = [item for item in seq if 1 <= item < self.vocab_size]
            if len(seq) < 2:
                continue
            
            # Create masked sequence
            input_ids = seq.copy()
            masked_lm_labels = [0] * len(seq)  # 0 = not masked
            
            # Mask some items
            num_masked = max(1, int(len(seq) * mask_prob))
            masked_positions = np.random.choice(len(seq), size=num_masked, replace=False)
            
            for pos in masked_positions:
                # 80% of the time, replace with [MASK] token (vocab_size-1)
                # 10% of the time, keep original
                # 10% of the time, replace with random item
                rand = np.random.random()
                if rand < 0.8:
                    input_ids[pos] = self.vocab_size - 1  # [MASK] token
                elif rand < 0.9:
                    # Keep original
                    pass
                else:
                    # Random item
                    input_ids[pos] = np.random.randint(1, self.vocab_size - 1)
                
                masked_lm_labels[pos] = seq[pos]  # True label
            
            samples.append({
                "input_ids": input_ids,
                "masked_lm_labels": masked_lm_labels,
            })
        
        return samples

    def _prepare_batch(
        self,
        batch: List[Dict]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare batch for training.
        
        Args:
            batch: List of training samples
        
        Returns:
            Tuple of (input_ids, masked_lm_labels, attention_mask)
        """
        batch_size = len(batch)
        max_len = max(len(s["input_ids"]) for s in batch)
        max_len = min(max_len, self.max_seq_length)
        
        input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        masked_lm_labels = torch.zeros(batch_size, max_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        
        for i, sample in enumerate(batch):
            seq_len = min(len(sample["input_ids"]), max_len)
            input_ids[i, :seq_len] = torch.tensor(sample["input_ids"][:seq_len])
            masked_lm_labels[i, :seq_len] = torch.tensor(sample["masked_lm_labels"][:seq_len])
            attention_mask[i, :seq_len] = 1
        
        return input_ids, masked_lm_labels, attention_mask

    def _evaluate_split(self, split: Dict[int, List[int]], k: int) -> Dict[str, float]:
        """Compute average Recall@K and NDCG@K for validation.
        
        Used for per-epoch validation during training.
        Similar to LRURecRetriever - full ranking evaluation.
        """
        if self.model is None:
            return {"recall": 0.0, "ndcg": 0.0, "num_users": 0}
        
        users = [u for u in sorted(split.keys()) if u in self.user_history and split.get(u)]
        recalls, ndcgs = [], []

        if not users:
            return {"recall": 0.0, "ndcg": 0.0, "num_users": 0}

        batch_size = max(1, self.batch_size)
        self.model.eval()

        with torch.no_grad():
            for start in range(0, len(users), batch_size):
                batch_users = users[start : start + batch_size]
                
                # Get sequences for batch
                seq_batch = []
                for uid in batch_users:
                    seq = self.user_history.get(uid, [])
                    # Filter to valid range
                    seq = [item for item in seq if 1 <= item < self.vocab_size]
                    if len(seq) == 0:
                        continue
                    seq_batch.append(seq[-self.max_seq_length:])
                
                if not seq_batch:
                    continue
                
                # Pad sequences
                max_len = max(len(seq) for seq in seq_batch)
                max_len = min(max_len, self.max_seq_length)
                
                input_ids = torch.zeros(len(seq_batch), max_len, dtype=torch.long).to(self.device)
                attention_mask = torch.zeros(len(seq_batch), max_len, dtype=torch.long).to(self.device)
                
                for i, seq in enumerate(seq_batch):
                    seq_len = min(len(seq), max_len)
                    input_ids[i, :seq_len] = torch.tensor(seq[:seq_len])
                    attention_mask[i, :seq_len] = 1
                
                # Predict scores for all items [batch_size, max_len, vocab_size]
                prediction_scores = self.model(input_ids, attention_mask)
                
                # Get scores at last position [batch_size, vocab_size]
                last_scores = prediction_scores[:, -1, :]  # [batch_size, vocab_size]
                
                # Process each user in batch
                for i, uid in enumerate(batch_users):
                    if i >= len(seq_batch):
                        continue
                    
                    scores = last_scores[i].cpu().numpy()  # [vocab_size]
                    seq = self.user_history.get(uid, [])
                    gt_items = split.get(uid, [])
                    
                    # Filter gt_items to valid range
                    valid_gt_items = [item for item in gt_items if 1 <= item < self.vocab_size]
                    if not valid_gt_items:
                        continue
                    
                    # Mask out history items and padding (0)
                    history_set = set(seq)
                    for item in history_set:
                        if 1 <= item < self.vocab_size:
                            scores[item] = -1e9
                    scores[0] = -1e9  # Padding token
                    
                    # Get top-K items
                    top_k_indices = np.argsort(scores)[::-1][:k]
                    top_k_items = [int(idx) for idx in top_k_indices if 1 <= idx < self.vocab_size]
                    
                    # Compute metrics
                    recall = recall_at_k(valid_gt_items, top_k_items, k)
                    ndcg = ndcg_at_k(valid_gt_items, top_k_items, k)
                    
                    recalls.append(recall)
                    ndcgs.append(ndcg)

        return {
            "recall": float(np.mean(recalls)) if recalls else 0.0,
            "ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "num_users": len(users),
        }

    def retrieve(self, user_id: int, exclude_items: Set[int] = None) -> List[int]:
        """Retrieve top-K candidates cho một user.
        
        Args:
            user_id: ID của user
            exclude_items: Set các items cần loại trừ (đã tương tác)
            
        Returns:
            List[int]: Top-K item IDs (sorted by score descending)
        """
        self._validate_fitted()
        
        if self.model is None:
            raise RuntimeError("BERT4RecRetriever model not initialized")
        
        exclude_items = exclude_items or set()
        
        # Get user history
        history = self.user_history.get(user_id, [])
        if len(history) == 0:
            return []
        
        # Filter history to valid range [1, vocab_size-1]
        valid_history = [item for item in history if 1 <= item < self.vocab_size]
        if not valid_history:
            return []
        
        # Prepare history tensor
        history_seq = valid_history[-self.max_seq_length:]  # Truncate to max length
        history_tensor = torch.tensor([history_seq], dtype=torch.long).to(self.device)
        attention_mask = torch.ones(1, len(history_seq), dtype=torch.long).to(self.device)
        
        # Predict scores for all items
        self.model.eval()
        with torch.no_grad():
            prediction_scores = self.model(history_tensor, attention_mask)  # [1, seq_len, vocab_size]
            # Get scores at last position [1, vocab_size]
            scores = prediction_scores[0, -1, :].cpu().numpy()  # [vocab_size]
        
        # Mask out history items and padding (0)
        history_set = set(valid_history)
        for item in history_set:
            if 1 <= item < self.vocab_size:
                scores[item] = -1e9
        scores[0] = -1e9  # Padding token
        
        # Mask out exclude_items
        for item in exclude_items:
            if 1 <= item < self.vocab_size:
                scores[item] = -1e9
        
        # Get top-K items
        k = min(self.top_k, self.vocab_size - 1)
        top_k_indices = np.argsort(scores)[::-1][:k]
        top_k_items = [int(idx) for idx in top_k_indices if 1 <= idx < self.vocab_size]
        
        return top_k_items

