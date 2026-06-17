from typing import Dict, List, Set

from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from retrieval.base import BaseRetriever
from retrieval.models.neural_lru import NeuralLRUConfig, NeuralLRURec
from evaluation.metrics import recall_at_k, ndcg_at_k


class _NeuralLRUTrainDataset(Dataset):
    """Simple sequence dataset for training NeuralLRURec.

    For each user sequence, we create (tokens, labels):
    - `tokens`: history without the last item, padded/truncated to max_len.
    - `labels`: full history, padded/truncated to max_len.

    This works with a standard CrossEntropyLoss over all positions.
    """

    def __init__(self, user2seq: Dict[int, List[int]], max_len: int) -> None:
        self.max_len = max_len
        self.samples: List[tuple[List[int], List[int]]] = []

        for seq in user2seq.values():
            if len(seq) < 2:
                continue
            labels = seq[-self.max_len :]
            tokens = seq[:-1][-self.max_len :]

            pad_t = self.max_len - len(tokens)
            pad_l = self.max_len - len(labels)
            tokens = [0] * pad_t + tokens
            labels = [0] * pad_l + labels
            self.samples.append((tokens, labels))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        tokens, labels = self.samples[idx]
        return torch.LongTensor(tokens), torch.LongTensor(labels)


class LRURecRetriever(BaseRetriever):
    """Neural LRURec-based retriever using an internal implementation.

    - Does NOT import or depend on `LlamaRec/`.
    - `fit` runs a small number of training epochs on user histories.
    - `retrieve` feeds a user's sequence and returns top-K items from the
      predicted scores at the last position.
    """

    def __init__(
        self,
        top_k: int = 50,
        max_len: int = 50,
        hidden_units: int = 64,
        num_blocks: int = 2,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        batch_size: int = 128,
        patience: int | None = None,
        num_workers: int = 0,
        num_epochs: int = 3,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(top_k=top_k)
        self.user_history: Dict[int, List[int]] = {}
        self.max_len = max_len
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.batch_size = batch_size
        self.patience = patience
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay

        self.model: NeuralLRURec | None = None
        self.item_count: int | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, train_data: Dict[int, List[int]], **kwargs) -> None:
        """Train NeuralLRURec on user→sequence data.

        Expected kwargs:
        - item_count: total number of items (len(smap)).
        """

        item_count = kwargs.get("item_count")
        if item_count is None:
            raise ValueError("LRURecRetriever.fit requires 'item_count' in kwargs")
        self.item_count = int(item_count)
        self.user_history = train_data
        val_data: Dict[int, List[int]] | None = kwargs.get("val_data")

        cfg = NeuralLRUConfig(
            num_items=self.item_count,
            hidden_units=self.hidden_units,
            num_blocks=self.num_blocks,
            dropout=self.dropout,
            attn_dropout=self.attn_dropout,
        )

        model = NeuralLRURec(cfg).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        train_dataset = _NeuralLRUTrainDataset(train_data, max_len=self.max_len)
        if len(train_dataset) == 0:
            self.model = model
            self.is_fitted = True
            return

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        best_state = None
        best_val_recall = -1.0
        epochs_no_improve = 0
        best_epoch = 0

        model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            seen = 0
            for tokens, labels in train_loader:
                tokens = tokens.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                scores = model(tokens)  # [B, L, num_items+1]
                logits = scores.view(-1, scores.size(-1))
                labels_flat = labels.view(-1)
                loss = criterion(logits, labels_flat)
                loss.backward()
                optimizer.step()

                batch_size = tokens.size(0)
                seen += batch_size
                total_loss += float(loss.item()) * batch_size

            avg_loss = total_loss / max(1, seen)
            log_msg = f"[LRURecRetriever] Epoch {epoch+1}/{self.num_epochs} - loss: {avg_loss:.4f}"

            # Optional validation on val_data after each epoch.
            if val_data is not None and len(val_data) > 0:
                # Temporarily attach current model for retrieval-based eval.
                self.model = model
                self.is_fitted = True
                val_metrics = self._evaluate_split(val_data, k=min(10, self.top_k))
                val_recall = val_metrics["recall"]
                log_msg += f", val_Recall@{min(10, self.top_k)}: {val_recall:.4f}, val_NDCG: {val_metrics['ndcg']:.4f}"

                if val_recall > best_val_recall:
                    best_val_recall = val_recall
                    best_state = deepcopy(model.state_dict())
                    log_msg += " [BEST]"
                    epochs_no_improve = 0
                    best_epoch = epoch + 1
                else:
                    epochs_no_improve += 1

                # Early stopping nếu không cải thiện sau `patience` epoch.
                if self.patience is not None and epochs_no_improve >= self.patience:
                    print(f"[LRURecRetriever] Early stopping at epoch {epoch+1} (no improvement for {self.patience} epochs)")
                    print(log_msg)
                    break

            print(log_msg)

        # Load best model (by validation recall) if we have one.
        if best_state is not None:
            model.load_state_dict(best_state)

        self.model = model
        self.is_fitted = True
        self.best_state = best_epoch

    def _evaluate_split(self, split: Dict[int, List[int]], k: int) -> Dict[str, float]:
        """Compute average Recall@K and NDCG@K for a given split.

        Used for per-epoch validation during training.
        """
        users = [u for u in sorted(split.keys()) if split.get(u)]
        recalls, ndcgs = [], []

        if not users:
            return {"recall": 0.0, "ndcg": 0.0, "num_users": 0}

        batch_size = max(1, self.batch_size)

        for start in range(0, len(users), batch_size):
            batch_users = users[start : start + batch_size]

            seq_batch = []
            hist_batch = []
            for u in batch_users:
                seq = self.user_history.get(u, [])
                if not seq:
                    # Không có history thì bỏ user này khỏi batch
                    continue
                seq_tokens = seq[-self.max_len :]
                pad_len = self.max_len - len(seq_tokens)
                seq_tokens = [0] * pad_len + seq_tokens

                seq_batch.append(seq_tokens)
                hist_batch.append(seq)

            if not seq_batch:
                continue

            inputs = torch.LongTensor(seq_batch).to(self.device)
            with torch.no_grad():
                self.model.eval()
                scores_batch = self.model(inputs)[:, -1, :]  # [B, num_items+1]

            for i, u in enumerate(batch_users[: len(seq_batch)]):
                gt_items = split.get(u, [])
                if not gt_items:
                    continue

                scores = scores_batch[i].clone()

                # Mask history and padding index 0 giống retrieve()
                for item in hist_batch[i]:
                    if 0 < item <= self.item_count:
                        scores[item] = -1e9
                scores[0] = -1e9

                topk_k = min(self.top_k, self.item_count)
                _, indices = torch.topk(scores, k=topk_k)
                recs = [int(idx.item()) for idx in indices]

                recalls.append(recall_at_k(recs, gt_items, k))
                ndcgs.append(ndcg_at_k(recs, gt_items, k))

        if not recalls:
            return {"recall": 0.0, "ndcg": 0.0, "num_users": 0}

        return {
            "recall": float(sum(recalls) / len(recalls)),
            "ndcg": float(sum(ndcgs) / len(ndcgs)),
            "num_users": len(recalls),
        }

    def retrieve(self, user_id: int, exclude_items: Set[int] = None) -> List[int]:
        self._validate_fitted()
        if self.model is None or self.item_count is None:
            raise RuntimeError("LRURecRetriever model not initialized")

        exclude_items = exclude_items or set()
        seq = self.user_history.get(user_id, [])
        if not seq:
            return []

        # Build input sequence for this user.
        seq_tokens = seq[-self.max_len :]
        pad_len = self.max_len - len(seq_tokens)
        seq_tokens = [0] * pad_len + seq_tokens

        with torch.no_grad():
            self.model.eval()
            inputs = torch.LongTensor([seq_tokens]).to(self.device)
            scores = self.model(inputs)[:, -1, :]  # [1, num_items+1]
            scores = scores.squeeze(0)

        # Mask out history and padding index 0.
        for item in seq:
            if 0 < item <= self.item_count:
                scores[item] = -1e9
        scores[0] = -1e9

        for item in exclude_items:
            if 0 < item <= self.item_count:
                scores[item] = -1e9

        k = min(self.top_k, self.item_count)
        _, indices = torch.topk(scores, k=k)
        return [int(i.item()) for i in indices]
