"""MMGCN-based retriever using multimodal graph convolutional network."""

from typing import Dict, List, Set, Any, Optional
import torch
import torch.nn.functional as F
import numpy as np

from retrieval.base import BaseRetriever
from retrieval.models.mmgcn import Net


class MMGCNRetriever(BaseRetriever):
    """Retriever sử dụng MMGCN (Multimodal Graph Convolutional Network).
    
    Wrapper cho Net model để implement BaseRetriever interface.
    MMGCN sử dụng visual và text features từ CLIP embeddings.
    """

    def __init__(
        self,
        top_k: int = 50,
        dim_x: int = 64,
        aggr_mode: str = "add",
        concate: bool = False,
        num_layer: int = 2,
        has_id: bool = True,
        reg_weight: float = 1e-4,
        batch_size: int = 512,
        num_epochs: int = 10,
        lr: float = 1e-3,
        patience: Optional[int] = None,
    ) -> None:
        """
        Args:
            top_k: Số lượng candidates trả về
            dim_x: Embedding dimension
            aggr_mode: Aggregation mode ("add", "mean", "max")
            concate: Có concat features không
            num_layer: Số GCN layers
            has_id: Có dùng ID embedding không
            reg_weight: Regularization weight
            batch_size: Batch size cho training
            num_epochs: Số epochs
            lr: Learning rate
            patience: Early stopping patience
        """
        super().__init__(top_k=top_k)
        self.dim_x = dim_x
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.num_layer = num_layer
        self.has_id = has_id
        self.reg_weight = reg_weight
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.patience = patience
        
        self.model: Optional[Net] = None
        self.num_user: Optional[int] = None
        self.num_item: Optional[int] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.user_history: Dict[int, List[int]] = {}  # Store training history for masking

    def fit(
        self,
        train_data: Dict[int, List[int]],
        **kwargs: Any
    ) -> None:
        """Train MMGCN model.
        
        Args:
            train_data: Dict {user_id: [item_ids]} - training interactions
            **kwargs: Additional arguments:
                - num_user: int - số users
                - num_item: int - số items
                - v_feat: np.ndarray - visual features [num_items, D]
                - t_feat: np.ndarray - text features [num_items, D]
                - edge_index: np.ndarray - graph edges [2, E]
        """
        self.num_user = kwargs.get("num_user")
        self.num_item = kwargs.get("num_item")
        v_feat = kwargs.get("v_feat")
        t_feat = kwargs.get("t_feat")
        edge_index = kwargs.get("edge_index")
        
        if any(x is None for x in [self.num_user, self.num_item, v_feat, t_feat, edge_index]):
            raise ValueError(
                "MMGCNRetriever.fit requires: num_user, num_item, v_feat, t_feat, edge_index"
            )
        
        # Build user-item dict for training
        user_item_dict = {u: items for u, items in train_data.items()}
        
        # Store training history for masking in evaluation
        self.user_history = train_data
        
        # Print graph statistics
        num_edges = edge_index.shape[1] if hasattr(edge_index, 'shape') else len(edge_index[0]) if isinstance(edge_index, (list, tuple)) else 0
        num_nodes = self.num_user + self.num_item
        print(f"[MMGCNRetriever] Graph statistics:")
        print(f"  Nodes: {num_nodes} (users: {self.num_user}, items: {self.num_item})")
        print(f"  Edges: {num_edges}")
        print(f"  Note: MMGCN processes ALL nodes in each forward pass, regardless of batch size!")
        
        # Initialize model
        words_tensor = None  # Not used in current implementation
        self.model = Net(
            v_feat=v_feat,
            t_feat=t_feat,
            words_tensor=words_tensor,
            edge_index=edge_index,
            batch_size=self.batch_size,
            num_user=self.num_user,
            num_item=self.num_item,
            aggr_mode=self.aggr_mode,
            concate=self.concate,
            num_layer=self.num_layer,
            has_id=self.has_id,
            user_item_dict=user_item_dict,
            reg_weight=self.reg_weight,
            dim_x=self.dim_x,
        ).to(self.device)
        
        # Training loop
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best_state = None
        best_val_recall = -1.0
        epochs_no_improve = 0
        
        val_data = kwargs.get("val_data")
        test_data = kwargs.get("test_data", {})  # Get test_data if provided (for negative sampling)
        
        # Prepare training samples: mỗi positive item trong training data sẽ có 1 negative tương ứng
        train_samples = []
        all_items = set(range(1, self.num_item + 1))
        
        for user_id, items in train_data.items():
            if len(items) < 1:
                continue
            
            # ✅ CRITICAL FIX: Exclude val and test items from negative candidates
            # Spec requires: "Negative items must never appear in user's interaction history,
            # exclude validation and test items"
            user_train_items = set(items)  # Training history
            user_val_items = set(val_data.get(user_id, [])) if val_data else set()
            user_test_items = set(test_data.get(user_id, [])) if test_data else set()
            
            # Negative candidates: all items EXCEPT train, val, and test items
            excluded_items = user_train_items | user_val_items | user_test_items
            neg_candidates = list(all_items - excluded_items)
            
            if len(neg_candidates) == 0:
                # Fallback: if no valid negatives (shouldn't happen in practice), skip this user
                continue
            
            # Với mỗi positive item trong training data, sample 1 negative tương ứng
            for pos_item in items:
                neg_item = np.random.choice(neg_candidates)
                train_samples.append((user_id, pos_item, neg_item))
        
        print(f"[MMGCNRetriever] Generated {len(train_samples)} training samples")
        expected_batches = (len(train_samples) + self.batch_size - 1) // self.batch_size
        print(f"[MMGCNRetriever] Batch size: {self.batch_size}, Expected batches per epoch: {expected_batches}")
        
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            num_batches = 0
            total_samples_processed = 0
            
            # Shuffle training samples
            np.random.shuffle(train_samples)
            
            # Process in batches
            for i in range(0, len(train_samples), self.batch_size):
                batch_samples = train_samples[i:i + self.batch_size]
                
                # Prepare batch tensors
                user_ids_batch = []
                pos_items_batch = []
                neg_items_batch = []
                
                for user_id, pos_item, neg_item in batch_samples:
                    user_ids_batch.append(user_id - 1)  # 0-indexed
                    pos_items_batch.append(self.num_user + pos_item - 1)  # Graph index
                    neg_items_batch.append(self.num_user + neg_item - 1)  # Graph index
                
                # Convert to tensors: [batch_size] for users, [batch_size*2] for items (pos+neg)
                user_tensor = torch.tensor(user_ids_batch, dtype=torch.long).to(self.device)
                # Repeat each user_id twice (for pos and neg)
                user_tensor = user_tensor.repeat_interleave(2)  # [batch_size*2]
                
                item_tensor = torch.tensor(
                    [[pos, neg] for pos, neg in zip(pos_items_batch, neg_items_batch)],
                    dtype=torch.long
                ).view(-1).to(self.device)  # [batch_size*2]
                
                optimizer.zero_grad()
                
                loss, _, _, _, _ = self.model.loss(
                    user_tensor,
                    item_tensor
                )
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                total_samples_processed += len(batch_samples)
            
            avg_loss = total_loss / max(1, num_batches)
            
            # Verify all samples were processed
            if epoch == 0:
                print(f"[MMGCNRetriever] Epoch {epoch+1}: Processed {total_samples_processed}/{len(train_samples)} samples in {num_batches} batches")
                if total_samples_processed != len(train_samples):
                    print(f"[MMGCNRetriever] WARNING: Not all samples were processed! Expected {len(train_samples)}, got {total_samples_processed}")
            
            # Validation
            if val_data is not None:
                self.model.eval()
                with torch.no_grad():
                    self.model.forward()  # Update self.result
                
                val_recall = self._evaluate_split(val_data, k=min(10, self.top_k))
                
                if val_recall > best_val_recall:
                    best_val_recall = val_recall
                    best_state = self.model.state_dict().copy()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                
                print(f"[MMGCNRetriever] Epoch {epoch+1}/{self.num_epochs} - "
                      f"loss: {avg_loss:.4f}, val_Recall@{min(10, self.top_k)}: {val_recall:.4f}")
                
                if self.patience and epochs_no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                print(f"[MMGCNRetriever] Epoch {epoch+1}/{self.num_epochs} - loss: {avg_loss:.4f}")
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        self.is_fitted = True

    def _evaluate_split(self, split: Dict[int, List[int]], k: int) -> float:
        """Compute average Recall@K for validation (optimized with batch processing)."""
        if self.model is None:
            return 0.0
        
        self.model.eval()
        with torch.no_grad():
            self.model.forward()  # Update self.result
        
        user_tensor = self.model.result[:self.num_user]  # [num_user, dim]
        item_tensor = self.model.result[self.num_user:]  # [num_item, dim]
        
        # Prepare batch data
        user_ids_list = []
        gt_items_list = []
        history_items_list = []
        
        skipped_users = 0
        for user_id, gt_items in split.items():
            if user_id > self.num_user or user_id < 1:
                skipped_users += 1
                continue
            user_ids_list.append(user_id)
            gt_items_list.append(gt_items)
            history_items_list.append(self.user_history.get(user_id, []))
        
        if skipped_users > 0:
            print(f"[MMGCNRetriever] Warning: Skipped {skipped_users} users with user_id > {self.num_user} or < 1")
        
        if not user_ids_list:
            print(f"[MMGCNRetriever] Warning: No valid users in split (total: {len(split)}, skipped: {skipped_users}, num_user: {self.num_user})")
            return 0.0
        
        recalls = []
        # Batch size for evaluation (can be adjusted based on GPU memory)
        batch_size = 512
        
        with torch.no_grad():
            for i in range(0, len(user_ids_list), batch_size):
                batch_user_ids = user_ids_list[i:i + batch_size]
                batch_gt_items = gt_items_list[i:i + batch_size]
                batch_history_items = history_items_list[i:i + batch_size]
                
                # Get user embeddings for batch
                # Convert user_ids from 1-indexed to 0-indexed for indexing into user_tensor
                batch_user_indices = torch.tensor([uid - 1 for uid in batch_user_ids], dtype=torch.long).to(self.device)
                batch_user_emb = user_tensor[batch_user_indices]  # [batch_size, dim]
                
                # Compute scores for all items for batch users [batch_size, num_items]
                scores_batch = torch.matmul(batch_user_emb, item_tensor.t())  # [batch_size, num_items]
                
                # Mask history items for each user in batch
                for j, (user_id, history_items) in enumerate(zip(batch_user_ids, batch_history_items)):
                    for item in history_items:
                        if 1 <= item <= self.num_item:
                            item_idx = item - 1  # Convert to 0-indexed
                            scores_batch[j, item_idx] = -1e9
                
                # Get top-K for each user in batch
                _, top_items_batch = torch.topk(scores_batch, k=k, dim=1)  # [batch_size, k]
                top_items_batch = (top_items_batch.cpu().numpy() + 1)  # Convert back to 1-indexed
                
                # Compute recall for each user in batch
                for j, (top_items, gt_items) in enumerate(zip(top_items_batch, batch_gt_items)):
                    top_items_list = top_items.tolist()
                    hits = len(set(top_items_list) & set(gt_items))
                    if len(gt_items) > 0:
                        recalls.append(hits / len(gt_items))  # ✅ FIX: Use standard recall formula
        
        return float(np.mean(recalls)) if recalls else 0.0

    def retrieve(
        self,
        user_id: int,
        exclude_items: Set[int] = None
    ) -> List[int]:
        """Retrieve top-K candidates cho một user.
        
        Args:
            user_id: ID của user
            exclude_items: Set các items cần loại trừ
            
        Returns:
            List[int]: Top-K item IDs
        """
        self._validate_fitted()
        
        if self.model is None or self.num_user is None or self.num_item is None:
            raise RuntimeError("MMGCNRetriever model not initialized")
        
        if user_id > self.num_user or user_id < 1:
            return []
        
        exclude_items = exclude_items or set()
        
        # Get user and item embeddings
        self.model.eval()
        with torch.no_grad():
            self.model.forward()  # Update self.result
        
        # Convert user_id from 1-indexed to 0-indexed for indexing into result
        user_emb = self.model.result[user_id - 1:user_id]  # [1, dim]
        item_emb = self.model.result[self.num_user:]  # [num_item, dim]
        
        # Compute scores
        scores = torch.matmul(user_emb, item_emb.t()).squeeze(0)
        
        # Mask excluded items
        for item in exclude_items:
            if 1 <= item <= self.num_item:
                item_idx = item - 1  # Convert to 0-indexed
                scores[item_idx] = -1e9
        
        # Get top-K
        k = min(self.top_k, self.num_item)
        _, indices = torch.topk(scores, k=k)
        
        # Convert back to 1-indexed item IDs
        return [int(idx.item() + 1) for idx in indices]

