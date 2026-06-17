"""VBPR-based retriever using Visual Bayesian Personalized Ranking."""

from typing import Dict, List, Set, Any, Optional
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from retrieval.base import BaseRetriever
from retrieval.models.vbpr import VBPR
from dataset.paths import get_clip_embeddings_path


class VBPRRetriever(BaseRetriever):
    """Retriever sử dụng VBPR (Visual Bayesian Personalized Ranking).
    
    VBPR combines collaborative filtering with visual features from images.
    Based on the implementation at https://github.com/aaossa/VBPR-PyTorch
    
    Reference:
        He, R., & McAuley, J. (2016). VBPR: visual bayesian personalized ranking 
        from implicit feedback. AAAI.
    """

    def __init__(
        self,
        top_k: int = 50,
        dim_gamma: int = 20,
        dim_theta: int = 20,
        batch_size: int = 64,
        num_epochs: int = 10,
        lr: float = 5e-4,
        lambda_reg: float = 0.01,
        optimizer: str = "adam",  # "sgd" or "adam" (default: adam for better convergence)
        patience: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            top_k: Số lượng candidates trả về
            dim_gamma: Dimension of user/item latent factors (default: 20)
            dim_theta: Dimension of visual projection (default: 20)
            batch_size: Batch size cho training (default: 64)
            num_epochs: Số epochs (default: 10)
            lr: Learning rate (default: 5e-4, following VBPR paper)
            lambda_reg: Regularization weight (default: 0.01)
            optimizer: Optimizer to use ("sgd" or "adam", default: "sgd")
            patience: Early stopping patience (None = no early stopping)
            device: Device to use ("cuda" or "cpu", auto-detect if None)
        """
        super().__init__(top_k=top_k)
        self.dim_gamma = dim_gamma
        self.dim_theta = dim_theta
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.lambda_reg = lambda_reg
        self.optimizer_name = optimizer.lower()
        self.patience = patience
        
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: Optional[VBPR] = None
        self.num_user: Optional[int] = None
        self.num_item: Optional[int] = None
        self.visual_features: Optional[torch.Tensor] = None
        self.user_history: Dict[int, List[int]] = {}  # Store training history for masking

    def fit(
        self,
        train_data: Dict[int, List[int]],
        **kwargs: Any
    ) -> None:
        """Train VBPR model.
        
        Args:
            train_data: Dict {user_id: [item_ids]} - training interactions
            **kwargs: Additional arguments:
                - num_user: int - số users
                - num_item: int - số items
                - visual_features: torch.Tensor - visual features [num_items, D] (optional)
                - dataset_code: str - dataset code (if visual_features not provided)
                - min_rating: int - min rating (if loading from file)
                - min_uc: int - min user count (if loading from file)
                - min_sc: int - min item count (if loading from file)
                - val_data: Dict[int, List[int]] - validation data for early stopping
        """
        self.num_user = kwargs.get("num_user")
        self.num_item = kwargs.get("num_item")
        
        if self.num_user is None or self.num_item is None:
            raise ValueError("VBPRRetriever.fit requires: num_user, num_item")
        
        # Load visual features
        visual_features = kwargs.get("visual_features")
        if visual_features is None:
            # Try to load from CLIP embeddings
            dataset_code = kwargs.get("dataset_code")
            min_rating = kwargs.get("min_rating")
            min_uc = kwargs.get("min_uc")
            min_sc = kwargs.get("min_sc")
            
            if all(x is not None for x in [dataset_code, min_rating, min_uc, min_sc]):
                visual_features = self._load_visual_features(
                    dataset_code, min_rating, min_uc, min_sc, self.num_item
                )
            else:
                raise ValueError(
                    "VBPR requires visual_features. Provide either:\n"
                    "  1. visual_features tensor directly, or\n"
                    "  2. dataset_code, min_rating, min_uc, min_sc to load from CLIP embeddings"
                )
        
        # Ensure visual_features is torch.Tensor
        if isinstance(visual_features, np.ndarray):
            visual_features = torch.from_numpy(visual_features).float()
        elif not isinstance(visual_features, torch.Tensor):
            raise TypeError(f"visual_features must be torch.Tensor or np.ndarray, got {type(visual_features)}")
        
        # Validate shape
        if visual_features.dim() != 2:
            raise ValueError(f"visual_features must be 2D [num_items, D], got shape {visual_features.shape}")
        
        if visual_features.size(0) != self.num_item:
            raise ValueError(
                f"visual_features shape mismatch: expected [{self.num_item}, D], "
                f"got {visual_features.shape}"
            )
        
        self.visual_features = visual_features.to(self.device)
        
        # Store training history for masking in evaluation
        self.user_history = train_data
        
        # Initialize model
        print(f"[VBPRRetriever] Model configuration:")
        print(f"  dim_gamma: {self.dim_gamma} {'(⚠️ Consider increasing to 64 for better performance)' if self.dim_gamma < 64 else ''}")
        print(f"  dim_theta: {self.dim_theta} {'(⚠️ Consider increasing to 64 for better performance)' if self.dim_theta < 64 else ''}")
        print(f"  lr: {self.lr} {'(⚠️ Consider increasing to 1e-3 if performance is low)' if self.lr < 1e-3 else ''}")
        print(f"  lambda_reg: {self.lambda_reg}")
        print(f"  visual_dim: {visual_features.size(1)}")
        
        self.model = VBPR(
            n_users=self.num_user,
            n_items=self.num_item,
            visual_features=self.visual_features,
            dim_gamma=self.dim_gamma,
            dim_theta=self.dim_theta,
        ).to(self.device)
        
        # Training loop
        # Note: VBPR paper uses SGD with lr=5e-4. If lr is too small (e.g., 1e-4), 
        # model may not learn effectively. Consider using Adam or increasing lr.
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            print(f"[VBPRRetriever] Using Adam optimizer with lr={self.lr}")
        else:
            if self.lr < 1e-3:
                print(f"[VBPRRetriever] WARNING: Learning rate {self.lr} may be too small for SGD.")
                print(f"  VBPR paper uses 5e-4. Consider using Adam optimizer (--optimizer adam) or increasing lr to 5e-4.")
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
            print(f"[VBPRRetriever] Using SGD optimizer with lr={self.lr}")
        best_state = None
        best_val_recall = -1.0
        epochs_no_improve = 0
        
        val_data = kwargs.get("val_data")
        test_data = kwargs.get("test_data", {})  # Get test_data if provided (for negative sampling)
        
        # Prepare training samples (user, pos_item, neg_item)
        # ✅ CRITICAL FIX: Pass val_data and test_data to exclude them from negative sampling
        train_samples = self._prepare_training_samples(train_data, val_data=val_data, test_data=test_data)
        
        if len(train_samples) == 0:
            raise ValueError("No training samples generated! Check train_data and num_item.")
        
        print(f"[VBPRRetriever] Generated {len(train_samples)} training samples")
        expected_batches = (len(train_samples) + self.batch_size - 1) // self.batch_size
        print(f"[VBPRRetriever] Batch size: {self.batch_size}, Expected batches per epoch: {expected_batches}")
        
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            num_batches = 0
            total_samples_processed = 0
            
            # Shuffle training samples
            np.random.shuffle(train_samples)
            
            # Training batches
            for i in range(0, len(train_samples), self.batch_size):
                batch = train_samples[i:i + self.batch_size]
                if len(batch) == 0:
                    continue
                
                user_ids = torch.tensor([s[0] for s in batch], dtype=torch.long).to(self.device)
                pos_ids = torch.tensor([s[1] for s in batch], dtype=torch.long).to(self.device)
                neg_ids = torch.tensor([s[2] for s in batch], dtype=torch.long).to(self.device)
                
                optimizer.zero_grad()
                
                loss = self.model.loss(
                    user_ids, pos_ids, neg_ids, lambda_reg=self.lambda_reg
                )
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                total_samples_processed += len(batch)
            
            avg_loss = total_loss / max(1, num_batches)
            
            # Verify all samples were processed
            if epoch == 0:
                print(f"[VBPRRetriever] Epoch {epoch+1}: Processed {total_samples_processed}/{len(train_samples)} samples in {num_batches} batches")
                if total_samples_processed != len(train_samples):
                    print(f"[VBPRRetriever] WARNING: Not all samples were processed! Expected {len(train_samples)}, got {total_samples_processed}")
            
            # Validation
            if val_data is not None:
                val_recall = self._evaluate_split(val_data, k=min(10, self.top_k))
                
                if val_recall > best_val_recall:
                    best_val_recall = val_recall
                    best_state = self.model.state_dict().copy()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                
                print(f"[VBPRRetriever] Epoch {epoch+1}/{self.num_epochs} - "
                      f"loss: {avg_loss:.4f}, val_Recall@{min(10, self.top_k)}: {val_recall:.4f}")
                
                if self.patience and epochs_no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                print(f"[VBPRRetriever] Epoch {epoch+1}/{self.num_epochs} - loss: {avg_loss:.4f}")
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        self.is_fitted = True

    def _prepare_training_samples(
        self,
        train_data: Dict[int, List[int]],
        val_data: Optional[Dict[int, List[int]]] = None,
        test_data: Optional[Dict[int, List[int]]] = None
    ) -> List[tuple]:
        """Prepare (user_id, pos_item_id, neg_item_id) training samples.
        
        Args:
            train_data: Dict {user_id: [item_ids]} - training interactions
            val_data: Optional dict {user_id: [item_ids]} - validation interactions (for negative exclusion)
            test_data: Optional dict {user_id: [item_ids]} - test interactions (for negative exclusion)
            
        Returns:
            List of (user_id, pos_item_id, neg_item_id) tuples
        """
        samples = []
        all_items = set(range(1, self.num_item + 1))
        val_data = val_data or {}
        test_data = test_data or {}
        
        for user_id, items in train_data.items():
            if len(items) < 1:
                continue
            
            # ✅ CRITICAL FIX: Exclude val and test items from negative candidates
            # Spec requires: "Negative items must never appear in user's interaction history,
            # exclude validation and test items"
            user_train_items = set(items)  # Training history
            user_val_items = set(val_data.get(user_id, []))
            user_test_items = set(test_data.get(user_id, []))
            
            # Negative candidates: all items EXCEPT train, val, and test items
            excluded_items = user_train_items | user_val_items | user_test_items
            neg_candidates = list(all_items - excluded_items)
            
            if len(neg_candidates) == 0:
                # Fallback: if no valid negatives (shouldn't happen in practice), skip this user
                continue
            
            # For each positive item, sample a negative
            for pos_item in items:
                neg_item = np.random.choice(neg_candidates)
                samples.append((user_id - 1, pos_item - 1, neg_item - 1))  # Convert to 0-indexed
        
        return samples

    def _evaluate_split(self, split: Dict[int, List[int]], k: int) -> float:
        """Compute average Recall@K for validation (optimized with batch processing)."""
        if self.model is None:
            return 0.0
        
        self.model.eval()
        recalls = []
        
        # Prepare batch data
        user_ids_list = []
        gt_items_list = []
        history_items_list = []
        
        skipped_users = 0
        for user_id, gt_items in split.items():
            # ✅ FIX: Check both user_id > num_user and user_id < 1 to match Hit calculation
            if user_id > self.num_user or user_id < 1:
                skipped_users += 1
                continue
            user_ids_list.append(user_id - 1)  # Convert to 0-indexed
            gt_items_list.append(gt_items)
            history_items_list.append(self.user_history.get(user_id, []))
        
        if skipped_users > 0:
            print(f"[VBPRRetriever] Warning: Skipped {skipped_users} users with user_id > {self.num_user} or < 1")
        
        if not user_ids_list:
            print(f"[VBPRRetriever] Warning: No valid users in split (total: {len(split)}, skipped: {skipped_users})")
            return 0.0
        
        # Batch size for evaluation (can be adjusted based on GPU memory)
        batch_size = 512
        device = self.device
        
        with torch.no_grad():
            for i in range(0, len(user_ids_list), batch_size):
                batch_user_ids = user_ids_list[i:i + batch_size]
                batch_gt_items = gt_items_list[i:i + batch_size]
                batch_history_items = history_items_list[i:i + batch_size]
                
                # Convert to tensor
                user_ids_tensor = torch.tensor(batch_user_ids, dtype=torch.long).to(device)
                
                # Predict scores for batch [batch_size, n_items]
                scores_batch = self.model.predict_batch(user_ids_tensor)  # [batch_size, n_items]
                
                # Mask history items for each user in batch
                for j, history_items in enumerate(batch_history_items):
                    for item in history_items:
                        if 1 <= item <= self.num_item:
                            item_idx = item - 1  # Convert to 0-indexed
                            scores_batch[j, item_idx] = -1e9
                
                # Get top-K for each user in batch
                _, top_items_batch = torch.topk(scores_batch, k=min(k, self.num_item), dim=1)  # [batch_size, k]
                top_items_batch = (top_items_batch.cpu().numpy() + 1)  # Convert back to 1-indexed
                
                # Compute recall for each user in batch
                # ✅ FIX: Use standard recall formula (hits / len(gt_items)) instead of min(k, len(gt_items))
                for j, (top_items, gt_items) in enumerate(zip(top_items_batch, batch_gt_items)):
                    top_items_list = top_items.tolist()
                    hits = len(set(top_items_list) & set(gt_items))
                    if len(gt_items) > 0:
                        recalls.append(hits / len(gt_items))  # Standard recall formula
        
        return float(np.mean(recalls)) if recalls else 0.0

    def _load_visual_features(
        self,
        dataset_code: str,
        min_rating: int,
        min_uc: int,
        min_sc: int,
        num_items: int
    ) -> torch.Tensor:
        """Load visual features from CLIP embeddings.
        
        Args:
            dataset_code: Dataset code
            min_rating: Minimum rating threshold
            min_uc: Minimum user count
            min_sc: Minimum item count
            num_items: Number of items
            
        Returns:
            Visual features tensor [num_items, D]
        """
        clip_path = get_clip_embeddings_path(dataset_code, min_rating, min_uc, min_sc)
        
        if not clip_path.exists():
            raise FileNotFoundError(
                f"CLIP embeddings not found at {clip_path}. "
                "Please run data_prepare.py with --use_image flag first."
            )
        
        clip_payload = torch.load(clip_path, map_location="cpu")
        image_embs = clip_payload.get("image_embs")
        
        if image_embs is None:
            raise ValueError("CLIP embeddings file does not contain image_embs")
        
        # Skip row 0 (padding), use rows 1..num_items
        visual_features = image_embs[1:num_items+1]  # [num_items, D]
        
        return visual_features

    def retrieve(
        self,
        user_id: int,
        exclude_items: Set[int] = None
    ) -> List[int]:
        """Retrieve top-K candidates cho một user.
        
        Args:
            user_id: ID của user (1-indexed)
            exclude_items: Set các items cần loại trừ
            
        Returns:
            List[int]: Top-K item IDs (1-indexed)
        """
        self._validate_fitted()
        
        if self.model is None or self.num_user is None or self.num_item is None:
            raise RuntimeError("VBPRRetriever model not initialized")
        
        if user_id > self.num_user or user_id < 1:
            return []
        
        exclude_items = exclude_items or set()
        
        # Get scores for all items
        self.model.eval()
        with torch.no_grad():
            scores = self.model.predict_all(user_id - 1)  # Convert to 0-indexed
        
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

