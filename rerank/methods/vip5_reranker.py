"""VIP5-based reranker using multimodal visual and textual features.

This implementation follows the original VIP5 source code as closely as possible.
Reference: https://github.com/jeykigung/VIP5
"""

from typing import Dict, List, Tuple, Any, Optional
import torch
import numpy as np
from pathlib import Path
import sys
import random
from copy import deepcopy

from rerank.base import BaseReranker
from dataset.paths import get_clip_embeddings_path
from evaluation.metrics import recall_at_k

# Import VIP5 model from original implementation
try:
    from rerank.models.vip5_modeling import VIP5, VIP5Seq2SeqLMOutput
    from rerank.models.vip5_utils import prepare_vip5_input, build_rerank_prompt, calculate_whole_word_ids
    VIP5_AVAILABLE = True
    VIP5_IMPORT_ERROR = None
except ImportError as e:
    VIP5_AVAILABLE = False
    # Store error for later use
    VIP5_IMPORT_ERROR = str(e)
    # Create dummy classes to allow import (will raise error when used)
    VIP5 = None
    VIP5Seq2SeqLMOutput = None
    def prepare_vip5_input(*args, **kwargs):
        raise ImportError(f"VIP5 not available: {VIP5_IMPORT_ERROR}")
    def build_rerank_prompt(*args, **kwargs):
        raise ImportError(f"VIP5 not available: {VIP5_IMPORT_ERROR}")
    def calculate_whole_word_ids(*args, **kwargs):
        raise ImportError(f"VIP5 not available: {VIP5_IMPORT_ERROR}")

# Try to import tokenizer from VIP5
VIP5_TOKENIZER_AVAILABLE = False
try:
    vip5_src_path = Path(__file__).parent.parent.parent / "retrieval" / "vip5_temp" / "src"
    if vip5_src_path.exists() and str(vip5_src_path) not in sys.path:
        sys.path.insert(0, str(vip5_src_path))
    try:
        from tokenization import P5Tokenizer  # type: ignore
        VIP5_TOKENIZER_AVAILABLE = True
    except ImportError:
        # Fallback to T5Tokenizer
        from transformers import T5Tokenizer
        P5Tokenizer = T5Tokenizer  # type: ignore
except (ImportError, Exception):
    # Fallback to T5Tokenizer
    from transformers import T5Tokenizer
    P5Tokenizer = T5Tokenizer  # type: ignore


class VIP5Reranker(BaseReranker):
    """Reranker sử dụng VIP5 (Visual Item Preference) model.
    
    VIP5 là một mô hình multimodal T5-based kết hợp visual và textual features
    để dự đoán user-item preferences. Implementation này theo sát source code gốc.
    
    Reference: https://github.com/jeykigung/VIP5
    """
    
    def __init__(
        self,
        top_k: int = 50,
        checkpoint_path: Optional[str] = None,
        backbone: str = "t5-small",
        tokenizer_path: Optional[str] = None,
        image_feature_type: str = "vitb32",
        image_feature_size_ratio: int = 2,
        max_text_length: Optional[int] = None,
        batch_size: int = 32,
        num_epochs: int = 10,
        lr: float = 1e-4,
        patience: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            top_k: Số lượng items trả về sau rerank
            checkpoint_path: Đường dẫn đến VIP5 checkpoint (optional)
            backbone: T5 model backbone (default: "t5-small")
            tokenizer_path: Path to tokenizer (optional, uses backbone if None)
            image_feature_type: CLIP feature type (vitb32, vitb16, vitl14, rn50, rn101)
            image_feature_size_ratio: Number of visual tokens per item (default: 2)
            max_text_length: Maximum text sequence length (optional, lấy từ config nếu None)
            batch_size: Batch size cho training (default: 32)
            num_epochs: Số epochs (default: 10)
            lr: Learning rate (default: 1e-4)
            patience: Early stopping patience (None = no early stopping)
            device: Device để chạy model ("cuda" hoặc "cpu")
        """
        super().__init__(top_k=top_k)
        self.checkpoint_path = checkpoint_path
        self.backbone = backbone
        self.tokenizer_path = tokenizer_path
        self.image_feature_type = image_feature_type
        self.image_feature_size_ratio = image_feature_size_ratio
        
        # Get max_text_length from config if not provided
        if max_text_length is None:
            try:
                from config import arg
                max_text_length = getattr(arg, 'vip5_max_text_length', 128)
            except ImportError:
                max_text_length = 128  # Default fallback
        
        self.max_text_length = max_text_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.patience = patience
        
        # Image feature dimensions from VIP5
        image_feature_dim_dict = {
            'vitb32': 512,
            'vitb16': 512,
            'vitl14': 768,
            'rn50': 1024,
            'rn101': 512
        }
        self.image_feature_dim = image_feature_dim_dict.get(image_feature_type, 512)
        
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if not VIP5_AVAILABLE:
            raise ImportError(
                f"VIP5 model is not available. Please ensure VIP5 source code is available. "
                f"Error: {VIP5_IMPORT_ERROR}"
            )
        # Type annotation: VIP5 is imported from vip5_modeling
        self.model: Optional[Any] = None  # Will be VIP5 instance after fit()
        self.tokenizer = None
        
        # CLIP embeddings cache
        self.visual_embeddings: Optional[torch.Tensor] = None  # [num_items, visual_dim]
        self.text_embeddings: Optional[torch.Tensor] = None     # [num_items, text_dim]
        self.item_id_to_idx: Dict[int, int] = {}  # item_id -> embedding index
        self.item_id_to_text: Dict[int, str] = {}  # item_id -> item text (for tokenization)
        
        # Store user history for training
        self.user_history: Dict[int, List[int]] = {}  # user_id -> [item_ids]
        
    def _load_clip_embeddings(
        self,
        dataset_code: str,
        min_rating: int,
        min_uc: int,
        min_sc: int,
        num_items: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load CLIP embeddings for visual and text features.
        
        Args:
            dataset_code: Dataset code
            min_rating: Minimum rating threshold
            min_uc: Minimum user count
            min_sc: Minimum item count
            num_items: Number of items
            
        Returns:
            Tuple of (visual_embeddings, text_embeddings)
        """
        clip_path = get_clip_embeddings_path(dataset_code, min_rating, min_uc, min_sc)
        
        if not clip_path.exists():
            raise FileNotFoundError(
                f"CLIP embeddings not found at {clip_path}. "
                "Please run data_prepare.py with --use_image and --use_text flags first."
            )
        
        clip_payload = torch.load(clip_path, map_location="cpu")
        image_embs = clip_payload.get("image_embs")  # Shape: [num_items+1, D] (row 0 is padding)
        text_embs = clip_payload.get("text_embs")    # Shape: [num_items+1, D] (row 0 is padding)
        
        if image_embs is None:
            raise ValueError("VIP5 requires image embeddings, but image_embs is None")
        if text_embs is None:
            raise ValueError("VIP5 requires text embeddings, but text_embs is None")
        
        # Skip row 0 (padding), use rows 1..num_items
        visual_emb = image_embs[1:num_items+1]  # [num_items, D]
        text_emb = text_embs[1:num_items+1]     # [num_items, D]
        
        return visual_emb, text_emb
    
    def fit(
        self,
        train_data: Dict[int, List[int]],
        **kwargs: Any
    ) -> None:
        """Fit VIP5 reranker with training.
        
        Args:
            train_data: Dict {user_id: [item_ids]} - training interactions
            **kwargs: Additional arguments:
                - dataset_code: str - dataset code
                - min_rating: int - minimum rating threshold
                - min_uc: int - minimum user count
                - min_sc: int - minimum item count
                - num_items: int - number of items
                - item_id2text: Dict[int, str] - mapping item_id -> text (optional)
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
        
        # Get dataset info
        dataset_code = kwargs.get("dataset_code")
        min_rating = kwargs.get("min_rating", 3)
        min_uc = kwargs.get("min_uc", 20)
        min_sc = kwargs.get("min_sc", 20)
        num_items = kwargs.get("num_items")
        self.item_id_to_text = kwargs.get("item_id2text", {})
        val_data = kwargs.get("val_data")
        
        if num_items is None:
            # Infer from train_data
            all_items = set()
            for items in train_data.values():
                all_items.update(items)
            num_items = max(all_items) if all_items else 0
        
        if dataset_code is None:
            raise ValueError("VIP5Reranker.fit requires dataset_code in kwargs")
        
        # Store user history
        self.user_history = train_data
        
        # Load CLIP embeddings
        print("Loading CLIP embeddings for VIP5...")
        visual_emb, text_emb = self._load_clip_embeddings(
            dataset_code, min_rating, min_uc, min_sc, num_items
        )
        # Move embeddings to device for faster evaluation
        self.visual_embeddings = visual_emb.to(self.device)
        self.text_embeddings = text_emb.to(self.device)
        
        # Build item_id to embedding index mapping
        # Assuming item IDs are 1-indexed and embeddings are in order
        for item_id in range(1, num_items + 1):
            self.item_id_to_idx[item_id] = item_id - 1  # 0-indexed
        
        # Initialize tokenizer
        print("Initializing VIP5 tokenizer...")
        try:
            if VIP5_TOKENIZER_AVAILABLE and self.tokenizer_path:
                self.tokenizer = P5Tokenizer.from_pretrained(
                    self.tokenizer_path,
                    max_length=self.max_text_length,
                    do_lower_case=False
                )
            else:
                # Use T5Tokenizer as fallback
                from transformers import T5Tokenizer
                self.tokenizer = T5Tokenizer.from_pretrained(self.backbone)
        except Exception as e:
            print(f"Warning: Failed to load tokenizer, using T5Tokenizer. Error: {e}")
            from transformers import T5Tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(self.backbone)
        
        # Initialize or load VIP5 model
        from transformers.models.t5.configuration_t5 import T5Config
        
        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            print(f"Loading VIP5 checkpoint from {self.checkpoint_path}...")
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            
            # Load config from checkpoint or use default
            if "config" in checkpoint:
                config = checkpoint["config"]
            else:
                # Create config from backbone
                config = T5Config.from_pretrained(self.backbone)
                # Add VIP5-specific config
                config.feat_dim = self.image_feature_dim
                config.n_vis_tokens = self.image_feature_size_ratio
                config.use_vis_layer_norm = True
                config.use_adapter = kwargs.get("use_adapter", True)  # ✅ Default: True
                adapter_config = kwargs.get("adapter_config", None)
                # ✅ Create default adapter_config if use_adapter=True but adapter_config=None
                if config.use_adapter and adapter_config is None:
                    try:
                        from rerank.models.adapters.config import AdapterConfig
                        adapter_config = AdapterConfig()
                        adapter_config.tasks = ["direct"]  # Direct Task for reranking
                        adapter_config.d_model = config.d_model  # For adapter
                        adapter_config.use_single_adapter = False
                        adapter_config.reduction_factor = 16  # Default reduction factor
                        adapter_config.track_z = False  # Don't track intermediate activations by default
                    except ImportError:
                        print("Warning: AdapterConfig not available. Adapters may not work correctly.")
                        adapter_config = None
                config.adapter_config = adapter_config
                config.add_adapter_cross_attn = kwargs.get("add_adapter_cross_attn", False)
                config.use_lm_head_adapter = kwargs.get("use_lm_head_adapter", False)
                config.whole_word_embed = True
                config.category_embed = True
            
            self.model = VIP5(config)
            
            # Load state dict
            if "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"], strict=False)
            elif "model" in checkpoint:
                self.model.load_state_dict(checkpoint["model"], strict=False)
            else:
                # Try loading directly
                self.model.load_state_dict(checkpoint, strict=False)
        else:
            print("Initializing new VIP5 model (no checkpoint provided)...")
            # Create config
            config = T5Config.from_pretrained(self.backbone)
            # Add VIP5-specific config
            config.feat_dim = self.image_feature_dim
            config.n_vis_tokens = self.image_feature_size_ratio
            config.use_vis_layer_norm = True
            config.use_adapter = kwargs.get("use_adapter", True)  # ✅ Default: True
            adapter_config = kwargs.get("adapter_config", None)
            # ✅ Create default adapter_config if use_adapter=True but adapter_config=None
            if config.use_adapter and adapter_config is None:
                try:
                    from rerank.models.adapters.config import AdapterConfig
                    adapter_config = AdapterConfig()
                    adapter_config.tasks = ["direct"]  # Direct Task for reranking
                    adapter_config.d_model = config.d_model  # For adapter
                    adapter_config.use_single_adapter = False
                    adapter_config.reduction_factor = 16  # Default reduction factor
                except ImportError:
                    print("Warning: AdapterConfig not available. Adapters may not work correctly.")
                    adapter_config = None
            config.adapter_config = adapter_config
            config.add_adapter_cross_attn = kwargs.get("add_adapter_cross_attn", False)
            config.use_lm_head_adapter = kwargs.get("use_lm_head_adapter", False)
            config.whole_word_embed = True
            config.category_embed = True
            
            self.model = VIP5(config)
            # Load pretrained T5 weights
            from transformers import T5ForConditionalGeneration
            pretrained = T5ForConditionalGeneration.from_pretrained(self.backbone)
            # Copy compatible weights
            self.model.shared.load_state_dict(pretrained.shared.state_dict())
            self.model.decoder.load_state_dict(pretrained.decoder.state_dict(), strict=False)
        
        self.model = self.model.to(self.device)
        
        # ✅ OPTIMIZE: Freeze base model and only train adapters if adapters are enabled
        # Access config from model (may have been updated during model initialization)
        use_adapter = self.model.config.use_adapter or self.model.config.use_lm_head_adapter
        if use_adapter:
            print("Adapters enabled: Freezing base model, only training adapter parameters...")
            # Freeze all parameters first
            for name, param in self.model.named_parameters():
                param.requires_grad = False
            
            # Unfreeze only adapter parameters
            adapter_param_names = []
            for name, param in self.model.named_parameters():
                if "adapter" in name.lower() or "output_adapter" in name.lower():
                    param.requires_grad = True
                    adapter_param_names.append(name)
            
            if len(adapter_param_names) == 0:
                print("Warning: Adapters enabled but no adapter parameters found! Training all parameters.")
                # Fallback: unfreeze all if no adapters found
                for param in self.model.parameters():
                    param.requires_grad = True
            else:
                print(f"Found {len(adapter_param_names)} adapter parameter groups:")
                for name in adapter_param_names[:5]:  # Show first 5
                    print(f"  - {name}")
                if len(adapter_param_names) > 5:
                    print(f"  ... and {len(adapter_param_names) - 5} more")
            
            # Calculate trainable parameters percentage
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            trainable_percentage = (trainable_params / total_params * 100) if total_params > 0 else 0
            print(f"Trainable parameters: {trainable_percentage:.2f}% ({trainable_params:,}/{total_params:,})")
        else:
            print("Adapters disabled: Training all model parameters (full fine-tuning)...")
        
        # Training
        print("Preparing training samples...")
        train_samples = self._prepare_training_samples(train_data)
        
        if len(train_samples) == 0:
            print("Warning: No training samples generated. Skipping training.")
            self.model.eval()
            self.is_fitted = True
            return
        
        print(f"Generated {len(train_samples)} training samples")
        
        # Setup optimizer - only optimize trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise ValueError("No trainable parameters found! Check adapter configuration.")
        optimizer = torch.optim.AdamW(trainable_params, lr=self.lr, weight_decay=1e-4)
        
        # Training loop
        best_state = None
        best_val_recall = -1.0
        epochs_no_improve = 0
        
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            num_batches = 0
            
            # Shuffle training samples
            random.shuffle(train_samples)
            
            # Training batches
            for i in range(0, len(train_samples), self.batch_size):
                batch = train_samples[i:i + self.batch_size]
                if len(batch) == 0:
                    continue
                
                # Prepare batch
                batch_dict = self._prepare_batch(batch)
                
                # Move to device
                input_ids = batch_dict["input_ids"].to(self.device)
                whole_word_ids = batch_dict["whole_word_ids"].to(self.device)
                category_ids = batch_dict["category_ids"].to(self.device)
                vis_feats = batch_dict["vis_feats"].to(self.device)
                target_ids = batch_dict["target_ids"].to(self.device)
                loss_weights = batch_dict["loss_weights"].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                output = self.model(
                    input_ids=input_ids,
                    whole_word_ids=whole_word_ids,
                    category_ids=category_ids,
                    vis_feats=vis_feats,
                    labels=target_ids,
                    return_dict=True,
                    task="direct",  # ✅ Use Direct Task for training
                    reduce_loss=False  # Get per-token loss
                )
                
                # Compute loss
                loss = output["loss"]  # Shape: [batch_size * seq_len] (per-token loss)
                
                # Reshape loss to [batch_size, seq_len] and compute per-sample loss
                batch_size = target_ids.size(0)
                seq_len = target_ids.size(1)
                
                if loss.dim() > 0 and loss.numel() > 0:
                    # Reshape loss from [batch_size * seq_len] to [batch_size, seq_len]
                    loss = loss.view(batch_size, seq_len)
                    
                    # Mask out padding tokens (where target_ids == -100)
                    target_mask = (target_ids != -100).float()  # [batch_size, seq_len]
                    
                    # Compute per-sample loss: sum over sequence, divide by number of valid tokens
                    per_sample_loss = (loss * target_mask).sum(dim=1) / target_mask.sum(dim=1).clamp(min=1.0)  # [batch_size]
                    
                    # Apply loss weights (length-aware normalization)
                    loss = (per_sample_loss * loss_weights).mean()
                else:
                    # Scalar loss (shouldn't happen with reduce_loss=False, but handle it)
                    loss = loss * loss_weights.mean() if loss.dim() == 0 else loss.mean()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / max(1, num_batches)
            
            # Validation
            if val_data is not None:
                # ✅ Pass rerank_mode and retrieval_candidates from kwargs
                rerank_mode = kwargs.get("rerank_mode", "ground_truth")
                retrieval_candidates = kwargs.get("retrieval_candidates", None)
                val_recall = self._evaluate_split(
                    val_data, 
                    k=min(10, self.top_k),
                    rerank_mode=rerank_mode,
                    retrieval_candidates=retrieval_candidates
                )
                
                if val_recall > best_val_recall:
                    best_val_recall = val_recall
                    best_state = deepcopy(self.model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                
                print(f"[VIP5Reranker] Epoch {epoch+1}/{self.num_epochs} - "
                      f"loss: {avg_loss:.4f}, val_Recall@{min(10, self.top_k)}: {val_recall:.4f}")
                
                if self.patience and epochs_no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                print(f"[VIP5Reranker] Epoch {epoch+1}/{self.num_epochs} - loss: {avg_loss:.4f}")
        
        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        self.model.eval()
        self.is_fitted = True
        print(f"VIP5Reranker training completed. Model on device: {self.device}")
    
    def rerank(
        self,
        user_id: int,
        candidates: List[int],
        **kwargs: Any
    ) -> List[Tuple[int, float]]:
        """Rerank candidates cho một user sử dụng VIP5 với beam search generation (theo cách gốc).
        
        Args:
            user_id: ID của user
            candidates: List các item IDs cần rerank
            **kwargs: Additional arguments:
                - user_history: List[int] - user's interaction history (optional)
                - num_beams: int - beam size for generation (default: min(len(candidates), 20))
                - num_return_sequences: int - number of sequences to return (default: min(len(candidates), top_k))
                - skip_validation: bool - skip fitted validation (for internal use during training)
        
        Returns:
            List[Tuple[int, float]]: [(item_id, score)] đã sort giảm dần
            Score = negative rank (rank 1 -> -1, rank 2 -> -2, ...)
        """
        # Skip validation if called during training (from _evaluate_split)
        skip_validation = kwargs.get("skip_validation", False)
        if not skip_validation:
            self._validate_fitted()
        
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("VIP5 model chưa được load. Gọi fit() trước!")
        
        if not candidates:
            return []
        
        # Get visual features for candidates
        candidate_visual = []
        valid_candidates = []
        
        for item_id in candidates:
            if item_id in self.item_id_to_idx:
                idx = self.item_id_to_idx[item_id]
                valid_candidates.append(item_id)
                candidate_visual.append(self.visual_embeddings[idx])
        
        if not valid_candidates:
            return []
        
        # ✅ VIP5 Direct Task (B-5): Recommend từ danh sách candidates
        # Theo code gốc VIP5 (data.py line 452-471), với direct task:
        # - Template: "Which item of the following to recommend for {} ? \n {}"
        # - Prompt chứa TẤT CẢ candidates: "item_1 <extra_id_0> <extra_id_0>, item_2 <extra_id_0> <extra_id_0>, ..."
        # - Visual features cho TẤT CẢ candidates
        # - Model generate top-K items sử dụng beam search
        # Reference: https://github.com/jeykigung/VIP5/blob/main/src/data.py#L452-L471
        
        # Build prompt với Direct Task template (B-5)
        visual_token_placeholder = " <extra_id_0>" * self.image_feature_size_ratio
        candidates_with_visual = visual_token_placeholder.join([f"item_{c}" for c in valid_candidates]) + visual_token_placeholder
        direct_prompt = f"Which item of the following to recommend for user_{user_id} ? \n {candidates_with_visual}"
        
        # Prepare visual features cho TẤT CẢ candidates
        if len(candidate_visual) > 0:
            all_candidates_visual_tensor = torch.stack(candidate_visual)  # [num_candidates, feat_dim]
        else:
            all_candidates_visual_tensor = torch.zeros(1, self.image_feature_dim, device=self.device)
        
        # Prepare VIP5 input
        vip5_input = prepare_vip5_input(
            direct_prompt,
            all_candidates_visual_tensor,
            self.tokenizer,
            max_length=self.max_text_length,
            image_feature_size_ratio=self.image_feature_size_ratio,
        )
        
        # Move to device
        input_ids = vip5_input["input_ids"].to(self.device)
        whole_word_ids = vip5_input["whole_word_ids"].to(self.device)
        category_ids = vip5_input["category_ids"].to(self.device)
        vis_feats = vip5_input["vis_feats"].to(self.device)
        
        # ✅ Generate với beam search (theo cách VIP5 gốc)
        # Get beam search parameters from kwargs or use defaults
        num_beams = kwargs.get("num_beams", min(len(valid_candidates), 20))
        num_return_sequences = kwargs.get("num_return_sequences", min(len(valid_candidates), self.top_k))
        
        self.model.eval()
        with torch.no_grad():
            # Generate top-K items sử dụng beam search
            # Note: VIP5.generate() cần hỗ trợ whole_word_ids, category_ids, vis_feats, task
            # Fix: use_cache=False để tránh lỗi reorder_cache với tuple past_key_values
            try:
                beam_outputs = self.model.generate(
                    input_ids=input_ids,
                    whole_word_ids=whole_word_ids,
                    category_ids=category_ids,
                    vis_feats=vis_feats,
                    task="direct",
                    max_length=64,  # gen_max_length from VIP5
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    no_repeat_ngram_size=0,
                    early_stopping=True,
                    use_cache=False,  # ✅ Fix: Disable cache to avoid reorder_cache error with tuple past_key_values
                )
            except (TypeError, AttributeError) as e:
                # Fallback: Nếu generate() không hỗ trợ custom parameters, dùng encoder_outputs
                # Encode trước, sau đó generate với encoder_outputs
                encoder_outputs = self.model.encoder(
                    input_ids=input_ids,
                    whole_word_ids=whole_word_ids,
                    category_ids=category_ids,
                    vis_feats=vis_feats,
                    attention_mask=vip5_input["attention_mask"].to(self.device),
                    return_dict=True,
                    task="direct",
                )
                # Generate với encoder_outputs
                beam_outputs = self.model.generate(
                    encoder_outputs=encoder_outputs,
                    task="direct",
                    max_length=64,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    no_repeat_ngram_size=0,
                    early_stopping=True,
                    use_cache=False,  # ✅ Fix: Disable cache to avoid reorder_cache error
                )
            
            # Decode generated sequences
            generated_sents = self.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
        
        # ✅ Score = negative rank (theo cách VIP5 gốc)
        # Item rank 1 -> score -1, rank 2 -> score -2, ...
        scores_dict = {}
        for rank, generated_text in enumerate(generated_sents):
            try:
                # Extract item_id from generated text (e.g., "item_123" -> 123)
                item_id = int(generated_text.replace("item_", "").strip())
                if item_id in valid_candidates:
                    # Score = negative rank (rank 0 -> score -1, rank 1 -> score -2, ...)
                    scores_dict[item_id] = -(rank + 1)
            except (ValueError, AttributeError):
                # Skip invalid generated text
                continue
        
        # Fill missing candidates with worst scores
        for item_id in valid_candidates:
            if item_id not in scores_dict:
                scores_dict[item_id] = -len(valid_candidates) - 1
        
        # Convert to list and sort by score descending (higher score = better rank)
        scores = [(item_id, scores_dict[item_id]) for item_id in valid_candidates]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k
        return scores[:self.top_k]
    
    def _prepare_training_samples(
        self,
        train_data: Dict[int, List[int]]
    ) -> List[Dict]:
        """Prepare training samples for VIP5 Direct Task (B-5).
        
        Args:
            train_data: Dict {user_id: [item_ids]} - training interactions
            
        Returns:
            List of training samples with keys: user_id, candidates, target_item, source_text, target_text
        """
        samples = []
        
        # Direct Task template B-5
        template = "Which item of the following to recommend for user_{} ? \n {}"
        
        # Get all items for negative sampling
        all_items = set()
        for items in train_data.values():
            all_items.update(items)
        all_items = list(all_items)
        
        for user_id, items in train_data.items():
            if len(items) < 2:
                continue  # Need at least 2 items
            
            # For Direct Task, we need: candidates list + target item
            # Similar to VIP5 original implementation (data.py line 452-471)
            target_item = items[-1]  # Last item is target
            
            # Sample negative candidates (exclude user's history)
            user_items_set = set(items)
            negative_candidates = [item for item in all_items if item not in user_items_set]
            
            # ✅ Get max_candidates from config (default: 100)
            try:
                from config import arg
                max_candidates = getattr(arg, 'vip5_max_candidates', 100)
            except (ImportError, AttributeError):
                max_candidates = 100
            
            # Sample (max_candidates - 1) negatives + 1 positive = max_candidates total
            num_negatives = min(max_candidates - 1, len(negative_candidates))
            if num_negatives > 0:
                sampled_negatives = random.sample(negative_candidates, num_negatives)
            else:
                sampled_negatives = []
            
            # Combine: negatives + positive target
            candidates = sampled_negatives + [target_item]
            random.shuffle(candidates)  # Shuffle to avoid bias
            
            # Build source text with visual token placeholders
            # Format: "item_1 <extra_id_0> <extra_id_0>, item_2 <extra_id_0> <extra_id_0>, ..."
            visual_token_placeholder = " <extra_id_0>" * self.image_feature_size_ratio
            candidates_with_visual = visual_token_placeholder.join([f"item_{item_id}" for item_id in candidates]) + visual_token_placeholder
            
            source_text = template.format(user_id, candidates_with_visual)
            target_text = f"item_{target_item}"
            
            samples.append({
                "user_id": user_id,
                "candidates": candidates,  # All candidates (for visual features)
                "target_item": target_item,
                "source_text": source_text,
                "target_text": target_text,
            })
        
        return samples
    
    def _prepare_batch(
        self,
        batch: List[Dict]
    ) -> Dict[str, torch.Tensor]:
        """Prepare batch for VIP5 training.
        
        Args:
            batch: List of training samples
            
        Returns:
            Dict with keys: input_ids, whole_word_ids, category_ids, vis_feats, target_ids, loss_weights
        """
        batch_size = len(batch)
        
        # Tokenize source texts
        source_texts = [s["source_text"] for s in batch]
        target_texts = [s["target_text"] for s in batch]
        
        # Tokenize inputs
        tokenized_inputs = self.tokenizer(
            source_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt"
        )
        input_ids = tokenized_inputs["input_ids"]  # [B, L]
        
        # Tokenize targets
        tokenized_targets = self.tokenizer(
            target_texts,
            padding="max_length",
            truncation=True,
            max_length=64,  # gen_max_length from VIP5
            return_tensors="pt"
        )
        target_ids = tokenized_targets["input_ids"]  # [B, L_t]
        
        # Calculate whole_word_ids
        whole_word_ids_list = []
        for source_text in source_texts:
            tokenized_text = self.tokenizer.tokenize(source_text)
            input_ids_text = self.tokenizer.encode(source_text, add_special_tokens=False)
            whole_word_ids = calculate_whole_word_ids(tokenized_text, input_ids_text, self.tokenizer)
            # Pad to max_length
            if len(whole_word_ids) < self.max_text_length:
                whole_word_ids = whole_word_ids + [0] * (self.max_text_length - len(whole_word_ids))
            else:
                whole_word_ids = whole_word_ids[:self.max_text_length]
            whole_word_ids_list.append(whole_word_ids)
        whole_word_ids = torch.LongTensor(whole_word_ids_list)  # [B, L]
        
        # Calculate category_ids (1 for <extra_id_0>, 0 for others)
        if hasattr(self.tokenizer, 'convert_tokens_to_ids'):
            extra_id_0_token_id = self.tokenizer.convert_tokens_to_ids('<extra_id_0>')
        else:
            extra_id_0_token_id = self.tokenizer.encode('<extra_id_0>', add_special_tokens=False)[0]
        
        category_ids = (input_ids == extra_id_0_token_id).long()  # [B, L]
        
        # Prepare visual features
        max_vis_tokens = category_ids.sum(dim=1).max().item()
        vis_feats_list = []
        
        for sample in batch:
            # ✅ Direct Task (B-5): Visual features cho TẤT CẢ candidates (không chỉ history)
            candidates = sample.get("candidates", [])  # Direct Task uses candidates
            if not candidates:
                # Fallback to history if candidates not available (backward compatibility)
                candidates = sample.get("history", [])
            
            # Get visual features for all candidates
            candidates_visual = []
            for item_id in candidates:
                if item_id in self.item_id_to_idx:
                    idx = self.item_id_to_idx[item_id]
                    candidates_visual.append(self.visual_embeddings[idx])
            
            if len(candidates_visual) == 0:
                # No valid items, use zeros (on same device as visual_embeddings)
                vis_feats = torch.zeros(max_vis_tokens, self.image_feature_dim, device=self.device)
            else:
                # Stack and repeat for image_feature_size_ratio
                candidates_visual_tensor = torch.stack(candidates_visual)  # [num_candidates, feat_dim]
                # Repeat each item image_feature_size_ratio times
                vis_feats = candidates_visual_tensor.repeat_interleave(self.image_feature_size_ratio, dim=0)  # [num_candidates * ratio, feat_dim]
                
                # Pad or truncate to max_vis_tokens
                if vis_feats.size(0) < max_vis_tokens:
                    # Create padding on same device as vis_feats
                    padding = torch.zeros(max_vis_tokens - vis_feats.size(0), self.image_feature_dim, device=vis_feats.device)
                    vis_feats = torch.cat([vis_feats, padding], dim=0)
                else:
                    vis_feats = vis_feats[:max_vis_tokens]
            
            vis_feats_list.append(vis_feats)
        
        vis_feats = torch.stack(vis_feats_list)  # [B, max_vis_tokens, feat_dim]
        
        # Loss weights (length-aware normalization)
        target_lengths = (target_ids != self.tokenizer.pad_token_id).sum(dim=1).float()
        loss_weights = 1.0 / target_lengths.clamp(min=1.0)  # [B]
        
        # Mask target_ids: set padding to -100 (ignore in loss)
        target_mask = (target_ids != self.tokenizer.pad_token_id)
        target_ids_masked = target_ids.clone()
        target_ids_masked[~target_mask] = -100
        
        return {
            "input_ids": input_ids,
            "whole_word_ids": whole_word_ids,
            "category_ids": category_ids,
            "vis_feats": vis_feats,
            "target_ids": target_ids_masked,
            "loss_weights": loss_weights,
        }
    
    def _evaluate_split(self, split: Dict[int, List[int]], k: int, **kwargs: Any) -> float:
        """Compute average Recall@K for validation.
        
        Args:
            split: Dict {user_id: [gt_item_ids]} - validation/test split
            k: Top-K for recall calculation
            **kwargs: Additional arguments:
                - rerank_mode: str - "retrieval" or "ground_truth" (default: "ground_truth")
                - retrieval_candidates: Dict[int, List[int]] - candidates from Stage 1 (for "retrieval" mode)
            
        Returns:
            Average Recall@K
        """
        if self.model is None or self.tokenizer is None:
            return 0.0
        
        # ✅ Get rerank_mode from kwargs (default: "ground_truth")
        rerank_mode = kwargs.get("rerank_mode", "ground_truth")
        retrieval_candidates = kwargs.get("retrieval_candidates", None)
        
        # ✅ Get eval_candidates from config (default: 20)
        try:
            from config import arg
            max_eval_candidates = getattr(arg, 'rerank_eval_candidates', 20)
        except (ImportError, AttributeError):
            max_eval_candidates = 20
        
        self.model.eval()
        recalls = []
        
        # ✅ OPTIMIZATION 1: Load candidates ONCE before loop (not inside loop!)
        pre_generated_candidates = None
        if rerank_mode == "ground_truth":
            try:
                from evaluation.utils import load_rerank_candidates
                from config import arg
                
                # Load candidates ONCE for all users
                pre_generated_candidates = load_rerank_candidates(
                    dataset_code=getattr(arg, 'dataset', 'beauty'),
                    min_rating=getattr(arg, 'min_rating', 0),
                    min_uc=getattr(arg, 'min_uc', 5),
                    min_sc=getattr(arg, 'min_sc', 5),
                )
            except Exception:
                pre_generated_candidates = None
        
        # ✅ OPTIMIZATION 2: Pre-compute all_items list once
        all_items_list = list(self.item_id_to_idx.keys()) if hasattr(self, 'item_id_to_idx') else []
        
        with torch.no_grad():
            # ✅ OPTIMIZATION 3: Add progress tracking for large splits
            total_users = len(split)
            processed = 0
            
            for user_id, gt_items in split.items():
                if user_id not in self.user_history:
                    continue
                
                # Get user history
                history = self.user_history[user_id]
                if len(history) == 0:
                    continue
                
                # ✅ Mode 1: "retrieval" - Use candidates from Stage 1
                if rerank_mode == "retrieval" and retrieval_candidates is not None:
                    if user_id in retrieval_candidates:
                        candidates = retrieval_candidates[user_id][:max_eval_candidates]
                        # Ensure at least one ground truth is in candidates
                        if not any(item in candidates for item in gt_items):
                            if len(candidates) < max_eval_candidates:
                                candidates.append(gt_items[0])
                            else:
                                candidates[0] = gt_items[0]
                    else:
                        # Fallback to ground_truth mode if no retrieval candidates
                        rerank_mode = "ground_truth"
                
                # ✅ Mode 2: "ground_truth" - Use pre-loaded candidates
                if rerank_mode == "ground_truth":
                    if pre_generated_candidates is not None:
                        # Use pre-loaded candidates (FAST!)
                        if user_id in pre_generated_candidates.get("val", {}):
                            candidates = pre_generated_candidates["val"][user_id]
                        elif user_id in pre_generated_candidates.get("test", {}):
                            candidates = pre_generated_candidates["test"][user_id]
                        else:
                            # Fallback: sample random candidates if user not in pre-generated
                            if not all_items_list:
                                continue
                            history_set = set(history)
                            candidate_pool = [item for item in all_items_list if item not in history_set]
                            if not candidate_pool:
                                continue
                            max_eval_candidates_actual = min(max_eval_candidates, len(candidate_pool))
                            candidates = random.sample(candidate_pool, max_eval_candidates_actual) if len(candidate_pool) > max_eval_candidates_actual else candidate_pool
                            if not any(item in candidates for item in gt_items):
                                candidates[0] = gt_items[0]
                    else:
                        # Fallback: sample random candidates if loading failed
                        if not all_items_list:
                            continue
                        history_set = set(history)
                        candidate_pool = [item for item in all_items_list if item not in history_set]
                        if not candidate_pool:
                            continue
                        max_eval_candidates_actual = min(max_eval_candidates, len(candidate_pool))
                        candidates = random.sample(candidate_pool, max_eval_candidates_actual) if len(candidate_pool) > max_eval_candidates_actual else candidate_pool
                        if not any(item in candidates for item in gt_items):
                            candidates[0] = gt_items[0]
                
                # ✅ Shuffle candidates to avoid bias (GT item should not always be first)
                random.shuffle(candidates)
                
                # Rerank candidates using Direct Task (B-5) - same as rerank() method
                scores = []
                
                # ✅ VIP5 Direct Task (B-5): Recommend từ danh sách candidates
                # ✅ OPTIMIZATION 4: Batch get visual features (faster than loop)
                valid_candidates = []
                candidate_indices = []
                for item_id in candidates:
                    if item_id in self.item_id_to_idx:
                        idx = self.item_id_to_idx[item_id]
                        valid_candidates.append(item_id)
                        candidate_indices.append(idx)
                
                if not valid_candidates:
                    continue
                
                # Batch get visual features using indexing (faster than loop)
                candidate_visual = self.visual_embeddings[candidate_indices]  # [num_candidates, feat_dim]
                
                # Build prompt với Direct Task template (B-5)
                visual_token_placeholder = " <extra_id_0>" * self.image_feature_size_ratio
                candidates_with_visual = visual_token_placeholder.join([f"item_{c}" for c in valid_candidates]) + visual_token_placeholder
                direct_prompt = f"Which item of the following to recommend for user_{user_id} ? \n {candidates_with_visual}"
                
                # candidate_visual is already a tensor [num_candidates, feat_dim] from batch indexing
                all_candidates_visual_tensor = candidate_visual
                
                # Encode prompt MỘT LẦN với tất cả candidates
                vip5_input = prepare_vip5_input(
                    direct_prompt,
                    all_candidates_visual_tensor,
                    self.tokenizer,
                    max_length=self.max_text_length,
                    image_feature_size_ratio=self.image_feature_size_ratio,
                )
                
                # Move to device
                input_ids = vip5_input["input_ids"].to(self.device)
                whole_word_ids = vip5_input["whole_word_ids"].to(self.device)
                category_ids = vip5_input["category_ids"].to(self.device)
                vis_feats = vip5_input["vis_feats"].to(self.device)
                attention_mask = vip5_input["attention_mask"].to(self.device)
                
                # ✅ Use rerank() method với beam search generation (theo cách VIP5 gốc)
                # Skip validation since we're in the middle of training
                reranked = self.rerank(
                    user_id=user_id,
                    candidates=valid_candidates,
                    num_beams=min(len(valid_candidates), 20),
                    num_return_sequences=min(len(valid_candidates), k),
                    skip_validation=True,  # ✅ Skip validation during training
                )
                
                # Get top-K item IDs từ reranked results
                top_k_items = [item_id for item_id, _ in reranked[:k]]
                
                # Compute recall (công thức chuẩn: hits / len(gt_items))
                hits = len(set(top_k_items) & set(gt_items))
                if len(gt_items) > 0:
                    recalls.append(hits / len(gt_items))
                
                # ✅ OPTIMIZATION 5: Progress tracking
                processed += 1
                if processed % 100 == 0:
                    print(f"[VIP5 _evaluate_split] Processed {processed}/{total_users} users...", end='\r')
        
        if processed > 0 and processed % 100 != 0:
            print(f"[VIP5 _evaluate_split] Processed {processed}/{total_users} users")
        
        return float(np.mean(recalls)) if recalls else 0.0

