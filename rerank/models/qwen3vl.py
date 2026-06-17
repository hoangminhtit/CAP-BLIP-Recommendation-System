"""Qwen3-VL model wrapper for reranking with different prompt modes.

Supports 4 modes:
1. raw_image: Use raw images directly in prompt
2. caption: Use image captions
3. semantic_summary: Use image semantic summaries
4. semantic_summary_small: Use semantic summaries with smaller text model
"""

import os
import torch
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import numpy as np


def _truncate_item_text(text: str, max_chars: int = 200) -> str:
    """Truncate item text metadata to prevent it from being too long.
    
    Args:
        text: Item text metadata
        max_chars: Maximum characters per item text (default: 200)
        
    Returns:
        Truncated text
    """
    if len(text) <= max_chars:
        return text
    # Truncate and add ellipsis
    return text[:max_chars - 3] + "..."


def resize_image_for_qwen3vl(img: Image.Image, max_size: int = 448) -> Image.Image:
    """Resize image for Qwen3-VL while maintaining aspect ratio.
    
    Args:
        img: PIL Image to resize
        max_size: Maximum size for the longer side (default: 448)
        
    Returns:
        Resized PIL Image
    """
    width, height = img.size
    
    # If image is already smaller than max_size, return as is
    if max(width, height) <= max_size:
        return img
    
    # Calculate new size maintaining aspect ratio
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    # Resize with high-quality resampling
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

try:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from unsloth import FastLanguageModel, FastVisionModel
    QWEN3VL_AVAILABLE = True
    FAST_VISION_MODEL_AVAILABLE = True
except ImportError:
    QWEN3VL_AVAILABLE = False
    FAST_VISION_MODEL_AVAILABLE = False
    # Try to import FastVisionModel separately
    try:
        from unsloth import FastVisionModel
        FAST_VISION_MODEL_AVAILABLE = True
    except ImportError:
        FAST_VISION_MODEL_AVAILABLE = False


class Qwen3VLModel:
    """Qwen3-VL model wrapper for reranking.
    
    Supports multiple prompt modes for different input types.
    """
    
    def __init__(
        self,
        mode: str = "raw_image",
        model_name: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            mode: Prompt mode - "raw_image", "caption", "semantic_summary", "semantic_summary_small"
            model_name: Model name (auto-selected based on mode if None)
            device: Device to run on (auto if None)
        """
        if not QWEN3VL_AVAILABLE:
            raise ImportError(
                "Qwen3-VL dependencies not available. Install with:\n"
                "pip install transformers unsloth\n"
                "pip install git+https://github.com/huggingface/transformers"
            )
        
        self.mode = mode.lower()
        if self.mode not in ["raw_image", "caption", "semantic_summary", "semantic_summary_small"]:
            raise ValueError(
                f"Invalid mode: {mode}. Must be one of: "
                "raw_image, caption, semantic_summary, semantic_summary_small"
            )
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Select model based on mode
        if self.mode == "semantic_summary_small":
            # Use smaller text-only model for semantic summaries
            self.model_name = model_name or "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"
            self._load_text_model()
        else:
            # Use Qwen3-VL for image-based modes
            self.model_name = model_name or "unsloth/Qwen3-VL-2B-Instruct"
            self._load_vl_model()
    
    def _load_vl_model(self):
        """Load Qwen3-VL model for image processing with LoRA adapters.
        
        Uses Unsloth's FastVisionModel to enable LoRA for efficient fine-tuning.
        LoRA is applied to vision layers, language layers, attention modules, and MLP modules.
        """
        print(f"Loading Qwen3-VL model with LoRA: {self.model_name}")
        try:
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Try to use FastVisionModel from Unsloth for LoRA support
            if FAST_VISION_MODEL_AVAILABLE:
                print("Using Unsloth FastVisionModel for LoRA support...")
                try:
                    # Get max_seq_length from config
                    try:
                        from config import arg
                        max_seq_length = getattr(arg, 'qwen_max_seq_length', 2048)
                    except ImportError:
                        max_seq_length = 2048  # Default fallback
                    
                    # Load model with FastVisionModel (supports LoRA)
                    self.model, _ = FastVisionModel.from_pretrained(
                        model_name=self.model_name,
                        max_seq_length=max_seq_length,
                        dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                        load_in_4bit=True,  # 4-bit quantization enabled by default
                        use_gradient_checkpointing="unsloth",  # Memory efficient
                    )
                    
                    # Add LoRA adapters using FastVisionModel.get_peft_model
                    print("Adding LoRA adapters to Qwen3-VL model...")
                    self.model = FastVisionModel.get_peft_model(
                        self.model,
                        finetune_vision_layers=True,      # Fine-tune vision layers
                        finetune_language_layers=True,    # Fine-tune language layers
                        finetune_attention_modules=True,   # Fine-tune attention modules
                        finetune_mlp_modules=True,         # Fine-tune MLP modules
                        r=16,                             # LoRA rank (higher = more accuracy, but may overfit)
                        lora_alpha=16,                    # Recommended: alpha >= r
                        lora_dropout=0,                   # Usually 0 for optimization
                        bias="none",                      # Usually "none" for optimization
                        random_state=3407,                # For reproducibility
                        use_rslora=False,                 # Rank Stabilized LoRA (optional)
                        loftq_config=None,                # LoftQ config (optional)
                    )
                    
                    if self.device.type == "cpu":
                        self.model = self.model.to(self.device)
                    
                    self.model.eval()
                    print(f"Qwen3-VL model loaded with LoRA on {self.device}")
                    print("  LoRA config: r=16, alpha=16, dropout=0")
                    print("  Fine-tuning: vision_layers=True, language_layers=True, attention=True, MLP=True")
                    return
                    
                except Exception as e:
                    print(f"Warning: Failed to load with FastVisionModel: {e}")
                    print("Falling back to standard Qwen3VLForConditionalGeneration...")
            
            # Fallback: Use standard transformers API (no LoRA)
            print("Loading Qwen3-VL model without LoRA (fallback)...")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_name,
                dtype="auto" if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
                trust_remote_code=True,
            )
            if self.device.type == "cpu":
                self.model = self.model.to(self.device)
            self.model.eval()
            print(f"Qwen3-VL model loaded on {self.device} (without LoRA)")
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Qwen3-VL model: {e}\n"
                f"Note: Qwen3-VL requires latest transformers. Install with:\n"
                f"pip install git+https://github.com/huggingface/transformers\n"
                f"For LoRA support, ensure unsloth is installed: pip install unsloth[colab-new]"
            )
    
    def _load_text_model(self):
        """Load smaller text-only model for semantic summaries with 4-bit quantization.
        
        Note:
            All Unsloth models are loaded with 4-bit quantization by default
            to reduce memory usage while maintaining performance.
        """
        print(f"Loading Qwen text model with 4-bit quantization: {self.model_name}")
        try:
            # Get max_seq_length from config
            try:
                from config import arg
                max_seq_length = getattr(arg, 'qwen_max_seq_length', 2048)
            except ImportError:
                max_seq_length = 2048  # Default fallback
            
            print(f"  Max sequence length: {max_seq_length}")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=max_seq_length,
                dtype=torch.float16,
                load_in_4bit=True,  # 4-bit quantization enabled by default for all Unsloth models
            )
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=8,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                use_gradient_checkpointing=True,
            )
            self.processor = None  # Text model uses tokenizer, not processor
            print(f"Qwen text model loaded on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load Qwen text model: {e}")
    
    def predict_probs(
        self,
        user_history: List[str],
        candidates: List[int],
        item_meta: Dict[int, Dict[str, Any]],
        num_candidates: Optional[int] = None,
    ) -> np.ndarray:
        """Predict probabilities for candidates.
        
        Args:
            user_history: List of item texts/images in user history
            candidates: List of candidate item IDs
            item_meta: Dict {item_id: {image_path, caption, semantic_summary, text}}
            num_candidates: Number of candidates (for validation)
            
        Returns:
            numpy array of probabilities [num_candidates]
        """
        if num_candidates is None:
            num_candidates = len(candidates)
        
        if self.mode == "raw_image":
            return self._predict_probs_raw_image(user_history, candidates, item_meta, num_candidates)
        elif self.mode == "caption":
            return self._predict_probs_caption(user_history, candidates, item_meta, num_candidates)
        elif self.mode == "semantic_summary":
            return self._predict_probs_semantic_summary_vl(user_history, candidates, item_meta, num_candidates)
        else:  # semantic_summary_small
            return self._predict_probs_semantic_summary_text(user_history, candidates, item_meta, num_candidates)
    
    def _build_rerank_prompt(self, user_history: List[str], candidate_texts: List[str]) -> str:
        """Build reranking prompt from user history and candidate texts."""
        history_text = "\n".join([f"- {h}" for h in user_history]) if user_history else "No previous interactions."
        
        cand_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidate_texts)])
        num_candidates = len(candidate_texts)
        answer_format = f"Answer with only one number (1-{num_candidates})." if num_candidates > 1 else "Answer with only one number (1)."
        
        prompt = f"""You are a recommendation ranking assistant.

Choose exactly ONE item the user is most likely to interact with next.

User history:
{history_text}

Candidate items:
{cand_text}

{answer_format}
""".strip()
        
        return prompt
    
    def _predict_probs_raw_image(
        self,
        user_history: List[str],  # Can be List[str] (texts/image_paths) or List[int] (item_ids)
        candidates: List[int],
        item_meta: Dict[int, Dict[str, Any]],
        num_candidates: int,
    ) -> np.ndarray:
        """Predict using raw images in prompt.
        
        NOTE: This is the ONLY method that loads and uses images directly.
        Other modes (caption, semantic_summary) only use text representations.
        
        Args:
            user_history: Can be:
                - List[int]: item_ids (preferred) - will load text and image from item_meta
                - List[str]: texts or image_paths (legacy) - will use as-is for history_text
        """
        # ✅ Build history with images and texts from item_meta
        # Check if user_history contains item_ids (int) or texts (str)
        history_images = []
        history_texts = []
        
        if user_history and isinstance(user_history[0], int):
            # user_history is List[int] (item_ids) - load text and image from item_meta
            for item_id in user_history:
                meta = item_meta.get(item_id, {})
                text = meta.get("text", f"item_{item_id}")
                text = _truncate_item_text(text, max_chars=200)
                image_path = meta.get("image_path") or meta.get("image")
                
                if image_path and os.path.isfile(image_path):
                    try:
                        img = Image.open(image_path).convert("RGB")
                        img = resize_image_for_qwen3vl(img, max_size=448)
                        history_images.append(img)
                        history_texts.append(text)
                    except Exception:
                        history_images.append(None)
                        history_texts.append(text)
                else:
                    history_images.append(None)
                    history_texts.append(text)
        else:
            # user_history is List[str] (texts/image_paths) - legacy format
            # Try to load images if they are paths, otherwise use as text
            for h in user_history:
                if isinstance(h, str) and os.path.isfile(h):
                    # It's an image path - try to load
                    try:
                        img = Image.open(h).convert("RGB")
                        img = resize_image_for_qwen3vl(img, max_size=448)
                        history_images.append(img)
                        history_texts.append(h)  # Use path as text fallback
                    except Exception:
                        history_images.append(None)
                        history_texts.append(h)
                else:
                    # It's text
                    history_images.append(None)
                    history_texts.append(h)
        
        # Build candidate images and texts
        candidate_images = []
        candidate_texts = []
        
        for item_id in candidates:
            meta = item_meta.get(item_id, {})
            text = meta.get("text", f"item_{item_id}")
            # ✅ Truncate item text metadata to prevent it from being too long
            text = _truncate_item_text(text, max_chars=200)
            image_path = meta.get("image_path") or meta.get("image")
            
            if image_path and os.path.isfile(image_path):
                try:
                    img = Image.open(image_path).convert("RGB")
                    # Resize image for Qwen3-VL (max 448px on longer side)
                    img = resize_image_for_qwen3vl(img, max_size=448)
                    candidate_images.append(img)
                    candidate_texts.append(text)
                except Exception:
                    candidate_images.append(None)
                    candidate_texts.append(text)
            else:
                candidate_images.append(None)
                candidate_texts.append(text)
        
        # Build prompt with images
        # ✅ Use history_texts (not raw user_history) to ensure we have proper text
        history_text = "\n".join([f"- {h}" for h in history_texts]) if history_texts else "No previous interactions."
        
        # Create messages with images (Qwen3-VL format)
        content = [
            {"type": "text", "text": f"""You are a recommendation ranking assistant.

Choose exactly ONE item the user is most likely to interact with next.

User history:"""}
        ]
        
        # ✅ Add history images and texts
        for img, text in zip(history_images, history_texts):
            if img is not None:
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": f"- {text}"})
        
        content.append({
            "type": "text",
            "text": "\nCandidate items:"
        })
        
        # Add candidate images and text
        for i, (text, img) in enumerate(zip(candidate_texts, candidate_images)):
            if img is not None:
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": f"{i+1}. {text}"})
        
        content.append({
            "type": "text",
            "text": f"\nAnswer with only one number (1-{num_candidates})."
        })
        
        messages = [{"role": "user", "content": content}]
        
        # Process and generate
        with torch.no_grad():
            # ✅ For raw_image mode, use larger max_length because images consume many visual tokens
            # Each image can consume 256-1024 visual tokens, so with 20 candidates we need more space
            # Qwen3-VL supports up to 8192 tokens, but we use config value to balance memory and completeness
            # Get max_seq_length from config
            try:
                from config import arg
                base_max_len = getattr(arg, 'qwen_max_seq_length', 2048)
            except ImportError:
                base_max_len = 2048  # Default fallback
            # Use larger for raw_image mode (2x base), smaller for text-only
            max_len = base_max_len * 2 if self.mode == "raw_image" else base_max_len
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                truncation=True,  # ✅ Truncate if too long
                max_length=max_len,  # ✅ Larger for raw_image mode (4096), smaller for text-only (2048)
            )
            
            # Move to device - handle nested structures (Qwen3-VL may have complex input format)
            def move_to_device(obj, dev):
                """Recursively move tensors to device."""
                if isinstance(obj, torch.Tensor):
                    return obj.to(dev)
                elif isinstance(obj, dict):
                    return {k: move_to_device(v, dev) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return type(obj)(move_to_device(item, dev) for item in obj)
                else:
                    return obj
            
            # ✅ Ensure all inputs are moved to device
            inputs = move_to_device(inputs, self.device)
            
            # ✅ Double-check: explicitly move input_ids and attention_mask if they exist
            if isinstance(inputs, dict):
                if "input_ids" in inputs and isinstance(inputs["input_ids"], torch.Tensor):
                    inputs["input_ids"] = inputs["input_ids"].to(self.device)
                if "attention_mask" in inputs and isinstance(inputs["attention_mask"], torch.Tensor):
                    inputs["attention_mask"] = inputs["attention_mask"].to(self.device)
                # Handle any other tensor keys that might exist
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor) and value.device != self.device:
                        inputs[key] = value.to(self.device)
            
            # Extract probabilities for numbers 1 to num_candidates from logits
            logits = self.model(**inputs).logits[:, -1]  # [vocab_size]
            
            # Get token IDs for numbers
            prob_array = np.zeros(num_candidates)
            for i in range(1, num_candidates + 1):
                num_str = str(i)
                token_id = self.processor.tokenizer.convert_tokens_to_ids(num_str)
                if token_id != self.processor.tokenizer.unk_token_id:
                    # Get probability for this token
                    prob = torch.softmax(logits, dim=-1)[0, token_id].item()
                    prob_array[i - 1] = prob
            
            # Normalize
            if prob_array.sum() > 0:
                prob_array = prob_array / prob_array.sum()
            else:
                # Fallback: uniform distribution
                prob_array = np.ones(num_candidates) / num_candidates
            
            return prob_array
    
    def _predict_probs_caption(
        self,
        user_history: List[str],
        candidates: List[int],
        item_meta: Dict[int, Dict[str, Any]],
        num_candidates: int,
    ) -> np.ndarray:
        """Predict using image captions.
        
        NOTE: This method does NOT load images. It only uses caption text from item_meta.
        Only raw_image mode loads and uses images directly.
        """
        candidate_texts = []
        for item_id in candidates:
            meta = item_meta.get(item_id, {})
            caption = meta.get("caption")
            text = meta.get("text", f"item_{item_id}")
            # ✅ Truncate item text metadata to prevent it from being too long
            text = _truncate_item_text(text, max_chars=200)
            
            if caption:
                candidate_texts.append(f"{text} (Image: {caption})")
            else:
                candidate_texts.append(text)
        
        prompt = self._build_rerank_prompt(user_history, candidate_texts)
        
        # Use VL model but with text-only input
        messages = [{"role": "user", "content": prompt}]
        
        with torch.no_grad():
            # ✅ For caption mode (text-only), use max_seq_length from config
            try:
                from config import arg
                max_length = getattr(arg, 'qwen_max_seq_length', 2048)
            except ImportError:
                max_length = 2048  # Default fallback
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,  # ✅ Text-only mode, use from config
            )
            
            # Move to device - handle nested structures (Qwen3-VL may have complex input format)
            def move_to_device(obj, dev):
                """Recursively move tensors to device."""
                if isinstance(obj, torch.Tensor):
                    return obj.to(dev)
                elif isinstance(obj, dict):
                    return {k: move_to_device(v, dev) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return type(obj)(move_to_device(item, dev) for item in obj)
                else:
                    return obj
            
            # ✅ Ensure all inputs are moved to device
            inputs = move_to_device(inputs, self.device)
            
            # ✅ Double-check: explicitly move input_ids and attention_mask if they exist
            if isinstance(inputs, dict):
                if "input_ids" in inputs and isinstance(inputs["input_ids"], torch.Tensor):
                    inputs["input_ids"] = inputs["input_ids"].to(self.device)
                if "attention_mask" in inputs and isinstance(inputs["attention_mask"], torch.Tensor):
                    inputs["attention_mask"] = inputs["attention_mask"].to(self.device)
                # Handle any other tensor keys that might exist
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor) and value.device != self.device:
                        inputs[key] = value.to(self.device)
            
            logits = self.model(**inputs).logits[:, -1]  # [vocab_size]
            
            # Extract probabilities for numbers
            prob_array = np.zeros(num_candidates)
            for i in range(1, num_candidates + 1):
                num_str = str(i)
                token_id = self.processor.tokenizer.convert_tokens_to_ids(num_str)
                if token_id != self.processor.tokenizer.unk_token_id:
                    prob = torch.softmax(logits, dim=-1)[0, token_id].item()
                    prob_array[i - 1] = prob
            
            if prob_array.sum() > 0:
                prob_array = prob_array / prob_array.sum()
            else:
                # Fallback: uniform distribution
                prob_array = np.ones(num_candidates) / num_candidates
            
            return prob_array
    
    def _predict_probs_semantic_summary_vl(
        self,
        user_history: List[str],
        candidates: List[int],
        item_meta: Dict[int, Dict[str, Any]],
        num_candidates: int,
    ) -> np.ndarray:
        """Predict using semantic summaries with VL model.
        
        NOTE: This method does NOT load images. It only uses semantic_summary text from item_meta.
        Only raw_image mode loads and uses images directly.
        """
        candidate_texts = []
        for item_id in candidates:
            meta = item_meta.get(item_id, {})
            semantic_summary = meta.get("semantic_summary")
            text = meta.get("text", f"item_{item_id}")
            # ✅ Truncate item text metadata to prevent it from being too long
            text = _truncate_item_text(text, max_chars=200)
            
            if semantic_summary:
                candidate_texts.append(f"{text} (Semantic: {semantic_summary})")
            else:
                candidate_texts.append(text)
        
        prompt = self._build_rerank_prompt(user_history, candidate_texts)
        
        messages = [{"role": "user", "content": prompt}]
        
        with torch.no_grad():
            # ✅ For semantic_summary mode (text-only), use max_seq_length from config
            try:
                from config import arg
                max_length = getattr(arg, 'qwen_max_seq_length', 2048)
            except ImportError:
                max_length = 2048  # Default fallback
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,  # ✅ Text-only mode, use from config
            )
            
            # Move to device - handle nested structures (Qwen3-VL may have complex input format)
            def move_to_device(obj, dev):
                """Recursively move tensors to device."""
                if isinstance(obj, torch.Tensor):
                    return obj.to(dev)
                elif isinstance(obj, dict):
                    return {k: move_to_device(v, dev) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return type(obj)(move_to_device(item, dev) for item in obj)
                else:
                    return obj
            
            inputs = move_to_device(inputs, self.device)
            
            logits = self.model(**inputs).logits[:, -1]
            
            prob_array = np.zeros(num_candidates)
            for i in range(1, num_candidates + 1):
                num_str = str(i)
                token_id = self.processor.tokenizer.convert_tokens_to_ids(num_str)
                if token_id != self.processor.tokenizer.unk_token_id:
                    prob = torch.softmax(logits, dim=-1)[0, token_id].item()
                    prob_array[i - 1] = prob
            
            if prob_array.sum() > 0:
                prob_array = prob_array / prob_array.sum()
            else:
                # Fallback: uniform distribution
                prob_array = np.ones(num_candidates) / num_candidates
            
            return prob_array
    
    def _predict_probs_semantic_summary_text(
        self,
        user_history: List[str],
        candidates: List[int],
        item_meta: Dict[int, Dict[str, Any]],
        num_candidates: int,
    ) -> np.ndarray:
        """Predict using semantic summaries with smaller text model."""
        candidate_texts = []
        for item_id in candidates:
            meta = item_meta.get(item_id, {})
            semantic_summary = meta.get("semantic_summary")
            text = meta.get("text", f"item_{item_id}")
            # ✅ Truncate item text metadata to prevent it from being too long
            text = _truncate_item_text(text, max_chars=200)
            
            if semantic_summary:
                candidate_texts.append(f"{text} (Semantic: {semantic_summary})")
            else:
                candidate_texts.append(text)
        
        prompt = self._build_rerank_prompt(user_history, candidate_texts)
        
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            logits = self.model(**inputs).logits[:, -1]
            
            prob_array = np.zeros(num_candidates)
            for i in range(1, num_candidates + 1):
                num_str = str(i)
                token_id = self.tokenizer.convert_tokens_to_ids(num_str)
                if token_id != self.tokenizer.unk_token_id:
                    prob = torch.softmax(logits, dim=-1)[0, token_id].item()
                    prob_array[i - 1] = prob
            
            if prob_array.sum() > 0:
                prob_array = prob_array / prob_array.sum()
            else:
                # Fallback: uniform distribution
                prob_array = np.ones(num_candidates) / num_candidates
            
            return prob_array

