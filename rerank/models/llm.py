
##%%writefile /kaggle/working/rerank/models/llm.py
from unsloth import FastLanguageModel
import torch
import torch.nn.functional as F
import string
import pandas as pd
import ast
import numpy as np
import os
import math
import logging

# Legacy: Keep for backward compatibility, but now we use numbers
# Use both uppercase and lowercase for up to 52 candidates (A-Z, a-z)
LETTERS = list(string.ascii_uppercase) + list(string.ascii_lowercase)  # A-Z, a-z (52 letters)


def build_prompt_from_candidates(user_history, candidate_ids, item_id2text, max_candidates=None):
    """Build prompt for LLM reranking.
    
    Args:
        user_history: List of item texts in user history
        candidate_ids: List of candidate item IDs
        item_id2text: Mapping from item_id to item text
        max_candidates: Maximum number of candidates (None = no limit, uses all)
        
    Returns:
        Formatted prompt string with candidate labels (A, B, C, ... or a, b, c, ...)
    """
    if max_candidates is not None and len(candidate_ids) > max_candidates:
        candidate_ids = candidate_ids[:max_candidates]
    
    history_text = "\n".join([f"- {h}" for h in user_history])

    candidates = [item_id2text.get(cid, f"item_{cid}") for cid in candidate_ids]
    num_candidates = len(candidates)
    
    # Use letters (A-Z, a-z) for up to 52 candidates (LlamaRec style)
    if num_candidates > len(LETTERS):
        raise ValueError(
            f"Too many candidates ({num_candidates}). Maximum supported: {len(LETTERS)} candidates "
            f"(using letters A-Z, a-z). Consider reducing max_candidates or using number labels."
        )
    
    # Use letters instead of numbers (LlamaRec style)
    cand_text = "\n".join(
        [f"{LETTERS[i]}. {c}" for i, c in enumerate(candidates)]
    )

    # Answer format with letters
    if num_candidates <= 26:
        answer_format = f"Answer with only one letter (A-{LETTERS[num_candidates-1]})."
    else:
        answer_format = f"Answer with only one letter (A-Z, a-{LETTERS[num_candidates-1]})."

    prompt = f"""
You are a recommendation ranking assistant.

Choose exactly ONE item the user is most likely to interact with next.

User history:
{history_text}

Candidate items:
{cand_text}

{answer_format}
""".strip()

    return prompt

def rank_candidates(probs, candidate_ids):
    """Rank candidates by probabilities.
    
    Args:
        probs: Array of probabilities (one per candidate)
        candidate_ids: List of candidate item IDs
        
    Returns:
        List of candidate IDs sorted by probability (descending)
        
    Raises:
        ValueError: If len(probs) != len(candidate_ids)
    """
    if len(probs) != len(candidate_ids):
        raise ValueError(
            f"Mismatch: {len(candidate_ids)} candidates but {len(probs)} probabilities. "
            f"Each candidate must have exactly one probability."
        )
    
    ranked = sorted(
        zip(candidate_ids, probs),
        key=lambda x: x[1],
        reverse=True
    )
    return [cid for cid, _ in ranked]

def recall_at_k(ranked_items, gt_item, k):
    return int(gt_item in ranked_items[:k])


def ndcg_at_k(ranked_items, gt_item, k):
    if gt_item not in ranked_items[:k]:
        return 0.0
    rank = ranked_items.index(gt_item) + 1
    return 1.0 / math.log2(rank + 1)

class LLMModel:
    def __init__(self, train_data=None, model_name=None, verbose=1):
            # Priority: Use Unsloth models by default for better performance and 4-bit quantization
            # ✅ Track debug print count to limit frequency during eval
            self._debug_predict_count = 0
            self._max_debug_prints = 3  # Only print debug for first 3 samples
            self.model_name = model_name or "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"
            self.train_data = train_data
            # Verbosity level: 0 (minimal), 1 (normal), 2 (verbose/debug)
            try:
                from config import arg
                self.verbose = getattr(arg, 'qwen_verbose', verbose)
            except ImportError:
                self.verbose = verbose

    def load_model(self, use_torch_compile=False, max_seq_length=None):
        """Load LLM model with 4-bit quantization (default for Unsloth models).
        
        Args:
            use_torch_compile: Whether to use torch.compile() for faster inference
            max_seq_length: Maximum sequence length (None = get from config, default: 2048)
        
        Note:
            - All Unsloth models are loaded with 4-bit quantization by default
            - If self.model_name points to a path with adapter weights, Unsloth will automatically
              load the adapter. Otherwise, it loads base model and prepares for training.
        """
        # Get max_seq_length from config if not provided
        if max_seq_length is None:
            try:
                from config import arg
                max_seq_length = getattr(arg, 'qwen_max_seq_length', 2048)
            except ImportError:
                max_seq_length = 2048  # Default fallback
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        print(f"Loading LLM model: {self.model_name}")
        print(f"  Max sequence length: {max_seq_length}")
        print(f"  Note: Unsloth will automatically load adapter if present in the model path")
        
        # Unsloth automatically detects and loads adapter weights if present
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.model_name,
            max_seq_length = max_seq_length,
            dtype = torch.float16,
            load_in_4bit = True,  # 4-bit quantization enabled by default for all Unsloth models
            device_map={'': local_rank},
        )
        
        # Only add LoRA if model doesn't already have adapter weights
        # Unsloth's from_pretrained automatically loads adapter if present, so we check
        # If adapter is already loaded, get_peft_model will reuse it
        try:
            # Get LoRA parameters from config
            try:
                from config import arg
                lora_r = getattr(arg, 'qwen_lora_r', 8)
                lora_alpha = getattr(arg, 'qwen_lora_alpha', 16)
                lora_dropout = getattr(arg, 'qwen_lora_dropout', 0.05)
            except ImportError:
                lora_r = 8
                lora_alpha = 16
                lora_dropout = 0.05
            
            # Check if model already has adapter (from pretrained path)
            if hasattr(self.model, 'peft_config') and self.model.peft_config:
                print(f"  ✅ Adapter weights loaded from pretrained model")
            else:
                # No adapter found, prepare for training by adding LoRA
                print(f"  No adapter found, adding LoRA for training...")
                print(f"    LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
                self.model = FastLanguageModel.get_peft_model(
                    self.model,
                    r = lora_r,
                    target_modules = ["q_proj","k_proj","v_proj","o_proj"],
                    lora_alpha = lora_alpha,
                    lora_dropout = lora_dropout,
                    bias = "none",
                    use_gradient_checkpointing = True,
                )
        except Exception as e:
            # Fallback: always add LoRA if check fails
            print(f"  Adding LoRA (fallback)...")
            try:
                from config import arg
                lora_r = getattr(arg, 'qwen_lora_r', 8)
                lora_alpha = getattr(arg, 'qwen_lora_alpha', 16)
                lora_dropout = getattr(arg, 'qwen_lora_dropout', 0.05)
            except ImportError:
                lora_r = 8
                lora_alpha = 16
                lora_dropout = 0.05
        self.model = FastLanguageModel.get_peft_model(
            self.model,
                r = lora_r,
            target_modules = ["q_proj","k_proj","v_proj","o_proj"],
                lora_alpha = lora_alpha,
                lora_dropout = lora_dropout,
            bias = "none",
            use_gradient_checkpointing = True,
        )
        
        # Compile model if requested (PyTorch 2.0+)
        if use_torch_compile and hasattr(torch, 'compile'):
            try:
                print("Compiling LLM model with torch.compile() for faster inference...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("LLM model compiled successfully!")
            except Exception as e:
                print(f"Warning: torch.compile() failed: {e}. Continuing without compilation.")
    def train(self, batch_size=None):

        from datasets import Dataset
        from trl import SFTTrainer, SFTConfig
        from unsloth.chat_templates import train_on_responses_only

        # Setup logging to file
        log_file = "training_eval_log.txt"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='a'),  # Append mode
                logging.StreamHandler()  # Also print to console
            ]
        )
        logger = logging.getLogger(__name__)
        logger.info("Starting LLM training and evaluation logging")

        hf_train_dataset = Dataset.from_list(self.train_data)
        
        # ✅ Format messages to text using chat template (like notebook Cell 7)
        import re  # For removing thinking content
        
        def formatting_prompts_func(examples):
            """Format conversations to text using chat template.
            
            When batched=True: examples["messages"] is a list of message lists
            When batched=False: examples["messages"] is a single message list
            """
            messages_list = examples["messages"]
            
            def clean_thinking_content(text):
                """Remove thinking content from text if present."""
                # Remove <think>...</think> tags (Qwen3 thinking format)
                text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
                # Remove <think>...</think> tags (Qwen3 format)  
                text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
                # Remove empty lines between assistant tag and response (if thinking was removed)
                # This ensures format: <|im_start|>assistant\nA<|im_end|> instead of <|im_start|>assistant\n\nA<|im_end|>
                text = re.sub(r'(<\|im_start\|>assistant\n)\n+', r'\1', text)
                # Clean up extra newlines but preserve structure
                # Don't strip() to preserve leading/trailing structure for train_on_responses_only
                text = re.sub(r'\n\n+', '\n', text)  # Replace multiple newlines with single newline
                return text
            
            # Check if batched (list of lists) or single (single list)
            if isinstance(messages_list, list) and len(messages_list) > 0:
                # Check if first element is a list (batched) or dict (single message list)
                if isinstance(messages_list[0], list):
                    # Batched: list of message lists
                    texts = []
                    for messages in messages_list:
                        text = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False,
                            enable_thinking=False  # ✅ Disable thinking mode for training consistency
                        )
                        # ✅ Remove thinking content if present (some tokenizers may still add it)
                        text = clean_thinking_content(text)
                        texts.append(text)
                    return {"text": texts}
                elif isinstance(messages_list[0], dict):
                    # Single: messages_list is already a list of message dicts
                    text = self.tokenizer.apply_chat_template(
                        messages_list,
                        tokenize=False,
                        add_generation_prompt=False,
                        enable_thinking=False  # ✅ Disable thinking mode for training consistency
                    )
                    # ✅ Remove thinking content if present (some tokenizers may still add it)
                    text = clean_thinking_content(text)
                    return {"text": text}
            
            raise ValueError(f"Invalid messages format: {type(messages_list)}")
        
        # ✅ Map to format as text (like notebook Cell 7) - use batched=True for efficiency
        hf_train_dataset = hf_train_dataset.map(
            formatting_prompts_func,
            batched=True,  # Process in batches for efficiency (like notebook)
        )
        
        # ✅ Debug: Print first sample to verify format (only if verbose >= 2)
        if self.verbose >= 2:
            print("\n[LLMModel] Debug: First training sample format:")
            print("-" * 80)
            first_text = hf_train_dataset[0]["text"]
            print(first_text[:500] + "..." if len(first_text) > 500 else first_text)
            print("-" * 80)
            # Check if response part exists
            if "<|im_start|>assistant" in first_text:
                print("✅ Response part found: <|im_start|>assistant")
            else:
                print("❌ Response part NOT found: <|im_start|>assistant")
            if "<|im_start|>user" in first_text:
                print("✅ Instruction part found: <|im_start|>user")
            else:
                print("❌ Instruction part NOT found: <|im_start|>user")
            print("-" * 80)

        # Get training parameters from config if available
        try:
            from config import arg
            num_epochs = getattr(arg, 'rerank_epochs', 2)
            # Use batch_size parameter if provided, otherwise get from config
            if batch_size is None:
                batch_size = getattr(arg, 'rerank_batch_size', 16)
            # Get learning rate from config (default: 1e-4, was hardcoded to 2e-5)
            learning_rate = getattr(arg, 'rerank_lr', 1e-4)
            # Get gradient accumulation steps from config
            gradient_accumulation_steps = getattr(arg, 'qwen_gradient_accumulation_steps', 2)
            # Get warmup steps from config
            warmup_steps = getattr(arg, 'qwen_warmup_steps', 20)
        except ImportError:
            num_epochs = 1
            if batch_size is None:
                batch_size = 16  # Default fallback
            learning_rate = 1e-4  # Default fallback
            gradient_accumulation_steps = 2  # Default fallback
            warmup_steps = 20  # Default fallback
        
        # ✅ Use SFTConfig and SFTTrainer (like notebook Cell 8)
        print(hf_train_dataset[0]["text"])
        training_args = SFTConfig(
            dataset_text_field="text",  # Field name in dataset
        output_dir="./qwen_rerank",
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,  # ✅ Use from config
            learning_rate=learning_rate,  # ✅ Use from config (default: 1e-4)
            num_train_epochs=num_epochs,
            logging_steps=5,
        save_steps=500,
        report_to="tensorboard",  # ✅ Changed from "none" to enable logging
            fp16=True,
            optim="adamw_8bit",
            ddp_find_unused_parameters = False,
            dataloader_pin_memory = False,
            load_best_model_at_end=False, 
            lr_scheduler_type = "cosine",
            warmup_steps = warmup_steps,  # ✅ Use from config
            weight_decay = 0.1,
            disable_tqdm=False,  # ✅ Enable progress bar
        )
        
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=hf_train_dataset,
            args=training_args,
        )
        
        # ✅ Use train_on_responses_only to automatically mask prompt tokens (like notebook)
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
        
        # ✅ Check if in eval mode - skip training if so
        try:
            from config import arg
            rerank_action = getattr(arg, 'rerank_action', 'train') or 'train'
        except ImportError:
            rerank_action = 'train'
        
        # ✅ Actually train the model! (skip if eval mode)
        if rerank_action != "eval":
                logger.info(f"[LLMModel] Starting training for {num_epochs} epochs...")
                logger.info(f"[LLMModel] Training config: lr={learning_rate}, batch_size={batch_size}, epochs={num_epochs}")
                logger.info(f"[LLMModel] Dataset size: {len(hf_train_dataset)} samples")
                logger.info(f"[LLMModel] Total steps: {len(hf_train_dataset) // (batch_size * 2) * num_epochs}")
                
                trainer.train()
                
                # ✅ Save final model (vì load_best_model_at_end=False)
                logger.info(f"[LLMModel] Saving final model...")
                trainer.save_model()  # Save to output_dir
                logger.info(f"[LLMModel] Model saved to {training_args.output_dir}")
                
                # ✅ Log final training loss
                logger.info(f"[LLMModel] Training completed!")
                logger.info(f"[LLMModel] Check training logs above for loss progression")
                logger.info(f"[LLMModel] If loss did not decrease, check:")
                logger.info(f"  - Learning rate (current: {learning_rate})")
                logger.info(f"  - Training data quality")
                logger.info(f"  - Model size (current: {self.model_name})")
    
    def predict(self, prompt, num_candidates=None):
        """Predict probabilities for candidates using letters (A, B, C, ... or a, b, c, ...) - LlamaRec style.
        
        Args:
            prompt: Input prompt (plain text, will be converted to chat template format)
            num_candidates: Number of candidates (if None, infers from prompt)
            
        Returns:
            numpy array of probabilities [num_candidates]
        """
        # Setup logging to file (append mode)
        log_file = "training_eval_log.txt"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='a'),  # Append mode
                logging.StreamHandler()  # Also print to console
            ]
        )
        logger = logging.getLogger(__name__)
        
        # ✅ Increment debug counter FIRST (before any debug prints)
        self._debug_predict_count += 1
        
        # Get max_length from config
        try:
            from config import arg
            max_length = getattr(arg, 'qwen_max_seq_length', 2048)
        except ImportError:
            max_length = 2048  # Default fallback
        
        # ✅ Convert plain text prompt to chat template format (consistency with training)
        # Training uses apply_chat_template with system message, so inference should too
        # ✅ Remove "You are a recommendation ranking assistant." from prompt if present (already in system message)
        system_msg = "You are a recommendation ranking assistant."
        if prompt.strip().startswith(system_msg):
            # Remove system message from prompt (already in system message)
            prompt = prompt.strip()[len(system_msg):].strip()
            # Remove leading newline if present
            if prompt.startswith("\n"):
                prompt = prompt[1:]
        
        # ✅ Add system message to match training format
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template with generation prompt (adds <|im_start|>assistant\n at the end)
        # This ensures consistency with training format
        # ✅ Disable thinking mode to ensure direct answer prediction (for reranking, we want direct letter answers)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # ✅ Add <|im_start|>assistant\n for generation
            enable_thinking=False  # ✅ Disable thinking mode for direct answer prediction
        )
        
        # Tokenize the chat template formatted text
        inputs = self.tokenizer(
            text, 
            return_tensors="pt",
            truncation=True,  # ✅ Truncate if too long
            max_length=max_length,  # ✅ Use from config
        ).to(self.model.device)

        # ✅ Debug: Check prompt format and tokenization (only if verbose >= 2, and only for first few samples)
        if self.verbose >= 2 and self._debug_predict_count <= self._max_debug_prints:
            # Check if prompt ends correctly (should end with <|im_start|>assistant\n)
            if not text.rstrip().endswith("<|im_start|>assistant"):
                print(f"[DEBUG] Sample {self._debug_predict_count}: Prompt may not end correctly. Last 50 chars: {text[-50:]}")
            
            # Check token count
            input_ids = inputs["input_ids"][0]
            print(f"[DEBUG] Sample {self._debug_predict_count}: Prompt token count: {len(input_ids)}")
            if self._debug_predict_count == 1:  # Only print full details for first sample
                print(f"[DEBUG] Sample {self._debug_predict_count}: Last 10 tokens: {input_ids[-10:].tolist()}")
                print(f"[DEBUG] Sample {self._debug_predict_count}: Last 10 token texts: {[self.tokenizer.decode([t]) for t in input_ids[-10:]]}")

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits[:, -1]  # [vocab_size]
        
        logger.info(f"[Eval] Predicted for sample {self._debug_predict_count}, num_candidates={num_candidates}")
        
        # Infer num_candidates from original prompt (before chat template) if not provided
        # Note: We use the original prompt for inference, not the chat template formatted text
        # because the chat template adds special tokens that might interfere with parsing
        if num_candidates is None:
            # Count "Candidate items:" section in original prompt
            if "Candidate items:" in prompt:
                candidates_section = prompt.split("Candidate items:")[1].split("Answer")[0]
                num_candidates = candidates_section.strip().count("\n") + 1
            else:
                # Fallback: try to infer from answer format
                if "Answer with only one letter" in prompt:
                    # Extract letter range like "A-Z" or "A-Z, a-z"
                    import re
                    # Try to match patterns like "A-Z" or "A-Z, a-z"
                    match = re.search(r'\(([A-Z])-([A-Za-z])\)', prompt)
                    if match:
                        start_letter = match.group(1)
                        end_letter = match.group(2)
                        start_idx = LETTERS.index(start_letter) if start_letter in LETTERS else 0
                        end_idx = LETTERS.index(end_letter) if end_letter in LETTERS else len(LETTERS) - 1
                        num_candidates = end_idx - start_idx + 1
                    else:
                        num_candidates = min(20, len(LETTERS))  # Default fallback
                else:
                    num_candidates = min(20, len(LETTERS))  # Default fallback
        
        # Validate num_candidates
        if num_candidates > len(LETTERS):
            print(f"[WARNING] num_candidates ({num_candidates}) exceeds max letters ({len(LETTERS)}). Truncating to {len(LETTERS)}.")
            num_candidates = len(LETTERS)
        
        # Get token IDs for letters A-Z, a-z (LlamaRec style)
        letter_tokens = []
        for i in range(num_candidates):
            letter = LETTERS[i]
            
            # Strategy 1: Try direct letter token
            token_id = self.tokenizer.convert_tokens_to_ids(letter)
            if token_id != self.tokenizer.unk_token_id:
                # ✅ Verify it's a single token
                encoded = self.tokenizer.encode(letter, add_special_tokens=False)
                if len(encoded) == 1:
                    letter_tokens.append((i, letter, token_id))
                    continue
                elif self.verbose >= 2 and self._debug_predict_count <= 1:
                    print(f"[DEBUG] Letter '{letter}' encodes to {len(encoded)} tokens: {encoded}")
            
            # Strategy 2: Try with space prefix (common in BPE tokenizers)
            space_letter = " " + letter
            token_id = self.tokenizer.convert_tokens_to_ids(space_letter)
            if token_id != self.tokenizer.unk_token_id:
                # ✅ Verify it's a single token
                encoded = self.tokenizer.encode(space_letter, add_special_tokens=False)
                if len(encoded) == 1:
                    letter_tokens.append((i, letter, token_id))
                    continue
                elif self.verbose >= 2 and self._debug_predict_count <= 1:
                    print(f"[DEBUG] Space+letter ' {letter}' encodes to {len(encoded)} tokens: {encoded}")
            
            # Strategy 3: Try encoding and taking first token (fallback)
            encoded = self.tokenizer.encode(letter, add_special_tokens=False)
            if len(encoded) > 0 and encoded[0] != self.tokenizer.unk_token_id:
                # ⚠️ Warning: This may not be a single token
                if len(encoded) > 1 and self.verbose >= 1:
                    print(f"[WARNING] Letter '{letter}' encodes to {len(encoded)} tokens, using first token only")
                letter_tokens.append((i, letter, encoded[0]))
                continue
        
        # ✅ Debug: Verify all letter tokens are single tokens (only if verbose >= 2, and only for first sample)
        if self.verbose >= 2 and self._debug_predict_count == 1:
            for cand_idx, letter, token_id in letter_tokens[:5]:  # Check first 5
                # Check if letter encodes to single token
                encoded = self.tokenizer.encode(letter, add_special_tokens=False)
                if len(encoded) > 1:
                    print(f"[DEBUG] Letter '{letter}' (candidate {cand_idx}) encodes to {len(encoded)} tokens: {encoded}")
                else:
                    print(f"[DEBUG] Letter '{letter}' (candidate {cand_idx}) = token {token_id} (single token ✓)")
        
        # Debug: Check if we found letter tokens (only warn if verbose >= 1)
        if len(letter_tokens) < num_candidates:
            if self.verbose >= 1:
                print(f"[WARNING] Only found {len(letter_tokens)}/{num_candidates} letter tokens!")
                if self.verbose >= 2:
                    print(f"  Found letters: {[l for _, l, _ in letter_tokens[:10]]}...")
            # If we found very few tokens, this is a problem
            if len(letter_tokens) == 0:
                if self.verbose >= 1:
                    print(f"[ERROR] No letter tokens found! Falling back to uniform distribution.")
                    print(f"[ERROR] This will cause recall = random! Check tokenizer compatibility.")
                return np.ones(num_candidates) / num_candidates
        
        # Extract probabilities for letter tokens
        token_ids = [tid for _, _, tid in letter_tokens]
        
        # ✅ Apply temperature scaling if specified (default: 1.0 = no scaling)
        # Temperature < 1.0: sharper distribution (more confident)
        # Temperature > 1.0: smoother distribution (less confident)
        # Temperature = 1.0: standard softmax (default)
        try:
            from config import arg
            temperature = getattr(arg, 'qwen_temperature', 1.0)
        except ImportError:
            temperature = 1.0  # Default: no temperature scaling
        
        if temperature != 1.0:
            # Apply temperature scaling: logits / temperature
            scaled_logits = logits[:, token_ids] / temperature
            probs = F.softmax(scaled_logits, dim=-1)
        else:
            # Standard softmax (no temperature scaling)
            probs = F.softmax(logits[:, token_ids], dim=-1)

        # Map back to candidate indices (0-indexed)
        prob_array = np.zeros(num_candidates)
        for idx, (cand_idx, letter, token_id) in enumerate(letter_tokens):
            if cand_idx < num_candidates:
                prob_array[cand_idx] = probs[0, idx].item()
        
        # Normalize probabilities (in case some letters weren't found)
        prob_sum = prob_array.sum()
        if prob_sum > 0:
            prob_array = prob_array / prob_sum
        else:
            # Fallback: uniform distribution if all probabilities are zero
            if self.verbose >= 1:
                print(f"[WARNING] All probabilities are zero, using uniform distribution")
                print(f"[WARNING] This will cause recall = random! Model may not have learned anything.")
            prob_array = np.ones(num_candidates) / num_candidates
        
        # ✅ Debug: Check if probabilities are uniform (model chưa học được gì) - only warn if verbose >= 2, and only for first sample
        if self.verbose >= 2 and self._debug_predict_count <= 1:
            prob_std = np.std(prob_array)
            expected_uniform_std = 0.0  # Uniform distribution has std = 0
            if prob_std < 0.01:  # Nearly uniform
                print(f"[WARNING] Probabilities are nearly uniform (std={prob_std:.4f})!")
                print(f"[WARNING] Model may not have learned anything. Check training loss.")
        
        return prob_array
    def evaluate(self,df, user2history, item_id2text):
        recalls = {1: [], 5: [], 10: []}
        ndcgs = {5: [], 10: []}

        for _, row in df.iterrows():
            user_id = row["user_index"]
            gt_item = row["label"]

            candidate_ids = ast.literal_eval(row["candidate_ids"])
            history = user2history[user_id][-10:]

            prompt = build_prompt_from_candidates(history, candidate_ids,item_id2text)
            probs = self.predict(prompt)
            ranked_items = rank_candidates(probs, candidate_ids)

            for k in recalls:
                recalls[k].append(recall_at_k(ranked_items, gt_item, k))

            for k in ndcgs:
                ndcgs[k].append(ndcg_at_k(ranked_items, gt_item, k))

        return {
            **{f"Recall@{k}": sum(v)/len(v) for k, v in recalls.items()},
            **{f"NDCG@{k}": sum(v)/len(v) for k, v in ndcgs.items()}
        }
    
    def predict_probs(self, prompt, num_candidates=None):
        """Alias for predict for backward compatibility."""
        return self.predict(prompt, num_candidates)
