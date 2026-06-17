"""Standalone training script for rerank models (Stage 2).

This script allows training rerank models independently from retrieval.
Can work in two modes:
1. With pre-trained retrieval: Load retrieval model and use its candidates
2. Ground truth mode: Use ground truth + random negatives (no retrieval needed)
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
from typing import Dict, List, Optional


class TeeLogger:
    """Redirect stdout/stderr to both terminal and file."""
    def __init__(self, file_path, mode='w'):
        self.file = open(file_path, mode, encoding='utf-8')
        self.terminal = sys.stdout if 'stdout' in str(file_path) else sys.stderr
        
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()  # Ensure immediate write
        
    def flush(self):
        self.terminal.flush()
        self.file.flush()
        
    def close(self):
        self.file.close()

# Note: config import is moved to main() to avoid argument parsing conflicts
from evaluation.utils import load_dataset_from_csv, evaluate_split
from rerank.registry import get_reranker_class
from retrieval.registry import get_retriever_class
from pytorch_lightning import seed_everything


def main():
    # Import config (it now includes script-specific arguments as optional)
    from config import arg
    
    # Setup logging to file
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_log_{timestamp}.txt"
    
    # Redirect stdout and stderr to both terminal and file
    tee_stdout = TeeLogger(log_file, mode='w')
    tee_stderr = TeeLogger(log_file, mode='a')
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr
    
    print(f"[INFO] Training log will be saved to: {log_file.absolute()}")
    print(f"[INFO] Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Get script-specific arguments from arg (with defaults and validation)
    rerank_method_val = getattr(arg, 'rerank_method', None)
    if rerank_method_val is None:
        raise ValueError("--rerank_method is required. Please specify: --rerank_method qwen|qwen3vl|vip5")
    
    valid_rerank_methods = ["qwen", "qwen3vl", "vip5"]
    if rerank_method_val not in valid_rerank_methods:
        raise ValueError(f"Invalid rerank_method: {rerank_method_val}. Must be one of {valid_rerank_methods}")
    
    class Args:
        rerank_method = rerank_method_val
        rerank_top_k = getattr(arg, 'rerank_top_k', 50) or 50
        mode = getattr(arg, 'mode', 'ground_truth') or 'ground_truth'
        retrieval_method = getattr(arg, 'retrieval_method', 'lrurec') or 'lrurec'
        retrieval_top_k = getattr(arg, 'retrieval_top_k', 200) or 200
        metric_k = getattr(arg, 'metric_k', 10) or 10
        qwen3vl_mode = getattr(arg, 'qwen3vl_mode', 'raw_image') or 'raw_image'
    
    args = Args()
    
    seed_everything(arg.seed)
    
    print("=" * 80)
    print("Standalone Rerank Training")
    print("=" * 80)
    print(f"Rerank Method: {args.rerank_method}")
    print(f"Mode: {args.mode}")
    if args.mode == "retrieval":
        print(f"Retrieval Method: {args.retrieval_method} (top_k={args.retrieval_top_k})")
    print("=" * 80)
    
    # Load dataset
    print("\n[1/3] Loading dataset...")
    from evaluation.utils import load_dataset_from_csv
    data = load_dataset_from_csv(
        arg.dataset_code,
        arg.min_rating,
        arg.min_uc,
        arg.min_sc
    )
    train = data["train"]
    val = data["val"]
    test = data["test"]
    item_count = data["item_count"]
    
    print(f"  Train users: {len(train)}")
    print(f"  Val users: {len(val)}")
    print(f"  Test users: {len(test)}")
    print(f"  Items: {item_count}")
    
    # Create reranker
    print(f"\n[2/3] Creating reranker ({args.rerank_method})...")
    RerankerCls = get_reranker_class(args.rerank_method)
    
    # Prepare reranker kwargs
    reranker_kwargs = {"top_k": args.rerank_top_k}
    
    # Add method-specific kwargs
    if args.rerank_method in ["qwen", "qwen3vl"]:
        # Unified reranker: use qwen_mode and qwen_model
        qwen_mode = getattr(arg, 'qwen_mode', None)
        qwen_model = getattr(arg, 'qwen_model', None)
        
        # Backward compatibility: if qwen3vl_mode is set, use it
        if args.rerank_method == "qwen3vl" and qwen_mode is None:
            qwen3vl_mode_legacy = getattr(arg, 'qwen3vl_mode', 'caption')
            # Map legacy modes to new modes
            mode_mapping = {
                'caption': 'caption',
                'VIU': 'VIU',
                'viu_small': 'VIU'  # Map to VIU
            }
            qwen_mode = mode_mapping.get(qwen3vl_mode_legacy, 'caption')
            # For viu_small, use qwen3-0.6b model
            if qwen3vl_mode_legacy == 'viu_small' and qwen_model is None:
                qwen_model = 'qwen3-0.6b'
        
        if qwen_mode:
            reranker_kwargs["mode"] = qwen_mode
        if qwen_model:
            reranker_kwargs["model"] = qwen_model
        
        max_candidates = getattr(arg, 'qwen_max_candidates', None)
        if max_candidates is not None:
            reranker_kwargs["max_candidates"] = max_candidates
    
    reranker = RerankerCls(**reranker_kwargs)
    
    # Prepare training kwargs
    training_kwargs = {
        "val_data": val,  # For early stopping
        "num_epochs": arg.rerank_epochs,
        "batch_size": arg.rerank_batch_size,
        "lr": arg.rerank_lr,
        "patience": arg.rerank_patience,
    }
    
    # Add dataset_code and related params for VIP5Reranker
    if args.rerank_method == "vip5":
        training_kwargs["dataset_code"] = arg.dataset_code
        training_kwargs["min_rating"] = arg.min_rating
        training_kwargs["min_uc"] = arg.min_uc
        training_kwargs["min_sc"] = arg.min_sc
        training_kwargs["num_items"] = item_count
    
    # Add text/image features if available
    item_id2text = {}
    if "meta" in data:
        item_id2text = {}
        user_history_text = {}
        item_meta = {}
        
        for item_id, meta in data["meta"].items():
            text = meta.get("text") if meta else None
            if text:
                item_id2text[item_id] = text
            item_meta[item_id] = meta if meta else {}
        
        # Build user history texts
        for user_id, items in train.items():
            user_history_text[user_id] = [
                item_id2text.get(item_id, f"item_{item_id}")
                for item_id in items
                if item_id in item_id2text
            ]
        
        if item_id2text:
            training_kwargs["item_id2text"] = item_id2text
            training_kwargs["user_history"] = user_history_text
            # ✅ Also add to reranker_kwargs so reranker can use it during eval
            reranker_kwargs["item_id2text"] = item_id2text
            reranker_kwargs["user_history"] = user_history_text
        
        # Add item_meta for multimodal modes (caption, VIU)
        if args.rerank_method in ["qwen", "qwen3vl"]:
            # Get mode from kwargs or config
            qwen_mode = reranker_kwargs.get("mode", "text_only")
            if qwen_mode in ["caption", "VIU"]:
                training_kwargs["item_meta"] = item_meta
                # ✅ Also add to reranker_kwargs so reranker can use it during eval
                reranker_kwargs["item_meta"] = item_meta
            else:
                # ✅ For text_only mode, also add item_meta if available (for ground_truth mode)
                if item_meta:
                    reranker_kwargs["item_meta"] = item_meta
    
    # Prepare training data for Qwen reranker (text_only mode)
    if args.rerank_method in ["qwen", "qwen3vl"]:
        # Check if mode is text_only
        qwen_mode = reranker_kwargs.get("mode", "text_only")
        if qwen_mode == "text_only":
            if not item_id2text:
                print("Warning: item_id2text not available. Qwen reranker will be loaded without training.")
                print("  Note: Qwen reranker requires item text for training. Ensure dataset has text data.")
            else:
                print("\n[2.5/3] Preparing training data for Qwen LLM reranker...")
                import random
                from rerank.models.llm import build_prompt_from_candidates
                
                # Build training samples: for each user, create samples with history + candidates + target
                training_samples = []
                all_items = set(range(1, item_count + 1))
                
                for user_id, items in train.items():
                    if len(items) < 2:
                        continue  # Need at least 2 items for history and target
                    
                    # Randomly select a split point for history and target
                    if len(items) > 3:
                        end_pos = random.randint(1, len(items) - 1)
                    else:
                        end_pos = len(items) - 1
                    
                    history_items = items[:end_pos]
                    target_item = items[end_pos]
                    
                    # Get history texts
                    history_texts = [
                        item_id2text.get(item_id, f"item_{item_id}")
                        for item_id in history_items
                        if item_id in item_id2text
                    ]
                    history_texts = history_texts[-10:]  # Max 10 items in history
                    
                    if not history_texts:
                        continue  # Skip if no history texts available
                    
                    # Generate candidates: target + random negatives (similar to ground_truth mode)
                    user_items_set = set(items)
                    negative_candidates = [item for item in all_items if item not in user_items_set]
                    
                    # Get num_negatives from config (default: 19 for 20 total candidates)
                    try:
                        total_candidates = getattr(arg, 'rerank_eval_candidates', 20)
                        num_negatives = total_candidates - 1  # 1 for ground truth
                    except (ImportError, AttributeError):
                        num_negatives = 19  # Default fallback (1 GT + 19 negatives = 20 total)
                    
                    num_negatives = min(num_negatives, len(negative_candidates))
                    if num_negatives > 0:
                        negatives = random.sample(negative_candidates, num_negatives)
                    else:
                        negatives = []
                    
                    candidates = [target_item] + negatives
                    random.shuffle(candidates)  # Shuffle so target is not always first
                    
                    # Find target index in candidates (0-indexed for letter mapping)
                    target_idx = candidates.index(target_item)
                    
                    # Validate number of candidates
                    from rerank.models.llm import LETTERS
                    if len(candidates) > len(LETTERS):
                        raise ValueError(
                            f"Too many candidates ({len(candidates)}). Maximum supported: {len(LETTERS)} candidates "
                            f"(using letters A-Z, a-z). Consider reducing num_candidates."
                        )
                    
                    # Build prompt (now uses letters instead of numbers)
                    prompt = build_prompt_from_candidates(
                        history_texts,
                        candidates,
                        item_id2text,
                        max_candidates=None
                    )
                    
                    # ✅ Remove "You are a recommendation ranking assistant." from prompt if present
                    # (already in system message to avoid duplication)
                    system_msg = "You are a recommendation ranking assistant."
                    if prompt.strip().startswith(system_msg):
                        # Remove system message from prompt (already in system message)
                        prompt = prompt.strip()[len(system_msg):].strip()
                        # Remove leading newline if present
                        if prompt.startswith("\n"):
                            prompt = prompt[1:]
                    
                    # Use letter index (LlamaRec style) instead of number
                    target_letter = LETTERS[target_idx]  # Letter index (A, B, C, ...)
                    
                    # Format for Unsloth (messages format)
                    training_samples.append({
                        "messages": [
                            {
                                "role": "system",
                                "content": system_msg
                            },
                            {
                                "role": "user",
                                "content": prompt
                            },
                            {
                                "role": "assistant",
                                "content": target_letter  # Answer is the target candidate letter (A, B, C, ...)
                            }
                        ]
                    })
                
                print(f"  Prepared {len(training_samples)} training samples for Qwen LLM")
                if training_samples:
                    training_kwargs["train_data_for_llm"] = training_samples
                else:
                    print("  Warning: No training samples prepared. Qwen reranker will be loaded without training.")
    
    # Load retrieval model if needed
    if args.mode == "retrieval":
        print(f"\n[2.5/3] Loading pre-trained retrieval model ({args.retrieval_method})...")
        RetrieverCls = get_retriever_class(args.retrieval_method)
        retriever = RetrieverCls(top_k=args.retrieval_top_k)
        retriever.fit(train, item_count=item_count, val_data=val)
        print("  Retrieval model loaded and ready")
    
    # Check action: train or eval
    rerank_action = getattr(arg, 'rerank_action', 'train') or 'train'
    
    # ✅ Always call fit() - it will handle both train and eval modes
    # In eval mode, fit() will load model and data but skip trainer.train()
    if rerank_action == "eval":
        print(f"\n[3/3] Loading pretrained reranker (eval mode)...")
        print(f"  Model path: {reranker_kwargs.get('model', 'N/A')}")
        print("  Note: Will load model and data, but skip training (trainer.train() will be skipped)")
    else:
        print(f"\n[3/3] Training reranker...")
    
    # ✅ Always call fit() - it will skip trainer.train() internally if eval mode
    reranker.fit(train, **training_kwargs)
    
    if rerank_action == "eval":
        print("  ✅ Pretrained model loaded (training skipped)")
    else:
        print("  ✅ Rerank training completed!")
    
    # Evaluate
    print(f"\n[4/4] Evaluating reranker...")
    ks = [1, 5, 10, 20]
    
    # Try to load pre-generated candidates for faster evaluation
    precomputed_candidates = None
    try:
        from evaluation.utils import load_rerank_candidates
        precomputed_candidates = load_rerank_candidates(
            dataset_code=arg.dataset_code,
            min_rating=arg.min_rating,
            min_uc=arg.min_uc,
            min_sc=arg.min_sc,
        )
        if precomputed_candidates.get("val") or precomputed_candidates.get("test"):
            print("  Using pre-generated candidates for faster evaluation")
    except Exception as e:
        # If loading fails, continue without precomputed candidates
        pass
    
    def recommend_fn(user_id, ground_truth=None, split_name="val"):
        """Recommendation function for evaluation."""
        # Try to use precomputed candidates first (faster)
        if precomputed_candidates:
            split_candidates = precomputed_candidates.get(split_name, {})
            if user_id in split_candidates:
                candidates = split_candidates[user_id]
                if candidates:
                    scored = reranker.rerank(user_id, candidates)
                    return [item_id for item_id, _ in scored]
        
        # Fallback to original logic
        if args.mode == "ground_truth":
            # Ground truth mode: use gt + random negatives
            if ground_truth is None:
                return []
            
            import random
            # Get rerank_eval_candidates from config (default: 20)
            try:
                max_candidates = getattr(arg, 'rerank_eval_candidates', 20)
            except (ImportError, AttributeError):
                max_candidates = 20
            
            # Get all items
            all_items = set(range(1, item_count + 1))
            # Exclude user's history
            user_history = set(train.get(user_id, []))
            exclude_set = user_history - set(ground_truth)
            candidate_pool = all_items - exclude_set - set(ground_truth)
            
            # Sample candidates: ensure at least one ground truth, but limit total candidates
            # This makes evaluation more realistic (not all ground truth items are guaranteed)
            num_gt_in_candidates = min(1, len(ground_truth))  # Only ensure 1 GT item
            gt_in_candidates = random.sample(ground_truth, num_gt_in_candidates) if len(ground_truth) > 0 else []
            
            # Calculate how many negatives we can add
            num_negatives = max_candidates - len(gt_in_candidates)
            num_negatives = max(0, min(num_negatives, len(candidate_pool)))
            
            if num_negatives > 0:
                negatives = random.sample(list(candidate_pool), num_negatives)
                candidates = gt_in_candidates + negatives
            else:
                candidates = gt_in_candidates
            
            # Shuffle to avoid bias
            random.shuffle(candidates)
            
            if not candidates:
                return []
            
            scored = reranker.rerank(user_id, candidates)
            return [item_id for item_id, _ in scored]
        
        else:  # retrieval mode
            # Use retrieval to get candidates
            exclude_set = set(train.get(user_id, []))
            candidates = retriever.retrieve(user_id, exclude_items=exclude_set)
            
            if not candidates:
                return []
            
            scored = reranker.rerank(user_id, candidates)
            return [item_id for item_id, _ in scored]
    
    # Evaluate on val and test
    # Create wrapper functions to pass split_name
    def val_recommend_fn(user_id, ground_truth=None):
        return recommend_fn(user_id, ground_truth, split_name="val")
    
    def test_recommend_fn(user_id, ground_truth=None):
        return recommend_fn(user_id, ground_truth, split_name="test")
    
    val_metrics = evaluate_split(val_recommend_fn, val, k=args.metric_k, ks=ks, ground_truth_mode=(args.mode == "ground_truth"))
    test_metrics = evaluate_split(test_recommend_fn, test, k=args.metric_k, ks=ks, ground_truth_mode=(args.mode == "ground_truth"))
    
    # Print results
    print("=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Rerank Method: {args.rerank_method}")
    print("-" * 80)
    print(f"Val Metrics:")
    print(f"  {'Metric':<12} {'@1':>10} {'@5':>10} {'@10':>10} {'@20':>10}")
    for metric_name in ["recall", "ndcg", "hit"]:
        values = [val_metrics.get(f"{metric_name}@{k}", 0.0) for k in ks]
        print(f"  {metric_name.capitalize():<12} {values[0]:>10.4f} {values[1]:>10.4f} {values[2]:>10.4f} {values[3]:>10.4f}")
    
    print(f"\nTest Metrics:")
    print(f"  {'Metric':<12} {'@1':>10} {'@5':>10} {'@10':>10} {'@20':>10}")
    for metric_name in ["recall", "ndcg", "hit"]:
        values = [test_metrics.get(f"{metric_name}@{k}", 0.0) for k in ks]
        print(f"  {metric_name.capitalize():<12} {values[0]:>10.4f} {values[1]:>10.4f} {values[2]:>10.4f} {values[3]:>10.4f}")
    print("=" * 80)
    
    # Log completion
    print()
    print(f"[INFO] Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] Log saved to: {log_file.absolute()}")
    
    # Restore stdout and stderr
    sys.stdout = tee_stdout.terminal
    sys.stderr = tee_stderr.terminal
    tee_stdout.close()
    tee_stderr.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Ensure error is logged before exiting
        print(f"\n[ERROR] Training failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise

