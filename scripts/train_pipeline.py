"""Training script for end-to-end two-stage pipeline.

This script trains both retrieval (Stage 1) and reranking (Stage 2) models
in sequence, then evaluates the full pipeline.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
from typing import Dict, List, Optional

# Note: config import is moved to main() to avoid argument parsing conflicts
from dataset import dataset_factory
from evaluation.metrics import recall_at_k, ndcg_at_k
from evaluation.utils import evaluate_split, load_dataset_from_csv
from pipelines.base import PipelineConfig, RetrievalConfig, RerankConfig, TwoStagePipeline
from pytorch_lightning import seed_everything


def evaluate_pipeline(
    pipeline: TwoStagePipeline,
    split: Dict[int, List[int]],
    k: int = 10,
    ks: Optional[List[int]] = None,
    ground_truth_mode: bool = False
) -> Dict[str, float]:
    """Evaluate pipeline on a split. Wrapper for evaluation.utils.evaluate_split.
    
    Args:
        pipeline: TwoStagePipeline instance
        split: Dict {user_id: [item_ids]} - ground truth
        k: Cutoff for metrics (used if ks is None)
        ks: List of K values to evaluate (e.g., [5, 10, 20]). If None, uses [k]
        ground_truth_mode: If True, use ground_truth rerank mode
    """
    return evaluate_split(pipeline.recommend, split, k=k, ks=ks, ground_truth_mode=ground_truth_mode)


def main():
    # Import config (it now includes script-specific arguments as optional)
    from config import arg, EXPERIMENT_ROOT
    
    # Get script-specific arguments from arg (with defaults)
    class Args:
        retrieval_method = getattr(arg, 'retrieval_method', 'lrurec') or 'lrurec'
        retrieval_top_k = getattr(arg, 'retrieval_top_k', 200) or 200
        rerank_method = getattr(arg, 'rerank_method', 'qwen') or 'qwen'
        rerank_top_k = getattr(arg, 'rerank_top_k', 50) or 50
        metric_k = getattr(arg, 'metric_k', 10) or 10
        rerank_mode = getattr(arg, 'rerank_mode', 'retrieval') or 'retrieval'
        qwen_mode = getattr(arg, 'qwen_mode', None)
        qwen_model = getattr(arg, 'qwen_model', None)
        qwen3vl_mode = getattr(arg, 'qwen3vl_mode', None)  # Legacy
    
    args = Args()
    
    seed_everything(arg.seed)
    
    print("=" * 80)
    print("Two-Stage Pipeline Training")
    print("=" * 80)
    print(f"Retrieval: {args.retrieval_method} (top_k={args.retrieval_top_k})")
    print(f"Rerank: {args.rerank_method} (top_k={args.rerank_top_k}, mode={args.rerank_mode})")
    
    print("=" * 80)
    
    # Load dataset
    print("\n[1/4] Loading dataset...")
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
    
    # Create pipeline config
    retrieval_cfg = RetrievalConfig(
        method=args.retrieval_method,
        top_k=args.retrieval_top_k
    )
    # Get num_negatives from config (default: 19 for 20 total candidates)
    try:
        num_negatives = getattr(arg, 'rerank_eval_candidates', 20) - 1  # 1 for ground truth
    except (ImportError, AttributeError):
        num_negatives = 19  # Default fallback (1 GT + 19 negatives = 20 total)
    
    rerank_cfg = RerankConfig(
        method=args.rerank_method,
        top_k=args.rerank_top_k,
        mode=args.rerank_mode,
        num_negatives=num_negatives,  # From config (rerank_eval_candidates - 1)
        # Use unified reranker options
        qwen_mode=getattr(arg, 'qwen_mode', None) if args.rerank_method.lower() in ["qwen", "qwen3vl"] else None,
        qwen_model=getattr(arg, 'qwen_model', None) if args.rerank_method.lower() in ["qwen", "qwen3vl"] else None,
        # Legacy: backward compatibility
        qwen3vl_mode=args.qwen3vl_mode if args.rerank_method.lower() == "qwen3vl" else None
    )
    pipeline_cfg = PipelineConfig(
        retrieval=retrieval_cfg,
        rerank=rerank_cfg
    )
    
    # Create pipeline
    pipeline = TwoStagePipeline(pipeline_cfg)
    
    # Prepare reranker kwargs (standardize hyperparameters for fair comparison)
    reranker_kwargs = {
        "vocab_size": item_count + 1,  # For BERT4Rec
        "val_data": val,  # For early stopping
        # Standardize hyperparameters from config (for trainable rerankers)
        "num_epochs": arg.rerank_epochs,
        "batch_size": arg.rerank_batch_size,
        "lr": arg.rerank_lr,
        "patience": arg.rerank_patience,
    }
    
    # Add text/image features if available (for Qwen/Qwen3-VL)
    if "meta" in data:
        item_id2text = {}
        user_history_text = {}
        item_meta = {}  # For Qwen3-VL
        
        for item_id, meta in data["meta"].items():
            text = meta.get("text") if meta else None
            if text:
                item_id2text[item_id] = text
            
            # Store full meta for Qwen3-VL
            item_meta[item_id] = meta if meta else {}
        
        # Build user history texts
        for user_id, items in train.items():
            user_history_text[user_id] = [
                item_id2text.get(item_id, f"item_{item_id}")
                for item_id in items
                if item_id in item_id2text
            ]
        
        if item_id2text:
            reranker_kwargs["item_id2text"] = item_id2text
            reranker_kwargs["user_history"] = user_history_text
        
        # Add item_meta for multimodal modes (caption, VIU)
        if args.rerank_method.lower() in ["qwen", "qwen3vl"]:
            # Get mode from config
            qwen_mode_val = qwen_mode or (args.qwen3vl_mode if args.rerank_method.lower() == "qwen3vl" else "text_only")
            if qwen_mode_val in ["caption", "VIU"]:
                reranker_kwargs["item_meta"] = item_meta
    
    # Train Stage 1
    print(f"\n[2/4] Training Stage 1 ({args.retrieval_method})...")
    retriever_kwargs = {
        "item_count": item_count,
        "val_data": val,
    }
    pipeline.fit(
        train,
        retriever_kwargs={"item_count": item_count, "val_data": val},
        reranker_kwargs=reranker_kwargs  # Pass standardized reranker kwargs
    )
    
    # Evaluate Stage 1
    print(f"\n[3/4] Evaluating Stage 1...")
    ks = [5, 10, 20]
    val_metrics_stage1 = evaluate_pipeline(pipeline, val, k=args.metric_k, ks=ks)
    test_metrics_stage1 = evaluate_pipeline(pipeline, test, k=args.metric_k, ks=ks)
    
    print(f"  Val Metrics:")
    print(f"    {'Metric':<12} {'@5':>10} {'@10':>10} {'@20':>10}")
    for metric_name in ["recall", "ndcg", "hit"]:
        values = [val_metrics_stage1.get(f"{metric_name}@{k}", 0.0) for k in ks]
        print(f"    {metric_name.capitalize():<12} {values[0]:>10.4f} {values[1]:>10.4f} {values[2]:>10.4f}")
    
    print(f"  Test Metrics:")
    print(f"    {'Metric':<12} {'@5':>10} {'@10':>10} {'@20':>10}")
    for metric_name in ["recall", "ndcg", "hit"]:
        values = [test_metrics_stage1.get(f"{metric_name}@{k}", 0.0) for k in ks]
        print(f"    {metric_name.capitalize():<12} {values[0]:>10.4f} {values[1]:>10.4f} {values[2]:>10.4f}")
    
    # Note: Stage 2 training is handled in pipeline.fit() via reranker_kwargs
    if args.rerank_method.lower() not in ("none", ""):
        print(f"\n[4/4] Stage 2 ({args.rerank_method}) training completed in pipeline.fit()")
    
    # Evaluate full pipeline
    print(f"\n[5/5] Evaluating Full Pipeline...")
    ground_truth_mode = (args.rerank_mode == "ground_truth")
    val_metrics_full = evaluate_pipeline(pipeline, val, k=args.metric_k, ks=ks, ground_truth_mode=ground_truth_mode)
    test_metrics_full = evaluate_pipeline(pipeline, test, k=args.metric_k, ks=ks, ground_truth_mode=ground_truth_mode)
    
    print("=" * 80)
    print("Final Results")
    print("=" * 80)
    print(f"Stage 1 Only - Val:")
    print(f"  {'Metric':<12} {'@5':>10} {'@10':>10} {'@20':>10}")
    for metric_name in ["recall", "ndcg", "hit"]:
        values = [val_metrics_stage1.get(f"{metric_name}@{k}", 0.0) for k in ks]
        print(f"  {metric_name.capitalize():<12} {values[0]:>10.4f} {values[1]:>10.4f} {values[2]:>10.4f}")
    
    print(f"\nStage 1 Only - Test:")
    print(f"  {'Metric':<12} {'@5':>10} {'@10':>10} {'@20':>10}")
    for metric_name in ["recall", "ndcg", "hit"]:
        values = [test_metrics_stage1.get(f"{metric_name}@{k}", 0.0) for k in ks]
        print(f"  {metric_name.capitalize():<12} {values[0]:>10.4f} {values[1]:>10.4f} {values[2]:>10.4f}")
    
    print(f"\nFull Pipeline (Stage 1 + Stage 2) - Val:")
    print(f"  {'Metric':<12} {'@5':>10} {'@10':>10} {'@20':>10}")
    for metric_name in ["recall", "ndcg", "hit"]:
        values = [val_metrics_full.get(f"{metric_name}@{k}", 0.0) for k in ks]
        print(f"  {metric_name.capitalize():<12} {values[0]:>10.4f} {values[1]:>10.4f} {values[2]:>10.4f}")
    
    print(f"\nFull Pipeline (Stage 1 + Stage 2) - Test:")
    print(f"  {'Metric':<12} {'@5':>10} {'@10':>10} {'@20':>10}")
    for metric_name in ["recall", "ndcg", "hit"]:
        values = [test_metrics_full.get(f"{metric_name}@{k}", 0.0) for k in ks]
        print(f"  {metric_name.capitalize():<12} {values[0]:>10.4f} {values[1]:>10.4f} {values[2]:>10.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

