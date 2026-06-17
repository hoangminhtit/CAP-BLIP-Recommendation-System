"""Offline evaluation for retrieval / rerank / full pipeline.

Modes:
- retrieval : Stage 1 only (retrieval-only)
- full      : Stage 1 + Stage 2
- rerank_only : Stage 2 only, using Stage 1 to generate candidates + inject ground truth

Metrics: Recall@K, NDCG@K on val/test split.
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from evaluation.metrics import recall_at_k, ndcg_at_k, hit_at_k
from pipelines.base import PipelineConfig, RetrievalConfig, RerankConfig, TwoStagePipeline
from retrieval import get_retriever_class
from rerank import get_reranker_class


def load_dataset(
    dataset_code: str,
    min_rating: int,
    min_uc: int,
    min_sc: int,
) -> Dict:
    """Load dataset.pkl dựa trên tham số filter.

    Thư mục: data/preprocessed/{code}_min_rating{min_rating}-min_uc{min_uc}-min_sc{min_sc}/dataset.pkl
    """
    # Prefer CSV loader via dataset_factory
    from datasets import dataset_factory
    args = type("X", (), {
        "dataset_code": dataset_code,
        "min_rating": min_rating,
        "min_uc": min_uc,
        "min_sc": min_sc,
    })
    dataset = dataset_factory(args).load_dataset()
    return dataset


def evaluate_users(
    users: List[int],
    recs_by_user: Dict[int, List[int]],
    ground_truth: Dict[int, List[int]],
    ks: List[int] = [5, 10, 20],
) -> Dict[str, float]:
    """Tính trung bình Recall@K, NDCG@K, và Hit@K trên một tập user cho nhiều K values.
    
    Args:
        users: List of user IDs
        recs_by_user: Dict {user_id: [recommended_item_ids]}
        ground_truth: Dict {user_id: [ground_truth_item_ids]}
        ks: List of K values to evaluate (default: [5, 10, 20])
        
    Returns:
        Dict with keys: recall@K, ndcg@K, hit@K for each K in ks
    """
    metrics_by_k = {k: {"recalls": [], "ndcgs": [], "hits": []} for k in ks}
    
    for u in users:
        gt_items = ground_truth.get(u, [])
        recs = recs_by_user.get(u, [])
        if not gt_items:
            continue
        
        for k in ks:
            r = recall_at_k(recs, gt_items, k)
            n = ndcg_at_k(recs, gt_items, k)
            h = hit_at_k(recs, gt_items, k)
            metrics_by_k[k]["recalls"].append(r)
            metrics_by_k[k]["ndcgs"].append(n)
            metrics_by_k[k]["hits"].append(h)

    result = {}
    for k in ks:
        if metrics_by_k[k]["recalls"]:
            result[f"recall@{k}"] = float(sum(metrics_by_k[k]["recalls"]) / len(metrics_by_k[k]["recalls"]))
            result[f"ndcg@{k}"] = float(sum(metrics_by_k[k]["ndcgs"]) / len(metrics_by_k[k]["ndcgs"]))
            result[f"hit@{k}"] = float(sum(metrics_by_k[k]["hits"]) / len(metrics_by_k[k]["hits"]))
        else:
            result[f"recall@{k}"] = 0.0
            result[f"ndcg@{k}"] = 0.0
            result[f"hit@{k}"] = 0.0
    
    return result


def run_retrieval_only(
    train: Dict[int, List[int]],
    eval_split: Dict[int, List[int]],
    retrieval_method: str,
    retrieval_top_k: int,
    ks: List[int] = [5, 10, 20],
) -> Dict[str, float]:
    """Stage 1 only: dùng TwoStagePipeline với rerank_method = none."""
    retrieval_cfg = RetrievalConfig(method=retrieval_method, top_k=retrieval_top_k)
    rerank_cfg = RerankConfig(method="none", top_k=max(ks) if ks else 20)
    cfg = PipelineConfig(retrieval=retrieval_cfg, rerank=rerank_cfg)
    pipeline = TwoStagePipeline(cfg)
    pipeline.fit(train)

    users = sorted(eval_split.keys())
    recs_by_user: Dict[int, List[int]] = {}
    for u in users:
        recs_by_user[u] = pipeline.recommend(u)

    return evaluate_users(users, recs_by_user, eval_split, ks)


def run_full_pipeline(
    train: Dict[int, List[int]],
    eval_split: Dict[int, List[int]],
    retrieval_method: str,
    retrieval_top_k: int,
    rerank_method: str,
    rerank_top_k: int,
    ks: List[int] = [5, 10, 20],
    rerank_mode: str = "retrieval",
    qwen3vl_mode: Optional[str] = None,
) -> Dict[str, float]:
    """Stage 1 + Stage 2: dùng TwoStagePipeline bình thường.
    
    Args:
        rerank_mode: "retrieval" (use Stage 1 candidates) or "ground_truth" (gt + negatives)
        qwen3vl_mode: Qwen3-VL mode (only used if rerank_method=qwen3vl)
        ks: List of K values to evaluate
    """
    retrieval_cfg = RetrievalConfig(method=retrieval_method, top_k=retrieval_top_k)
    rerank_cfg = RerankConfig(
        method=rerank_method,
        top_k=rerank_top_k,
        mode=rerank_mode,
        num_negatives=19,
        qwen3vl_mode=qwen3vl_mode
    )
    cfg = PipelineConfig(retrieval=retrieval_cfg, rerank=rerank_cfg)
    pipeline = TwoStagePipeline(cfg)
    pipeline.fit(train)

    users = sorted(eval_split.keys())
    recs_by_user: Dict[int, List[int]] = {}
    
    ground_truth_mode = (rerank_mode == "ground_truth")
    for u in users:
        if ground_truth_mode:
            gt_items = eval_split.get(u, [])
            recs_by_user[u] = pipeline.recommend(u, ground_truth=gt_items)
        else:
            recs_by_user[u] = pipeline.recommend(u)

    return evaluate_users(users, recs_by_user, eval_split, ks)


def run_rerank_only(
    train: Dict[int, List[int]],
    eval_split: Dict[int, List[int]],
    retrieval_method: str,
    retrieval_top_k: int,
    rerank_method: str,
    rerank_top_k: int,
    ks: List[int] = [5, 10, 20],
    qwen3vl_mode: Optional[str] = None,
) -> Dict[str, float]:
    """Stage 2 only: dùng Stage 1 để tạo candidates + inject ground truth.

    - Stage 1 chỉ để tạo pool candidates (top-K Stage 1).
    - Sau đó union với ground_truth để đảm bảo reranker luôn nhìn thấy item đúng.
    - Metrics đo trên ranking cuối của Stage 2.
    """
    RetrieverCls = get_retriever_class(retrieval_method)
    RerankerCls = get_reranker_class(rerank_method)

    retriever = RetrieverCls(top_k=retrieval_top_k)
    
    # Create reranker with mode if Qwen3-VL
    reranker_kwargs = {"top_k": rerank_top_k}
    if rerank_method.lower() == "qwen3vl" and qwen3vl_mode is not None:
        reranker_kwargs["mode"] = qwen3vl_mode
    reranker = RerankerCls(**reranker_kwargs)

    retriever.fit(train)
    reranker.fit(train)

    users = sorted(eval_split.keys())
    recs_by_user: Dict[int, List[int]] = {}

    for u in users:
        gt_items = eval_split.get(u, [])
        base_cands = retriever.retrieve(u)
        # Union base candidates + ground truth, giữ thứ tự, loại trùng
        seen = set()
        merged: List[int] = []
        for item_id in base_cands + gt_items:
            if item_id not in seen:
                seen.add(item_id)
                merged.append(item_id)

        if not merged:
            recs_by_user[u] = []
            continue

        scored = reranker.rerank(u, merged)
        ranked_items = [item_id for item_id, _ in scored]
        recs_by_user[u] = ranked_items

    return evaluate_users(users, recs_by_user, eval_split, ks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline evaluation for two-stage recommendation")

    # Dataset / filtering (cần khớp với data_prepare.py)
    parser.add_argument("--dataset_code", type=str, default="beauty")
    parser.add_argument("--min_rating", type=int, default=3)
    parser.add_argument("--min_uc", type=int, default=5)
    parser.add_argument("--min_sc", type=int, default=5)

    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Dùng val hay test để eval")
    parser.add_argument("--K", type=int, default=10, help="Cutoff cho Recall@K, NDCG@K (legacy, will evaluate @5, @10, @20)")

    parser.add_argument("--mode", type=str, default="full", choices=["retrieval", "full", "rerank_only"], help="Chế độ eval")

    # Retrieval config
    parser.add_argument("--retrieval_method", type=str, default="lrurec")
    parser.add_argument("--retrieval_top_k", type=int, default=200)

    # Rerank config
    parser.add_argument("--rerank_method", type=str, default="qwen")
    parser.add_argument("--rerank_top_k", type=int, default=50)
    parser.add_argument("--rerank_mode", type=str, default="retrieval", choices=["retrieval", "ground_truth"],
                       help="Rerank mode: 'retrieval' (use Stage 1 candidates) or 'ground_truth' (gt + negatives)")
    parser.add_argument("--qwen3vl_mode", type=str, default="raw_image",
                       choices=["raw_image", "caption", "semantic_summary", "semantic_summary_small"],
                       help="Qwen3-VL mode (only used if rerank_method=qwen3vl)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = load_dataset(
        dataset_code=args.dataset_code,
        min_rating=args.min_rating,
        min_uc=args.min_uc,
        min_sc=args.min_sc,
    )

    train: Dict[int, List[int]] = dataset["train"]
    val: Dict[int, List[int]] = dataset["val"]
    test: Dict[int, List[int]] = dataset["test"]

    eval_split = val if args.split == "val" else test

    # Evaluate at K=1, 5, 10, 20
    ks = [1, 5, 10, 20]
    
    if args.mode == "retrieval":
        metrics = run_retrieval_only(
            train=train,
            eval_split=eval_split,
            retrieval_method=args.retrieval_method,
            retrieval_top_k=args.retrieval_top_k,
            ks=ks,
        )
        mode_name = "Stage 1 only (retrieval)"

    elif args.mode == "full":
        metrics = run_full_pipeline(
            train=train,
            eval_split=eval_split,
            retrieval_method=args.retrieval_method,
            retrieval_top_k=args.retrieval_top_k,
            rerank_method=args.rerank_method,
            rerank_top_k=args.rerank_top_k,
            ks=ks,
            rerank_mode=args.rerank_mode,
            qwen3vl_mode=args.qwen3vl_mode if args.rerank_method.lower() == "qwen3vl" else None,
        )
        mode_name = f"Full pipeline (Stage 1 + Stage 2, rerank_mode={args.rerank_mode})"
        if args.rerank_method.lower() == "qwen3vl":
            mode_name += f", qwen3vl_mode={args.qwen3vl_mode}"

    else:  # rerank_only
        metrics = run_rerank_only(
            train=train,
            eval_split=eval_split,
            retrieval_method=args.retrieval_method,
            retrieval_top_k=args.retrieval_top_k,
            rerank_method=args.rerank_method,
            rerank_top_k=args.rerank_top_k,
            ks=ks,
            qwen3vl_mode=args.qwen3vl_mode if args.rerank_method.lower() == "qwen3vl" else None,
        )
        mode_name = "Stage 2 only (rerank-only với pool từ Stage 1 + ground truth)"
        if args.rerank_method.lower() == "qwen3vl":
            mode_name += f", qwen3vl_mode={args.qwen3vl_mode}"

    print("=" * 80)
    print(f"Mode      : {mode_name}")
    print(f"Dataset   : {args.dataset_code}")
    print(f"Split     : {args.split}")
    print(f"Retriever : {args.retrieval_method} (top_k={args.retrieval_top_k})")
    print(f"Reranker  : {args.rerank_method} (top_k={args.rerank_top_k}, mode={args.rerank_mode})")
    print("-" * 80)
    print(f"{'Metric':<12} {'@1':>10} {'@5':>10} {'@10':>10} {'@20':>10}")
    print("-" * 80)
    for metric_name in ["recall", "ndcg", "hit"]:
        values = [metrics.get(f"{metric_name}@{k}", 0.0) for k in ks]
        print(f"{metric_name.capitalize():<12} {values[0]:>10.4f} {values[1]:>10.4f} {values[2]:>10.4f} {values[3]:>10.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
