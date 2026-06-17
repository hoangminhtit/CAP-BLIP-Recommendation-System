"""Evaluate chat-reranking JSON outputs against a prepared fold."""

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


def parse_list(value) -> List[int]:
    if isinstance(value, list):
        return [int(v) for v in value]
    if pd.isna(value):
        return []
    text = str(value)
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = ast.literal_eval(text)
    return [int(v) for v in parsed]


def load_truth(path: Path) -> Dict[int, List[int]]:
    df = pd.read_csv(path, sep="\t", names=["userid", "itemid", "rating"])
    return df.groupby("userid")["itemid"].apply(lambda s: s.astype(int).tolist()).to_dict()


def recall_at_k(recs: List[int], gt: List[int], k: int) -> float:
    if not gt:
        return 0.0
    return len(set(recs[:k]) & set(gt)) / len(set(gt))


def hit_at_k(recs: List[int], gt: List[int], k: int) -> float:
    return 1.0 if set(recs[:k]) & set(gt) else 0.0


def ndcg_at_k(recs: List[int], gt: List[int], k: int) -> float:
    gt_set = set(gt)
    dcg = 0.0
    for idx, item in enumerate(recs[:k], start=1):
        if item in gt_set:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(gt_set), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def complete_ranking(row) -> List[int]:
    reranked = parse_list(row.get("reranked_recs", []))
    original = parse_list(row.get("recs", []))
    seen = set()
    output = []
    for item in reranked + original:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasetpath", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--ks", default="1,5,10,20")
    parser.add_argument("--save_metrics", type=Path, default=None)
    args = parser.parse_args()

    truth_path = args.datasetpath / "fold_0" / f"{args.split}_data.csv"
    truth = load_truth(truth_path)
    df = pd.read_json(args.output_json)
    ks = [int(k.strip()) for k in args.ks.split(",") if k.strip()]

    metrics = {"num_users": 0}
    values = {f"recall@{k}": [] for k in ks}
    values.update({f"ndcg@{k}": [] for k in ks})
    values.update({f"hit@{k}": [] for k in ks})

    for _, row in df.iterrows():
        user = int(row["userid"])
        gt = truth.get(user, [])
        if not gt:
            continue
        recs = complete_ranking(row)
        if not recs:
            continue
        metrics["num_users"] += 1
        for k in ks:
            values[f"recall@{k}"].append(recall_at_k(recs, gt, k))
            values[f"ndcg@{k}"].append(ndcg_at_k(recs, gt, k))
            values[f"hit@{k}"].append(hit_at_k(recs, gt, k))

    for name, vals in values.items():
        metrics[name] = float(sum(vals) / len(vals)) if vals else 0.0

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.save_metrics:
        args.save_metrics.parent.mkdir(parents=True, exist_ok=True)
        with args.save_metrics.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
