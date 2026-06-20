"""Prepare this repo's datasets for the chat-reranking experiments.

The original chat-reranking code expects:
- item helper pickles such as itemid_to_name.pkl
- fold_0/train_data.csv and fold_0/test_data.csv
- recs/baselines/<baseline>.tsv with columns userid, itemid, score

This script builds those files from data/preprocessed/*/dataset_single_export.csv
and, when available, experiments/retrieval/*/retrieved.csv.
"""

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


def preprocessed_dir(data_root: Path, dataset_code: str, min_rating: int, min_uc: int, min_sc: int) -> Path:
    return data_root / "preprocessed" / f"{dataset_code}_min_rating{min_rating}-min_uc{min_uc}-min_sc{min_sc}"


def clean_item_name(value, item_id: int, max_chars: int) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return f"item_{item_id}"
    text = " ".join(str(value).split())
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].strip()
    return text or f"item_{item_id}"


def build_item_maps(df: pd.DataFrame, max_chars: int) -> Tuple[Dict[int, str], Dict[str, int]]:
    item_rows = df.drop_duplicates(subset=["item_new_id"]).copy()
    itemid_to_name = {}
    used_names = {}

    for _, row in item_rows.iterrows():
        item_id = int(row["item_new_id"])
        base_name = clean_item_name(row.get("item_text"), item_id, max_chars)
        name = base_name
        if name in used_names and used_names[name] != item_id:
            name = f"{base_name} [item {item_id}]"
        used_names[name] = item_id
        itemid_to_name[item_id] = name

    itemname_to_id = {name: item_id for item_id, name in itemid_to_name.items()}
    return itemid_to_name, itemname_to_id


def export_split_files(df: pd.DataFrame, out_dir: Path, sample_users: int = 0, sample_seed: int = 42) -> None:
    fold_dir = out_dir / "fold_0"
    fold_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split][["user_id", "item_new_id"]].copy()
        split_df["rating"] = 1
        split_df.to_csv(fold_dir / f"{split}_data.csv", sep="\t", header=False, index=False)

    users = sorted(df["user_id"].dropna().astype(int).unique())
    items = sorted(df["item_new_id"].dropna().astype(int).unique())
    pd.Series(users).to_csv(fold_dir / "users.csv", header=False, index=False)
    pd.Series(items).to_csv(fold_dir / "items.csv", header=False, index=False)

    test_users = sorted(df[df["split"] == "test"]["user_id"].dropna().astype(int).unique())
    if sample_users and sample_users > 0 and sample_users < len(test_users):
        rng = random.Random(sample_seed)
        test_users = sorted(rng.sample(test_users, sample_users))
    pd.Series(test_users).to_csv(fold_dir / "sample_test_users.csv", sep="\t", header=False, index=False)

    sample_test = df[(df["split"] == "test") & (df["user_id"].isin(test_users))][["user_id", "item_new_id"]].copy()
    sample_test["rating"] = 1
    sample_test.to_csv(fold_dir / "sample_test_data.csv", sep="\t", header=False, index=False)


def split_user_order(df: pd.DataFrame, split: str) -> List[int]:
    split_df = df[df["split"] == split].reset_index(drop=False).rename(columns={"index": "row_order"})
    users = (
        split_df.sort_values("row_order")
        .groupby("user_id", sort=False)["item_new_id"]
        .size()
        .index
        .astype(int)
        .tolist()
    )
    return users


def load_retrieved_candidates(path: Path, source_df: pd.DataFrame, split: str, top_m: int) -> pd.DataFrame:
    retrieved = pd.read_csv(path)
    retrieved = retrieved[retrieved["split"] == split].copy()
    user_order = split_user_order(source_df, split)

    rows = []
    for _, row in retrieved.iterrows():
        user_index = int(row["user_index"])
        if user_index >= len(user_order):
            continue
        user_id = user_order[user_index]
        candidate_ids = json.loads(row["candidate_ids"])[:top_m]
        scores = json.loads(row.get("candidate_scores", "[]")) if "candidate_scores" in row else []
        for rank, item_id in enumerate(candidate_ids):
            score = scores[rank] if rank < len(scores) else float(top_m - rank)
            rows.append({"userid": user_id, "itemid": int(item_id), "rating": float(score)})

    return pd.DataFrame(rows)


def build_popularity_candidates(df: pd.DataFrame, split: str, top_m: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    train = df[df["split"] == "train"]
    popularity = train["item_new_id"].astype(int).value_counts()
    ranked_items = popularity.index.astype(int).tolist()
    all_items = sorted(df["item_new_id"].dropna().astype(int).unique())

    rows = []
    for user_id in split_user_order(df, split):
        history = set(train[train["user_id"] == user_id]["item_new_id"].astype(int).tolist())
        candidates = [item for item in ranked_items if item not in history]
        if len(candidates) < top_m:
            tail = [item for item in all_items if item not in history and item not in candidates]
            rng.shuffle(tail)
            candidates.extend(tail)
        for rank, item_id in enumerate(candidates[:top_m]):
            rows.append({"userid": user_id, "itemid": int(item_id), "rating": float(top_m - rank)})

    return pd.DataFrame(rows)


def write_helper_pickles(out_dir: Path, itemid_to_name: Dict[int, str], itemname_to_id: Dict[str, int]) -> None:
    helper_pairs = {
        "itemid_to_name.pkl": itemid_to_name,
        "itemname_to_id.pkl": itemname_to_id,
        "itemid_to_namegenres.pkl": itemid_to_name,
        "itemnamegenres_to_id.pkl": itemname_to_id,
        "itemid_to_nameplot.pkl": itemid_to_name,
        "itemnameplot_to_id.pkl": itemname_to_id,
    }
    for filename, obj in helper_pairs.items():
        with (out_dir / filename).open("wb") as fp:
            pickle.dump(obj, fp)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_code", default="beauty")
    parser.add_argument("--min_rating", type=int, default=3)
    parser.add_argument("--min_uc", type=int, default=20)
    parser.add_argument("--min_sc", type=int, default=20)
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--retrieval_method", default="lrurec")
    parser.add_argument("--retrieved_csv", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--top_m", type=int, default=20)
    parser.add_argument("--baseline_name", default=None)
    parser.add_argument("--max_name_chars", type=int, default=160)
    parser.add_argument("--fallback_popularity", action="store_true")
    parser.add_argument(
        "--sample_users",
        type=int,
        default=0,
        help="Number of test users to write to fold_0/sample_test_users.csv. 0 means all test users.",
    )
    parser.add_argument("--sample_seed", type=int, default=42)
    args = parser.parse_args()

    source_dir = preprocessed_dir(args.data_root, args.dataset_code, args.min_rating, args.min_uc, args.min_sc)
    source_csv = source_dir / "dataset_single_export.csv"
    if not source_csv.exists():
        raise FileNotFoundError(f"Missing {source_csv}. Run data_prepare.py first.")

    out_dir = args.output_dir or (Path("experiments") / "chat-reranking" / "prepared" / args.dataset_code)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source_csv)
    itemid_to_name, itemname_to_id = build_item_maps(df, args.max_name_chars)
    write_helper_pickles(out_dir, itemid_to_name, itemname_to_id)
    export_split_files(df, out_dir, args.sample_users, args.sample_seed)

    retrieved_csv = args.retrieved_csv
    if retrieved_csv is None:
        retrieved_csv = Path("experiments") / "retrieval" / args.retrieval_method / args.dataset_code / f"seed{args.seed}" / "retrieved.csv"

    baseline_name = args.baseline_name or f"{args.retrieval_method}-{args.split}-top{args.top_m}.tsv"
    if retrieved_csv.exists():
        baseline = load_retrieved_candidates(retrieved_csv, df, args.split, args.top_m)
        source = str(retrieved_csv)
    elif args.fallback_popularity:
        baseline = build_popularity_candidates(df, args.split, args.top_m, args.seed)
        source = "popularity fallback"
    else:
        raise FileNotFoundError(
            f"Missing {retrieved_csv}. Run scripts/train_retrieval.py first or pass --fallback_popularity."
        )

    rec_dir = out_dir / "recs" / "baselines"
    rec_dir.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(rec_dir / baseline_name, sep="\t", header=False, index=False)

    (out_dir / "recs" / "reranked").mkdir(parents=True, exist_ok=True)
    (out_dir / "recs" / "reranked" / "to_delete").mkdir(parents=True, exist_ok=True)

    manifest = {
        "source_csv": str(source_csv),
        "candidate_source": source,
        "dataset_code": args.dataset_code,
        "split": args.split,
        "top_m": args.top_m,
        "baseline_name": baseline_name,
        "num_items": len(itemid_to_name),
        "num_users": int(df["user_id"].nunique()),
        "sample_users": int(args.sample_users),
        "sample_seed": int(args.sample_seed),
        "num_candidate_rows": int(len(baseline)),
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Prepared chat-reranking dataset at: {out_dir}")
    print(f"Baseline file: {rec_dir / baseline_name}")
    print(f"Candidate source: {source}")


if __name__ == "__main__":
    main()
