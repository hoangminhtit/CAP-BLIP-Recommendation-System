# python scripts/sample_candidates_from_csv.py \
#   --dataset_code beauty \
#   --min_rating 4 \
#   --min_uc 6 --min_sc 6 \
#   --num_candidates 50 \
#   --sample_seed 42

"""Sample candidate lists for reranking from preprocessed CSV.

This script regenerates rerank_candidates.csv after user/item sampling,
ensuring candidates are based on the filtered dataset.
"""

import argparse
import random
import json
import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from dataset.paths import get_preprocessed_csv_path, get_preprocessed_folder_path


def _parse_args():
    parser = argparse.ArgumentParser(description="Sample candidate lists for reranking from preprocessed CSV")
    parser.add_argument("--data_path", type=str, default="data", help="Path to data folder")
    parser.add_argument("--dataset_code", type=str, default="beauty", help="Dataset code")
    parser.add_argument("--min_rating", type=int, default=4, help="Minimum rating used in preprocessing")
    parser.add_argument("--min_uc", type=int, default=6, help="Minimum user count used in preprocessing")
    parser.add_argument("--min_sc", type=int, default=6, help="Minimum item count used in preprocessing")
    parser.add_argument("--num_candidates", type=int, default=50, help="Number of candidates per user")
    parser.add_argument("--sample_seed", type=int, default=42, help="Random seed for reproducible sampling")
    return parser.parse_args()


def main():
    args = _parse_args()

    csv_path = get_preprocessed_csv_path(args.dataset_code, args.min_rating, args.min_uc, args.min_sc, args.data_path)
    if not csv_path.exists():
        print(f"[sample_candidates] CSV not found at {csv_path}. Run data_prepare.py or sample_users_from_csv.py first.")
        return

    df = pd.read_csv(csv_path)
    required_cols = ["user_id", "item_new_id", "split"]
    if not all(col in df.columns for col in required_cols):
        print(f"[sample_candidates] Required columns {required_cols} not found in CSV.")
        return

    # Group by split to get train/val/test interactions
    train_dict = {}
    val_dict = {}
    test_dict = {}

    for _, row in df.iterrows():
        user_id = int(row["user_id"])
        item_id = int(row["item_new_id"])
        split = row["split"]

        if split == "train":
            if user_id not in train_dict:
                train_dict[user_id] = []
            train_dict[user_id].append(item_id)
        elif split == "val":
            if user_id not in val_dict:
                val_dict[user_id] = []
            val_dict[user_id].append(item_id)
        elif split == "test":
            if user_id not in test_dict:
                test_dict[user_id] = []
            test_dict[user_id].append(item_id)

    # Get all items from dataset
    all_items = set()
    for items in train_dict.values():
        all_items.update(items)
    for items in val_dict.values():
        all_items.update(items)
    for items in test_dict.values():
        all_items.update(items)
    all_items = sorted(list(all_items))

    if not all_items:
        print("[sample_candidates] WARNING: No items found. Skipping candidate list generation.")
        return

    # Set random seed for reproducibility
    random.seed(args.sample_seed)

    # Generate candidates for val split
    val_candidates = {}
    for user_id, gt_items in val_dict.items():
        # Get user's history (train items)
        user_history = set(train_dict.get(user_id, []))
        # Exclude history items from candidate pool
        candidate_pool = [item for item in all_items if item not in user_history]

        if len(candidate_pool) < args.num_candidates:
            print(f"[sample_candidates] WARNING: User {user_id} has only {len(candidate_pool)} candidates (requested {args.num_candidates})")
            candidates = candidate_pool
        else:
            # Sample random candidates
            candidates = random.sample(candidate_pool, args.num_candidates)

        # Ensure at least one ground truth is in candidates (if available)
        if gt_items and not any(item in candidates for item in gt_items):
            if len(candidates) < args.num_candidates:
                candidates.append(gt_items[0])
            else:
                candidates[0] = gt_items[0]

        # Shuffle to avoid bias
        random.shuffle(candidates)
        val_candidates[user_id] = candidates

    # Generate candidates for test split
    test_candidates = {}
    for user_id, gt_items in test_dict.items():
        # Get user's history (train + val items)
        user_history = set(train_dict.get(user_id, []))
        user_history.update(val_dict.get(user_id, []))
        # Exclude history items from candidate pool
        candidate_pool = [item for item in all_items if item not in user_history]

        if len(candidate_pool) < args.num_candidates:
            print(f"[sample_candidates] WARNING: User {user_id} has only {len(candidate_pool)} candidates (requested {args.num_candidates})")
            candidates = candidate_pool
        else:
            # Sample random candidates
            candidates = random.sample(candidate_pool, args.num_candidates)

        # Ensure at least one ground truth is in candidates (if available)
        if gt_items and not any(item in candidates for item in gt_items):
            if len(candidates) < args.num_candidates:
                candidates.append(gt_items[0])
            else:
                candidates[0] = gt_items[0]

        # Shuffle to avoid bias
        random.shuffle(candidates)
        test_candidates[user_id] = candidates

    # Save candidate lists to CSV
    candidate_rows = []
    for user_id, candidates in val_candidates.items():
        candidate_rows.append({
            "user_id": user_id,
            "split": "val",
            "candidates": json.dumps(candidates),
        })
    for user_id, candidates in test_candidates.items():
        candidate_rows.append({
            "user_id": user_id,
            "split": "test",
            "candidates": json.dumps(candidates),
        })

    if candidate_rows:
        candidates_df = pd.DataFrame(candidate_rows)
        folder = get_preprocessed_folder_path(args.dataset_code, args.min_rating, args.min_uc, args.min_sc, args.data_path)
        candidates_csv = folder / "rerank_candidates.csv"
        candidates_df.to_csv(candidates_csv, index=False)
        print(f"[sample_candidates] Saved rerank candidate lists to: {candidates_csv}")
        print(f"  Val users with candidates: {len(val_candidates)}")
        print(f"  Test users with candidates: {len(test_candidates)}")
        print(f"  Total candidates generated: {len(candidate_rows)}")


if __name__ == "__main__":
    main()