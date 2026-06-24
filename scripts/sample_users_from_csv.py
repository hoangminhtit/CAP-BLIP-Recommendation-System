# python scripts/sample_users_from_csv.py \
#   --dataset_code beauty 
#   --min_rating 4 
#   --min_uc 6 --min_sc 6 \
#   --sample_users 200 
#   --sample_seed 123

"""Randomly sample a subset of users from dataset_single_export.csv.

Use this after data_prepare.py to shrink data size for quick experiments.
Also filters out unused items (items not interacted with by sampled users).
Additionally filters items with fewer than --min_sc interactions and users with fewer than --min_uc interactions.
"""

import argparse
import random
import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from dataset.paths import get_preprocessed_csv_path, get_preprocessed_folder_path


def _parse_args():
    parser = argparse.ArgumentParser(description="Sample users from preprocessed CSV, filter unused items, and apply min_sc/min_uc filtering")
    parser.add_argument("--data_path", type=str, default="data", help="Path to data folder")
    parser.add_argument("--dataset_code", type=str, default="beauty", help="Dataset code")
    parser.add_argument("--min_rating", type=int, default=4, help="Minimum rating used in preprocessing")
    parser.add_argument("--min_uc", type=int, default=6, help="Minimum user count used in preprocessing")
    parser.add_argument("--min_sc", type=int, default=6, help="Minimum item count used in preprocessing")
    parser.add_argument("--sample_users", type=int, default=100, help="Number of users to keep; if <=0 or >= total, keeps all")
    parser.add_argument("--sample_seed", type=int, default=42, help="Random seed for reproducible sampling")
    parser.add_argument("--output_csv", type=str, default=None, help="Optional output file path; defaults to preprocessed folder with sampled suffix")
    return parser.parse_args()


def main():
    args = _parse_args()

    csv_path = get_preprocessed_csv_path(args.dataset_code, args.min_rating, args.min_uc, args.min_sc, args.data_path)
    if not csv_path.exists():
        print(f"[sample_users] CSV not found at {csv_path}. Run data_prepare.py first.")
        return

    df = pd.read_csv(csv_path)
    if "user_id" not in df.columns:
        print("[sample_users] Column user_id not found in CSV; cannot sample.")
        return

    users = sorted(df["user_id"].unique().tolist())
    total_users = len(users)
    n = args.sample_users
    if n is None or n <= 0 or n >= total_users:
        chosen = users
    else:
        rng = random.Random(args.sample_seed)
        rng.shuffle(users)
        chosen = users[:n]

    sampled_df = df[df["user_id"].isin(chosen)].copy()

    # Filter out unused items (items not interacted with by sampled users)
    used_items = set(sampled_df["item_new_id"].unique())
    original_item_count = df["item_new_id"].nunique()
    sampled_item_count = len(used_items)
    
    # Keep only rows where item_new_id is in used_items
    # This removes item metadata rows for unused items
    sampled_df = sampled_df[sampled_df["item_new_id"].isin(used_items)].copy()
    
    print(f"[sample_users] Filtered unused items: {original_item_count} -> {sampled_item_count} items")

    # Additional filtering: remove items with fewer than min_sc interactions
    item_counts = sampled_df.groupby("item_new_id").size()
    valid_items = item_counts[item_counts >= args.min_sc].index
    if len(valid_items) < len(item_counts):
        sampled_df = sampled_df[sampled_df["item_new_id"].isin(valid_items)].copy()
        print(f"[sample_users] Filtered items with < {args.min_sc} interactions: {len(item_counts)} -> {len(valid_items)} items")

    # Additional filtering: remove users with fewer than min_uc interactions
    user_counts = sampled_df.groupby("user_id").size()
    valid_users = user_counts[user_counts >= args.min_uc].index
    if len(valid_users) < len(user_counts):
        sampled_df = sampled_df[sampled_df["user_id"].isin(valid_users)].copy()
        print(f"[sample_users] Filtered users with < {args.min_uc} interactions: {len(user_counts)} -> {len(valid_users)} users")

    # Ensure train/val/test share the same user set (val/test are last interactions per user)
    if "split" in sampled_df.columns:
        train_users = set(sampled_df[sampled_df["split"] == "train"]["user_id"].unique())
        val_users = set(sampled_df[sampled_df["split"] == "val"]["user_id"].unique())
        test_users = set(sampled_df[sampled_df["split"] == "test"]["user_id"].unique())

        keep_users = train_users & val_users & test_users
        dropped_users = (train_users | val_users | test_users) - keep_users

        if dropped_users:
            before_rows = len(sampled_df)
            sampled_df = sampled_df[sampled_df["user_id"].isin(keep_users)].copy()
            after_rows = len(sampled_df)
            print(
                f"[sample_users] Dropped users missing in any split (train/val/test must match): "
                f"{len(dropped_users)} users, rows {before_rows} -> {after_rows}"
            )
        else:
            print("[sample_users] Train/val/test user sets already aligned")

    final_user_count = sampled_df["user_id"].nunique()
    final_item_count = sampled_df["item_new_id"].nunique()

    if args.output_csv:
        out_path = Path(args.output_csv)
    else:
        folder = get_preprocessed_folder_path(args.dataset_code, args.min_rating, args.min_uc, args.min_sc, args.data_path)
        out_path = folder / f"dataset_single_export.csv"

    sampled_df.to_csv(out_path, index=False)

    split_counts = sampled_df.groupby("split")["user_id"].nunique().to_dict() if "split" in sampled_df.columns else {}
    split_counts_items = sampled_df.groupby("split")["item_new_id"].nunique().to_dict() if "split" in sampled_df.columns else {}

    print("[sample_users] Sampling complete")
    print(f"  Input users: {total_users}, items: {original_item_count}")
    print(f"  After sampling: {len(chosen)} users, {sampled_item_count} items")
    print(f"  Final (after filtering): {final_user_count} users, {final_item_count} items")
    print(f"  Output: {out_path}")
    if split_counts:
        print(f"  Users per split: {split_counts}")
    if split_counts_items:
        print(f"  Items per split: {split_counts_items}")


if __name__ == "__main__":
    main()
