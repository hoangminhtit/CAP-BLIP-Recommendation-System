"""Training script for rerank models (Stage 2).

This script trains rerank models (e.g., Qwen) on candidates from Stage 1 retrieval.
"""

import random
import string
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rerank.models.llm import LLMModel

LETTERS = list(string.ascii_uppercase[:20])  # A-T


def build_training_samples(user2items, all_items, max_history=5):
    """Build training samples for LLM reranking."""
    samples = []

    for user, interactions in user2items.items():
        if len(interactions) < 3:
            continue

        # sort nếu có timestamp
        interactions = interactions[-(max_history + 1):]

        pos_item = interactions[-1]
        history_items = interactions[:-1]

        pos_id = pos_item["item_new_id"]
        pos_text = pos_item["item_text"]

        # sample negatives
        neg_items = random.sample(
            [i for i in all_items if i["item_new_id"] != pos_id],
            19
        )

        candidates = [(pos_id, pos_text)] + [
            (i["item_new_id"], i["item_text"]) for i in neg_items
        ]
        random.shuffle(candidates)

        label_idx = [i for i, (cid, _) in enumerate(candidates) if cid == pos_id][0]
        label = LETTERS[label_idx]

        samples.append({
            "user_id": user,
            "history": [i["item_text"] for i in history_items],
            "candidates": [c[1] for c in candidates],
            "label": label
        })

    return samples


def build_prompt(sample):
    """Build prompt for training sample."""
    history_text = "\n".join(
        [f"- {h}" for h in sample["history"]]
    )

    cand_text = "\n".join(
        [f"{LETTERS[i]}. {c}" for i, c in enumerate(sample["candidates"])]
    )

    prompt = f"""
You are a recommendation ranking assistant.

Choose exactly ONE item the user is most likely to interact with next.

User history:
{history_text}

Candidate items:
{cand_text}

Answer with only one letter (A-T).
""".strip()

    return prompt


def to_unsloth_format(samples):
    """Convert samples to Unsloth training format."""
    data = []

    for s in samples:
        data.append({
            "messages": [
                {
                    "role": "system",
                    "content": "You are a recommendation ranking assistant."
                },
                {
                    "role": "user",
                    "content": build_prompt(s)
                },
                {
                    "role": "assistant",
                    "content": s["label"]   # ONLY label token
                }
            ]
        })

    return data


def main():
    from config import arg
    from dataset.paths import get_preprocessed_csv_path, get_retrieved_csv_path
    from evaluation.utils import load_dataset_from_csv
    
    # Load dataset using config
    dataset_path = get_preprocessed_csv_path(
        arg.dataset_code,
        arg.min_rating,
        arg.min_uc,
        arg.min_sc
    )
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        print("Please run data_prepare.py first!")
        return
    
    df = pd.read_csv(dataset_path)

    df_train = df[df["split"] == "train"]
    user2items = defaultdict(list)

    for _, row in df_train.iterrows():
        user2items[row["user_id"]].append(row)
    
    all_items = (
        df_train[["item_new_id", "item_text"]]
        .drop_duplicates()
        .to_dict("records")
    )

    # Build training samples
    train_samples = build_training_samples(user2items, all_items)
    train_data = to_unsloth_format(train_samples)
    
    # Train model
    model = LLMModel(train_data=train_data, model_name="unsloth/Qwen3-0.6B-Base-bnb-4bit")
    model.load_model()
    model.train()

    # Evaluate - Load retrieved candidates using config
    # TODO: Make retrieval method configurable via args
    retrieval_method = "lrurec"  # Default, should be from config/args
    retrieved_path = get_retrieved_csv_path(retrieval_method, arg.dataset_code, arg.seed)
    if not retrieved_path.exists():
        print(f"Retrieved candidates not found at {retrieved_path}")
        print("Please run scripts/train_retrieval.py first!")
        return
    
    df_can = pd.read_csv(retrieved_path)
    df_val = df_can[df_can["split"] == "val"]
    df_test = df_can[df_can["split"] == "test"]

    # item_id -> text
    item_df = pd.read_csv(dataset_path)[["item_new_id", "item_text"]].drop_duplicates()
    item_id2text = dict(zip(item_df.item_new_id, item_df.item_text))

    df_inter = pd.read_csv(dataset_path)
    user2history = defaultdict(list)

    for _, row in df_inter.iterrows():
        user2history[row["user_id"]].append(row["item_text"])

    val_metrics = model.evaluate(df_val, user2history, item_id2text)
    test_metrics = model.evaluate(df_test, user2history, item_id2text)

    print("=" * 80)
    print("Rerank Evaluation Results")
    print("=" * 80)
    print("VAL:", val_metrics)
    print("TEST:", test_metrics)
    print("=" * 80)


if __name__ == "__main__":
    main()

