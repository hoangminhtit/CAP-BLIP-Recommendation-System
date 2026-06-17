from dataset import *
from dataset.clip_embeddings import maybe_extract_clip_embeddings
from dataset.blip2_captions import maybe_generate_blip2_captions
from dataset.qwen3vl_viu import maybe_generate_viu
from config import *
from pytorch_lightning import seed_everything
import torch
import json
import pandas as pd
import random
from pathlib import Path


def main(args):
    seed_everything(args.seed)
    dataset = dataset_factory(args)
    data = dataset.load_dataset()
    maybe_extract_clip_embeddings(dataset, data, args)
    
    # Generate BLIP2/BLIP captions if enabled
    # Captions will be saved directly to CSV (no need for separate .pt file)
    captions = maybe_generate_blip2_captions(dataset, data, args)
    
    # Generate Qwen3 VL VIU if enabled
    # VIU will be saved directly to CSV
    viu_data = maybe_generate_viu(dataset, data, args)

    # Export a single CSV with columns:
    # {Item_id, user_id, item_new_id, item_text, item_image_embedding,
    #  item_text_embedding, item_image_path, item_caption, item_viu, split}
    preproc_folder = Path(dataset._get_preprocessed_folder_path())
    clip_path = preproc_folder.joinpath("clip_embeddings.pt")

    clip_payload = None
    if clip_path.exists():
        clip_payload = torch.load(clip_path)

    image_embs = None
    text_embs = None
    if clip_payload:
        image_embs = clip_payload.get("image_embs")
        text_embs = clip_payload.get("text_embs")

    # Build inverse smap to recover original item ids
    smap = data.get("smap", {})
    inv_smap = {new: orig for orig, new in smap.items()}

    meta = data.get("meta", {})

    rows = []
    def add_rows_for_split(split_name, split_dict):
        for user, items in split_dict.items():
            for item in items:
                orig_item = inv_smap.get(item, None)
                info = meta.get(item, {})
                text = info.get("text") if info else None
                image_path = info.get("image_path") or info.get("image") if info else None
                
                # Get caption if available
                caption = None
                if captions is not None:
                    caption = captions.get(item)
                
                # Get VIU if available
                viu = None
                if viu_data is not None:
                    viu = viu_data.get(item)

                img_emb = None
                txt_emb = None
                if image_embs is not None and 0 <= item < image_embs.size(0):
                    emb = image_embs[item]
                    if emb is not None:
                        img_emb = emb.tolist()
                if text_embs is not None and 0 <= item < text_embs.size(0):
                    emb = text_embs[item]
                    if emb is not None:
                        txt_emb = emb.tolist()

                rows.append({
                    "Item_id": orig_item,
                    "user_id": int(user),
                    "item_new_id": int(item),
                    "item_text": text,
                    "item_image_embedding": json.dumps(img_emb) if img_emb is not None else "",
                    "item_text_embedding": json.dumps(txt_emb) if txt_emb is not None else "",
                    "item_image_path": image_path or "",
                    "item_caption": caption or "",
                    "item_viu": viu or "",
                    "split": split_name,
                })

    train_dict = data.get("train", {})
    val_dict = data.get("val", {})
    test_dict = data.get("test", {})
    
    add_rows_for_split("train", train_dict)
    add_rows_for_split("val", val_dict)
    add_rows_for_split("test", test_dict)
    
    # Warn if val or test sets are empty
    if len(val_dict) == 0:
        print(f"[data_prepare] WARNING: Validation set is empty!")
        print(f"  This will cause evaluation to fail. Check dataset filtering parameters.")
    if len(test_dict) == 0:
        print(f"[data_prepare] WARNING: Test set is empty!")
        print(f"  This will cause evaluation to fail. Check dataset filtering parameters.")
    
    if rows:
        df = pd.DataFrame(rows)
        out_csv = preproc_folder.joinpath("dataset_single_export.csv")
        df.to_csv(out_csv, index=False)
        print(f"Saved single CSV export to: {out_csv}")
        print(f"  Train users: {len(train_dict)}, Val users: {len(val_dict)}, Test users: {len(test_dict)}")
    
    # ✅ Generate candidate lists for val and test (for rerank evaluation)
    # Use rerank_eval_candidates (unified argument for both evaluation and data preparation)
    num_candidates = getattr(args, 'rerank_eval_candidates', 20)
    print(f"\n[data_prepare] Generating candidate lists for val/test (num_candidates={num_candidates})...")
    
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
        print("[data_prepare] WARNING: No items found. Skipping candidate list generation.")
    else:
        # Generate candidates for val split
        val_candidates = {}
        for user_id, gt_items in val_dict.items():
            # Get user's history (train items)
            user_history = set(train_dict.get(user_id, []))
            # Exclude history items from candidate pool
            candidate_pool = [item for item in all_items if item not in user_history]
            
            if len(candidate_pool) < num_candidates:
                print(f"[data_prepare] WARNING: User {user_id} has only {len(candidate_pool)} candidates (requested {num_candidates})")
                candidates = candidate_pool
            else:
                # Sample random candidates
                candidates = random.sample(candidate_pool, num_candidates)
            
            # Ensure at least one ground truth is in candidates (if available)
            if gt_items and not any(item in candidates for item in gt_items):
                if len(candidates) < num_candidates:
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
            
            if len(candidate_pool) < num_candidates:
                print(f"[data_prepare] WARNING: User {user_id} has only {len(candidate_pool)} candidates (requested {num_candidates})")
                candidates = candidate_pool
            else:
                # Sample random candidates
                candidates = random.sample(candidate_pool, num_candidates)
            
            # Ensure at least one ground truth is in candidates (if available)
            if gt_items and not any(item in candidates for item in gt_items):
                if len(candidates) < num_candidates:
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
            candidates_csv = preproc_folder.joinpath("rerank_candidates.csv")
            candidates_df.to_csv(candidates_csv, index=False)
            print(f"Saved rerank candidate lists to: {candidates_csv}")
            print(f"  Val users with candidates: {len(val_candidates)}")
            print(f"  Test users with candidates: {len(test_candidates)}")

if __name__ == "__main__":
    main(arg)