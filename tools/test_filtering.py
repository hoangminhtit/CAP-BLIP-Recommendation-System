"""
Script để test chức năng lọc items dựa trên text và image
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_code', type=str, default='beauty')
parser.add_argument('--min_rating', type=int, default=3)
parser.add_argument('--min_uc', type=int, default=5)
parser.add_argument('--min_sc', type=int, default=5)
parser.add_argument('--use_image', action='store_true', default=False)
parser.add_argument('--use_text', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

print("=" * 80)
print("TESTING FILTERING WITH SETTINGS:")
print("=" * 80)
print(f"Dataset: {args.dataset_code}")
print(f"use_text: {args.use_text}")
print(f"use_image: {args.use_image}")
print()

# Construct path to preprocessed data
preprocessed_root = Path('data/preprocessed')
folder_name = f'{args.dataset_code}_min_rating{args.min_rating}-min_uc{args.min_uc}-min_sc{args.min_sc}'

# Load via dataset factory (will prefer CSV if present)
from datasets import dataset_factory
dataset = dataset_factory(args).load_dataset()
    
    print("-" * 80)
    print("DATASET STATISTICS")
    print("-" * 80)
    print(f"Total users: {len(dataset['train'])}")
    print(f"Total items: {len(dataset['smap'])}")
    print(f"Total metadata entries: {len(dataset['meta'])}")
    print()
    
    # Analyze metadata
    print("-" * 80)
    print("METADATA ANALYSIS")
    print("-" * 80)
    
    items_with_text = 0
    items_with_image = 0
    items_with_both = 0
    items_with_none = 0
    
    for item_id, meta_info in dataset['meta'].items():
        has_text = False
        has_image = False
        
        if isinstance(meta_info, dict):
            has_text = meta_info.get('text') is not None and len(meta_info.get('text', '')) > 0
            has_image = meta_info.get('image') is not None and len(meta_info.get('image', '')) > 0
        else:
            # Old format (just string)
            has_text = meta_info is not None and len(str(meta_info)) > 0
        
        if has_text and has_image:
            items_with_both += 1
        elif has_text:
            items_with_text += 1
        elif has_image:
            items_with_image += 1
        else:
            items_with_none += 1
    
    print(f"Items with BOTH text and image: {items_with_both}")
    print(f"Items with ONLY text: {items_with_text}")
    print(f"Items with ONLY image: {items_with_image}")
    print(f"Items with NEITHER: {items_with_none}")
    print()
    
    # Show sample metadata
    print("-" * 80)
    print("SAMPLE METADATA (first 3 items)")
    print("-" * 80)
    for i, (item_id, meta_info) in enumerate(list(dataset['meta'].items())[:3]):
        print(f"\nItem {item_id}:")
        if isinstance(meta_info, dict):
            print(f"  Text: {meta_info.get('text', 'N/A')[:100]}...")
            print(f"  Image: {meta_info.get('image', 'N/A')[:80]}...")
        else:
            print(f"  Old format: {str(meta_info)[:100]}...")
    
    print("\n" + "=" * 80)
    
else:
    print(f"Dataset not found at: {dataset_path}")
    print("Please run data_prepare.py first to download and preprocess the dataset.")
    print()
    print("To test with filtering enabled, run:")
    print(f"  python data_prepare.py --use_text --use_image")
    print("Or:")
    print(f"  python data_prepare.py --use_text")
