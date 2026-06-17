import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
from config import arg
from datasets import dataset_factory

# Load via dataset factory (will prefer CSV if present)
dataset = dataset_factory(arg).load_dataset()
    
    print("=" * 80)
    print("DATASET STRUCTURE")
    print("=" * 80)
    
    # Hiển thị các keys
    print(f"\nKeys trong dataset: {list(dataset.keys())}")
    
    # Train data
    print("\n" + "-" * 80)
    print("1. TRAIN DATA (dict)")
    print("-" * 80)
    print(f"   - Số lượng users: {len(dataset['train'])}")
    print(f"   - Ví dụ user 1: {dataset['train'].get(1, 'N/A')}")
    print(f"   - Ví dụ user 2: {dataset['train'].get(2, 'N/A')}")
    print(f"   - Format: {{user_id: [item_id_1, item_id_2, ...]}}")
    
    # Val data
    print("\n" + "-" * 80)
    print("2. VALIDATION DATA (dict)")
    print("-" * 80)
    print(f"   - Số lượng users: {len(dataset['val'])}")
    print(f"   - Ví dụ user 1: {dataset['val'].get(1, 'N/A')}")
    print(f"   - Ví dụ user 2: {dataset['val'].get(2, 'N/A')}")
    print(f"   - Format: {{user_id: [item_id]}}")
    
    # Test data
    print("\n" + "-" * 80)
    print("3. TEST DATA (dict)")
    print("-" * 80)
    print(f"   - Số lượng users: {len(dataset['test'])}")
    print(f"   - Ví dụ user 1: {dataset['test'].get(1, 'N/A')}")
    print(f"   - Ví dụ user 2: {dataset['test'].get(2, 'N/A')}")
    print(f"   - Format: {{user_id: [item_id]}}")
    
    # Meta data
    print("\n" + "-" * 80)
    print("4. META DATA (dict)")
    print("-" * 80)
    print(f"   - Số lượng items: {len(dataset['meta'])}")
    sample_items = list(dataset['meta'].items())[:3]
    for item_id, title in sample_items:
        print(f"   - Item {item_id}: {title}")
    print(f"   - Format: {{item_id: 'title'}}")
    
    # User mapping
    print("\n" + "-" * 80)
    print("5. USER MAPPING (umap)")
    print("-" * 80)
    print(f"   - Số lượng users: {len(dataset['umap'])}")
    sample_users = list(dataset['umap'].items())[:3]
    for original_id, new_id in sample_users:
        print(f"   - Original: {original_id} -> New: {new_id}")
    print(f"   - Format: {{original_user_id: new_user_id}}")
    
    # Item mapping
    print("\n" + "-" * 80)
    print("6. ITEM MAPPING (smap)")
    print("-" * 80)
    print(f"   - Số lượng items: {len(dataset['smap'])}")
    sample_items = list(dataset['smap'].items())[:3]
    for original_id, new_id in sample_items:
        print(f"   - Original: {original_id} -> New: {new_id}")
    print(f"   - Format: {{original_item_id: new_item_id}}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total users: {len(dataset['train'])}")
    print(f"Total items: {len(dataset['smap'])}")
    print(f"Train interactions per user: varies (all except last 2)")
    print(f"Val interactions per user: 1 (second to last)")
    print(f"Test interactions per user: 1 (last)")
    print("=" * 80)
    
else:
    print(f"Dataset not found at: {dataset_path}")
    print("Please run data_prepare.py first to download and preprocess the dataset.")
