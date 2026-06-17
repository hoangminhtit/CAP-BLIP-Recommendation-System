"""
Script để test chức năng download images
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
args = parser.parse_args()

print("=" * 80)
print("TESTING DOWNLOADED IMAGES")
print("=" * 80)
print(f"Dataset: {args.dataset_code}")
print()

# Construct path to preprocessed data
preprocessed_root = Path('data/preprocessed')
folder_name = f'{args.dataset_code}_min_rating{args.min_rating}-min_uc{args.min_uc}-min_sc{args.min_sc}'
images_folder = preprocessed_root.joinpath(folder_name).joinpath('images')

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
    
    # Analyze downloaded images
    print("-" * 80)
    print("IMAGE DOWNLOAD ANALYSIS")
    print("-" * 80)
    
    items_with_local_image = 0
    items_with_url_only = 0
    items_without_image = 0
    
    for item_id, meta_info in dataset['meta'].items():
        if isinstance(meta_info, dict):
            has_local = 'image_path' in meta_info and meta_info['image_path'] is not None
            has_url = meta_info.get('image') is not None and len(meta_info.get('image', '')) > 0
            
            if has_local:
                items_with_local_image += 1
            elif has_url:
                items_with_url_only += 1
            else:
                items_without_image += 1
        else:
            items_without_image += 1
    
    print(f"Items with DOWNLOADED images: {items_with_local_image}")
    print(f"Items with URL only (not downloaded): {items_with_url_only}")
    print(f"Items without image: {items_without_image}")
    print()
    
    # Check images folder
    if images_folder.exists():
        image_files = list(images_folder.glob('*.*'))
        print(f"Images folder: {images_folder}")
        print(f"Total image files on disk: {len(image_files)}")
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in image_files)
        total_size_mb = total_size / (1024 * 1024)
        print(f"Total size: {total_size_mb:.2f} MB")
    else:
        print(f"Images folder not found: {images_folder}")
    
    print()
    
    # Show sample metadata with local images
    print("-" * 80)
    print("SAMPLE METADATA WITH DOWNLOADED IMAGES (first 3)")
    print("-" * 80)
    count = 0
    for item_id, meta_info in dataset['meta'].items():
        if isinstance(meta_info, dict) and 'image_path' in meta_info:
            print(f"\nItem {item_id}:")
            print(f"  Text: {meta_info.get('text', 'N/A')[:80]}...")
            print(f"  URL: {meta_info.get('image', 'N/A')[:60]}...")
            print(f"  Local: {meta_info.get('image_path', 'N/A')}")
            
            # Verify file exists
            if Path(meta_info['image_path']).exists():
                file_size = Path(meta_info['image_path']).stat().st_size
                print(f"  Size: {file_size / 1024:.1f} KB ✓")
            else:
                print(f"  File not found! ✗")
            
            count += 1
            if count >= 3:
                break
    
    if count == 0:
        print("No items with downloaded images found.")
    
    print("\n" + "=" * 80)
    
else:
    print(f"Dataset not found at: {dataset_path}")
    print("Please run data_prepare.py first with --use_image flag:")
    print(f"  python data_prepare.py --use_image")
    print("Or:")
    print(f"  python data_prepare.py --use_text --use_image")
