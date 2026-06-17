"""Unified cleanup utility for preprocessed data and experiment results.

This script consolidates clean_preprocessed.py and cleanup_experiments.py
into a single tool with subcommands.
"""

import argparse
import shutil
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import arg, EXPERIMENT_ROOT


def clean_preprocessed():
    """Clean preprocessed data folder."""
    preprocessed_root = Path('data/preprocessed')
    folder_name = f'{arg.dataset_code}_min_rating{arg.min_rating}-min_uc{arg.min_uc}-min_sc{arg.min_sc}'
    dataset_folder = preprocessed_root.joinpath(folder_name)
    
    print("=" * 80)
    print("CLEANING OLD PREPROCESSED DATA")
    print("=" * 80)
    print(f"Dataset: {arg.dataset_code}")
    print(f"Folder: {dataset_folder}")
    print()
    
    if dataset_folder.exists():
        print(f"Removing: {dataset_folder}")
        shutil.rmtree(dataset_folder)
        print("✓ Successfully removed old preprocessed data")
    else:
        print("No preprocessed data found. Nothing to clean.")
    
    print()
    print("=" * 80)
    print("Now run: python data_prepare.py")
    print("Or with filtering: python data_prepare.py --use_text --use_image")
    print("=" * 80)


def clean_experiments(method: str, dataset: str = None, seed: int = None, all_datasets: bool = False):
    """Clean experiment results.
    
    Args:
        method: Retrieval method name (e.g., lrurec)
        dataset: Dataset code (e.g., beauty, games, ml_100k)
        seed: Seed used in experiments (e.g., 42)
        all_datasets: Remove all datasets under this method
    """
    def remove_dir(p: Path) -> None:
        if p.exists() and p.is_dir():
            print(f"[cleanup] Removing folder: {p}")
            shutil.rmtree(p)
        else:
            print(f"[cleanup] Skip (not found): {p}")
    
    base = Path(EXPERIMENT_ROOT) / "retrieval" / method
    
    if all_datasets:
        # Remove all datasets for this method.
        if not base.exists():
            print(f"[cleanup] No experiments found for method '{method}'.")
            return
        for ds_dir in base.iterdir():
            if ds_dir.is_dir():
                if seed is not None:
                    seed_dir = ds_dir / f"seed{seed}"
                    remove_dir(seed_dir)
                else:
                    remove_dir(ds_dir)
        return
    
    # If not all-datasets, we require dataset.
    if not dataset:
        raise SystemExit("--dataset is required unless you specify --all-datasets")
    
    ds_base = base / dataset
    if seed is not None:
        target = ds_base / f"seed{seed}"
        remove_dir(target)
    else:
        remove_dir(ds_base)


def main():
    parser = argparse.ArgumentParser(
        description="Clean preprocessed data or experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean preprocessed data
  python tools/clean.py preprocessed
  
  # Clean specific experiment
  python tools/clean.py experiments --method lrurec --dataset beauty --seed 42
  
  # Clean all experiments for a method
  python tools/clean.py experiments --method lrurec --all-datasets
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Preprocessed subcommand
    preprocessed_parser = subparsers.add_parser(
        'preprocessed',
        help='Clean preprocessed data folder'
    )
    
    # Experiments subcommand
    experiments_parser = subparsers.add_parser(
        'experiments',
        help='Clean experiment results'
    )
    experiments_parser.add_argument(
        '--method',
        type=str,
        required=True,
        help='Retrieval method name (e.g., lrurec)'
    )
    experiments_parser.add_argument(
        '--dataset',
        type=str,
        help='Dataset code (e.g., beauty, games, ml_100k)'
    )
    experiments_parser.add_argument(
        '--seed',
        type=int,
        help='Seed used in experiments (e.g., 42)'
    )
    experiments_parser.add_argument(
        '--all-datasets',
        action='store_true',
        help='Remove all datasets under this method'
    )
    
    args = parser.parse_args()
    
    if args.command == 'preprocessed':
        clean_preprocessed()
    elif args.command == 'experiments':
        clean_experiments(
            method=args.method,
            dataset=args.dataset,
            seed=args.seed,
            all_datasets=args.all_datasets
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

