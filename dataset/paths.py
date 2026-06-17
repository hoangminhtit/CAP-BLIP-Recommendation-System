"""Path utilities for dataset and experiment files.

This module provides helper functions to get standardized paths for:
- Preprocessed datasets
- Experiment results
- Retrieved candidates
- Model checkpoints
"""

from pathlib import Path
from typing import Optional

from config import EXPERIMENT_ROOT, RAW_DATASET_ROOT_FOLDER


def get_preprocessed_folder_path(
    dataset_code: str,
    min_rating: int,
    min_uc: int,
    min_sc: int
) -> Path:
    """Get path to preprocessed dataset folder.
    
    Args:
        dataset_code: Dataset code (beauty, games, ml-100k)
        min_rating: Minimum rating threshold
        min_uc: Minimum user count
        min_sc: Minimum item count
        
    Returns:
        Path to preprocessed folder
    """
    preprocessed_root = Path(RAW_DATASET_ROOT_FOLDER) / "preprocessed"
    folder_name = f"{dataset_code}_min_rating{min_rating}-min_uc{min_uc}-min_sc{min_sc}"
    return preprocessed_root / folder_name


def get_preprocessed_csv_path(
    dataset_code: str,
    min_rating: int,
    min_uc: int,
    min_sc: int
) -> Path:
    """Get path to preprocessed CSV file.
    
    Args:
        dataset_code: Dataset code
        min_rating: Minimum rating threshold
        min_uc: Minimum user count
        min_sc: Minimum item count
        
    Returns:
        Path to dataset_single_export.csv
    """
    folder = get_preprocessed_folder_path(dataset_code, min_rating, min_uc, min_sc)
    return folder / "dataset_single_export.csv"


def get_experiment_path(
    stage: str,
    method: str,
    dataset_code: str,
    seed: int
) -> Path:
    """Get path to experiment results folder.
    
    Args:
        stage: Stage name ("retrieval", "rerank", "pipeline")
        method: Method name (e.g., "lrurec", "qwen")
        dataset_code: Dataset code
        seed: Random seed
        
    Returns:
        Path to experiment folder
    """
    return Path(EXPERIMENT_ROOT) / stage / method / dataset_code / f"seed{seed}"


def get_retrieved_csv_path(
    method: str,
    dataset_code: str,
    seed: int
) -> Path:
    """Get path to retrieved candidates CSV file.
    
    Args:
        method: Retrieval method name
        dataset_code: Dataset code
        seed: Random seed
        
    Returns:
        Path to retrieved.csv
    """
    exp_path = get_experiment_path("retrieval", method, dataset_code, seed)
    return exp_path / "retrieved.csv"


def get_retrieved_metrics_path(
    method: str,
    dataset_code: str,
    seed: int
) -> Path:
    """Get path to retrieved metrics JSON file.
    
    Args:
        method: Retrieval method name
        dataset_code: Dataset code
        seed: Random seed
        
    Returns:
        Path to retrieved_metrics.json
    """
    exp_path = get_experiment_path("retrieval", method, dataset_code, seed)
    return exp_path / "retrieved_metrics.json"


def get_clip_embeddings_path(
    dataset_code: str,
    min_rating: int,
    min_uc: int,
    min_sc: int
) -> Path:
    """Get path to CLIP embeddings file.
    
    Args:
        dataset_code: Dataset code
        min_rating: Minimum rating threshold
        min_uc: Minimum user count
        min_sc: Minimum item count
        
    Returns:
        Path to clip_embeddings.pt
    """
    folder = get_preprocessed_folder_path(dataset_code, min_rating, min_uc, min_sc)
    return folder / "clip_embeddings.pt"


def get_rerank_candidates_path(
    dataset_code: str,
    min_rating: int,
    min_uc: int,
    min_sc: int
) -> Path:
    """Get path to rerank candidates CSV file.
    
    This file contains pre-generated candidate lists for val/test splits
    to ensure consistent evaluation across all rerankers.
    
    Args:
        dataset_code: Dataset code
        min_rating: Minimum rating threshold
        min_uc: Minimum user count
        min_sc: Minimum item count
        
    Returns:
        Path to rerank_candidates.csv
    """
    folder = get_preprocessed_folder_path(dataset_code, min_rating, min_uc, min_sc)
    return folder / "rerank_candidates.csv"

