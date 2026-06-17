"""Data I/O utilities for loading and saving datasets.

This module provides standardized functions for loading and saving
dataset files in various formats (CSV, pickle, etc.).
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .paths import get_preprocessed_csv_path
from evaluation.utils import load_dataset_from_csv as _load_dataset_from_csv


def load_dataset_from_csv(
    dataset_code: str,
    min_rating: int,
    min_uc: int,
    min_sc: int
) -> Dict:
    """Load dataset from CSV export.
    
    This is a wrapper around evaluation.utils.load_dataset_from_csv
    that uses the path utilities.
    
    Args:
        dataset_code: Dataset code (beauty, games, ml-100k)
        min_rating: Minimum rating threshold
        min_uc: Minimum user count
        min_sc: Minimum item count
        
    Returns:
        Dict with keys: train, val, test, meta, smap, item_count
    """
    return _load_dataset_from_csv(dataset_code, min_rating, min_uc, min_sc)


def load_csv_dataframe(
    dataset_code: str,
    min_rating: int,
    min_uc: int,
    min_sc: int
) -> pd.DataFrame:
    """Load dataset CSV as pandas DataFrame.
    
    Args:
        dataset_code: Dataset code
        min_rating: Minimum rating threshold
        min_uc: Minimum user count
        min_sc: Minimum item count
        
    Returns:
        DataFrame with dataset data
    """
    csv_path = get_preprocessed_csv_path(dataset_code, min_rating, min_uc, min_sc)
    
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found at {csv_path}. Run data_prepare.py first."
        )
    
    return pd.read_csv(csv_path)


def validate_dataset_format(data: Dict) -> bool:
    """Validate dataset format after loading.
    
    Args:
        data: Dataset dictionary
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    required_keys = {"train", "val", "test", "meta", "smap"}
    missing_keys = required_keys - set(data.keys())
    
    if missing_keys:
        raise ValueError(f"Missing required keys in dataset: {missing_keys}")
    
    # Validate train/val/test are dicts
    for key in ["train", "val", "test"]:
        if not isinstance(data[key], dict):
            raise ValueError(f"Dataset['{key}'] must be a dict, got {type(data[key])}")
    
    # Validate meta is dict
    if not isinstance(data["meta"], dict):
        raise ValueError(f"Dataset['meta'] must be a dict, got {type(data['meta'])}")
    
    return True

