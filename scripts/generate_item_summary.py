"""Standalone item summary generation task.

Usage (run after data_prepare.py):
    python scripts/generate_item_summary.py \
        --dataset_code beauty --min_rating 4 --min_uc 6 --min_sc 6 \
        --generate_item_summary --summary_source auto --summary_language en

Requirements:
- dataset_single_export.csv must exist (produced by data_prepare.py)
- unsloth installed for Qwen3 text models.
"""

import argparse
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import parser  # reuse global parser to stay consistent with existing flags

# Import side-effects parse_args in config.py; ensure we parse CLI here

def main():
    args = parser.parse_args()

    # Late import to keep startup light
    from dataset.item_summary import generate_item_summaries_from_csv

    generate_item_summaries_from_csv(args)


if __name__ == "__main__":
    main()
