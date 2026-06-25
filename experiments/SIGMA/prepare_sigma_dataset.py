"""Convert this repo's preprocessed CSV into RecBole atomic files for SIGMA.

The SIGMA implementation under experiments/SIGMA is a RecBole sequential model.
RecBole expects an atomic interaction file:

    <data_path>/<dataset>/<dataset>.inter

with typed headers such as user_id:token and item_id:token. This script converts
data/preprocessed/*/dataset_single_export.csv into that format.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def preprocessed_csv_path(
    data_root: Path,
    dataset_code: str,
    min_rating: int,
    min_uc: int,
    min_sc: int,
) -> Path:
    folder = f"{dataset_code}_min_rating{min_rating}-min_uc{min_uc}-min_sc{min_sc}"
    return data_root / "preprocessed" / folder / "dataset_single_export.csv"


def candidate_preprocessed_csv_paths(
    data_root: Path,
    dataset_code: str,
    min_rating: int,
    min_uc: int,
    min_sc: int,
) -> list[Path]:
    """Return likely CSV locations for repo-root and Kaggle cwd workflows."""
    candidates = [
        preprocessed_csv_path(data_root, dataset_code, min_rating, min_uc, min_sc),
        preprocessed_csv_path(Path.cwd() / "data", dataset_code, min_rating, min_uc, min_sc),
        preprocessed_csv_path(REPO_ROOT / "data", dataset_code, min_rating, min_uc, min_sc),
    ]
    unique = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def build_sigma_interactions(df: pd.DataFrame) -> pd.DataFrame:
    required = {"user_id", "item_new_id", "split"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"dataset_single_export.csv is missing columns: {missing}")

    split_order = {"train": 0, "val": 1, "test": 2}
    work = df.copy()
    work["_row_order"] = range(len(work))
    work["_split_order"] = work["split"].map(split_order)
    if work["_split_order"].isna().any():
        bad_splits = sorted(work.loc[work["_split_order"].isna(), "split"].dropna().unique().tolist())
        raise ValueError(f"Unknown split values: {bad_splits}. Expected train/val/test.")

    # dataset_single_export.csv is already written in chronological split order.
    # Because original timestamps are not exported, use a deterministic per-user
    # order that preserves train -> val -> test and row order inside each split.
    work = work.sort_values(["user_id", "_split_order", "_row_order"]).copy()
    work["timestamp"] = work.groupby("user_id").cumcount() + 1

    out = pd.DataFrame(
        {
            "user_id:token": work["user_id"].astype(int),
            "item_id:token": work["item_new_id"].astype(int),
            "rating:float": 1.0,
            "timestamp:float": work["timestamp"].astype(float),
        }
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare RecBole atomic data for SIGMA.")
    parser.add_argument("--dataset_code", default="beauty")
    parser.add_argument("--min_rating", type=int, default=3)
    parser.add_argument("--min_uc", type=int, default=6)
    parser.add_argument("--min_sc", type=int, default=6)
    parser.add_argument("--data_root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--source_csv", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, default=REPO_ROOT / "experiments" / "SIGMA" / "dataset")
    parser.add_argument(
        "--sigma_dataset_name",
        default=None,
        help="RecBole dataset folder/name. Default: sigma_<dataset_code>.",
    )
    args = parser.parse_args()

    def resolve_repo_path(path: Path) -> Path:
        return path if path.is_absolute() else REPO_ROOT / path

    data_root = resolve_repo_path(args.data_root)
    output_root = resolve_repo_path(args.output_root)

    if data_root.suffix == ".inter":
        raise ValueError(
            "--data_root must point to the repo data directory, not the output .inter file. "
            "Use --output_root for SIGMA/RecBole output. Example: "
            "--data_root /kaggle/working/CAP-BLIP-Recommendation-System/data "
            "--output_root /kaggle/working/CAP-BLIP-Recommendation-System/experiments/SIGMA/dataset"
        )

    if args.source_csv:
        source_csv = resolve_repo_path(args.source_csv)
        searched_paths = [source_csv]
    else:
        searched_paths = candidate_preprocessed_csv_paths(
            data_root,
            args.dataset_code,
            args.min_rating,
            args.min_uc,
            args.min_sc,
        )
        source_csv = next((path for path in searched_paths if path.exists()), searched_paths[0])

    if not source_csv.exists():
        raise FileNotFoundError(
            "Missing dataset_single_export.csv. Run data_prepare.py first, pass --source_csv, "
            "or ensure --data_root points to the directory that contains preprocessed/.\n"
            "Searched:\n  " + "\n  ".join(str(path) for path in searched_paths)
        )

    sigma_dataset_name = args.sigma_dataset_name or f"sigma_{args.dataset_code}"
    out_dir = output_root / sigma_dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{sigma_dataset_name}.inter"

    df = pd.read_csv(source_csv)
    inter = build_sigma_interactions(df)
    inter.to_csv(out_file, sep="\t", index=False)

    num_users = int(inter["user_id:token"].nunique())
    num_items = int(inter["item_id:token"].nunique())
    print(f"Prepared SIGMA RecBole dataset: {out_file}")
    print(f"  Source CSV : {source_csv}")
    print(f"  Dataset    : {sigma_dataset_name}")
    print(f"  Users      : {num_users}")
    print(f"  Items      : {num_items}")
    print(f"  Interactions: {len(inter)}")
    print()
    print("Run SIGMA with:")
    print(
        f"  python {REPO_ROOT / 'experiments' / 'SIGMA' / 'model' / 'run.py'} "
        f"--dataset {sigma_dataset_name} --data_path {output_root}"
    )


if __name__ == "__main__":
    main()
