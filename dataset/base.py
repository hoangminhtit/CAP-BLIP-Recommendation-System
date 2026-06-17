import pickle
import shutil
import tempfile
import os
from pathlib import Path
import gzip
from abc import *
from .utils import *
from config import RAW_DATASET_ROOT_FOLDER

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, args):
        self.args = args
        self.min_rating = args.min_rating
        self.min_uc = args.min_uc
        self.min_sc = args.min_sc

        assert self.min_uc >= 2, 'Need at least 2 ratings per user for validation and test'

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @classmethod
    def raw_code(cls):
        return cls.code()

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return []

    @classmethod
    @abstractmethod
    def url(cls):
        pass

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def load_ratings_df(self):
        pass

    @abstractmethod
    def maybe_download_raw_dataset(self):
        pass

    def _load_dataset_from_csv(self):
        """Helper method to load dataset from CSV. Extracted to avoid duplication."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required to load CSV datasets")
        
        preproc_folder = self._get_preprocessed_folder_path()
        csv_path = preproc_folder.joinpath('dataset_single_export.csv')
        
        if not csv_path.is_file():
            return None
        
        df = pd.read_csv(csv_path)
        # Reconstruct train/val/test, meta, smap. umap cannot be recovered from CSV.
        df = df.reset_index(drop=False).rename(columns={"index": "row_order"})
        grouped = (
            df.sort_values("row_order")
              .groupby(["split", "user_id"])["item_new_id"]
              .apply(lambda s: s.astype(int).tolist())
        )

        train, val, test = {}, {}, {}
        for (split, user), items in grouped.items():
            user = int(user)
            if split == "train":
                train[user] = items
            elif split == "val":
                val[user] = items
            else:
                test[user] = items

        meta_df = df.drop_duplicates(subset=["item_new_id"]).set_index("item_new_id")
        meta = {}
        for item_new_id, row in meta_df.iterrows():
            text = row.get("item_text") if not pd.isna(row.get("item_text")) else None
            image_path = row.get("item_image_path") if not pd.isna(row.get("item_image_path")) else None
            meta[int(item_new_id)] = {"text": text, "image_path": image_path}

        smap = {}
        map_df = df[~df["Item_id"].isna()].drop_duplicates(subset=["Item_id"]).copy()
        for _, row in map_df.iterrows():
            try:
                orig = row["Item_id"]
                new = int(row["item_new_id"])
                smap[orig] = new
            except Exception:
                continue

        return {"train": train, "val": val, "test": test, "meta": meta, "umap": {}, "smap": smap}

    def load_dataset(self):
        preproc_folder = self._get_preprocessed_folder_path()
        csv_path = preproc_folder.joinpath('dataset_single_export.csv')

        # If CSV already exists, skip preprocessing and load it
        if csv_path.is_file():
            result = self._load_dataset_from_csv()
            if result is not None:
                return result

        # Otherwise run preprocessing and then load (either CSV or legacy pickle)
        self.preprocess()
        result = self._load_dataset_from_csv()
        if result is not None:
            return result

        # fallback to pickle for compatibility
        dataset_path = self._get_preprocessed_dataset_path()
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Neither CSV nor pickle dataset found. "
                f"Expected CSV at {csv_path} or pickle at {dataset_path}"
            )
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset

    def filter_triplets(self, df):
        print('Filtering triplets')
        if self.min_sc > 1 or self.min_uc > 1:
            item_sizes = df.groupby('sid').size()
            good_items = item_sizes.index[item_sizes >= self.min_sc]
            user_sizes = df.groupby('uid').size()
            good_users = user_sizes.index[user_sizes >= self.min_uc]
            while len(good_items) < len(item_sizes) or len(good_users) < len(user_sizes):
                if self.min_sc > 1:
                    item_sizes = df.groupby('sid').size()
                    good_items = item_sizes.index[item_sizes >= self.min_sc]
                    df = df[df['sid'].isin(good_items)]

                if self.min_uc > 1:
                    user_sizes = df.groupby('uid').size()
                    good_users = user_sizes.index[user_sizes >= self.min_uc]
                    df = df[df['uid'].isin(good_users)]

                item_sizes = df.groupby('sid').size()
                good_items = item_sizes.index[item_sizes >= self.min_sc]
                user_sizes = df.groupby('uid').size()
                good_users = user_sizes.index[user_sizes >= self.min_uc]
        return df
    
    def densify_index(self, df):
        print('Densifying index')
        umap = {u: i for i, u in enumerate(set(df['uid']), start=1)}
        smap = {s: i for i, s in enumerate(set(df['sid']), start=1)}
        df['uid'] = df['uid'].map(umap)
        df['sid'] = df['sid'].map(smap)
        return df, umap, smap

    def split_df(self, df, user_count):
        print('Splitting')
        user_group = df.groupby('uid')
        user2items = user_group.progress_apply(
            lambda d: list(d.sort_values(by=['timestamp', 'sid'])['sid']),
            include_groups=False  # Fix FutureWarning
        )
        train, val, test = {}, {}, {}
        for i in range(user_count):
            user = i + 1
            items = user2items[user]
            train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
        return train, val, test

    def _get_rawdata_root_path(self):
        return Path(RAW_DATASET_ROOT_FOLDER)

    def _get_rawdata_folder_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath(self.raw_code())

    def _get_preprocessed_root_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath('preprocessed')

    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()
        folder_name = '{}_min_rating{}-min_uc{}-min_sc{}' \
            .format(self.code(), self.min_rating, self.min_uc, self.min_sc)
        return preprocessed_root.joinpath(folder_name)

    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('dataset.pkl')
    
    def _get_images_folder_path(self):
        """Get folder path for storing downloaded images"""
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('images')
