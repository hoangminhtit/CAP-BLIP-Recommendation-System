from .base import AbstractDataset
from .utils import *

from datetime import date
from pathlib import Path
import pickle
import shutil
import tempfile
import os

import gzip
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


class AmazonFashionDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'fashion'

    @classmethod
    def url(cls):
        return [
            'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Amazon_Fashion.csv',
            'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Amazon_Fashion.json.gz',
        ]

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['amazon_fashion.csv', 'amazon_fashion_meta.json.gz']

    def maybe_download_raw_dataset(self):
        folder_path = self._get_rawdata_folder_path()
        if folder_path.is_dir() and \
           all(folder_path.joinpath(filename).is_file() for filename in self.all_raw_file_names()):
            print('Raw data already exists. Skip downloading')
            return

        print("Raw file doesn't exist. Downloading...")
        for idx, url in enumerate(self.url()):
            tmproot = Path(tempfile.mkdtemp())
            tmpfile = tmproot.joinpath('file')
            download(url, tmpfile)
            os.makedirs(folder_path, exist_ok=True)
            shutil.move(tmpfile, folder_path.joinpath(self.all_raw_file_names()[idx]))
            print()

    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        self.maybe_download_raw_dataset()
        df = self.load_ratings_df()
        meta_raw = self.load_meta_dict()

        print(f'Total items in metadata: {len(meta_raw)}')

        # Filter by min_rating first
        if self.min_rating > 0:
            initial_count = len(df)
            df = df[df['rating'] >= self.min_rating]
            print(f'Ratings after min_rating filter (rating >= {self.min_rating}): {len(df)}/{initial_count}')

        # Step 1: filter by text
        if self.args.use_text:
            valid_text_items = set()
            for item_id, meta_info in meta_raw.items():
                if meta_info['text'] is not None and len(meta_info['text']) > 0:
                    valid_text_items.add(item_id)
            print(f'Items with valid text: {len(valid_text_items)}')
            df = df[df['sid'].isin(valid_text_items)]
        else:
            df = df[df['sid'].isin(meta_raw)]

        # Step 2: filter triplets
        print(f'Ratings before filter_triplets: {len(df)}')
        df = self.filter_triplets(df)
        print(f'Ratings after filter_triplets: {len(df)}')

        # Step 3: remaining items (original item ids)
        remaining_items = set(df['sid'].unique())
        print(f'Items remaining after triplet filtering: {len(remaining_items)}')

        # Step 5: download images for remaining items
        if self.args.use_image:
            items_to_download = {k: v for k, v in meta_raw.items() if k in remaining_items}
            print(f'Downloading images for {len(items_to_download)} items (after filtering)...')

            image_folder = self._get_images_folder_path()
            downloaded_images, valid_image_items = download_and_verify_images_batch(
                items_to_download, image_folder, max_workers=20
            )

            for item_id, local_path in downloaded_images.items():
                if item_id in meta_raw:
                    meta_raw[item_id]['image_path'] = local_path

            print(f'Successfully downloaded: {len(downloaded_images)}/{len(items_to_download)} images')

            items_to_remove = remaining_items - valid_image_items
            if items_to_remove:
                print(f'Removing {len(items_to_remove)} items without valid images...')
                df = df[df['sid'].isin(valid_image_items)]
                print(f"Final items after image filtering: {df['sid'].nunique()}")

        # Step 6: densify index ONCE (after all filtering)
        df, umap, smap = self.densify_index(df)

        train, val, test = self.split_df(df, len(umap))
        meta = {smap[k]: v for k, v in meta_raw.items() if k in smap}

        preproc_folder = dataset_path.parent
        rows = []
        inv_smap = {v: k for k, v in smap.items()}

        def add_rows(split_name, split_dict):
            for user, items in split_dict.items():
                for item in items:
                    orig_item = inv_smap.get(item, None)
                    info = meta.get(item, {})
                    text = info.get('text') if info else None
                    image_path = info.get('image_path') or info.get('image') if info else None
                    rows.append({
                        'Item_id': orig_item,
                        'user_id': int(user),
                        'item_new_id': int(item),
                        'item_text': text,
                        'item_image_embedding': '',
                        'item_text_embedding': '',
                        'item_image_path': image_path or '',
                        'split': split_name,
                    })

        add_rows('train', train)
        add_rows('val', val)
        add_rows('test', test)

        df_out = pd.DataFrame(rows)
        out_csv = preproc_folder.joinpath('dataset_single_export.csv')
        df_out.to_csv(out_csv, index=False)

        dataset = {
            'train': train,
            'val': val,
            'test': test,
            'meta': meta,
            'umap': umap,
            'smap': smap,
        }

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath(self.all_raw_file_names()[0])
        df = pd.read_csv(file_path, header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df

    def load_meta_dict(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath(self.all_raw_file_names()[1])

        meta_dict = {}
        with gzip.open(file_path, 'rb') as f:
            for line in f:
                item = eval(line)
                asin = item.get('asin', '').strip()
                if not asin:
                    continue

                title = item.get('title', '').strip()
                description = item.get('description', '')
                if isinstance(description, list):
                    description = ' '.join(description).strip()
                else:
                    description = str(description).strip() if description else ''

                from dataset.utils import process_item_text
                max_text_length = getattr(self.args, 'max_text_length', 512)
                text = process_item_text(title, description, max_length=max_text_length)

                image = item.get('imUrl', '').strip()

                meta_dict[asin] = {
                    'text': text if text else None,
                    'image': image if image else None,
                    'title': title if title else None,
                }

        return meta_dict
