from .ml_100k import ML100KDataset

import re
import pandas as pd


class ML1MDataset(ML100KDataset):
    @classmethod
    def code(cls):
        return 'ml-1m'

    @classmethod
    def url(cls):
        return 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'

    @classmethod
    def all_raw_file_names(cls):
        return ['README', 'movies.dat', 'ratings.dat', 'users.dat']

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('ratings.dat')
        df = pd.read_csv(
            file_path,
            sep='::',
            engine='python',
            names=['uid', 'sid', 'rating', 'timestamp'],
            encoding='ISO-8859-1',
        )
        return df

    def load_meta_dict(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('movies.dat')
        df = pd.read_csv(
            file_path,
            sep='::',
            engine='python',
            names=['movieId', 'title', 'genres'],
            encoding='ISO-8859-1',
        )

        from dataset.utils import process_item_text
        max_text_length = getattr(self.args, 'max_text_length', 512)

        meta_dict = {}
        for row in df.itertuples(index=False):
            movie_id = int(row.movieId)
            raw_title = str(row.title)
            genres = str(row.genres).replace('|', ' ')

            year_match = re.search(r'\((\d{4})\)\s*$', raw_title)
            year = year_match.group(1) if year_match else ''
            title = re.sub(r'\s*\(\d{4}\)\s*$', '', raw_title).strip()

            # Move trailing articles to the front, e.g. "Matrix, The" -> "The Matrix".
            for article in ['The', 'A', 'An']:
                suffix = f', {article}'
                if title.endswith(suffix):
                    title = f'{article} {title[:-len(suffix)]}'.strip()
                    break

            full_title = f'{title} ({year})' if year else title
            text = process_item_text(full_title, genres, max_length=max_text_length)

            meta_dict[movie_id] = {
                'text': text if text else None,
                'image': None,
                'title': full_title if full_title else None,
                'genres': genres,
            }

        return meta_dict
