from .games import GamesDataset


class SportDataset(GamesDataset):
    @classmethod
    def code(cls):
        return 'sport'

    @classmethod
    def url(cls):
        return [
            'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Sports_and_Outdoors.csv',
            'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Sports_and_Outdoors.json.gz',
        ]

    @classmethod
    def all_raw_file_names(cls):
        return ['sport.csv', 'sport_meta.json.gz']
