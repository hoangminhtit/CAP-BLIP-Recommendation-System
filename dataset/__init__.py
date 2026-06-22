from .ml_100k import ML100KDataset
from .ml_1m import ML1MDataset
from .beauty import BeautyDataset
from .games import GamesDataset
from .clothes import ClothesDataset
from .sport import SportDataset

DATASETS = {
    ML100KDataset.code(): ML100KDataset,
    ML1MDataset.code(): ML1MDataset,
    BeautyDataset.code(): BeautyDataset,
    GamesDataset.code(): GamesDataset,
    ClothesDataset.code(): ClothesDataset,
    SportDataset.code(): SportDataset,
    'sports': SportDataset,

}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
