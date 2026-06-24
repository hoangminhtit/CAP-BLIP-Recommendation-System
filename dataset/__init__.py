from .ml_100k import ML100KDataset
from .beauty import BeautyDataset
from .games import GamesDataset
from .clothes import ClothesDataset
from .sports import SportsDataset
from .fashion import AmazonFashionDataset

DATASETS = {
    ML100KDataset.code(): ML100KDataset,
    BeautyDataset.code(): BeautyDataset,
    GamesDataset.code(): GamesDataset,
    ClothesDataset.code(): ClothesDataset,
    SportsDataset.code(): SportsDataset,
    AmazonFashionDataset.code(): AmazonFashionDataset,

}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
