import sys
import logging
import argparse
from logging import getLogger
from pathlib import Path
import numpy as np
from recbole.utils import init_logger, init_seed
#from recbole.trainer import Trainer
#from mamba4rec import Mamba4Rec
from gated_mamba import SIGMA
#from simple4rec import SMLPREC
#from gated_mamba_s import Mamba4Rec
#from bert4rec import BERT4Rec
#from gru4rec import GRU4Rec
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)
from custom_trainer import CustomTrainer  # 导入自定义的Trainer

def patch_numpy_for_recbole() -> None:
    """RecBole 1.2.0 still references NumPy aliases removed in NumPy 2.x."""
    if not hasattr(np, "float_"):
        np.float_ = np.float64
    if not hasattr(np, "complex_"):
        np.complex_ = np.complex128
    if not hasattr(np, "int_"):
        np.int_ = np.int64
    if not hasattr(np, "bool_"):
        np.bool_ = np.bool


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run SIGMA with grouped custom evaluation.")
    parser.add_argument("--config", default=None, help="Path to SIGMA config yaml.")
    parser.add_argument("--dataset", default=None, help="RecBole dataset name, e.g. sigma_beauty.")
    parser.add_argument("--data_path", default=None, help="Directory containing the dataset folder.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--gpu_id", default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]

    def resolve_repo_path(value):
        path = Path(value)
        return path if path.is_absolute() else repo_root / path

    config_file = resolve_repo_path(args.config) if args.config else script_dir / "config.yaml"
    config_dict = {}
    if args.dataset:
        config_dict["dataset"] = args.dataset
    if args.data_path:
        config_dict["data_path"] = str(resolve_repo_path(args.data_path))
    else:
        config_dict["data_path"] = str((script_dir.parent / "dataset").resolve())
    if args.epochs is not None:
        config_dict["epochs"] = args.epochs
    if args.train_batch_size is not None:
        config_dict["train_batch_size"] = args.train_batch_size
    if args.eval_batch_size is not None:
        config_dict["eval_batch_size"] = args.eval_batch_size
    if args.gpu_id is not None:
        config_dict["gpu_id"] = args.gpu_id

    patch_numpy_for_recbole()
    config = Config(model=SIGMA, config_file_list=[str(config_file)], config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = SIGMA(config, train_data.dataset).to(config['device'])
    logger.info(model)
    
    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = CustomTrainer(config, model)  # 使用自定义的Trainer

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, show_progress=config["show_progress"]
    )
    # 进行评估
    grouped_results = trainer.evaluate(test_data, show_progress=config["show_progress"])

    
    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {grouped_results}")
