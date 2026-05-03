from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.settings import Settings
from src.data.dataset_loader import OffensiveDatasetLoader
from src.data.preprocessor import TwitterTextPreprocessor
from src.data.torch_dataset import DataLoaderFactory


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args: argparse.Namespace) -> None:
    cfg = Settings()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    setup_logging()
    set_seed(cfg.split.random_seed)
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("MPNet + TransformerEncoder Pipeline")
    logger.info(f"Device: {device}")
    logger.info("=" * 70)

    logger.info("\n================================= 1. Loading and preprocessing data =================================")
    preprocessor = TwitterTextPreprocessor()
    loader = OffensiveDatasetLoader(preprocessor, cfg.labels, cfg.split)
    data = loader.load(args.data_path)

    logger.info("\n================================= 2. Tokenizer and DataLoaders ======================================")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.sbert_model_name)
    loader_factory = DataLoaderFactory(tokenizer, cfg.model, cfg.training)
    train_loader, val_loader, test_loader = loader_factory.make_all(data)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPNet + TransformerEncoder Training")
    parser.add_argument("--data_path", default="data/labeled_data.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--freeze_sbert", action="store_true")
    parser.add_argument("--accum_steps", type=int, default=1)
    main(parser.parse_args())