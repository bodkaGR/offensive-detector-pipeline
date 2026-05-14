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

from src.training.mlflow_tracker import MLflowTracker
from src.evaluation.metrics import print_results, save_report
from src.evaluation.visualizer import TrainingVisualizer
from src.model.components.focal_loss import FocalLoss
from src.training.optimizer_factory import DifferentialLROptimizerFactory
from src.training.scheduler_factory import WarmupCosineSchedulerFactory
from src.training.trainer import Trainer
from src.model.mpnet_transformer import MPNetTransformerClassifier
from src.config.settings import Settings, MLflowSettings
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

    mlflow_cfg = MLflowSettings(
        tracking_uri=cfg.mlflow.tracking_uri,
        experiment_name=args.experiment or cfg.mlflow.experiment_name,
        run_name=args.run_name or "",
        log_model=cfg.mlflow.log_model,
        log_plots=cfg.mlflow.log_plots,
        tags={
            **cfg.mlflow.tags,
            "freeze_sbert": str(args.freeze_sbert)
        },
    )

    logger.info("=" * 70)
    logger.info("MPNet + TransformerEncoder Pipeline")
    logger.info(f"Device: {device}")
    logger.info("=" * 70)

    logger.info("\n================================= 1. Loading and Preprocessing data =================================")
    preprocessor = TwitterTextPreprocessor()
    loader = OffensiveDatasetLoader(preprocessor, cfg.labels, cfg.split)
    data = loader.load(args.data_path)

    y_train = data["y_train"]
    off_pct = float(y_train.sum()) / max(len(y_train), 1)

    logger.info("\n================================= 2. Tokenizer and DataLoaders ======================================")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.sbert_model_name)
    loader_factory = DataLoaderFactory(tokenizer, cfg.model, cfg.training)
    train_loader, val_loader, test_loader = loader_factory.make_all(data)

    logger.info("\n================================= 3. Model Initialization ===========================================")
    model = MPNetTransformerClassifier(cfg.model, freeze_sbert=args.freeze_sbert)

    optimizer = DifferentialLROptimizerFactory(cfg.training).create(model)
    total_steps = len(train_loader) * args.epochs // args.accum_steps

    scheduler = WarmupCosineSchedulerFactory(cfg.training.warmup_ratio).create(optimizer, total_steps)
    criterion = FocalLoss(
        alpha=cfg.training.focal_alpha,
        gamma=cfg.training.focal_gamma,
        pos_weight=torch.tensor([5.0], device=device)
    )

    logger.info("\n================================= 4. Learning =======================================================")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        cfg=cfg.training,
        accum_steps=args.accum_steps,
    )

    with MLflowTracker(mlflow_cfg, cfg.model, cfg.training) as tracker:
        tracker.log_params()
        tracker.log_dataset_info(
            len(data["X_train"]), len(data["X_val"]), len(data["X_test"]), off_pct
        )

        original_fit = trainer.fit

        def fit_with_tracking(epochs: int) -> dict:
            history = original_fit(epochs)
            tracker.log_history(history)
            return history

        history = fit_with_tracking(args.epochs)

        logger.info("\n================================= 5. Saving Model ===================================================")
        model.save(cfg.paths.model_checkpoint, tokenizer=tokenizer)

        logger.info("\n================================= 6. Evaluation and Visualisation ===================================")
        visualizer = TrainingVisualizer(cfg.paths.plots)
        visualizer.plot_training_curves(history, title="MPNet + TransformerEncoder")

        metrics = trainer.evaluate(test_loader)
        print_results(metrics, title="MPNet + TransformerEncoder - Test")

        visualizer.plot_confusion_matrix(metrics["confusion_matrix"])
        visualizer.plot_roc_pr(metrics["labels"], metrics["probs"], name="MPNet + TransformerEncoder")
        visualizer.plot_probability_distribution(metrics["labels"], metrics["probs"])

        tracker.log_test_metrics(metrics)
        tracker.log_model_artifact(cfg.paths.model_checkpoint)
        tracker.log_tokenizer_artifact(cfg.paths.tokenizer_dir)
        tracker.log_plots(cfg.paths.plots)

    save_report(metrics, cfg.paths.reports)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPNet + TransformerEncoder Training")
    parser.add_argument("--data_path", default="data/labeled_data.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--freeze_sbert", action="store_true")
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--experiment", type=str, default="")
    parser.add_argument("--run_name", type=str, default="")
    main(parser.parse_args())