from __future__ import annotations

import logging
import os
import platform
import sys
import mlflow
import mlflow.pytorch
import torch

from src.config.settings import MLflowSettings, ModelSettings, TrainingSettings

logger = logging.getLogger(__name__)


class MLflowTracker:

    def __init__(
        self,
        mlflow_cfg: MLflowSettings,
        model_cfg: ModelSettings,
        train_cfg: TrainingSettings,
    ):
        self._mlflow_cfg = mlflow_cfg
        self._model_cfg = model_cfg
        self._train_cfg = train_cfg
        self._run = None
        self._active = False

    def __enter__(self) -> "MLflowTracker":
        self._setup_tracking()
        self._start_run()
        self._active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._active:
            if exc_type is not None:
                mlflow.set_tag("run_status", "FAILED")
                mlflow.set_tag("error", str(exc_val))
            else:
                mlflow.set_tag("run_status", "FINISHED")
            mlflow.end_run()
            self._active = False
        return False

    def _setup_tracking(self) -> None:
        try:
            mlflow.set_tracking_uri(self._mlflow_cfg.tracking_uri)
            mlflow.set_experiment(self._mlflow_cfg.experiment_name)
            logger.info(f"MLflow tracking URI: {self._mlflow_cfg.tracking_uri}")
            logger.info(f"MLflow experiment name: {self._mlflow_cfg.experiment_name}")
        except Exception as exc:
            logger.warning(f"MLflow setup failed: {exc}. Proceeding without tracking.")
            self._active = False

    def _start_run(self) -> None:
        run_name = self._mlflow_cfg.run_name or None
        self._run = mlflow.start_run(run_name=run_name)
        logger.info(f"MLflow run started: {self._run.info.run_id}")

        # System tags
        mlflow.set_tags({
            **self._mlflow_cfg.tags,
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "platform": platform.system(),
            "cuda_available": str(torch.cuda.is_available()),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        })

    def log_params(self) -> None:
        if not self._active:
            return

        params = {
            "sbert_model": self._model_cfg.sbert_model_name,
            "sbert_max_seq_len": self._model_cfg.sbert_max_seq_len,
            "sbert_hidden_dim": self._model_cfg.sbert_hidden_dim,
            "tc_num_layers": self._model_cfg.tc_num_layers,
            "tc_num_heads": self._model_cfg.tc_num_heads,
            "tc_ffn_dim": self._model_cfg.tc_ffn_dim,
            "tc_dropout": self._model_cfg.tc_dropout,
            "tc_head_dims": str(self._model_cfg.tc_head_hidden_dims),
            "epochs": self._train_cfg.epochs,
            "batch_size": self._train_cfg.batch_size,
            "learning_rate": self._train_cfg.learning_rate,
            "weight_decay": self._train_cfg.weight_decay,
            "warmup_ratio": self._train_cfg.warmup_ratio,
            "gradient_clip": self._train_cfg.gradient_clip,
            "early_stop": self._train_cfg.early_stop_patience,
            "focal_alpha": self._train_cfg.focal_alpha,
            "focal_gamma": self._train_cfg.focal_gamma,
        }

        mlflow.log_params(params)

    def log_dataset_info(self, n_train: int, n_val: int, n_test: int, n_offensive_pct: float) -> None:
        if not self._active:
            return

        mlflow.log_params({
            "dataset_train_size": n_train,
            "dataset_val_size": n_val,
            "dataset_test_size": n_test,
            "dataset_offensive_pct": round(n_offensive_pct, 3),
        })

    def log_epoch_metrics(
        self,
        epoch: int,
        tr_loss: float,
        vl_loss: float,
        tr_f1: float,
        vl_f1: float,
        tr_auc: float,
        vl_auc: float,
        lr: float,
    ) -> None:
        if not self._active:
            return

        mlflow.log_metrics({
            "train_loss": tr_loss,
            "val_loss": vl_loss,
            "train_f1": tr_f1,
            "val_f1": vl_f1,
            "train_auc": tr_auc,
            "val_auc": vl_auc,
            "learning_rate": lr,
        }, step=epoch)

    def log_best_epoch(self, epoch: int, val_f1: float) -> None:
        if not self._active:
            return

        mlflow.log_params({
            "best_epoch": epoch,
            "best_val_f1": val_f1,
        })

    def log_test_metrics(self, metrics: dict) -> None:
        if not self._active:
            return

        scalar_keys = {
            "accuracy", "f1_weighted", "f1_macro", "f1_binary", "roc_auc", "pr_auc",
        }

        metrics = {f"test_{key}": value for key, value in metrics.items() if key in scalar_keys}
        mlflow.log_metrics(metrics)

        confusion_matrix = metrics.get("confusion_matrix")
        if confusion_matrix is not None:
            import tempfile
            with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as f:
                f.write("Confusion Matrix (rows=true, cols=pred):\n")
                f.write(f"             clean  offensive\n")
                f.write(f"clean      {confusion_matrix[0][0]:6d}  {confusion_matrix[0][1]:9d}\n")
                f.write(f"offensive  {confusion_matrix[1][0]:6d}  {confusion_matrix[1][1]:9d}\n\n")
                f.write(metrics.get("classification_report", ""))
                tmp_path = f.name
            mlflow.log_artifact(tmp_path, artifact_path="evaluation")
            os.unlink(tmp_path)

        logger.info(f"MLflow: logged test metrics - "
                    f"F1={metrics.get('test_f1_weighted'):.4f}, "
                    f"AUC={metrics.get('test_roc_auc'):.4f}")

    def log_model_artifact(self, model_path: str) -> None:
        if not self._active or not self._mlflow_cfg.log_model:
            return

        if os.path.exists(model_path):
            mlflow.log_artifact(model_path, artifact_path="model")

    def log_tokenizer_artifact(self, tokenizer_dir: str) -> None:
        if not self._active or not self._mlflow_cfg.log_model:
            return

        if os.path.isdir(tokenizer_dir):
            mlflow.log_artifacts(tokenizer_dir, artifact_path="tokenizer")

    def log_plots(self, plots_dir: str) -> None:
        if not self._active or not self._mlflow_cfg.log_plots:
            return

        png_files = [file for file in os.listdir(plots_dir) if file.endswith(".png")]
        for file in png_files:
            mlflow.log_artifact(os.path.join(plots_dir, file), artifact_path="plots")

    def log_history(self, history: dir) -> None:
        if not self._active:
            return

        for epoch, (tl, vl, tf, vf, ta, va) in enumerate(zip(
            history["train_loss"], history["val_loss"],
            history["train_f1"], history["val_f1"],
            history["train_auc"], history["val_auc"],
        ), start=1):
            lr = history["lr"][epoch - 1] if "lr" in history else 0.0
            self.log_epoch_metrics(epoch, tl, vl, tf, vf, ta, va, lr)

    @property
    def run_url(self) -> str:
        if self._run is None:
            return ""
        return f"{self._mlflow_cfg.tracking_uri}/#/experiments/.../runs/{self._run.info.run_id}"

    @property
    def run_id(self) -> str:
        return self._run.info.run_id if self._run else ""