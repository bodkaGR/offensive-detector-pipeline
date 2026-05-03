from __future__ import annotations

import logging
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix,
    classification_report
)

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from typing_extensions import override

from src.config.settings import TrainingSettings
from src.model.interfaces.classifier import IClassifier
from src.training.interfaces.trainer import ITrainer

logger = logging.getLogger(__name__)


class Trainer(ITrainer):

    def __init__(
        self,
        model: IClassifier,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
        cfg: TrainingSettings,
        accum_steps: int = 1,
    ):
        self._model = model.to(device)
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._criterion = criterion
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._device = device
        self._cfg = cfg
        self._accum_steps = accum_steps

        self.history: dict[str, list] = {
            k: [] for k in [
                "train_loss", "val_loss",
                "train_f1", "val_f1",
                "train_auc", "val_auc",
                "lr",
            ]
        }

    @override
    def fit(self, epochs: int) -> dict:
        best_val_f1 = 0.0
        best_state = None
        patience = 0

        logger.info(f"Learning: {epochs} epochs | device={self._device}")
        logger.info("=" * 70)

        for epoch in range(1, epochs + 1):
            time_start = time.time()

            tr_loss, tr_f1, tr_auc = self._train_epoch()
            vl_loss, vl_f1, vl_auc = self._eval_epoch(self._val_loader)

            current_lr = self._scheduler.get_last_lr()[0]

            self._update_history(tr_loss, vl_loss, tr_f1, vl_f1, tr_auc, vl_auc, current_lr)

            logger.info(
                f"Epoch {epoch:2d}/{epochs} | "
                f"loss={tr_loss:.4f}/{vl_loss:.4f} | "
                f"F1={tr_f1:.4f}/{vl_f1:.4f} | "
                f"AUC={tr_auc:.4f}/{vl_auc:.4f} | "
                f"lr={current_lr:.2e} | {time.time() - time_start:.0f}s"
            )

            if vl_f1 > best_val_f1 + 1e-5:
                best_val_f1 = vl_f1
                best_state = {
                    k: v.cpu().clone()
                    for k, v in self._model.state_dict().items()
                }
                patience = 0
                logger.info(f"New best val_f1: {best_val_f1:.4f}")
            else:
                patience += 1
                if patience >= self._cfg.early_stop_patience:
                    logger.info(f"Early stopping: {epoch} epoch")
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
            logger.info(f"Restored the best model (val_f1={best_val_f1:.4f})")

        return self.history

    @override
    def evaluate(self, loader: DataLoader) -> dict:
        self._model.eval()
        all_probs, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                ids, mask, labels = self._move(batch)
                probs = self._model.predict_proba(ids, mask).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())

        probs = np.array(all_probs)
        labels = np.array(all_labels)
        # Use threshold from inference settings
        threshold = 0.5
        preds = (probs >= threshold).astype(int)

        return {
            "accuracy": round(accuracy_score(labels, preds), 4),
            "f1_weighted": round(f1_score(labels, preds, average="weighted", zero_division=0), 4),
            "f1_macro": round(f1_score(labels, preds, average="macro", zero_division=0), 4),
            "f1_binary": round(f1_score(labels, preds, average="binary", zero_division=0), 4),
            "roc_auc": round(roc_auc_score(labels, probs), 4),
            "pr_auc": round(average_precision_score(labels, probs), 4),
            "confusion_matrix": confusion_matrix(labels, preds),
            "classification_report": classification_report(
                labels, preds,
                target_names=["clean", "offensive"],
                zero_division=0
            ),
            "probs": probs,
            "labels": labels,
            "preds": preds,
        }

    def _move(self, batch: dict) -> tuple:
        return (
            batch["input_ids"].to(self._device),
            batch["attention_mask"].to(self._device),
            batch["label"].to(self._device),
        )

    def _train_epoch(self) -> tuple[float, float, float]:
        self._model.train()
        total_loss, all_probs, all_labels = 0.0, [], []
        n_batches = len(self._train_loader)

        self._optimizer.zero_grad()

        for step, batch in enumerate(self._train_loader):
            ids, mask, labels = self._move(batch)

            logits = self._model(ids, mask)
            loss = self._criterion(logits, labels) / self._accum_steps
            loss.backward()

            total_loss += loss.item() * self._accum_steps

            # Gradient accumulation: update weights every accum_steps batches
            if (step + 1) % self._accum_steps == 0 or (step + 1) == n_batches:
                nn.utils.clip_grad_norm_(
                    self._model.parameters(), self._cfg.gradient_clip
                )
                self._optimizer.step()
                self._scheduler.step()
                self._optimizer.zero_grad()

            with torch.no_grad():
                probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / n_batches
        f1, auc = self._compute_metrics(all_probs, all_labels)
        return avg_loss, f1, auc

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> tuple[float, float, float]:
        self._model.eval()
        total_loss, all_probs, all_labels = 0.0, [], []

        for batch in loader:
            ids, mask, labels = self._move(batch)
            logits = self._model(ids, mask)
            total_loss += self._criterion(logits, labels).item()
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader)
        f1, auc = self._compute_metrics(all_probs, all_labels)
        return avg_loss, f1, auc

    @staticmethod
    def _compute_metrics(probs: list, labels: list) -> tuple[float, float]:
        # Use threshold from inference settings
        threshold = 0.5
        preds = (np.array(probs) >= threshold).astype(int)
        f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        try:
            auc = roc_auc_score(labels, probs)
        except Exception:
            auc = 0.0
        return round(f1, 4), round(auc, 4)

    def _update_history(self, *values) -> None:
        keys = [
            "train_loss",
            "val_loss",
            "train_f1",
            "val_f1",
            "train_auc",
            "val_auc",
            "lr"
        ]
        for key, value in zip(keys, values):
            self.history[key].append(value)