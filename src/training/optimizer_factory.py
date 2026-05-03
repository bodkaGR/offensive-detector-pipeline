from __future__ import annotations

import logging

from torch.optim import AdamW

from src.config.settings import TrainingSettings
from src.model.mpnet_transformer import MPNetTransformerClassifier

logger = logging.getLogger(__name__)


class DifferentialLROptimizerFactory:

    NO_DECAY_PARAMS = {"bias", "LayerNorm.weight", "layer_norm.weight"}

    def __init__(self, cfg: TrainingSettings):
        self._cfg = cfg

    def create(self, model: MPNetTransformerClassifier) -> AdamW:
        sbert_wd, sbert_nwd = [], []
        new_wd, new_nwd = [], []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            no_wd = any(nd in name for nd in self.NO_DECAY_PARAMS)
            if name.startswith("sbert_encoder"):
                (sbert_nwd if no_wd else sbert_wd).append(param)
            else:
                (new_nwd if no_wd else new_wd).append(param)

        sbert_lr = self._cfg.learning_rate * 0.1

        param_groups = [
            {"params": sbert_wd, "lr": sbert_lr, "weight_decay": self._cfg.weight_decay},
            {"params": sbert_nwd, "lr": sbert_lr, "weight_decay": 0.0},
            {"params": new_wd, "lr": self._cfg.learning_rate, "weight_decay": self._cfg.weight_decay},
            {"params": new_nwd, "lr": self._cfg.learning_rate, "weight_decay": 0.0},
        ]

        logger.info(
            f"Optimizer: SBERT lr={sbert_lr:.1e} "
            f"(wd={len(sbert_wd)}, nwd={len(sbert_nwd)}) | "
            f"New lr={self._cfg.learning_rate:.1e} "
            f"(wd={len(new_wd)}, nwd={len(new_nwd)})"
        )
        return AdamW(param_groups, eps=1e-8)