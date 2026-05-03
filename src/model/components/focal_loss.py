from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        pos_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self._alpha = alpha
        self._gamma = gamma
        self._pos_weight = pos_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        targets = targets.float()
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self._pos_weight, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self._alpha * targets + (1 - self._alpha) * (1 - targets)
        weight = alpha_t * (1 - p_t) ** self._gamma
        return (weight * bce_loss).mean()