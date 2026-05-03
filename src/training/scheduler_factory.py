from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class WarmupCosineSchedulerFactory:

    def __init__(self, warmup_ratio: float = 0.1):
        self._warmup_ratio = warmup_ratio

    def create(
        self, optimizer: Optimizer, total_steps: int
    ) -> LambdaLR:
        warmup_steps = int(total_steps * self._warmup_ratio)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)