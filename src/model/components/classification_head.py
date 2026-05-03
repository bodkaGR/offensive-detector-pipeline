from __future__ import annotations

import torch
from torch import nn


class ClassificationHead(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim

        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = hidden_dim

        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)