from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class IClassifier(ABC):

    @abstractmethod
    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor: ...

    @abstractmethod
    def predict_proba(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor: ...

    @abstractmethod
    def save(self, path: str, tokenizer=None) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str, device: str) -> "IClassifier": ...