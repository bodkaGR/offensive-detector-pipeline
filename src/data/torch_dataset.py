from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from src.config.settings import ModelSettings, TrainingSettings


class OffensiveTextDataset(Dataset):
    def __init__(
        self,
        texts: np.ndarray,
        labels: np.ndarray,
        tokenizer: AutoTokenizer,
        max_length: int
    ):
        self._texts = texts
        self._labels = labels
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, index: int) -> dict:
        text = str(self._texts[index])
        label = int(self._labels[index])

        encoding = self._tokenizer(
            text,
            max_length=self._max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


class DataLoaderFactory:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model_cfg: ModelSettings,
        train_cfg: TrainingSettings,
    ):
        self._tokenizer = tokenizer
        self._max_len = model_cfg.sbert_max_seq_len
        self._batch_size = train_cfg.batch_size

    def make_train_loader(
        self, x: np.ndarray, y: np.ndarray
    ) -> DataLoader:
        dataset = OffensiveTextDataset(x, y, self._tokenizer, self._max_len)
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=True,
            pin_memory=False,
            drop_last=False,
        )

    def make_eval_loader(
        self, x: np.ndarray, y: np.ndarray
    ) -> DataLoader:
        dataset = OffensiveTextDataset(x, y, self._tokenizer, self._max_len)
        return DataLoader(
            dataset,
            batch_size=self._batch_size * 2,
            shuffle=False,
            pin_memory=False,
        )

    def make_all(self, data: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
        return (
            self.make_train_loader(data["X_train"], data["y_train"]),
            self.make_eval_loader(data["X_val"], data["y_val"]),
            self.make_eval_loader(data["X_test"], data["y_test"]),
        )
