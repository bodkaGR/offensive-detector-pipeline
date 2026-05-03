from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from typing_extensions import override
from sklearn.model_selection import train_test_split

from src.config.settings import LabelSettings, SplitSettings
from src.data.interfaces.dataset_loader import IDatasetLoader
from src.data.interfaces.preprocessor import ITextPreprocessor


logger = logging.getLogger(__name__)


class OffensiveDatasetLoader(IDatasetLoader):

    def __init__(
        self,
        preprocessor: ITextPreprocessor,
        label_cfg: LabelSettings,
        split_cfg: SplitSettings,
    ):
        self._preprocessor = preprocessor
        self._label_cfg = label_cfg
        self._split_cfg = split_cfg

    @override
    def load(self, path: str) -> dict:
        logger.info(f"Loading dataset from {path}")
        df = self._read_csv(path)
        df = self._preprocess(df)
        df = self._binarize_labels(df)

        self._log_stats(df)

        x = df["tweet_clean"].to_numpy(dtype=str)
        y = df["label"].to_numpy(dtype=int)
        y3 = df["class"].to_numpy(dtype=int)

        splits = self._split(x, y, y3)
        return { "df": df, **splits }

    def _read_csv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        if "unnamed: 0" in df.columns:
            df = df.drop(columns=["unnamed: 0"])
        return df

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Texts cleaning...")
        df["tweet_clean"] = df["tweet"].apply(self._preprocessor.clean)
        before = len(df)
        df = df[df["tweet_clean"].str.len() > 3].reset_index(drop=True)
        logger.info(f"Filtered {before - len(df)} empty tweets")
        return df

    def _binarize_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        df["label"] = df["class"].map(self._label_cfg.binary_map)
        return df

    def _log_stats(self, df: pd.DataFrame) -> None:
        total = len(df)
        distribution = df["label"].value_counts()
        logger.info(f"Dataset: {total:,} records")
        for label, name in self._label_cfg.binary_labels.items():
            count = distribution.get(label, 0)
            logger.info(f"  {name}: {count:,} ({count/total*100:.1f}%)")

    def _split(
        self,
        x: np.ndarray,
        y: np.ndarray,
        y3: np.ndarray,
    ) -> dict:
        cfg = self._split_cfg
        X_tmp, X_test, y_tmp, y_test, y3_tmp, y3_test = train_test_split(
            x, y, y3,
            test_size=cfg.test_size,
            stratify=y,
            random_state=cfg.random_seed,
        )
        val_ratio = cfg.val_size / (1 - cfg.test_size)

        X_train, X_val, y_train, y_val, y3_train, y3_val = train_test_split(
            X_tmp, y_tmp, y3_tmp,
            test_size=val_ratio,
            stratify=y_tmp,
            random_state=cfg.random_seed,
        )

        logger.info(f"Split: train={len(X_train):,} | val={len(X_val):,} | test={len(X_test):,}")
        return {
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val, "y_test": y_test,
            "y3_train": y3_train, "y3_val": y3_val, "y3_test": y3_test,
        }