from __future__ import annotations

import logging
import os

import torch
from transformers import AutoTokenizer

from src.config.settings import InferenceSettings, ModelSettings
from src.model.interfaces.classifier import IClassifier
from src.model.mpnet_transformer import MPNetTransformerClassifier

logger = logging.getLogger(__name__)


class OffensiveTextPredictor:

    def __init__(
        self,
        model: IClassifier,
        tokenizer: AutoTokenizer,
        inference_cfg: InferenceSettings,
        model_cfg: ModelSettings,
        device: str,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._threshold = inference_cfg.classification_threshold
        self._max_length = model_cfg.sbert_max_seq_len
        self._device = device
        self._model.eval()

    @classmethod
    def from_checkpoint(
            cls,
            model_path: str,
            tokenizer_path: str,
            inference_cfg: InferenceSettings,
            model_cfg: ModelSettings,
            device: str | None = None,
    ) -> "OffensiveTextPredictor":
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer_src = tokenizer_path if os.path.exists(tokenizer_path) else model_cfg.sbert_model_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)

        model = MPNetTransformerClassifier.load(model_path, device=device)
        return cls(model, tokenizer, inference_cfg, model_cfg, device)

    @torch.no_grad()
    def predict(self, texts: list[str] | str) -> list[dict]:
        if isinstance(texts, str):
            texts = [texts]

        encoding = self._tokenizer(
            texts,
            max_length=self._max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        ids = encoding["input_ids"].to(self._device)
        mask = encoding["attention_mask"].to(self._device)
        probs = self._model.predict_proba(ids, mask).cpu().numpy()

        results = []
        for i, (text, p_offensive) in enumerate(zip(texts, probs)):
            is_offensive = float(p_offensive) >= self._threshold
            results.append({
                "text": text,
                "is_offensive": bool(is_offensive),
                "label": "offensive" if is_offensive else "clean",
                "p_offensive": round(float(p_offensive), 4),
                "p_clean": round(1 - float(p_offensive), 4),
                "confidence": round(max(float(p_offensive), 1 - float(p_offensive)), 4),
            })
        return results

    def predict_csv(
        self,
        input_path: str,
        output_path: str,
        text_col: str = "tweet",
        batch_size: int = 64,
    ) -> None:
        import pandas as pd
        df = pd.read_csv(input_path)
        texts = df[text_col].fillna("").tolist()

        all_results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            all_results.extend(self.predict(batch))
            if i % (batch_size * 10) == 0:
                logger.info(f"Processed { i + min(batch_size, len(texts) - i):, }/{ len(texts):, }")

        df["is_offensive"] = [result["is_offensive"] for result in all_results]
        df["label"] = [result["label"] for result in all_results]
        df["p_offensive"] = [result["p_offensive"] for result in all_results]
        df["confidence"] = [result["confidence"] for result in all_results]
        df.to_csv(output_path, index=False)

        n_off = sum(result["is_offensive"] for result in all_results)
        logger.info(f"Saved: { output_path } | Offensive: { n_off:, }/{ len(all_results):, }")