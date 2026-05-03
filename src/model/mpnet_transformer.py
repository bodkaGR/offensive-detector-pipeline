from __future__ import annotations

import logging
import os.path

import torch
import torch.nn as nn
from transformers import AutoModel
from typing_extensions import override

from src.config.settings import ModelSettings
from src.model.components.attention_pooling import AttentionPooling
from src.model.components.classification_head import ClassificationHead
from src.model.components.transformer_block import TransformerEncoderBlock
from src.model.interfaces.classifier import IClassifier


logger = logging.getLogger(__name__)


class MPNetTransformerClassifier(nn.Module, IClassifier):

    def __init__(
        self, cfg: ModelSettings, freeze_sbert: bool = False,
    ):
        super().__init__()
        self._cfg = cfg
        self._freeze_sbert = freeze_sbert

        logger.info(f"Loading MPNet: {cfg.sbert_model_name}")
        self.sbert_encoder = AutoModel.from_pretrained(cfg.sbert_model_name)
        self._apply_freezing(freeze_sbert)

        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                hidden_dim=cfg.sbert_hidden_dim,
                num_heads=cfg.tc_num_heads,
                ffn_dim=cfg.tc_ffn_dim,
                dropout=cfg.tc_dropout,
                attn_dropout=cfg.tc_attn_dropout,
            )
            for _ in range(cfg.tc_num_layers)
        ])
        self.final_norm = nn.LayerNorm(cfg.sbert_hidden_dim)

        self.pooling = AttentionPooling(cfg.sbert_hidden_dim)

        self.head = ClassificationHead(
            input_dim=cfg.sbert_hidden_dim,
            hidden_dims=cfg.tc_head_hidden_dims,
            dropout=cfg.tc_head_dropout,
        )

        self._log_params()

    def _apply_freezing(self, freeze_sbert: bool) -> None:
        if freeze_sbert:
            for param in self.sbert_encoder.parameters():
                param.requires_grad = False
            logger.info("MPNet completely frozen")
            return

        if hasattr(self.sbert_encoder, "encoder"):
            for i, layer in enumerate(self.sbert_encoder.encoder.layer):
                if i < 6:
                    for param in layer.parameters():
                        param.requires_grad = False
        logger.info("MPNet: layers 1-6 frozen, layers 7-12 fine-tuned")

    def _log_params(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"Parameters: total={total/1e6:.1f}M | "
            f"trainable={trainable/1e6:.1f}M | "
            f"frozen={(total-trainable)/1e6:.1f}M"
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        token_embeddings = self.sbert_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        padding_mask = (attention_mask == 0)

        x = token_embeddings
        for block in self.transformer_blocks:
            x = block(x, key_padding_mask=padding_mask)
        x = self.final_norm(x)

        pooled = self.pooling(x, attention_mask)

        logits = self.head(pooled)
        return logits

    def predict_proba(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        logits = self.forward(input_ids, attention_mask)
        return torch.sigmoid(logits)

    @override
    def save(self, path: str, tokenizer=None) -> None:
        torch.save({
            "state_dict": self.state_dict(),
            "freeze_sbert": self._freeze_sbert,
            "hidden_dim": self._cfg.sbert_hidden_dim,
            "config": {
                "num_layers": len(self.transformer_blocks),
                "num_heads": self._cfg.tc_num_heads,
                "ffn_dim": self._cfg.tc_ffn_dim,
                "dropout": self._cfg.tc_dropout,
                "head_hidden_dims": self._cfg.tc_head_hidden_dims,
            },
        }, path)
        logger.info(f"Model saved to {path}")

        if tokenizer is not None:
            tok_dir = os.path.join(os.path.dirname(path), "tokenizer")
            tokenizer.save_pretrained(tok_dir)
            logger.info(f"Tokenizer saved to {tok_dir}")

    @classmethod
    @override
    def load(
        cls, path: str, device: str = "cpu"
    ) -> "MPNetTransformerClassifier":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]

        model_config = ModelSettings(
            tc_num_layers=config["num_layers"],
            tc_num_heads=config["num_heads"],
            tc_ffn_dim=config["ffn_dim"],
            tc_dropout=config["dropout"],
            tc_head_hidden_dims=config["head_hidden_dims"],
        )
        model = cls(model_config, freeze_sbert=checkpoint["freeze_sbert"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        logger.info(f"Model loaded: {path} -> {device}")
        return model