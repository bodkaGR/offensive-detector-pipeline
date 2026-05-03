from __future__ import annotations

import torch
import torch.nn as nn


class TransformerEncoderBlock(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        attn_dropout: float,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Pre-LN + MHA + residual
        residual = x

        x = self.norm1(x)
        attn_out, _ = self.attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
        )
        x = residual + self.drop1(attn_out)

        # Pre-LN + FFN + residual
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        return x