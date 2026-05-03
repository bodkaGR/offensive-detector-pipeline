from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.scorer = nn.Linear(hidden_dim, 1, bias=True)

    def forward(
        self,
        token_embeds: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        scores = self.scorer(token_embeds).squeeze(-1)  # (batch, seq_len)

        scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)  # (batch, seq_len)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        pooled = torch.bmm(
            attn_weights.unsqueeze(1),  # (batch, 1, seq_len)
            token_embeds,  # (batch, seq_len, hidden)
        ).squeeze(1)  # (batch, hidden)

        return pooled