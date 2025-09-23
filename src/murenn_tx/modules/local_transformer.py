from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LocalTransformerBlock"]


class LocalTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        dim_ff: int = 1024,
        dropout: float = 0.1,
        window: int = 64,
        posenc: str = "none",
    ):
        super().__init__()
        self.window = window
        self.posenc = posenc
        if posenc == "learnable":
            self.pos_emb = nn.Parameter(torch.zeros(1, window, d_model))
            nn.init.normal_(self.pos_emb, std=0.02)
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        W = self.window
        pad = (W - (T % W)) % W
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        x = x.view(B, -1, W, D).reshape(-1, W, D)
        if self.posenc == "learnable":
            x = x + self.pos_emb
        x = self.layer(x)
        x = x.view(B, -1, W, D).reshape(B, T + pad, D)
        return x[:, :T, :]
