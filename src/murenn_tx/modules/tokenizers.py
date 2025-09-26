from __future__ import annotations

import torch
import torch.nn as nn

from .frontend import same_padding_1d

__all__ = ["ConvTokenizer1D"]


class ConvTokenizer1D(nn.Module):
    def __init__(self, in_ch: int, d_model: int, kernel: int = 7, hop: int = 4):
        super().__init__()
        self.hop = hop
        # depthwise conv does anti-aliasing and subsampling
        self.conv1 = nn.Conv1d(
            in_ch, in_ch, kernel_size=kernel, groups=in_ch,
            stride=hop, padding=kernel // 2, bias=False
        )
        self.proj = nn.Conv1d(in_ch, d_model, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = self.conv1(x)          # (B, C, T/hop)
        x = self.proj(x)           # (B, d_model, T/hop)
        x = x.transpose(1, 2)      # (B, T/hop, d_model)
        x = self.norm(x)
        x = self.act(x)
        return x
