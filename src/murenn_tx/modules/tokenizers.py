from __future__ import annotations

import torch
import torch.nn as nn

from .frontend import same_padding_1d

__all__ = ["ConvTokenizer1D"]


class ConvTokenizer1D(nn.Module):
    def __init__(self, in_ch: int, d_model: int, kernel: int = 7, hop: int = 4):
        super().__init__()
        self.hop = hop
        self.conv1 = nn.Conv1d(in_ch, d_model, kernel_size=kernel, padding=same_padding_1d(kernel))
        self.act = nn.GELU()
        self.proj = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(self.act(self.conv1(x)))
        h = h[:, :, :: self.hop]
        h = h.transpose(1, 2).contiguous()
        return self.norm(h)
