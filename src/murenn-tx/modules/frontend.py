from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["same_padding_1d", "AntiAliasedDownsample1d", "AntiAliasedUpsample1d", "MRFrontEnd"]


def same_padding_1d(kernel_size: int, stride: int = 1, dilation: int = 1) -> int:
    return ((stride - 1) + dilation * (kernel_size - 1)) // 2


class AntiAliasedDownsample1d(nn.Module):
    def __init__(self, stride: int):
        super().__init__()
        assert stride >= 1
        self.stride = stride
        k = torch.tensor([1.,4.,6.,4.,1.]) / 16.0
        self.register_buffer('kernel', k.view(1,1,-1))
        self.pad = (self.kernel.shape[-1] - 1) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            return x
        x = F.conv1d(x, self.kernel.expand(x.size(1), 1, -1), groups=x.size(1), padding=self.pad)
        return x[:, :, :: self.stride]
    

class AntiAliasedUpsample1d(nn.Module):
    def __init__(self, scale: int):
        super().__init__()
        assert scale >= 1
        self.scale = scale
        k = torch.tensor([1.,4.,6.,4.,1.]) / 16.0
        self.register_buffer('kernel', k.view(1,1,-1))
        self.pad = (self.kernel.shape[-1] - 1) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 1:
            return x
        x = x.repeat_interleave(self.scale, dim=-1)
        x = F.conv1d(x, self.kernel.expand(x.size(1), 1, -1), groups=x.size(1), padding=self.pad)
        return x


class MRFrontEnd(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 32, n_scales: int = 3):
        super().__init__()
        self.n_scales = n_scales
        self.proj0 = nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=same_padding_1d(7))
        self.lp = AntiAliasedDownsample1d(stride=2)
        self.projs = nn.ModuleList([
        nn.Conv1d(base_channels, base_channels * (2 ** s), kernel_size=3, padding=same_padding_1d(3))
        for s in range(n_scales)
        ])

    def forward(self, x: torch.Tensor):
        xs = []
        z = self.proj0(x)
        for s in range(self.n_scales):
            if s > 0:
                z = self.lp(z)
            xs.append(self.projs[s](z))
        return xs