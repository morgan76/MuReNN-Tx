from __future__ import annotations

from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets.dtcwt.transform1d import DTCWTForward as DTCWTForward1D

__all__ = ["same_padding_1d", "AntiAliasedDownsample1d", "AntiAliasedUpsample1d", "MRFrontEnd"]


def same_padding_1d(kernel_size: int, stride: int = 1, dilation: int = 1) -> int:
    return ((stride - 1) + dilation * (kernel_size - 1)) // 2


class AntiAliasedDownsample1d(nn.Module):
    def __init__(self, stride: int):
        super().__init__()
        assert stride >= 1
        self.stride = stride
        k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
        self.register_buffer("kernel", k.view(1, 1, -1))
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
        k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
        self.register_buffer("kernel", k.view(1, 1, -1))
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
        self.proj0 = nn.Conv1d(
            in_channels, base_channels, kernel_size=7, padding=same_padding_1d(7)
        )
        self.lp = AntiAliasedDownsample1d(stride=2)
        self.projs = nn.ModuleList(
            [
                nn.Conv1d(
                    base_channels, base_channels * (2**s), kernel_size=3, padding=same_padding_1d(3)
                )
                for s in range(n_scales)
            ]
        )

    def forward(self, x: torch.Tensor):
        xs = []
        z = self.proj0(x)
        for s in range(self.n_scales):
            if s > 0:
                z = self.lp(z)
            xs.append(self.projs[s](z))
        return xs


class MuReNNFrontEnd(nn.Module):
    """
    MuReNN-style front-end: DTCWT pyramid + learnable per-level 1D conv banks
    applied separately to real/imag parts, then squared-magnitude pooling.

    Config mirrors the WASPAA Student:
      - stride: base stride for j=0; higher levels use stride // 2**(j-1)
      - octaves: list of octave indices; the count per index gives #filters at that level
      - Q_multiplier: kernel_size_j = Q_multiplier * (#filters at level j)
    Returns a list of tensors [(B, C_j, T_j) for j = 0..J], fine -> coarse.
    """

    def __init__(
        self,
        stride: int = 64,
        octaves: list[int] = (0, 1, 1, 2, 2, 2),  # example: levels 0..2 with counts {0:1, 1:2, 2:3}
        Q_multiplier: int = 16,
        include_scale: bool = False,
        # alternate_gh: bool = True,
    ):
        super().__init__()
        Q_ctr = Counter(octaves)
        assert len(Q_ctr) > 0, "octaves must not be empty"
        self.J_psi = max(Q_ctr)  # highest level index
        # ensure every level 0..J_psi has at least one filter
        for j in range(self.J_psi + 1):
            assert Q_ctr[j] > 0, f"octaves must include level {j} at least once"

        self.stride = stride
        self.Q_ctr = Q_ctr

        # Dual-tree complex wavelet transform
        self.tfm = DTCWTForward1D(
            J=1 + self.J_psi,
            complex=True,  # return complex highpasses
            include_scale=include_scale,
        )
        # Per-level learnable conv banks on real/imag parts
        psis = []
        for j in range(1 + self.J_psi):
            n_filters = Q_ctr[j]
            kernel_size = Q_multiplier * n_filters
            stride_j = stride if j == 0 else max(1, stride // (2 ** (j - 1)))
            psis.append(
                nn.Conv1d(
                    in_channels=1,
                    out_channels=n_filters,
                    kernel_size=kernel_size,
                    stride=stride_j,
                    bias=False,
                    padding=kernel_size // 2,
                )
            )
        self.psis = nn.ModuleList(psis)

    def forward(self, x: torch.Tensor):
        # x: (B,1,T)
        # DTCWT returns lowpass 'yl' and list of highpass complex subbands 'yh' per level
        try:
            yl, x_levels = self.tfm(x)  # preferred call
        except TypeError:
            yl, x_levels = self.tfm.forward(x)  # fallback

        xs = []
        N_min = None
        for j in range(1 + self.J_psi):
            # x_levels[j]: (B, 1, T_j) complex (in pytorch_wavelets, real/imag split via .real/.imag)
            x_level = x_levels[j].to(torch.complex64) / (2**j)

            Wx_real = self.psis[j](x_level.real)
            Wx_imag = self.psis[j](x_level.imag)
            Ux_j = (Wx_real * Wx_real + Wx_imag * Wx_imag).real  # (B, C_j, T'_j)

            N_min = Ux_j.shape[-1] if N_min is None else min(N_min, Ux_j.shape[-1])
            xs.append(Ux_j)

        # Crop all levels to the same time length (shortest)
        if N_min is not None:
            xs = [u[..., :N_min] for u in xs]

        # Fine -> coarse order (j=0 is finest). If you need low->high frequency flip across channels,
        # do it per level: xs[j] = torch.flip(xs[j], dims=(-2,))
        return xs
