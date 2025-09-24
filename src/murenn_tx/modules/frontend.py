from __future__ import annotations

from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from murenn import DTCWT as DTCWTForward 

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
    MuReNN front-end that matches the WASPAA implementation style:
    - 1D DTCWT pyramid
    - per-scale learnable complex convs (psis) on real/imag parts
    - concatenates energy maps across scales
    """
    def __init__(self, 
        in_channels: int = 1,
        base_channels: int = 32,
        n_scales: int = 3,
        sample_rate: int = 16000,
        kernel_size: int = 129,
        fmin: float = 30.0,
        fmax_ratio: float = 0.45,
        **kwargs,   
    ):
        super().__init__()
        assert in_channels == 1, "MuReNNFrontEnd expects mono input"

        self.stride = kwargs.get("stride", 64)
        Q_multiplier = kwargs.get("Q_multiplier", 16)
        octaves = kwargs.get("octaves")
        if octaves is None:
            # fallback: contiguous 0..n_scales-1
            octaves = list(range(n_scales))
        else:
            octaves = list(map(int, octaves))

        # ----- FIXED: proper scale/octave bookkeeping -----
        self.octaves = octaves
        self.n_scales = len(octaves)                       # number of scales
        self.J_psi = max(octaves)                          # highest octave index
        self.n_octaves = self.J_psi + 1                    # number of octaves

        # (Optional) sanity checks
        assert min(octaves) == 0, "octave indices must start at 0"
        assert set(octaves) == set(range(self.n_octaves)), "octave indices must be contiguous"

        # ----- DTCWT pyramid depth uses #octaves, not #scales -----
        try:
            self.tfm = DTCWTForward(J=self.n_octaves, include_scale=False, alternate_gh=True)
        except TypeError:
            self.tfm = DTCWTForward(J=self.n_octaves, include_scale=False)

        # ----- Per-scale settings -----
        # Define a per-scale Q factor from the octave index.
        # You can change this mapping; 2**octave is a reasonable default.
        Q_per_scale = [2 ** o for o in octaves]            # length = n_scales

        psis = []
        for j in range(self.n_scales):
            # kernel size grows with octave (and multiplier); enforce odd size
            k = int(Q_multiplier * Q_per_scale[j])
            if k < 3:
                k = 3
            if k % 2 == 0:
                k += 1

            # Choose stride per scale to roughly equalize time lengths across scales.
            # Note: DTCWT reduces temporal resolution by ~2^octave;
            # using a smaller stride at coarser scales helps keep T_j similar.
            o_j = octaves[j]
            stride_j = self.stride if o_j == 0 else max(1, self.stride // (2 ** (o_j - 1)))

            psi = nn.Conv1d(
                in_channels=1,
                out_channels=base_channels,   # channels per scale
                kernel_size=k,
                stride=stride_j,
                bias=False,
                padding=k // 2,
            )
            psis.append(psi)

        # ----- FIXED: use ModuleList for modules -----
        self.psis = nn.ModuleList(psis)

    def forward(self, x: torch.Tensor):
        # x: (B,1,T)
        yl, x_levels = self.tfm(x)  # list/tuple of per-octave signals

        Ux = []
        N_j = None
        # We have n_octaves DTCWT levels; map them to n_scales in the order of 'octaves'
        # If your DTCWT returns levels indexed by octave directly (0..J_psi),
        # you can iterate per scale using its octave tag.
        for j in range(self.n_scales):
            o_j = j if self.n_scales == self.n_octaves else None
            # If your 'octaves' list maps scale->octave, use it to pick the level:
            o_j = self.octaves[j]  # <-- uses the closure/octaves from __init__

            # Get the level for this octave; adapt this indexing to your DTCWT wrapper
            x_level = x_levels[o_j].to(dtype=torch.complex64) / (2 ** o_j)  # (B,1,Tj) complex

            Wx_real = self.psis[j](x_level.real)  # (B,C,T')
            Wx_imag = self.psis[j](x_level.imag)  # (B,C,T')
            Ux_j = Wx_real * Wx_real + Wx_imag * Wx_imag  # magnitude^2
            Ux_j = torch.real(Ux_j)

            if j == 0:
                N_j = Ux_j.shape[-1]
            else:
                Ux_j = Ux_j[:, :, :N_j]  # align time length

            Ux.append(Ux_j)

        # Return list of per-scale tensors (B, C, T) as expected downstream
        return Ux