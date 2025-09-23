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
        # Same “octaves” logic as in student.py: number of filters per scale
        # Here we make a simple mapping: scale j -> base_channels at j=0 then doubles
        # If you want exact parity with student.py's Counter(octaves), pass a list instead.
        #Q_per_scale = [base_channels * (2 ** j) for j in range(n_scales)]
        Q_per_scale = Counter(octaves)
        #Q_per_scale = [base_channels for j in range(n_scales)]
        #self.J_psi = n_scales - 1
        self.J_psi = max(Q_per_scale)
        n_scales = 1 + self.J_psi 

        # MuReNN’s DTCWT wrapper (1D). Some versions accept `alternate_gh`;
        # guard with try/except to support both.
        try:
            self.tfm = DTCWTForward(J=1 + self.J_psi, include_scale=False, alternate_gh=True)
        except TypeError:
            self.tfm = DTCWTForward(J=1 + self.J_psi, include_scale=False)

        # Per-scale complex conv banks (psis), matching student.py structure
        psis = []
        for j in range(1 + self.J_psi):
            kernel_size = Q_multiplier * Q_per_scale[j]
            stride_j = self.stride if j == 0 else max(1, self.stride // (2 ** (j - 1)))
            #stride_j = max(1, self.stride // (2 ** j))
            psi = nn.Conv1d(
                in_channels=1,
                #out_channels=Q_per_scale[j],
                out_channels=base_channels,
                kernel_size=kernel_size,
                stride=stride_j,
                bias=False,
                padding=kernel_size // 2,
            )
            psis.append(psi)
        self.psis = nn.ParameterList(psis)

    def forward(self, x: torch.Tensor):
        # x: (B,1,T)
        # DTCWT 1D forward
        yl, x_levels = self.tfm(x)  # same call style as student.py

        Ux = []
        N_j = None
        for j in range(1 + self.J_psi):
            # Ensure complex dtype (some backends return real tensors)
            x_level = x_levels[j].to(dtype=torch.complex64) / (2 ** j)  # (B,1,Tj) complex
            Wx_real = self.psis[j](x_level.real)  # (B,Qj,T')
            Wx_imag = self.psis[j](x_level.imag)  # (B,Qj,T')
            Ux_j = Wx_real * Wx_real + Wx_imag * Wx_imag  # magnitude^2
            Ux_j = torch.real(Ux_j)
            if j == 0:
                N_j = Ux_j.shape[-1]
            else:
                Ux_j = Ux_j[:, :, :N_j]  # align time length
            #print(N_j, Ux_j.shape)
            Ux.append(Ux_j)

        #Ux = torch.cat(Ux, dim=1)      # concat scales on channel axis
        #Ux = torch.flip(Ux, dims=(-2,))  # flip so freqs go low->high like the original
        # Return as list of scale maps to keep the rest of your pipeline unchanged:
        # make per-scale tensors by splitting along channels
        # (tokenizers expect list[B, C_s, T]); here we pack them as one scale
        #print("Ux.shape", Ux.shape)
        #return Ux

        return Ux