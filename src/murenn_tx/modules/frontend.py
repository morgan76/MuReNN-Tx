from __future__ import annotations

from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from murenn import DTCWT as DTCWTForward 
import numpy as np
import math

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
        sample_rate: int = 44100,
        kernel_size: int = 129,
        fmin: float = 30.0,
        fmax_ratio: float = 0.45,
        kappa: float = 2.5,
        seconds: float = 5.0,
        min_kernel: int | None = None,
        max_kernel: int | None = None,
        **kwargs,   
    ):
        super().__init__()
        assert in_channels == 1, "MuReNNFrontEnd expects mono input"

        self.stride = kwargs.get("stride", 64)
        Q_multiplier = kwargs.get("Q_multiplier", 16)
        octaves = kwargs.get("octaves")

        self.octaves = octaves
        Q_ctr = Counter(octaves)
        self.J_psi = max(Q_ctr)

        assert min(octaves) == 0, "octave indices must start at 0"

        self.sr = sample_rate
        self.fmax = fmax_ratio * sample_rate
        self.fmin = fmin
        self.kappa = kappa
        self.seconds = seconds

        # scale min/max kernel with sr if not provided
        min_kernel = 8
        max_kernel = 256

        if min_kernel is None:
            self.L_MIN = max(15, round(32 * sample_rate / 44100))   # ~1.45 ms @44.1k → ~64; ~23 @16k→32–34, bump to ≥15
        else:
            self.L_MIN = int(min_kernel)
        if max_kernel is None:
            self.L_MAX = round(4096 * sample_rate / 44100)          # ~93 ms @44.1k; ~1488 @16k
        else:
            self.L_MAX = int(max_kernel)

        try:
            self.tfm = DTCWTForward(J=1+self.J_psi, alternate_gh=True, include_scale=False)
            
        except TypeError:
            self.tfm = DTCWTForward(J=1+self.J_psi, include_scale=False)

        psis = []
        self.out_ch_per_scale = []
        self.kernel_sizes = []
        self.fc_centers = []  # for debugging/inspection

        def octave_edges(j):
            f_hi = self.fmax / (2 ** j)
            f_lo = max(self.fmin, self.fmax / (2 ** (j+1)))
            return f_lo, f_hi


        for j in range(1+self.J_psi):
            # kernel size grows with octave (and multiplier); enforce odd size

            Qj = Q_ctr[j]                         # bands in this octave
            f_lo, f_hi = octave_edges(j)
            centers = np.geomspace(max(1e-6, f_lo), f_hi, Qj+1)[1:] if Qj > 0 else []
            centers = centers.tolist()

            if len(centers) == 0:
                # degenerate (shouldn’t happen if Q_ctr[j]>0)
                fc = (f_lo + f_hi) * 0.5
            else:
                # geometric mean is natural on log-frequency grids
                fc = math.exp(sum(math.log(c) for c in centers) / len(centers))

            sr_j = self.sr / (2 ** j)
            L_j = int(round(kappa * self.sr / max(1.0, fc)))

            T0  = int(math.ceil(self.seconds * self.sr))          # full-rate samples for a clip
            T_j = max(1, int(math.ceil(T0 / (2 ** j))))           # expected length at scale j
            #alpha = 0.33
            alpha = 0.25
            L_j = min(max(min_kernel, min(L_j, 4096)), int(alpha * T_j))  # clamp [64,4096] and ≤ α·T_j
            if L_j % 2 == 0:
                L_j += 1
            #kernel_size = Q_multiplier*Q_ctr[j]
            kernel_size = L_j
            

            #if j == 0:
            #    stride_j = self.stride
            #else:
            #    stride_j = max(1, self.stride // (2**(j-1)))
            #stride_j = self.stride
            stride_j = max(1, self.stride // 2**j)

            print('Scale s =', j)
            print("-kernel size =", kernel_size)
            print("-stride_j =", stride_j)
            print("-out channels =", base_channels)
            print("-fc =", fc)
            print('sample rate =', self.sr)
            print('----')
            

            psi = torch.nn.Conv1d(
                in_channels=1,
                #out_channels=Q_ctr[j],
                out_channels=base_channels,
                #out_channels=channels[j],
                kernel_size=kernel_size,
                stride=stride_j,
                bias=False,
                padding=kernel_size//2)
            
            psis.append(psi)

        # ----- FIXED: use ModuleList for modules -----
        self.psis = nn.ModuleList(psis)

        self.norms = nn.ModuleList([
            nn.GroupNorm(num_groups=min(8, self.psis[j].out_channels),
                        num_channels=self.psis[j].out_channels)
            for j in range(1 + self.J_psi)
        ])

      
    def forward(self, x: torch.Tensor):
        # x: (B,1,T)
        yl, x_levels = self.tfm(x)  # list/tuple of per-octave signals

        #print([i.shape for i in x_levels])
        Ux = []
        N_j = None

        for j_psi in range(1+self.J_psi):
            x_level = x_levels[j_psi].type(torch.complex64) #/ (2**j_psi)
            #print('Scale ', j_psi)
            #print('x_level.shape =', x_level.shape)
            #print(f"[DBG] lvl{j_psi}_amax:", x_level.abs().max().item())
            Wx_real = self.psis[j_psi](x_level.real)
            Wx_imag = self.psis[j_psi](x_level.imag)
            #print(Wx_real.shape, Wx_imag.shape)
            Ux_j = Wx_real * Wx_real + Wx_imag * Wx_imag
            Ux_j = torch.real(Ux_j)
            #print("Ux_j.shape", Ux_j.shape)
            Ux_j = torch.log1p(1e4 * Ux_j)
            Ux_j = self.norms[j_psi](Ux_j)
            #print(Ux_j.min(), Ux_j.max())
            #Ux_j = torch.nn.functional.gelu(Ux_j)

            if j_psi == 0:
                N_j = Ux_j.shape[-1]
            else:
                Ux_j = Ux_j[:, :, :N_j]
            #print('---')

            #print(Ux_j.mean())
            #print(Ux_j.std())
            #print(Ux_j.abs().max())
            
            Ux.append(Ux_j)

        # Return list of per-scale tensors (B, C, T) as expected downstream
        return Ux
    