from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from .frontend import AntiAliasedUpsample1d

__all__ = ["CrossScaleFusion"]


class CrossScaleFusion(nn.Module):
    def __init__(self, d_model: int, n_scales: int):
        super().__init__()
        self.proj = nn.Linear(d_model * n_scales, d_model)

    def forward(self, seqs):
        B = seqs[0].size(0)
        d = seqs[0].size(-1)
        T_fine = seqs[0].size(1)
        aligned = []
        for xs in seqs:
            Ts = xs.size(1)
            if Ts == T_fine:
                aligned.append(xs)
            else:
                x_t = xs.transpose(1, 2)  # (B,d,Ts)
                if T_fine % Ts == 0 and T_fine >= Ts:
                    # exact integer upscale -> antialiased repeat
                    scale = T_fine // Ts
                    x_t = AntiAliasedUpsample1d(scale).to(x_t.device)(x_t)
                else:
                    # fractional (or downsample) -> interpolate to exact length
                    x_t = F.interpolate(x_t, size=T_fine, mode="linear", align_corners=False)
                # final guard to guarantee exact length
                L = x_t.size(-1)
                if L < T_fine:
                    x_t = F.pad(x_t, (0, T_fine - L))
                elif L > T_fine:
                    x_t = x_t[..., :T_fine]
                aligned.append(x_t.transpose(1, 2))

        h = torch.cat(aligned, dim=-1)
        return self.proj(h)
