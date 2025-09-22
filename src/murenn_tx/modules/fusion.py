from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

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
                # scale = T_fine // Ts
                x_t = xs.transpose(1, 2)
                # x_t = AntiAliasedUpsample1d(scale).to(x_t.device)(x_t)[:, :, :T_fine]
                x_t = F.interpolate(x_t, size=T_fine, mode="linear", align_corners=False)
                aligned.append(x_t.transpose(1, 2))
        h = torch.cat(aligned, dim=-1)
        return self.proj(h)
