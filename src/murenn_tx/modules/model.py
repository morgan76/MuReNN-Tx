from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
import torch.nn as nn
import math

from .frontend import MRFrontEnd, MuReNNFrontEnd
from .tokenizers import ConvTokenizer1D
from collections import Counter

class SinusoidalPE1D(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # (L, D)

    def forward(self, T):  # returns (T, D)
        return self.pe[:T]

class SinusoidalPE2D(nn.Module):
    def __init__(self, d_model: int, n_scales_max: int, max_len: int):
        super().__init__()
        assert d_model % 2 == 0
        self.time = SinusoidalPE1D(d_model // 2, max_len)
        self.scale = SinusoidalPE1D(d_model // 2, n_scales_max)

    def forward(self, s: int, T: int):
        pe_t = self.time(T)                 # (T, D/2)
        pe_s = self.scale(s+1)[-1].expand(T, -1)  # take row 's' -> (T, D/2)
        return torch.cat([pe_t, pe_s], dim=-1)    # (T, D)




@dataclass
class MuReNNTxConfig:
    sample_rate: int = 16000
    n_scales: int = 3
    base_channels: int = 32
    d_model: int = 256
    hop: int = 4
    local_window: int = 64
    nhead: int = 4
    ff_mult: int = 4
    depth_per_scale: int = 2
    global_depth: int = 2
    n_classes: int = 50
    frontend: Any = None
    local_posenc: str = "none"
    cross_band: int | None = 1
    cross_q_stride: int = 1
    scale_dropout_p: float = .1


class MuReNNTx(nn.Module):
    def __init__(self, cfg: MuReNNTxConfig):
        super().__init__()
        self.cfg = cfg
        fe_cfg = getattr(self.cfg, "frontend", None)
        if fe_cfg and (
            getattr(fe_cfg, "name", None) == "murenn"
            or (isinstance(fe_cfg, dict) and fe_cfg.get("name") == "murenn")
        ):
            self.fe = MuReNNFrontEnd(
                base_channels=self.cfg.base_channels,
                n_scales=self.cfg.n_scales,
                stride=getattr(fe_cfg, "stride", 64),
                octaves=getattr(fe_cfg, "octaves", [0, 1, 1, 2, 2, 2]),
                Q_multiplier=getattr(fe_cfg, "Q_multiplier", 16),
                include_scale=getattr(fe_cfg, "include_scale", False),
            )
        else:
            self.fe = MRFrontEnd(
                in_channels=1, base_channels=cfg.base_channels, n_scales=cfg.n_scales
            )
        #self.tokenizers = nn.ModuleList()
        self.locals = nn.ModuleList()
        octaves=getattr(fe_cfg, "octaves")
        octaves = Counter(octaves)
        assert len(octaves) == cfg.n_scales
        for s in range(cfg.n_scales):
            in_ch = octaves[s]
            #self.tokenizers.append(
            #    ConvTokenizer1D(in_ch, cfg.d_model, kernel=7, hop=cfg.hop * (2**s))
            #    ConvTokenizer1D(self.cfg.base_channels, cfg.d_model, kernel=7, hop=cfg.hop * (2**s))
            #)
            self.locals.append(
                nn.Sequential(
                    *[
                        nn.TransformerEncoderLayer(
                            d_model=cfg.d_model,
                            nhead=cfg.nhead,
                            dim_feedforward=cfg.ff_mult * cfg.d_model,
                            dropout=0.1,
                            batch_first=True,
                            norm_first=True,
                            activation="gelu",
                        )
                        for _ in range(cfg.depth_per_scale)
                    ]
                )
            )
        #self.global_blocks = nn.Sequential(
        #    *[
        #        nn.TransformerEncoderLayer(
        #            d_model=cfg.d_model,
        #            nhead=cfg.nhead,
        #            dim_feedforward=cfg.ff_mult * cfg.d_model,
        #            dropout=0.1,
        #            batch_first=True,
        #            norm_first=True,
        #            activation="gelu",
        #        )
        #        for _ in range(cfg.global_depth)
        #    ]
        #)

        self.in_norms = nn.ModuleList([nn.LayerNorm(cfg.d_model) for _ in range(cfg.n_scales)])
        self.out_norms = nn.ModuleList([nn.LayerNorm(cfg.d_model) for _ in range(cfg.n_scales)])

        self.local_cls = nn.Parameter(torch.zeros(1,cfg.n_scales,cfg.d_model))
        nn.init.normal_(self.local_cls, std=0.02)
        self.cls = nn.Parameter(torch.zeros(1,1,cfg.d_model))
        nn.init.normal_(self.cls, std=0.02)
        #self.head_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model * cfg.n_scales, cfg.n_classes)
        self.time_pe = SinusoidalPE1D(cfg.d_model, max_len=200_000)


    def forward(self, x: torch.Tensor):
        aux = {}

        pyramid = self.fe(x)
        #print(len(pyramid), self.cfg.n_scales)
        assert len(pyramid) == self.cfg.n_scales

        seqs = []
        for s, x_s in enumerate(pyramid):
            #print('x_s.shape', x_s.shape)
            #tok = self.tokenizers[s](x_s)
            #print("tok.shape", tok.shape)
            tok = x_s.permute(0, 2, 1)
            tok = self.in_norms[s](tok)
            B, T_s, D = tok.shape
            pe_t = self.time_pe(T_s).to(tok.device, tok.dtype)        
            tok = tok + pe_t

            local_cls = self.local_cls[:,s,:].expand(tok.size(0), 1, -1)
            x_cls_local = torch.cat([local_cls, tok], dim=1)
            tok = self.locals[s](x_cls_local)

            cls = tok[:, 0, :] + tok[:, 1:, :].mean(1)
            scale_vec = self.out_norms[s](cls)
            seqs.append(scale_vec)
        
        fused = torch.stack(seqs, dim=1) 

        B = fused.size(0)
        #xg = torch.cat([self.cls.expand(B, 1, -1), fused], dim=1)
        #glob = self.global_blocks(xg)
        #logits = self.head(self.head_norm(glob[:, 0]))
        fused = fused.reshape(B, -1)
        logits = self.head(fused)

        return logits