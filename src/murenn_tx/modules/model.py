from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import math

from .frontend import MRFrontEnd, MuReNNFrontEnd
from .fusion import CrossScaleFusion
from .cross_scale import UpwardCrossScaleFusion
from .local_transformer import LocalTransformerBlock
from .tokenizers import ConvTokenizer1D


@dataclass
class MuReNNTxConfig_:
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


class MuReNNTx_(nn.Module):
    def __init__(self, cfg: MuReNNTxConfig_):
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
                # alternate_gh=getattr(fe_cfg, "alternate_gh", True),
            )
        else:
            self.fe = MRFrontEnd(
                in_channels=1, base_channels=cfg.base_channels, n_scales=cfg.n_scales
            )
        self.tokenizers = nn.ModuleList()
        self.locals = nn.ModuleList()
        for s in range(cfg.n_scales):
            in_ch = cfg.base_channels #* (2**s)
            self.tokenizers.append(
                ConvTokenizer1D(in_ch, cfg.d_model, kernel=7, hop=cfg.hop * (2**s))
            )
            self.locals.append(
                nn.Sequential(
                    *[
                        LocalTransformerBlock(
                            cfg.d_model,
                            nhead=cfg.nhead,
                            dim_ff=cfg.ff_mult * cfg.d_model,
                            window=cfg.local_window,
                            posenc=getattr(cfg, "local_posenc", "none")
                        )
                        for _ in range(cfg.depth_per_scale)
                    ]
                )
            )
        #self.fuse = CrossScaleFusion(d_model=cfg.d_model, n_scales=cfg.n_scales)
        #self.fuse = UpwardCrossScaleFusion(d_model=cfg.d_model, n_scales=cfg.n_scales, nhead=cfg.nhead, band=cfg.cross_band, q_stride=cfg.cross_q_stride)
        self.global_blocks = nn.Sequential(
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
                for _ in range(cfg.global_depth)
            ]
        )
        self.cls = nn.Parameter(torch.zeros(1,1,cfg.d_model))
        nn.init.normal_(self.cls, std=0.02)
        self.head_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.n_classes)

    def forward_(self, x: torch.Tensor):
        aux = {}
        #print('input shape =', x.shape)
        pyramid = self.fe(x)
        #print('pyramid shape =', [i.shape for i in pyramid])
        seqs = []
        for s, x_s in enumerate(pyramid):
            tok = self.tokenizers[s](x_s)
            #print('tokenized s=', s, 'of shape', tok.shape)
            seqs.append(self.locals[s](tok))
            #print('shape after local transformer =', seqs[-1].shape)
        #print([i.shape for i in seqs])
        fused = self.fuse(seqs)
        #print('fused.shape =', fused.shape)
        glob = self.global_blocks(fused)
        #print('glob.shape =', glob.shape)
        logits = self.head(self.head_norm(glob).mean(dim=1))
        #print('############## \n')
        return logits
    
    def forward(self, x: torch.Tensor):
        aux = {}
        pyramid = self.fe(x)
        seqs = []
        for s, x_s in enumerate(pyramid):
            tok = self.tokenizers[s](x_s)
            seqs.append(self.locals[s](tok))
        fused = self.fuse(seqs)
        B = fused.size(0)
        xg = torch.cat([self.cls.expand(B, 1, -1), fused], dim=1)
        glob = self.global_blocks(xg)
        logits = self.head(self.head_norm(glob[:, 0]))
        return logits

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
        self.tokenizers = nn.ModuleList()
        self.locals = nn.ModuleList()
        for s in range(cfg.n_scales):
            in_ch = cfg.base_channels 
            self.tokenizers.append(
                ConvTokenizer1D(in_ch, cfg.d_model, kernel=7, hop=cfg.hop * (2**s))
            )
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
        self.global_blocks = nn.Sequential(
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
                for _ in range(cfg.global_depth)
            ]
        )
        self.local_cls = nn.Parameter(torch.zeros(1,cfg.n_scales,cfg.d_model))
        nn.init.normal_(self.local_cls, std=0.02)
        self.cls = nn.Parameter(torch.zeros(1,1,cfg.d_model))
        nn.init.normal_(self.cls, std=0.02)
        self.head_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.n_classes)
        d = cfg.d_model
        d_t = d // 2
        d_s = d - d_t  # handle odd d_model robustly

        self.time_pe = SinusoidalPE1D(d_t, max_len=200_000)       # in-scale time
        self.scale_pe = SinusoidalPE1D(d_s, max_len=cfg.n_scales)

    def forward(self, x: torch.Tensor):
        aux = {}
        #print('input shape =', x.shape)
        pyramid = self.fe(x)
        #print('pyramid shape =', [i.shape for i in pyramid])
        seqs = []
        for s, x_s in enumerate(pyramid):
            tok = self.tokenizers[s](x_s)
            B, T_s, D = tok.shape
            device = tok.device
            dtype = tok.dtype
            pe_t = self.time_pe(T_s).to(device=device, dtype=dtype)
            pe_s_row = self.scale_pe(self.cfg.n_scales)[s].to(device=device, dtype=dtype)  # (d_s,)
            pe_s = pe_s_row.unsqueeze(0).expand(T_s, -1)  # (T_s, d_s)
            pos = torch.cat([pe_t, pe_s], dim=-1)
            tok = tok + pos.unsqueeze(0)

        #    print('tokenized s=', s, 'of shape', tok.shape)
        #    print('local cls shape', self.local_cls.shape)
            local_cls = self.local_cls[:,s,:].expand(tok.size(0), 1, -1)
        #    print('local cls expanded shape', local_cls.shape)
            x_cls_local = torch.cat([local_cls, tok], dim=1)
        #    print('x_cls_local shape', x_cls_local.shape)
            tok = self.locals[s](x_cls_local)
            seqs.append(tok[:, 0, :])
        #    print('shape after local transformer =', seqs[-1].shape)
        #print([i.shape for i in seqs])
        #fused = self.fuse(seqs)
        fused = torch.stack(seqs, dim=1) 
        #print('fused.shape =', fused.shape)
        B = fused.size(0)
        xg = torch.cat([self.cls.expand(B, 1, -1), fused], dim=1)
        #print('xg.shape =', xg.shape)
        glob = self.global_blocks(xg)
        #print('glob.shape =', glob.shape)
        logits = self.head(self.head_norm(glob[:, 0]))
        return logits
