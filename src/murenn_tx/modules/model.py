from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .frontend import MRFrontEnd, MuReNNFrontEnd
from .fusion import CrossScaleFusion
from .local_transformer import LocalTransformerBlock
from .tokenizers import ConvTokenizer1D


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
        self.fuse = CrossScaleFusion(d_model=cfg.d_model, n_scales=cfg.n_scales)
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
        self.head_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.n_classes)

    def forward(self, x: torch.Tensor):
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
        print([i.shape for i in seqs])
        fused = self.fuse(seqs)
        print('fused.shape =', fused.shape)
        glob = self.global_blocks(fused)
        print('glob.shape =', glob.shape)
        logits = self.head(self.head_norm(glob).mean(dim=1))
        print('############## \n')
        return logits
