import torch
import torch.nn as nn
from .registry import register_model
from murenn_tx.modules.frontend import MuReNNFrontEnd, MRFrontEnd
from murenn_tx.utils.pe import SinusoidalPE1D
from collections import Counter


def _get(obj, key, default=None):
    # works for dataclass-like objects and dicts
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class MuReNNTx(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        print(cfg)
        fe_cfg = _get(cfg, "frontend", None)
        from collections import Counter
        if fe_cfg and _get(fe_cfg, "name", None) == "murenn":
            self.fe = MuReNNFrontEnd(
                base_channels=cfg.base_channels,
                n_scales=cfg.n_scales,
                stride=_get(fe_cfg, "stride", 64),
                octaves = Counter(_get(fe_cfg, "octaves", [0,1,1,2,2,2])),
                Q_multiplier=_get(fe_cfg, "Q_multiplier", 16),
                include_scale=_get(fe_cfg, "include_scale", False),
            )
        else:
            self.fe = MRFrontEnd(in_channels=1, base_channels=cfg.base_channels, n_scales=cfg.n_scales)

        self.locals = nn.ModuleList()
        
        octaves = Counter(_get(fe_cfg, "octaves"))
        assert len(octaves) == cfg.n_scales

        for s in range(cfg.n_scales):
            self.locals.append(nn.Sequential(*[
                nn.TransformerEncoderLayer(
                    d_model=cfg.d_model,
                    nhead=cfg.nhead,
                    dim_feedforward=cfg.ff_mult * cfg.d_model,
                    dropout=0.1,
                    batch_first=True,
                    norm_first=True,
                    activation="gelu",
                ) for _ in range(cfg.depth_per_scale)
            ]))

        self.in_norms  = nn.ModuleList([nn.LayerNorm(cfg.d_model) for _ in range(cfg.n_scales)])
        self.out_norms = nn.ModuleList([nn.LayerNorm(cfg.d_model) for _ in range(cfg.n_scales)])

        self.local_cls = nn.Parameter(torch.zeros(1, cfg.n_scales, cfg.d_model))
        nn.init.normal_(self.local_cls, std=0.02)

        self.head = nn.Linear(cfg.d_model * cfg.n_scales, cfg.n_classes)
        self.time_pe = SinusoidalPE1D(cfg.d_model, max_len=200_000)

    def forward(self, x):
        pyramid = self.fe(x)  # list of [B, C_s, T_s]
        assert len(pyramid) == self.cfg.n_scales

        seqs = []
        for s, x_s in enumerate(pyramid):
            tok = x_s.permute(0, 2, 1)              
            tok = self.in_norms[s](tok)
            B, T_s, _ = tok.shape
            tok = tok + self.time_pe(T_s).to(tok.device, tok.dtype)

            local_cls = self.local_cls[:, s, :].expand(B, 1, -1)
            tok = torch.cat([local_cls, tok], dim=1)
            tok = self.locals[s](tok)

            cls = tok[:, 0, :] + tok[:, 1:, :].mean(1)
            seqs.append(self.out_norms[s](cls))

        fused = torch.stack(seqs, dim=1).reshape(x.size(0), -1)
        return self.head(fused)

@register_model("tx")
def _build_tx(cfg):
    return MuReNNTx(cfg)