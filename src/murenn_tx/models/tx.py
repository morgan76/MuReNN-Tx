import torch
import torch.nn as nn
from .registry import register_model
from murenn_tx.modules.frontend import MuReNNFrontEnd, MRFrontEnd
from murenn_tx.utils.pe import SinusoidalPE1D
from murenn_tx.modules.tokenizers import ConvTokenizer1D
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

        # ---------- Front-end ----------
        fe_cfg = _get(cfg, "frontend", None)
        if fe_cfg and _get(fe_cfg, "name", None) == "murenn":
            # NOTE: pass the *list* of octave indices to FE (FE will count them internally)
            self.fe = MuReNNFrontEnd(
                base_channels=cfg.base_channels,
                n_scales=cfg.n_scales,
                stride=_get(fe_cfg, "stride", 512),  # mel-like hop @44.1k
                octaves=_get(fe_cfg, "octaves", [0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]),
                Q_multiplier=_get(fe_cfg, "Q_multiplier", 16),
                include_scale=_get(fe_cfg, "include_scale", False),
            )
        else:
            self.fe = MRFrontEnd(in_channels=1, base_channels=cfg.base_channels, n_scales=cfg.n_scales)

        S = cfg.n_scales
        d = cfg.d_model
        nhead = cfg.nhead
        ff = cfg.ff_mult * d
        drop = _get(cfg, "dropout", 0.1)
        depth_local = _get(cfg, "depth_per_scale", 1)
        depth_cross = _get(cfg, "depth_cross", 1)

        # ---------- Tokenizers (per scale) ----------
        # Expect FE outputs [B, C_s, T] → ConvTokenizer1D → [B, T, d]
        from murenn_tx.modules.tokenizers import ConvTokenizer1D  # your class
        self.tokenizers = nn.ModuleList([
            ConvTokenizer1D(in_ch=cfg.base_channels, d_model=d, hop=8)
            for _ in range(S)
        ])

        # ---------- Temporal encoders (stage A) : per scale ----------
        enc_layer = lambda: nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=ff,
            dropout=drop, batch_first=True, norm_first=True, activation="gelu"
        )
        #self.temporal_A = nn.ModuleList([
        #    nn.TransformerEncoder(enc_layer(), num_layers=depth_local) for _ in range(S)
        #])

        # ---------- Cross-scale encoder (stage B) : per time ----------
        self.cross_encoder = nn.TransformerEncoder(enc_layer(), num_layers=depth_cross)

        # ---------- Temporal encoders (stage C) : per scale (again) ----------
        self.temporal_C = nn.ModuleList([
            nn.TransformerEncoder(enc_layer(), num_layers=depth_local) for _ in range(S)
        ])

        # ---------- Norms & positional encodings ----------
        self.in_norms  = nn.ModuleList([nn.LayerNorm(d) for _ in range(S)])
        self.mid_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(S)])   # before stage C
        self.out_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(S)])

        self.time_pe  = SinusoidalPE1D(d, max_len=200_000)
        self.scale_pe = SinusoidalPE1D(d, max_len=S)  # PE over scale index

        # CLS per scale for the final temporal pass (stage C)
        self.local_cls = nn.Parameter(torch.zeros(1, S, d))
        nn.init.normal_(self.local_cls, std=0.02)

        # ---------- Head ----------
        self.head = nn.Sequential(
            nn.Linear(d * S, d),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d, cfg.n_classes)
        )

    def forward(self, x):
        """
        x: [B, 1, T]
        """
        pyramid = self.fe(x)  # list of [B, C_s, T_s]
        assert len(pyramid) == self.cfg.n_scales
        S = self.cfg.n_scales
        d = self.cfg.d_model
        B = x.size(0)

        # Align time across scales
        T = min(t.shape[-1] for t in pyramid)
        pyramid = [t[..., :T] for t in pyramid]

        # ---------- Stage A: per-scale tokenization + temporal encoder ----------
        tokens_A = []  # list of [B, T, d]
        for s in range(S):
            
            x_s = pyramid[s]                      # [B, C_s, T]
            #print("x_s.shape", x_s.shape)
            tok  = self.tokenizers[s](x_s)        # [B, T, d]
            #print("tok.shape", tok.shape)
            tok  = self.in_norms[s](tok)
            tok  = tok + self.time_pe(tok.shape[1]).to(tok.device, tok.dtype)
            #tok  = self.temporal_A[s](tok)        # [B, T, d]
            tokens_A.append(tok)

        # Stack into [B, S, T, d]
        X = torch.stack(tokens_A, dim=1)
        T = X.size(2)
        # ---------- Stage B: cross-scale per time ----------
        # reshape to (B*T, S, d), add scale PE, run cross encoder, reshape back
        BT = B * T
        X_bt = X.permute(0, 2, 1, 3).reshape(BT, S, d)              # [B*T, S, d]
        X_bt = X_bt + self.scale_pe(S).to(X_bt.device, X_bt.dtype)  # add scale PE
        X_bt = self.cross_encoder(X_bt)                              # [B*T, S, d]
        X = X_bt.reshape(B, T, S, d).permute(0, 2, 1, 3).contiguous()# [B, S, T, d]

        # ---------- Stage C: per-scale temporal encoder (with CLS) ----------
        seqs = []
        for s in range(S):
            tok = X[:, s, :, :]                         # [B, T, d]
            tok = self.mid_norms[s](tok)
            tok = tok + self.time_pe(tok.shape[1]).to(tok.device, tok.dtype)

            # prepend a per-scale CLS
            cls_s = self.local_cls[:, s, :].expand(B, 1, d)  # [B,1,d]
            tok = torch.cat([cls_s, tok], dim=1)             # [B, 1+T, d]
            tok = self.temporal_C[s](tok)                    # [B, 1+T, d]
            cls = tok[:, 0, :]                               # [B, d]
            seqs.append(self.out_norms[s](cls))

        fused = torch.cat(seqs, dim=-1)  # [B, d*S]
        return self.head(fused)

@register_model("tx")
def _build_tx(cfg):
    return MuReNNTx(cfg)