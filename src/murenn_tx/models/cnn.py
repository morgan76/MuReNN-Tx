import torch
import torch.nn as nn
from .registry import register_model
from murenn_tx.modules.frontend import MuReNNFrontEnd, MRFrontEnd

def conv_block(in_ch, out_ch, k=3, s=1, p=1, gn_groups=8):
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.GroupNorm(num_groups=min(gn_groups, out_ch), num_channels=out_ch),
        nn.GELU()
    )

class ScaleCNN(nn.Module):
    """A per-scale temporal CNN with downsampling + global pooling."""
    def __init__(self, in_ch, width=128, depth=3, pool_stride=2):
        super().__init__()
        layers = []
        ch = in_ch
        for d in range(depth):
            layers += [
                conv_block(ch, width, k=5, s=1, p=2),
                conv_block(width, width, k=3, s=1, p=1),
            ]
            # temporal downsample to build some invariance
            layers += [nn.Conv1d(width, width, kernel_size=3, stride=pool_stride, padding=1, bias=False)]
            ch = width
        self.net = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # [B, C, 1]
            nn.Flatten(1)             # [B, C]
        )

    def forward(self, x):  # x: [B, C, T]
        z = self.net(x)
        return self.head(z)  # [B, C]

class MuReNNCNN(nn.Module):
    """
    CNN baseline using the same MuReNN frontend.
    Per scale: a small temporal CNN -> pooled vector.
    Then fuse across scales and classify.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        fe_cfg = getattr(cfg, "frontend", None)
        if fe_cfg and getattr(fe_cfg, "name", None) == "murenn":
            self.fe = MuReNNFrontEnd(
                base_channels=cfg.base_channels,
                n_scales=cfg.n_scales,
                stride=getattr(fe_cfg, "stride", 64),
                octaves=getattr(fe_cfg, "octaves", [0,1,1,2,2,2]),
                Q_multiplier=getattr(fe_cfg, "Q_multiplier", 16),
                include_scale=getattr(fe_cfg, "include_scale", False),
            )
        else:
            self.fe = MRFrontEnd(in_channels=1, base_channels=cfg.base_channels, n_scales=cfg.n_scales)

        # assume FE outputs per-scale channel count == cfg.d_model (or map to it)
        self.per_scale = nn.ModuleList([
            ScaleCNN(in_ch=cfg.base_channels, width=cfg.d_model,
                     depth=getattr(cfg, "cnn_depth", 3),
                     pool_stride=getattr(cfg, "cnn_pool_stride", 2))
            for _ in range(cfg.n_scales)
        ])

        self.fuse = nn.Sequential(
            nn.LayerNorm(cfg.d_model * cfg.n_scales),
            nn.Linear(cfg.d_model * cfg.n_scales, cfg.d_model),
            nn.GELU(),
        )
        self.head = nn.Linear(cfg.d_model, cfg.n_classes)

        # optional: project FE channels per scale to d_model if needed
        proj_needed = getattr(cfg, "fe_out_to_d_model", False)
        if proj_needed:
            self.projects = nn.ModuleList([
                nn.Conv1d(in_channels=cfg.base_channels,
                          out_channels=cfg.d_model, kernel_size=1)
                for _ in range(cfg.n_scales)
            ])
        else:
            self.projects = None

    def forward(self, x):
        pyramid = self.fe(x)  # list of [B, C_s, T_s]
        assert len(pyramid) == self.cfg.n_scales

        pooled = []
        for s, x_s in enumerate(pyramid):
            if self.projects is not None:
                x_s = self.projects[s](x_s)
            per_scale = self.per_scale[s](x_s)
            pooled.append(per_scale)  # [B, d_model]
        
        h = torch.cat(pooled, dim=1)              # [B, d_model * n_scales]
        
        h = self.fuse(h)                          # [B, d_model]
        return self.head(h)                       # [B, n_classes]

@register_model("cnn")
def _build_cnn(cfg):
    return MuReNNCNN(cfg)