import torch
import torch.nn as nn
from .registry import register_model
from murenn_tx.modules.frontend import MuReNNFrontEnd

def conv2d_blk(c_in, c_out, k=(3,3), s=(1,1), p=(1,1), norm="bn"):
    layers = [nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=False)]
    if norm == "bn":
        layers.append(nn.BatchNorm2d(c_out))
    elif norm == "gn":
        layers.append(nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out))
    layers.append(nn.GELU())
    return nn.Sequential(*layers)

class SpecAugment2D(nn.Module):
    def __init__(self, freq_mask=12, time_mask=24, p=0.5):
        super().__init__()
        self.freq_mask, self.time_mask, self.p = freq_mask, time_mask, p
    def forward(self, x):  # x: [B,1,F,T]
        if not self.training or torch.rand(()) > self.p: return x
        B, C, F, T = x.shape
        # freq mask
        f = torch.randint(low=0, high=max(1, self.freq_mask+1), size=()).item()
        f0 = torch.randint(0, max(1, F - f + 1), ()).item()
        x[:, :, f0:f0+f, :] = 0
        # time mask
        t = torch.randint(low=0, high=max(1, self.time_mask+1), size=()).item()
        t0 = torch.randint(0, max(1, T - t + 1), ()).item()
        x[:, :, :, t0:t0+t] = 0
        return x

class MuReNN2DCNN(nn.Module):
    """
    MuReNN frontend → stack scales as a 2D 'scalogram' [B,1,F,T] → 2D CNN (MelCNN-like).
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        fe_cfg = getattr(cfg, "frontend", None) or object()
        self.fe = MuReNNFrontEnd(
            base_channels=getattr(cfg, "base_channels", 16),
            n_scales=getattr(cfg, "n_scales", 8),
            stride=getattr(fe_cfg, "stride", 512),      # 44.1k: mel-like hop
            octaves=getattr(fe_cfg, "octaves", [0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]),
            Q_multiplier=getattr(fe_cfg, "Q_multiplier", 16),
            include_scale=getattr(fe_cfg, "include_scale", False),
        )

        # 2D CNN trunk copied from MelCNN (width defaults)
        width = getattr(cfg, "cnn_width", 64)
        self.cnn = nn.Sequential(
            conv2d_blk(1,    width, norm="bn"),
            conv2d_blk(width, width, norm="bn"),
            nn.MaxPool2d((2, 2)),
            conv2d_blk(width, width*2, norm="bn"),
            conv2d_blk(width*2, width*2, norm="bn"),
            nn.MaxPool2d((2, 2)),
            conv2d_blk(width*2, width*4, norm="bn"),
            conv2d_blk(width*4, width*4, norm="bn"),
            nn.MaxPool2d((2, 2)),
        )

        self.specaug = SpecAugment2D(
            freq_mask=getattr(getattr(cfg, "mel", None) or object(), "freq_mask_param", 12),
            time_mask=getattr(getattr(cfg, "mel", None) or object(), "time_mask_param", 24),
            p=getattr(getattr(cfg, "mel", None) or object(), "specaug_p", 0.5),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        d_model = getattr(cfg, "d_model", width*4)
        self.proj = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(width*4),
            nn.Linear(width*4, d_model),
            nn.GELU(),
            nn.Dropout(p=getattr(cfg, "dropout", 0.1)),
        )
        self.head = nn.Linear(d_model, cfg.n_classes)

    def _stack_pyramid(self, pyramid, order="high2low"):
        """
        pyramid: list of [B, C_j, T_j]
        returns: [B,1,F,T] with F=sum_j C_j and all T equal (crop to min)
        """
        # Align time (crop to min length)
        T = min(x.shape[-1] for x in pyramid)
        bands = []
        seq = range(len(pyramid)) if order=="high2low" else reversed(range(len(pyramid)))
        for j in seq:
            xj = pyramid[j][..., :T]    # [B,Cj,T]
            bands.append(xj)
        S = torch.cat(bands, dim=1)     # [B, F, T]
        return S.unsqueeze(1)           # [B, 1, F, T]

    def forward(self, x):  # x: [B,1,T]
        pyr = self.fe(x)                       # list of [B, Cj, Tj]
        #print('pyr shapes =', [i.shape for i in pyr])
        S = self._stack_pyramid(pyr, "high2low")

        #T_list = [u.shape[-1] for u in pyr]
        #C_list = [u.shape[1]  for u in pyr]
        #m = [u.mean().item() for u in pyr]
        #v = [u.var().item()  for u in pyr]

        #print("Frames per scale:", T_list)        # should be ~equal after (A)
        #print("Channels per scale:", C_list)      # matches Q_ctr[j]
        #print("Mean,var per scale:", list(zip(m,v)))
        #print("S.shape", S.shape)
        # optional: normalize per-frequency band here if your FE doesn't (GroupNorm 1d over F)
        #if self.training:
        #    S = self.specaug(S)
        z = self.cnn(S)                        # [B, 4W, F', T']
        #print('z cnn shape =', z.shape)
        z = self.pool(z)                       # [B, 4W, 1, 1]
        #print('pool shape =', z.shape)
        z = self.proj(z)                       # [B, d_model]
        #print('proj shape =', z.shape)
        return self.head(z)

@register_model("murenn2dcnn")
def _build_murenn2dcnn(cfg):
    return MuReNN2DCNN(cfg)