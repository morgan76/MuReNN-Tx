from __future__ import annotations
import torch
import torch.nn as nn
import torchaudio

from murenn_tx.models.registry import register_model

def _maybe(value, default):
    return default if value is None else value

class SpecAugment2D(nn.Module):
    """Lightweight SpecAug: applies in training mode only."""
    def __init__(self, freq_mask_param=12, time_mask_param=24, p=0.5):
        super().__init__()
        self.fm = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.tm = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
        self.p = p

    def forward(self, x):  # x: [B, 1, M, T] (log-mel)
        if not self.training or self.p <= 0:
            return x
        if torch.rand(()) < self.p:
            # torchaudio expects [B, M, T]
            b, c, m, t = x.shape
            x_ = x.reshape(b * c, m, t)
            x_ = self.fm(x_)
            x_ = self.tm(x_)
            x = x_.reshape(b, c, m, t)
        return x


def conv2d_blk(c_in, c_out, k=(3,3), s=(1,1), p=(1,1), norm="bn"):
    layers = [nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=False)]
    if norm == "bn":
        layers.append(nn.BatchNorm2d(c_out))
    elif norm == "gn":
        layers.append(nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out))
    layers.append(nn.GELU())
    return nn.Sequential(*layers)


class MelCNN(nn.Module):
    """
    Log-Mel frontend + small 2D CNN (VGG-ish) + global pooling → linear head.
    Works on raw waveform [B, 1, T].
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # --- Mel params (favor YAML if provided; else pick sane defaults by sample rate) ---
        d = getattr(cfg, "data", None)
        sr = getattr(d, "sample_rate", 16000) if d is not None else 16000
        mel = getattr(cfg, "mel", None)

        n_mels      = _maybe(getattr(mel, "n_mels", None),        64)
        n_fft       = _maybe(getattr(mel, "n_fft", None),         1024 if sr < 32000 else 2048)
        hop_length  = _maybe(getattr(mel, "hop_length", None),    160  if sr == 16000 else 512)
        win_length  = _maybe(getattr(mel, "win_length", None),    None)
        f_min       = _maybe(getattr(mel, "f_min", None),         20.0)
        f_max       = _maybe(getattr(mel, "f_max", None),         sr / 2.0)
        power       = _maybe(getattr(mel, "power", None),         2.0)     # power mel
        top_db      = _maybe(getattr(mel, "top_db", None),        80.0)
        center      = _maybe(getattr(mel, "center", None),        True)
        pad_mode    = _maybe(getattr(mel, "pad_mode", None),      "reflect")

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            f_min=f_min, f_max=f_max, n_mels=n_mels, power=power, center=center, pad_mode=pad_mode,
            norm=None, mel_scale="htk"
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=top_db)

        # Optional SpecAugment (train only)
        aug_p         = _maybe(getattr(mel, "specaug_p", None),         0.5)
        freq_mask     = _maybe(getattr(mel, "freq_mask_param", None),   12)
        time_mask     = _maybe(getattr(mel, "time_mask_param", None),   24)
        self.specaug  = SpecAugment2D(freq_mask, time_mask, p=aug_p)

        # Simple loudness/RMS normalization toggle (applied before STFT)
        self.rms_target = _maybe(getattr(mel, "rms_target", None), 0.1)
        self.rms_norm   = _maybe(getattr(mel, "rms_norm", None),   False)

        # --- 2D CNN trunk ---
        # Input to CNN: [B, 1, n_mels, T']
        width = _maybe(getattr(cfg, "cnn_width", None), 64)
        self.cnn = nn.Sequential(
            conv2d_blk(1,   width, norm="bn"),       # -> [B, W, M, T]
            conv2d_blk(width, width, norm="bn"),
            nn.MaxPool2d((2, 2)),                    # downsample both (M, T)

            conv2d_blk(width, width*2, norm="bn"),   # -> [B, 2W, M/2, T/2]
            conv2d_blk(width*2, width*2, norm="bn"),
            nn.MaxPool2d((2, 2)),

            conv2d_blk(width*2, width*4, norm="bn"), # -> [B, 4W, M/4, T/4]
            conv2d_blk(width*4, width*4, norm="bn"),
            nn.MaxPool2d((2, 2)),
        )

        # Global pooling over (mel,time) and small head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        d_model = _maybe(getattr(cfg, "d_model", None), width*4)
        self.proj = nn.Sequential(
            nn.Flatten(1),                           # [B, 4W]
            nn.LayerNorm(width*4),
            nn.Linear(width*4, d_model),
            nn.GELU(),
            nn.Dropout(p=_maybe(getattr(cfg, "dropout", None), 0.1)),
        )
        self.head = nn.Linear(d_model, cfg.n_classes)

    @torch.no_grad()
    def _rms_norm(self, x):
        # x: [B,1,T]
        if not self.rms_norm:
            return x
        rms = x.pow(2).mean(dim=(1,2), keepdim=True).sqrt().clamp_min(1e-8)
        return x * (self.rms_target / rms)

    def _frontend(self, x):  # x: [B,1,T] -> log-mel as [B,1,M,T']
        # torchaudio transforms expect [B, T] or [B, C, T]?
        # MelSpectrogram handles [B, T] or [B, 1, T] if batch_first=False internally;
        # we'll reshape to [B, T] for simplicity.
        B, C, T = x.shape
        assert C == 1, "MelCNN expects mono waveform [B,1,T]"
        x = x.squeeze(1)                                # [B, T]
        mel = self.melspec(x)                           # [B, M, T']
        mel = self.to_db(mel)                           # log-mel (dB)
        mel = mel.unsqueeze(1)                          # [B, 1, M, T']
        return mel

    def forward(self, x):  # x: [B,1,T]
        x = self._rms_norm(x)
        S = self._frontend(x)           # [B,1,M,T']
        S = self.specaug(S)             # no-op at eval
        z = self.cnn(S)                 # [B, 4W, M', T']
        z = self.pool(z)                # [B, 4W, 1, 1]
        z = self.proj(z)                # [B, d_model]
        return self.head(z)             # [B, n_classes]


@register_model("mel_cnn")
def _build_mel_cnn(cfg):
    return MelCNN(cfg)