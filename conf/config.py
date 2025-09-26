from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class FrontendCfg:
    name: str = "murenn"
    stride: int = 64
    octaves: List[int] = field(default_factory=lambda: [0,1,1,2,2,2])
    Q_multiplier: int = 16
    include_scale: bool = False

@dataclass
class DataCfg:
    root: str = "data/ESC-50"
    batch_size: int = 32
    num_workers: int = 4
    sample_rate: int = 16000
    segment_seconds: float = 5.0
    fold: int = 1

@dataclass
class MelCfg:
    n_mels: int = 64
    n_fft: Optional[int] = None
    hop_length: Optional[int] = None
    win_length: Optional[int] = None
    f_min: float = 20.0
    f_max: Optional[float] = None
    power: float = 2.0
    top_db: float = 80.0
    center: bool = True
    pad_mode: str = "reflect"
    # aug / preprocessing
    specaug_p: float = 0.5
    freq_mask_param: int = 12
    time_mask_param: int = 24
    rms_norm: bool = False
    rms_target: float = 0.1

@dataclass
class MuReNNTxConfig:
    # backbone
    arch: str = "tx"  # "tx" or "cnn"

    # shared model params
    n_classes: int = 50
    n_scales: int = 6
    base_channels: int = 64
    d_model: int = 128
    nhead: int = 4
    ff_mult: int = 4
    depth_per_scale: int = 2

    # nested configs
    frontend: FrontendCfg = field(default_factory=FrontendCfg)
    data: DataCfg = field(default_factory=DataCfg)
    mel: MelCfg = field(default_factory=MelCfg)

    # training
    epochs: int = 100
    lr: float = 3e-4
    wd: float = 0.05
    seed: int = 1337
    check_val_every_n_epoch: int = 1
    dropout: float = 0.1

    # cnn-only knobs
    cnn_width: int = 32
    cnn_depth: int = 3
    cnn_pool_stride: int = 2
    fe_out_to_d_model: bool = False
    fe_ch_per_scale: int = 128

    def __post_init__(self):
        if isinstance(self.frontend, dict):
            self.frontend = FrontendCfg(**self.frontend)
        if isinstance(self.data, dict):
            self.data = DataCfg(**self.data)
        if isinstance(self.mel, dict):                  # NEW
            self.mel = MelCfg(**self.mel)