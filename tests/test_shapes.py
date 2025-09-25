import torch
from conf.config import MuReNNTxConfig
from murenn_tx.models.registry import build_model
import importlib; importlib.import_module("murenn_tx.models.tx")

def test_token_shapes_divisible():
    cfg = MuReNNTxConfig(n_classes=3, d_model=64, depth_per_scale=2)
    m = build_model("tx", cfg=cfg)
    x = torch.randn(1, 1, 16000)
    y = m(x)
    assert y.shape[-1] == 3