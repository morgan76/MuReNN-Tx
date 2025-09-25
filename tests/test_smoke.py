import torch
from conf.config import MuReNNTxConfig
from murenn_tx.models.registry import build_model
import importlib; importlib.import_module("murenn_tx.models.tx")  # ensure register (redundant if step A2 done)

def test_smoke():
    cfg = MuReNNTxConfig(n_classes=3, d_model=64)
    m = build_model("tx", cfg=cfg)
    x = torch.randn(2, 1, 16000)
    y = m(x)
    assert y.shape == (2, 3)