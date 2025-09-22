import torch

from murenn_tx.modules.model import MuReNNTx, MuReNNTxConfig


def test_smoke():
    cfg = MuReNNTxConfig(n_classes=3, d_model=64)
    m = MuReNNTx(cfg)
    x = torch.randn(2, 1, 16000)
    y = m(x)
    assert y.shape == (2, 3)
