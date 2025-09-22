import torch

from murenn_tx.modules.model import MuReNNTx, MuReNNTxConfig


def test_token_shapes_divisible():
    cfg = MuReNNTxConfig(n_classes=3, d_model=64, hop=4)
    m = MuReNNTx(cfg)
    x = torch.randn(1, 1, 16000)
    y = m(x)
    assert y.shape[-1] == 3
