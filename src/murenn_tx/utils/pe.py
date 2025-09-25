import torch, torch.nn as nn, math

class SinusoidalPE1D(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200_000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # (L, D)

    def forward(self, T: int):
        return self.pe[:T]