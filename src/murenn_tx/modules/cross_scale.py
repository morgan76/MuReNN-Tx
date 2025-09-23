from __future__ import annotations
import torch
import torch.nn as nn

__all__ = ["UpwardCrossScaleFusion"]


def _aligned_band_indices(Tq: int, Tk: int, band: int, device) -> torch.Tensor:
    """Map each fine query index to a (2b+1)-wide band in the neighbor (coarse)."""
    centers = torch.linspace(0, Tk - 1, Tq, device=device).round().long()
    offs = torch.arange(-band, band + 1, device=device).view(1, -1)
    idx = (centers.view(-1, 1) + offs).clamp_(0, Tk - 1)  # (Tq, 2b+1)
    return idx


class UpwardCrossScaleFusion(nn.Module):
    """
    Progressive coarse→fine message passing with (optional) banded cross-attention.
    For s = S-1..1:
        q = seqs[s-1]  (finer)
        k,v = seqs[s]  (coarser), restricted to a small temporal band
        q <- LN(q + Attn(q, k_band, v_band)) -> LN(q + FFN(q))
    Returns the updated finest stream (B, T0, d).
    """
    def __init__(self, d_model: int, n_scales: int, nhead: int = 4, band: int | None = 1,
                 kdim: int | None = None, vdim: int | None = None, q_stride: int = 1):
        super().__init__()
        self.n_scales = n_scales
        self.band = band
        self.q_stride = q_stride
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True,
                                          kdim=kdim or d_model, vdim=vdim or d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def _message_full(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # Optional query subsampling for full cross-attn
        q_sub = q[:, :: self.q_stride] if self.q_stride > 1 else q
        out, _ = self.attn(q_sub, kv, kv)
        if self.q_stride > 1:
            out = torch.nn.functional.interpolate(out.transpose(1, 2), size=q.size(1), mode="linear", align_corners=False).transpose(1, 2)
        return out

    def _message_banded(self, q: torch.Tensor, kv: torch.Tensor, band: int) -> torch.Tensor:
        B, Tq, d = q.shape
        Tk = kv.size(1)
        idx = _aligned_band_indices(Tq, Tk, band, q.device)  # (Tq, L)
        L = idx.size(1)
        gather_idx = idx.view(1, Tq, L, 1).expand(B, -1, -1, d)
        kv_exp = kv.unsqueeze(1).expand(B, Tq, Tk, d)
        band = torch.gather(kv_exp, 2, gather_idx)  # (B,Tq,L,d)
        qf = q.reshape(B * Tq, 1, d)
        kf = band.reshape(B * Tq, L, d)
        out, _ = self.attn(qf, kf, kf)  # (B*Tq,1,d)
        return out.view(B, Tq, d)

    def forward(self, seqs: list[torch.Tensor]) -> torch.Tensor:
        # seqs: fine->coarse list of (B, T_s, d) after local time self-attn
        y = list(seqs)
        for s in range(self.n_scales - 1, 0, -1):
            q = y[s - 1]
            kv = y[s]
            msg = self._message_full(q, kv) if self.band is None else self._message_banded(q, kv, self.band)
            q = self.norm1(q + msg)
            q = self.norm2(q + self.ff(q))
            y[s - 1] = q
        return y[0]
