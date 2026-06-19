"""SimVP-lite: pure-CNN encoder->translator->decoder forecaster (headline model).

Spatial resolution is kept constant (21x21 is already small) to avoid odd-size
down/up-sampling. forward(x): (B, t_in, C, H, W) -> (B, t_out, C, H, W).
Ref: Gao et al., SimVP (CVPR'22), arXiv:2206.05099.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvSC(nn.Module):
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=k // 2),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SimVP(nn.Module):
    def __init__(self, in_ch=2, t_in=10, t_out=15, hid=64, n_enc=2, n_trans=4, k=3):
        super().__init__()
        self.in_ch, self.t_in, self.t_out, self.hid = in_ch, t_in, t_out, hid
        enc = [ConvSC(in_ch, hid, k)] + [ConvSC(hid, hid, k) for _ in range(n_enc - 1)]
        self.enc = nn.Sequential(*enc)
        trans = [ConvSC(t_in * hid, t_out * hid, k)]
        trans += [ConvSC(t_out * hid, t_out * hid, k) for _ in range(n_trans - 1)]
        self.trans = nn.Sequential(*trans)
        dec = [ConvSC(hid, hid, k) for _ in range(n_enc - 1)]
        self.dec = nn.Sequential(*dec)
        self.head = nn.Conv2d(hid, in_ch, 1)

    def forward(self, x, y=None, ssprob=0.0):  # y/ssprob unused (non-recurrent)
        B, T, C, H, W = x.shape
        z = self.enc(x.reshape(B * T, C, H, W))            # (B*t_in, hid, H, W)
        z = z.reshape(B, T * self.hid, H, W)
        z = self.trans(z)                                  # (B, t_out*hid, H, W)
        z = z.reshape(B * self.t_out, self.hid, H, W)
        z = self.dec(z)
        o = torch.relu(self.head(z))                       # non-negative pressure
        return o.reshape(B, self.t_out, C, H, W)
