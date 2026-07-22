"""Tactile-map -> F/CoP forecasters. Two per-frame encoders behind an IDENTICAL GRU + one-shot
head, so the encoder is the only variable in the flatten-vs-CNN comparison.

Input  x: (B, t_in, 2, 32, 32) normalized map history.
Output  : (B, H, 6) forecast of the next H steps of the 6-dim F/CoP target (normalized units).
"""
from __future__ import annotations

import torch
import torch.nn as nn

IN_CH, GRID = 2, 32
FLAT = IN_CH * GRID * GRID          # 2048


class FlattenEncoder(nn.Module):
    """Flatten each frame -> linear -> embedding (no spatial structure exploited)."""

    def __init__(self, d: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(FLAT, d), nn.ReLU())

    def forward(self, x):                         # (B,t_in,2,32,32) -> (B,t_in,d)
        B, T = x.shape[:2]
        return self.proj(x.reshape(B * T, IN_CH, GRID, GRID)).reshape(B, T, -1)


class CNNEncoder(nn.Module):
    """Small conv stack per frame -> embedding (exploits spatial structure)."""

    def __init__(self, d: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(IN_CH, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),   # 32->16
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),   # 16->8
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, d), nn.ReLU())

    def forward(self, x):                         # (B,t_in,2,32,32) -> (B,t_in,d)
        B, T = x.shape[:2]
        return self.conv(x.reshape(B * T, IN_CH, GRID, GRID)).reshape(B, T, -1)


class Seq2Seq(nn.Module):
    """encoder -> GRU over t_in frames -> one-shot PROBABILISTIC head -> (mu, logvar), each (B,H,6).

    Predicts the RESIDUAL (change vs the last observed value) as a Gaussian per (step, channel):
    mean mu + log-variance lv. Trained with Gaussian NLL; lv clamped for stability."""

    def __init__(self, encoder: nn.Module, d: int, hidden: int, horizon: int, n_out: int = 6):
        super().__init__()
        self.encoder = encoder
        self.gru = nn.GRU(d, hidden, batch_first=True)
        self.mu = nn.Linear(hidden, horizon * n_out)
        self.lv = nn.Linear(hidden, horizon * n_out)
        self.H, self.n_out = horizon, n_out

    def forward(self, x):
        e = self.encoder(x)                       # (B,t_in,d)
        _, h = self.gru(e)                        # h: (1,B,hidden)
        last = h[-1]
        mu = self.mu(last).reshape(-1, self.H, self.n_out)
        lv = self.lv(last).clamp(-6, 4).reshape(-1, self.H, self.n_out)
        return mu, lv                             # (B,H,6), (B,H,6)


def build_model(encoder: str, horizon: int, d: int = 64, hidden: int = 64) -> Seq2Seq:
    enc = {"flatten": FlattenEncoder, "cnn": CNNEncoder}[encoder](d)
    return Seq2Seq(enc, d, hidden, horizon)
