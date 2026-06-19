"""Convolutional recurrent seq2seq forecasters: ConvGRU (primary) and ConvLSTM (baseline).

forward(x) : (B, t_in, C, H, W) -> (B, t_out, C, H, W).
ConvGRU is preferred for the small (N=82) tactile dataset: 3 gates vs LSTM's 4 -> fewer
params / less overfitting, while keeping spatial structure (conv state).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    def __init__(self, in_ch, hid, k=3):
        super().__init__()
        self.hid = hid
        p = k // 2
        self.conv_zr = nn.Conv2d(in_ch + hid, 2 * hid, k, padding=p)
        self.conv_h = nn.Conv2d(in_ch + hid, hid, k, padding=p)

    def forward(self, x, h):
        if h is None:
            h = x.new_zeros(x.size(0), self.hid, x.size(2), x.size(3))
        z, r = torch.chunk(torch.sigmoid(self.conv_zr(torch.cat([x, h], 1))), 2, 1)
        hh = torch.tanh(self.conv_h(torch.cat([x, r * h], 1)))
        h_new = (1 - z) * h + z * hh
        return h_new, h_new  # (output, state)


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid, k=3):
        super().__init__()
        self.hid = hid
        self.conv = nn.Conv2d(in_ch + hid, 4 * hid, k, padding=k // 2)

    def forward(self, x, state):
        if state is None:
            z = x.new_zeros(x.size(0), self.hid, x.size(2), x.size(3))
            state = (z, z)
        h, c = state
        i, f, g, o = torch.chunk(self.conv(torch.cat([x, h], 1)), 4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, (h, c)


def _make_cell(kind, in_ch, hid, k):
    return ConvGRUCell(in_ch, hid, k) if kind == "gru" else ConvLSTMCell(in_ch, hid, k)


class ConvRNNSeq2Seq(nn.Module):
    def __init__(self, cell="gru", in_ch=2, hid=64, layers=2, k=3, t_out=15):
        super().__init__()
        self.t_out = t_out
        cells = []
        ch = in_ch
        for _ in range(layers):
            cells.append(_make_cell(cell, ch, hid, k))
            ch = hid
        self.cells = nn.ModuleList(cells)
        self.readout = nn.Conv2d(hid, in_ch, 1)

    def _step(self, inp, states):
        x = inp
        new = []
        for l, cell in enumerate(self.cells):
            out, st = cell(x, states[l])
            new.append(st)
            x = out
        return x, new  # top hidden, updated states

    def forward(self, x, y=None, ssprob=0.0):
        B, t_in = x.shape[:2]
        states = [None] * len(self.cells)
        for t in range(t_in):
            top, states = self._step(x[:, t], states)
        t_out = y.shape[1] if y is not None else self.t_out
        inp = x[:, -1]
        outs = []
        for t in range(t_out):
            top, states = self._step(inp, states)
            o = torch.relu(self.readout(top))  # pressure is non-negative
            outs.append(o)
            if self.training and y is not None and float(torch.rand(())) < ssprob:
                inp = y[:, t]            # scheduled sampling: teacher forcing
            else:
                inp = o                   # free-running
        return torch.stack(outs, 1)
