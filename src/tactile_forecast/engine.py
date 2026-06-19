"""Training / evaluation loops with masked loss. Metrics are computed in the original
[0,1] pressure space (inverse transform applied) and only over valid taxels."""
from __future__ import annotations

import numpy as np
import torch

from . import tactile_utils as U


def masked_mse(pred, target, mask, active_weight=1.0, active_thr=0.05):
    """pred/target:(B,T,C,H,W); mask:(B,C,H,W). Mean SE over valid taxels."""
    m = mask.unsqueeze(1)  # (B,1,C,H,W)
    w = m
    if active_weight > 1.0:
        w = m * (1.0 + (active_weight - 1.0) * (target > active_thr).float())
    se = (pred - target) ** 2 * w
    return se.sum() / w.sum().clamp_min(1e-8)


def train_one_epoch(model, loader, opt, device, ssprob, active_weight):
    model.train()
    total, n = 0.0, 0
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        opt.zero_grad()
        pred = model(x, y=y, ssprob=ssprob)
        loss = masked_mse(pred, y, mask, active_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * x.size(0); n += x.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, inv, mask_np):
    """Returns (per_horizon_metrics, mean_skill) in [0,1] space."""
    model.eval()
    preds, targs, lasts = [], [], []
    for x, y, _ in loader:
        x = x.to(device)
        pred = model(x).cpu().numpy()
        preds.append(inv(pred))
        targs.append(inv(y.numpy()))
        lasts.append(inv(x[:, -1].cpu().numpy()))
    pred = np.concatenate(preds); targ = np.concatenate(targs); last = np.concatenate(lasts)
    metrics = U.horizon_metrics(pred, targ, last, mask_np)
    mean_skill = float(np.mean([metrics[h]["skill"] for h in metrics]))
    return metrics, mean_skill


def ss_schedule(epoch, ss_epochs):
    """Linear teacher-forcing decay 1->0 over ss_epochs (recurrent models only)."""
    if ss_epochs <= 0:
        return 0.0
    return max(0.0, 1.0 - epoch / ss_epochs)
