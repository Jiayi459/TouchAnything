"""Train the tactile-map -> F/CoP forecaster + export harness-aligned predictions.

Library (import; the CLI is scripts/train_tactile_map.py). Norms are fit on TRAIN only; VAL is
used for early stopping; TEST is only touched by export (scored later by the frozen harness).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..eval_harness.config import Config
from ..eval_harness.dataset import Norm, load_target
from ..eval_harness.splits import load_splits
from . import data as D
from .models import build_model


def build_datasets(cfg: Config, tm: dict, t_in: int, use_available: bool = False):
    """-> (train_ds, val_ds, MapNorm, target Norm, test_idxs). Norms fit on TRAIN only."""
    sp = load_splits(cfg)
    pick = (lambda k: D.available_idxs(cfg, sp[k])) if use_available else (lambda k: sp[k])
    tr, va, te = pick("train"), pick("val"), pick("test")

    maps_tr, tgts_tr = D.load_raw(cfg, tr, tm["baseline_frames"])
    mnorm = D.MapNorm.from_train(maps_tr, tm["alpha"])
    tnorm = Norm.from_train(tgts_tr)

    def make(idxs, maps=None, tgts=None):
        if maps is None:
            maps, tgts = D.load_raw(cfg, idxs, tm["baseline_frames"])
        mn = D.normalize(maps, mnorm)
        tn = {i: tnorm.z(t) for i, t in tgts.items()}
        return D.MapWindows(mn, tn, cfg, t_in)

    return make(tr, maps_tr, tgts_tr), make(va), mnorm, tnorm, te


def train(train_ds, val_ds, cfg: Config, encoder: str, tm: dict, seed: int = 0):
    """Train one (encoder, t_in) model; keep the best-VAL weights. Returns the model (eval mode)."""
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(encoder, cfg.horizon, tm["d"], tm["hidden"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=tm["lr"])
    tl = DataLoader(train_ds, batch_size=tm["batch"], shuffle=True)
    vl = DataLoader(val_ds, batch_size=tm["batch"]) if len(val_ds) else None
    best, best_state = np.inf, None
    for ep in range(tm["epochs"]):
        model.train()
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = ((model(x) - y) ** 2).mean()
            loss.backward(); opt.step()
        v = _val_mse(model, vl, dev) if vl else float(loss.item())
        if v < best:
            best, best_state = v, {k: t.cpu().clone() for k, t in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, best


@torch.no_grad()
def _val_mse(model, loader, dev):
    model.eval(); se = n = 0.0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        se += float(((model(x) - y) ** 2).sum()); n += y.numel()
    return se / max(n, 1)


@torch.no_grad()
def export(model, mnorm: "D.MapNorm", tnorm: Norm, cfg: Config, tm: dict, t_in: int,
           idxs: list[int], batch: int = 64) -> dict[int, np.ndarray]:
    """Predict at every harness origin per recording -> {idx: (n_origins, H, 6)} in RAW target units."""
    dev = next(model.parameters()).device
    out = {}
    for i in idxs:
        tn = tnorm.z(load_target(cfg, i))                     # (T,6) normalized target
        mn = mnorm.apply(D.load_map(cfg, i, tm["baseline_frames"]))
        n = min(len(tn), len(mn)); tn, mn = tn[:n], mn[:n]    # share the time axis
        X, ors = D.recording_windows(mn, cfg, t_in)
        if len(ors) == 0:
            continue
        deltas = []
        for s in range(0, len(X), batch):
            xb = torch.from_numpy(X[s:s + batch]).to(dev)
            deltas.append(model(xb).cpu().numpy())            # normalized RESIDUAL
        anchor = tn[ors][:, None, :]                          # (n,1,6) last observed value (normalized)
        out[i] = tnorm.unz(anchor + np.concatenate(deltas, 0))   # persistence + delta -> raw units
    return out


def save_preds(path: str, preds: dict[int, np.ndarray]) -> None:
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **{str(i): p.astype(np.float32) for i, p in preds.items()})
