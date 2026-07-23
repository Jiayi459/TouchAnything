"""Train + cross-validate the tactile-map -> F/CoP forecaster (probabilistic, residual).

Mirrors the F/CoP probGRU protocol (src/actionsense/action_dynamics.py): a probabilistic head
(mean + log-variance) trained with Gaussian NLL, 5-fold CV by recording, sigma calibration on a
VAL subset held out from TRAIN, and skill-vs-persistence + coverage reported per channel & step.
Target is the RESIDUAL over persistence, so persistence == predicting residual 0.
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

EPS = 1e-12


def recordings(cfg: Config, require_maps: bool = True) -> list[int]:
    """All Slice/Peel recordings in the frozen split (optionally only those with local maps)."""
    sp = load_splits(cfg)
    allrec = sorted(sp["train"] + sp["val"] + sp["test"])
    return D.available_idxs(cfg, allrec) if require_maps else allrec


def _dataset(cfg, tm, t_in, idxs, mnorm, tnorm):
    maps, tgts = D.load_raw(cfg, idxs, tm["baseline_frames"])
    return D.MapWindows(D.normalize(maps, mnorm), {i: tnorm.z(t) for i, t in tgts.items()}, cfg, t_in)


def train_model(train_ds, val_ds, cfg: Config, encoder: str, tm: dict, seed: int = 0):
    """Train one probabilistic model (Gaussian NLL); keep best-VAL-NLL weights."""
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(encoder, cfg.horizon, tm["d"], tm["hidden"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=tm["lr"])
    tl = DataLoader(train_ds, batch_size=tm["batch"], shuffle=True)
    best, best_state = np.inf, None
    for _ in range(tm["epochs"]):
        model.train()
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            mu, lv = model(x)
            loss = 0.5 * (lv + (y - mu) ** 2 * torch.exp(-lv)).mean()      # Gaussian NLL on residual
            opt.zero_grad(); loss.backward(); opt.step()
        v = _val_nll(model, val_ds, dev) if len(val_ds) else float(loss.item())
        if v < best:
            best, best_state = v, {k: t.cpu().clone() for k, t in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def _val_nll(model, ds, dev):
    model.eval(); s = n = 0.0
    for x, y in DataLoader(ds, batch_size=128):
        x, y = x.to(dev), y.to(dev)
        mu, lv = model(x)
        s += float((0.5 * (lv + (y - mu) ** 2 * torch.exp(-lv))).sum()); n += y.numel()
    return s / max(n, 1)


@torch.no_grad()
def _predict(model, ds, batch=128):
    """-> (mu, sd, resid_true), each (N,H,6) in normalized RESIDUAL space."""
    dev = next(model.parameters()).device
    mus, sds, ys = [], [], []
    for x, y in DataLoader(ds, batch_size=batch):
        mu, lv = model(x.to(dev))
        mus.append(mu.cpu().numpy()); sds.append(np.exp(0.5 * lv.cpu().numpy())); ys.append(y.numpy())
    if not mus:
        z = np.zeros((0, ds.H, 6)); return z, z, z
    return np.concatenate(mus), np.concatenate(sds), np.concatenate(ys)


def evaluate(model, ds, sigma_scale=1.0):
    """skill vs persistence (per channel, per step) + coverage@2sd. Persistence predicts residual 0."""
    mu, sd, resid = _predict(model, ds)
    em = (mu - resid) ** 2; ep = resid ** 2
    sk_ch = 1 - em.mean((0, 1)) / (ep.mean((0, 1)) + EPS)
    sk_step = 1 - em.mean(0) / (ep.mean(0) + EPS)
    cov = float((np.abs(resid - mu) <= 2 * sigma_scale * sd).mean())
    return sk_ch, sk_step, cov


def calibrate_sigma(model, ds, target=0.95):
    mu, sd, resid = _predict(model, ds)
    if len(mu) == 0:
        return 1.0
    return float(np.percentile(np.abs(resid - mu) / (sd + 1e-9), 100 * target) / 2.0)


def cross_validate(cfg: Config, tm: dict, encoder: str, t_in: int, recs: list[int],
                   folds: int = 5, seed: int = 0):
    """5-fold CV by recording. Norms + model fit on TRAIN; sigma calibrated on a VAL subset of TRAIN;
    skill + coverage measured on the held-out TEST fold. -> (sk_ch (folds,6), sk_step (folds,H,6),
    cov_raw, cov_cal)."""
    rng = np.random.default_rng(seed)
    fold_of = rng.integers(0, folds, size=len(recs))
    skc, sks, cr, cc = [], [], [], []
    for f in range(folds):
        te = [recs[i] for i in range(len(recs)) if fold_of[i] == f]
        tr = [recs[i] for i in range(len(recs)) if fold_of[i] != f]
        if len(te) < 1 or len(tr) < 4:
            continue
        r2 = np.random.default_rng(seed * 100 + f)
        idx = r2.permutation(len(tr)); nv = max(2, len(tr) // 6)
        val, trn = [tr[i] for i in idx[:nv]], [tr[i] for i in idx[nv:]]
        if encoder == "aggregate":                          # neural AR on the aggregate 6-dim F/CoP
            tnorm = Norm.from_train({i: load_target(cfg, i) for i in trn})
            mk = lambda ids: D.AggWindows({i: tnorm.z(load_target(cfg, i)) for i in ids}, cfg, t_in)  # noqa: E731
            train_ds, val_ds, test_ds = mk(trn), mk(val), mk(te)
        else:                                               # map input (flatten / cnn)
            maps_tr, tgts_tr = D.load_raw(cfg, trn, tm["baseline_frames"])
            mnorm = D.MapNorm.from_train(maps_tr, tm["alpha"]); tnorm = Norm.from_train(tgts_tr)
            train_ds = D.MapWindows(D.normalize(maps_tr, mnorm),
                                    {i: tnorm.z(t) for i, t in tgts_tr.items()}, cfg, t_in)
            val_ds = _dataset(cfg, tm, t_in, val, mnorm, tnorm)
            test_ds = _dataset(cfg, tm, t_in, te, mnorm, tnorm)
        model = train_model(train_ds, val_ds, cfg, encoder, tm, seed=seed)
        s = calibrate_sigma(model, val_ds)
        sk_ch, sk_step, c_raw = evaluate(model, test_ds, sigma_scale=1.0)
        _, _, c_cal = evaluate(model, test_ds, sigma_scale=s)
        skc.append(sk_ch); sks.append(sk_step); cr.append(c_raw); cc.append(c_cal)
    return np.array(skc), np.array(sks), float(np.mean(cr)), float(np.mean(cc))
