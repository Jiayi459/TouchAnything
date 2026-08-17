"""Training curves for the probGRU, plus a calibration check on its variance head.

Reads the checkpoints --save-model wrote: each carries the final training's per-epoch NLL,
the history sweep's best VAL NLL per input length, and which epoch was kept.

WHAT THE CURVE IS FOR. Early stopping keeps the lowest-VAL-NLL weights, so a run cannot be
hurt by training too long -- but where the minimum sits says something. On the 2026-08-17
run VAL bottomed at epoch 2 and rose monotonically after, while ActionSense's own code
notes its loss "overfits badly after ~epoch 10". Overfitting four times sooner under a
location-held-out split is not a nuisance to be tuned away; it says what the extra epochs
were learning did not survive a change of environment.

Train NLL is sampled every 5th epoch (log_train_every), so it is drawn with markers and
gaps rather than interpolated through points that were never measured.

CALIBRATION, from the saved forecasts rather than the checkpoint. The Gaussian head is
trained but never scored -- the harness measures point error only -- so a model whose
spread is wrong pays nothing and no MSE table shows it. Coverage does: the fraction of
truths falling inside +-2 sigma should be about 95%. Materially above that is a model
hedging with intervals wider than its errors, which is what the forecast plot suggests.

    python scripts/plot_opentouch_loss.py --models runs/models --preds runs/preds
"""
from __future__ import annotations

import argparse
import collections
import glob
import os

import numpy as np


def load_ckpts(d):
    import torch
    out = []
    for p in sorted(glob.glob(os.path.join(d, "*.pt"))):
        try:
            out.append((os.path.basename(p), torch.load(p, map_location="cpu",
                                                        weights_only=False)))
        except Exception as exc:                                   # pragma: no cover
            print(f"  skipped {p}: {type(exc).__name__}: {exc}")
    return out


def coverage(preds_dir, k_sigma=2.0):
    """Coverage and scale, POOLED and PER CHANNEL.

    Per channel is not a refinement, it is the point. F sits near 750,000 while CoP lives in
    [-1,1], so a pooled median sigma is whatever the two CoP channels say and tells you
    nothing about F -- and a pooled coverage can read 95% while one channel hedges and
    another is overconfident, cancelling. The first version reported only the pooled number
    and would have hidden exactly that.

    Both statistics are kept: coverage alone cannot separate a well-sized interval from a
    wide one that happens to sit in the right place, and the sigma-to-error ratio alone
    ignores where the mean is. For Gaussian errors the ratio is 1/0.6745 = 1.48; materially
    below that means the tails are heavier than the interval admits."""
    inside = tot = 0
    sig, err = [], []
    per = collections.defaultdict(lambda: [0, 0, [], []])   # channel -> in, tot, sig, err
    chans = None
    for p in sorted(glob.glob(os.path.join(preds_dir, "clip_*.npz"))):
        z = np.load(p, allow_pickle=True)
        if "sigma_prob_gru" not in z.files or "mu_prob_gru" not in z.files:
            continue
        y, ors = z["y"], z["origins"]
        mu, sd = z["mu_prob_gru"], z["sigma_prob_gru"]
        if len(ors) == 0 or mu.shape[0] == 0:
            continue          # a clip too short to yield any forecast origin still got a
        H = mu.shape[1]       # file written for it; there is simply nothing to score

        truth = np.stack([y[t + 1:t + 1 + H] for t in ors])
        d = np.abs(truth - mu)
        inside += int((d <= k_sigma * sd).sum()); tot += d.size
        sig.append(sd.ravel()); err.append(d.ravel())
        if chans is None:
            chans = [str(c) for c in z["channels"]] if "channels" in z.files else \
                    [f"c{j}" for j in range(d.shape[-1])]
        for j, c in enumerate(chans):
            e = per[c]
            e[0] += int((d[..., j] <= k_sigma * sd[..., j]).sum()); e[1] += d[..., j].size
            e[2].append(sd[..., j].ravel()); e[3].append(d[..., j].ravel())
    if tot == 0:
        return None
    by_ch = {c: (v[0] / max(v[1], 1), float(np.median(np.concatenate(v[2]))),
                 float(np.median(np.concatenate(v[3]))))
             for c, v in per.items()}
    return (inside / tot, float(np.median(np.concatenate(sig))),
            float(np.median(np.concatenate(err))), by_ch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="runs/models")
    ap.add_argument("--preds", default="runs/preds")
    ap.add_argument("--out", default="docs/opentouch_loss.png")
    a = ap.parse_args()

    cks = load_ckpts(a.models)
    if not cks:
        raise SystemExit(f"no *.pt under {a.models} -- rerun with --save-model")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax, axs = axes

    for name, ck in cks:
        h = ck.get("history", {})
        va = np.asarray(h.get("val_nll", []), dtype=float)
        tr = np.asarray(h.get("train_nll", []), dtype=float)
        if va.size == 0:
            continue
        tag = ck.get("split", name).replace("location-k4-", "").replace("-seed0", "")
        ep = np.arange(1, va.size + 1)
        line, = ax.plot(ep, va, lw=1.4, label=f"{tag} val")
        m = np.isfinite(tr)
        if m.any():
            ax.plot(ep[m], tr[m], "o--", ms=3, lw=0.8, alpha=0.55,
                    color=line.get_color(), label=f"{tag} train")
        b = int(np.nanargmin(va))
        ax.plot(ep[b], va[b], "*", ms=12, color=line.get_color())

        sw = ck.get("sweep") or {}
        if sw:
            xs = sorted(int(x) for x in sw)
            axs.plot([x / 30.0 for x in xs], [sw[x] if x in sw else sw[str(x)] for x in xs],
                     "o-", lw=1.4, label=tag)
            axs.plot(ck["t_in"] / 30.0, min(sw.values()), "*", ms=12,
                     color=axs.lines[-1].get_color())

    ax.set_xlabel("epoch"); ax.set_ylabel("Gaussian NLL")
    ax.set_title("probGRU training curves (star = kept weights)\n"
                 "train sampled every 5th epoch", fontsize=10)
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.25)
    axs.set_xlabel("input history (s)"); axs.set_ylabel("best VAL NLL")
    axs.set_title("history sweep, chosen on VAL\n"
                  "(≥2 s is mostly zero-padding: <50% of clips are long enough)",
                  fontsize=10)
    axs.legend(fontsize=7); axs.grid(alpha=0.25)

    cov = coverage(a.preds)
    if cov:
        frac, msig, merr, by_ch = cov
        fig.suptitle(f"±2σ coverage {frac:.1%} (nominal 95.4%) | median σ {msig:.4g} vs "
                     f"median |error| {merr:.4g}  (ratio {msig / max(merr, 1e-9):.2f})",
                     fontsize=10)
        print(f"±2 sigma coverage: {frac:.2%} (nominal 95.4%)")
        print(f"median sigma {msig:.5g} | median |error| {merr:.5g} "
              f"| ratio {msig / max(merr, 1e-9):.2f}   (pooled: dominated by whichever "
              f"channel has the smaller scale -- read the per-channel table below)")
        print(f"\n{'channel':10s} {'±2σ cov':>9s} {'median σ':>12s} "
              f"{'median |err|':>13s} {'σ/|err|':>8s}")
        for c, (f_, s_, e_) in by_ch.items():
            print(f"{c:10s} {f_:9.2%} {s_:12.5g} {e_:13.5g} {s_ / max(e_, 1e-12):8.2f}")
        print("Gaussian errors give σ/|err| = 1.48 and 95.4% coverage. Above -> the head "
              "hedges with intervals wider than its mistakes; below -> it is overconfident. "
              "The harness scores point error only, so neither shows up anywhere else, and "
              "the pooled row cannot separate the channels: F and CoP differ by orders of "
              "magnitude, so a pooled median is whatever CoP says.")
    fig.tight_layout(rect=(0, 0, 1, 0.93 if cov else 1.0))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=140)
    print(f"wrote {a.out} ({len(cks)} checkpoints)")


if __name__ == "__main__":
    raise SystemExit(main())
