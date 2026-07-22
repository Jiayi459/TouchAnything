"""Plot the tactile-map -> F/CoP results: flatten vs CNN vs baselines (skill vs persistence).

Scores the 6 exported model npz through the frozen harness (alongside persistence/seasonal/AR),
then draws:
  docs/tactile_map_skill_vs_history.png   mean skill vs history, flatten vs cnn (+ AR/persistence refs)
  docs/tactile_map_skill_bars.png         per-channel full-horizon skill for AR / flatten / CNN

    python scripts/plot_tactile_map.py
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.actionsense.eval_harness import evaluate as E, metrics          # noqa: E402
from src.actionsense.eval_harness.config import load_config              # noqa: E402
from src.actionsense.eval_harness.splits import load_splits             # noqa: E402

HISTS = [1, 3, 10]
RUN = "runs/tactile_map"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config(); sp = load_splits(cfg)
    results, norm, _ = E.fit_and_forecast(cfg, sp)
    for enc in ["flatten", "cnn"]:
        for h in HISTS:
            name = f"{enc}_{h}s"
            preds = {int(k): v for k, v in np.load(f"{RUN}/preds_{name}.npz").items()}
            results[name] = E.score_external(cfg, sp, name, preds, results, norm)
    ref = results["persistence"]["ch_mse"]
    ch = cfg.channels

    def mean_skill(m):
        return float(metrics.skill(results[m]["ch_mse"], ref).mean())

    # ---- (1) skill vs history: flatten vs cnn, with AR + persistence reference lines ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for enc, c in [("flatten", "C0"), ("cnn", "C1")]:
        ax.plot(HISTS, [mean_skill(f"{enc}_{h}s") for h in HISTS], "-o", color=c, lw=2, label=enc)
    ax.axhline(mean_skill("ar"), color="C2", ls="--", lw=1.5, label="AR (aggregate baseline)")
    ax.axhline(0, color="0.5", lw=1, label="persistence")
    ax.set_xscale("log"); ax.set_xticks(HISTS); ax.set_xticklabels([f"{h}s" for h in HISTS])
    ax.set_xlabel("input history"); ax.set_ylabel("mean skill vs persistence")
    ax.set_title("Tactile-map -> F/CoP: CNN exploits spatial structure, flatten does not")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("docs/tactile_map_skill_vs_history.png", dpi=120)
    print("[done] docs/tactile_map_skill_vs_history.png")

    # ---- (2) per-channel full-horizon skill: AR vs flatten(best) vs cnn(best) ----
    models = [("ar", "C2"), ("flatten_10s", "C0"), ("cnn_10s", "C1")]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(ch)); w = 0.8 / len(models)
    for k, (m, c) in enumerate(models):
        ss = metrics.skill(results[m]["ch_mse"], ref)
        ax.bar(x + k * w, ss, w, color=c, label=m)
    ax.axhline(0, color="0.5", lw=.8)
    ax.set_xticks(x + w); ax.set_xticklabels(ch); ax.set_ylabel("skill vs persistence")
    ax.set_title("Per-channel full-horizon skill (10 s history)"); ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig("docs/tactile_map_skill_bars.png", dpi=120)
    print("[done] docs/tactile_map_skill_bars.png")


if __name__ == "__main__":
    main()
