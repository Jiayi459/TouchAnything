"""Filmstrip of the raw 16x16 pressure map for OpenTouch clips, with CoP overlaid.

The companion to plot_opentouch_fcop.py: that one shows the three summary channels the
models actually forecast, this one shows the field they were reduced from, so the
summary can be checked against the thing it summarises. Selection is imported from that
script rather than reimplemented, so `--actions ... --n ...` picks the SAME clips and the
two figures line up; `--idx` takes the indices it prints if you want an exact set.

Descriptive only -- no model, no forecast, no R^2. Same Layer-2 constraint as the
companion script: this verifies the trait prior, it must not be used to reclassify
actions (SESSION_LOG 2026-08-12, Q3).

CoP overlay convention, taken from moments() in scripts/extract_opentouch.py: cx weights
xs = linspace(-1,1,W) along the COLUMN axis and cy weights ys = linspace(-1,1,H) along
the ROW axis, so col = (cx+1)/2*(W-1) and row = (cy+1)/2*(H-1) under imshow's default
origin="upper". If the red dot ever sits off the pressed region, that mapping -- not the
sensor -- is what to re-check.

Note the grid has only 169 live taxels of 256; dead cells read ~0 and simply stay dark.
No baseline correction has been applied anywhere upstream (deliberate).

    python scripts/plot_opentouch_tactile_map.py --actions "holding,picking up" --n 2
    python scripts/plot_opentouch_tactile_map.py --idx 12,345 --frames 10
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.plot_opentouch_fcop import load_manifest, pick  # noqa: E402


def frame_set(r, T, n):
    """n evenly spaced frames, plus the segmentation's onset/peak/post when present."""
    frames = {int(round(v)) for v in np.linspace(0, T - 1, n)}
    tags = {}
    for key, tag in (("onset_idx", "onset"), ("peak_idx", "peak"), ("post_idx", "post")):
        try:
            v = int(r.get(key))
        except (TypeError, ValueError):
            continue
        if 0 <= v < T:
            frames.add(v)
            tags[v] = tag
    return sorted(frames), tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.expanduser("~/opentouch/cache"))
    ap.add_argument("--idx", help="comma-separated clip indices (as printed by "
                                  "plot_opentouch_fcop.py)")
    ap.add_argument("--actions", help="comma-separated actions; same selection rule as "
                                      "plot_opentouch_fcop.py")
    ap.add_argument("--n", type=int, default=2, help="clips per action")
    ap.add_argument("--frames", type=int, default=8, help="evenly spaced frames per clip")
    ap.add_argument("--out", default="docs/opentouch_tactile_map.png")
    args = ap.parse_args()
    if not args.idx and not args.actions:
        raise SystemExit("give --idx or --actions")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_manifest(args.cache)
    if args.idx:
        want = [int(i) for i in args.idx.split(",") if i.strip()]
        by_idx = {r["idx"]: r for r in rows}
        missing = [i for i in want if i not in by_idx]
        if missing:
            raise SystemExit(f"no manifest entry for idx {missing}")
        picked = [(by_idx[i].get("action", "?"), by_idx[i]) for i in want]
    else:
        acts = [a.strip() for a in args.actions.split(",") if a.strip()]
        picked = [(a, r) for a in acts for r in pick(rows, a, args.n)]
        if not picked:
            raise SystemExit(f"no clips for {acts}")

    ncol = None
    fig = axes = None
    for row_i, (action, r) in enumerate(picked):
        path = os.path.join(args.cache, f"clip_{r['idx']}.npy")
        if not os.path.exists(path):
            raise SystemExit(
                f"{path} missing. The cache was built with --no-clips, so only the "
                f"summary channels exist; the raw maps need a re-extract without it.")
        p = np.load(path).astype(np.float32)[:, 0]      # (T,16,16), hand axis is 1
        st = np.load(os.path.join(args.cache, f"state_{r['idx']}.npy"))
        fps = r.get("fps_est") or 30.0
        frames, tags = frame_set(r, p.shape[0], args.frames)

        if fig is None:
            ncol = len(frames)
            fig, axes = plt.subplots(len(picked), ncol, squeeze=False,
                                     figsize=(1.5 * ncol, 1.9 * len(picked)))
        # Per-clip scale (99th pct, not max) so one hot taxel cannot flatten the field.
        vmax = float(np.percentile(p, 99.5)) or float(p.max()) or 1.0
        H, W = p.shape[1], p.shape[2]
        for col in range(ncol):
            ax = axes[row_i][col]
            ax.set_xticks([]); ax.set_yticks([])
            if col >= len(frames):
                ax.axis("off")
                continue
            f = frames[col]
            ax.imshow(p[f], cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
            F, cx, cy = st[f, 0, 0], st[f, 0, 1], st[f, 0, 2]
            if F > 0:
                ax.plot((cx + 1) / 2 * (W - 1), (cy + 1) / 2 * (H - 1),
                        "o", ms=4, mfc="none", mec="cyan", mew=1.2)
            tag = tags.get(f, "")
            ax.set_title(f"{f / fps:.2f}s{' ' + tag if tag else ''}", fontsize=7,
                         color="crimson" if tag == "peak" else "black")
            if col == 0:
                ax.set_ylabel(f"[{r['idx']}] {action}\n{r.get('object_name', '') or '?'}",
                              fontsize=7)

    fig.suptitle("OpenTouch raw pressure (16x16), cyan ring = CoP; "
                 "per-clip colour scale", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}  ({len(picked)} clips)")


if __name__ == "__main__":
    raise SystemExit(main())
