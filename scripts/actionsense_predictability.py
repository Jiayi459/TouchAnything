"""Training-free predictability probe for ActionSense, by kitchen-activity category.

ActionSense (NeurIPS 2022, MIT) wearables HDF5 (per subject session), from delpreto/ActionNet:
    tactile-glove-left / tactile-glove-right / tactile_data / {data (N,H,W), time_s}
    experiment-activities / activities / {data (rows [Activity,Start/Stop,Valid,Notes]), time_s}
Continuous recording -> we segment the tactile stream into per-activity clips using the
Start/Stop label markers, resample each clip to a common 30 Hz (to match EgoTouch/OpenTouch
frame-based metrics), stack both gloves -> (T,2,H,W), and probe.

Activity labels are verb-first phrases ('Peel a cucumber', 'Pour water ...') mapped through the
shared categorize_phrase() taxonomy. Metrics from src/tactile_forecast/predictability.py.
Subjects S00-S05 wore tactile gloves; S06-S09 did not (files auto-skip: no tactile stream).

Usage (run from repo root):
    python scripts/actionsense_predictability.py --data-dir ~/actionsense
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tactile_forecast.categories import categorize_phrase, TEMPORAL_PATTERN  # noqa: E402
from src.tactile_forecast import predictability as P  # noqa: E402

TACTILE_KEYS = ("tactile-glove-left", "tactile-glove-right")
BAD_RATINGS = ("Bad", "Maybe")


def _dec(x):
    return x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)


def activity_intervals(h5):
    """List of (label, start_s, end_s) from the experiment-activities stream."""
    grp = h5["experiment-activities"]["activities"]
    rows = [[_dec(v) for v in r] for r in grp["data"][:]]
    times = np.squeeze(np.array(grp["time_s"][:])).astype(float)
    out, open_lbl, open_t = [], None, None
    for i, row in enumerate(rows):
        label, startstop = row[0], row[1]
        rating = row[2] if len(row) > 2 else "Good"
        if rating in BAD_RATINGS:
            continue
        if startstop == "Start":
            open_lbl, open_t = label, times[i]
        elif startstop == "Stop" and open_lbl is not None:
            out.append((open_lbl, open_t, times[i]))
            open_lbl = None
    return out


def load_tactile(h5, key):
    """(data (N,H,W) float32, time_s (N,)) for one glove, or None."""
    if key not in h5 or "tactile_data" not in h5[key]:
        return None
    g = h5[key]["tactile_data"]
    d = np.asarray(g["data"][:], dtype=np.float32)
    t = np.squeeze(np.array(g["time_s"][:])).astype(float)
    if d.ndim == 2:  # (N, D) -> (N, s, s)
        s = int(round(d.shape[1] ** 0.5))
        d = d.reshape(d.shape[0], s, s)
    return d, t


def resample(X, t, target_t):
    """Linear-interpolate frames X:(n,H,W) at times t to target_t:(T,)."""
    n = X.shape[0]
    idx = np.clip(np.searchsorted(t, target_t), 1, n - 1)
    t0, t1 = t[idx - 1], t[idx]
    w = ((target_t - t0) / (t1 - t0 + 1e-12))[:, None, None]
    return X[idx - 1] * (1 - w) + X[idx] * w


def clip_for_interval(gloves, start, end, fps):
    """Stack both gloves' resampled frames for [start,end] -> (T,2,H,W) or None."""
    T = int(round((end - start) * fps))
    if T < 2:
        return None
    target_t = np.linspace(start, end, T)
    chans = []
    for gd in gloves:
        if gd is None:
            return None
        d, t = gd
        m = (t >= t[0]) & (t <= t[-1])  # guard
        if t.min() > start or t.max() < end or d.shape[0] < 2:
            return None
        chans.append(resample(d, t, target_t))
    return np.stack(chans, axis=1)  # (T, 2, H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(os.path.expanduser("~"), "actionsense"))
    ap.add_argument("--target-fps", type=int, default=30)
    ap.add_argument("--min-frames", type=int, default=35)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--out", default=os.path.join("docs", "predictability_actionsense.csv"))
    args = ap.parse_args()
    import h5py

    files = sorted(glob.glob(os.path.join(args.data_dir, "**", "*.hdf5"), recursive=True))
    print(f"HDF5 files: {len(files)} under {args.data_dir!r}")

    if args.inspect:
        for fp in files[:2]:
            with h5py.File(fp, "r") as h5:
                print(f"\n--- {os.path.basename(fp)} ---")
                print("top-level devices:", list(h5.keys())[:20])
                for k in TACTILE_KEYS:
                    gd = load_tactile(h5, k)
                    print(f"  {k}: {'MISSING' if gd is None else (gd[0].shape, 'Fs=%.1f' % ((gd[1].size-1)/(gd[1][-1]-gd[1][0])))}")
                ivs = activity_intervals(h5)
                print(f"  activities: {len(ivs)}; sample:", ivs[:3])
        return

    by_activity = P.new_group_dict()
    by_category = P.new_group_dict()
    by_pattern = P.new_group_dict()
    n_ok = n_iv = 0
    for fp in files:
        try:
            h5 = h5py.File(fp, "r")
        except OSError as e:
            print(f"  {os.path.basename(fp)}: CORRUPT/truncated ({e.args[0][:40]}...), skip")
            continue
        with h5:
            gloves = [load_tactile(h5, k) for k in TACTILE_KEYS]
            if all(g is None for g in gloves):
                print(f"  {os.path.basename(fp)}: no tactile stream, skip")
                continue
            try:
                ivs = activity_intervals(h5)
            except KeyError:
                print(f"  {os.path.basename(fp)}: no activity labels, skip")
                continue
            for label, start, end in ivs:
                n_iv += 1
                if label in ("None", ""):
                    continue
                clip = clip_for_interval(gloves, start, end, args.target_fps)
                if clip is None or clip.shape[0] < args.min_frames:
                    continue
                m = P.seq_metrics(clip)
                if not m:
                    continue
                cat = categorize_phrase(label)
                by_activity[label].append(m)
                by_category[cat].append(m)
                by_pattern[TEMPORAL_PATTERN.get(cat, "Other")].append(m)
                n_ok += 1
        print(f"  {os.path.basename(fp)}: cumulative usable clips={n_ok}")
    print(f"\nactivity intervals={n_iv}  usable(T>={args.min_frames})={n_ok}\n")

    def show(title, groups, min_n=1):
        rows = P.add_predictability_index(P.aggregate(
            {g: l for g, l in groups.items() if len(l) >= min_n}))
        print(f"===== {title} (ranked by predictability_index, higher=easier) =====")
        hdr = f"{'group':<48}{'n':>5}{'persH1':>8}{'persH15':>9}{'persH30':>9}{'period':>8}{'migr15':>8}{'PI':>7}"
        print(hdr); print("-" * len(hdr))
        for g in sorted(rows, key=lambda g: -rows[g]["pi"]):
            r = rows[g]
            print(f"{g[:47]:<48}{r['n']:>5}{r['pers_nmse_h1']:>8.3f}{r['pers_nmse_h15']:>9.3f}"
                  f"{r['pers_nmse_h30']:>9.3f}{r['periodicity']:>8.3f}"
                  f"{r['contact_migration']:>8.3f}{r['pi']:>7.2f}")
        print()
        return rows

    pat_rows = show("BY TEMPORAL-PATTERN CLASS (Axis B)", by_pattern)
    cat_rows = show("BY ACTION CATEGORY (mapped via verb taxonomy)", by_category)
    act_rows = show("BY RAW ACTIONSENSE ACTIVITY", by_activity)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import csv
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["grouping", "group", "n", *P.METRIC_KEYS, "predictability_index"])
        for gname, rows in [("temporal_pattern", pat_rows), ("action_category", cat_rows),
                            ("raw_activity", act_rows)]:
            for g, r in sorted(rows.items(), key=lambda kv: -kv[1]["pi"]):
                w.writerow([gname, g, r["n"], *[f"{r[k]:.5f}" for k in P.METRIC_KEYS],
                            f"{r['pi']:.4f}"])
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
