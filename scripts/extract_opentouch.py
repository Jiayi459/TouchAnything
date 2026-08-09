"""Extract the tiny, permanent OpenTouch cache from one (large) HDF5 shard.

Each OpenTouch shard is ~561 MB, almost all of it `rgb_images_jpeg`. For the
forecasting port we need only the tactile grid, the pose, and the clock. This script
pulls those out, joins the per-clip label row, and writes a cache that mirrors the
ActionSense layout so the eval harness loads it via a config swap, not a rewrite:

    <out>/state_<N>.npy      (T, 1, 6)   physical moments [F, CoPx, CoPy, sxx, syy, sxy]
    <out>/clip_<N>.npy       (T, 1, 16, 16) float16 raw pressure (for the map models)
    <out>/pose_<N>.npy       (T, 21, 3)  float16 Rokoko hand landmarks (may be absent)
    <out>/manifest.jsonl     one record per clip (append-only across shards)

Design notes (see SESSION_LOG 2026-08-06/07):
  * OpenTouch instruments ONLY the right hand ("We instrument only the right dominant
    hand to simplify hardware and standardize annotations", arXiv:2512.16842), so the
    hand axis has extent 1. ActionSense's layout is (T, 2, 6); we keep the axis so the
    harness indexing is identical.
  * The 16x16 grid carries only 169 live taxels; dead cells read ~0 and contribute zero
    weight to the moments, so F/CoP need no mask. Any per-taxel statistic DOES.
    `--taxel-stats` dumps the per-taxel activity so the dead set can be identified once.
  * NO baseline correction is applied here. The ActionSense DC-offset bug (P4) must not
    be blind-fixed on a different sensor: clips are segmented around a pressure peak, so
    a causal first-N-frames baseline may already sit in contact. `--taxel-stats` measures
    the resting level; the correction is chosen from that measurement, downstream.

Usage (one shard at a time; the streaming driver deletes each shard after this returns):
    python scripts/extract_opentouch.py --shard data/office_csail_p2.hdf5 \
        --labels final_annotations --out data/opentouch_states
    python scripts/extract_opentouch.py --shard <f> --labels <d> --out <d> --taxel-stats
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PRESSURE_KEY = "right_pressure"
POSE_KEYS = ("right_hand_landmarks", "hand_landmarks")
TS_KEYS = ("timestamps", "timestamp")


def load_labels(labels_dir: str) -> dict:
    """clip_id -> row dict, pooled over every *_merged.csv."""
    rows = {}
    paths = sorted(glob.glob(os.path.join(labels_dir, "**", "*.csv"), recursive=True))
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                cid = (r.get("clip_id") or "").strip()
                if cid:
                    rows[cid] = {k: (v or "").strip() for k, v in r.items()}
    return rows


def label_lookup(labels: dict, shard_stem: str, group: str, prefixes: list[str]):
    """Join a shard's clip group to its label row.

    The obvious key is "<shard_stem>::<group>". One shard family breaks that rule:
    grocery_target p3/p4 were merged into a single annotation file whose clip_ids are
    prefixed `grocery_target_p3_p4_merged_by_ts`. 26 shards vs 25 CSVs is exactly this.
    A silent miss here is what left 457 clips 'unlabeled' in the 2026-07-02 probe, so we
    try the direct key, then the raw group, then any prefix that contains the shard stem.
    """
    for key in (f"{shard_stem}::{group}", group):
        if key in labels:
            return labels[key], "direct"
    stem_core = shard_stem.replace("_merged_by_ts", "")
    for pre in prefixes:
        if stem_core in pre or pre in stem_core:
            key = f"{pre}::{group}"
            if key in labels:
                return labels[key], f"fallback:{pre}"
    return None, "MISS"


def moments(p: np.ndarray) -> np.ndarray:
    """(T,H,W) pressure -> (T,6) [F, CoPx, CoPy, sxx, syy, sxy], coords in [-1,1].

    Identical maths to src/actionsense/physical_state.py, minus the baseline correction
    (deliberately deferred; see the module docstring).
    """
    T, H, W = p.shape
    ys = np.linspace(-1.0, 1.0, H)[:, None]
    xs = np.linspace(-1.0, 1.0, W)[None, :]
    p = np.clip(p.astype(np.float64), 0.0, None)
    F = p.sum(axis=(1, 2))
    safe = np.where(F > 0, F, 1.0)
    cx = (p * xs).sum(axis=(1, 2)) / safe
    cy = (p * ys).sum(axis=(1, 2)) / safe
    dx = xs[None, :, :] - cx[:, None, None]
    dy = ys[None, :, :] - cy[:, None, None]
    sxx = (p * dx * dx).sum(axis=(1, 2)) / safe
    syy = (p * dy * dy).sum(axis=(1, 2)) / safe
    sxy = (p * dx * dy).sum(axis=(1, 2)) / safe
    out = np.stack([F, cx, cy, sxx, syy, sxy], axis=1)
    out[F <= 0, 1:] = 0.0          # CoP undefined with no contact; masked downstream
    return out


def next_index(out_dir: str) -> int:
    mf = os.path.join(out_dir, "manifest.jsonl")
    if not os.path.exists(mf):
        return 0
    n = 0
    with open(mf) as f:
        for line in f:
            if line.strip():
                n = max(n, json.loads(line)["idx"] + 1)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True, help="one *.hdf5 shard")
    ap.add_argument("--labels", required=True, help="final_annotations dir")
    ap.add_argument("--out", required=True, help="cache dir (appended to)")
    ap.add_argument("--no-clips", action="store_true",
                    help="skip raw 16x16 maps (breaks the deferred map-model branch)")
    ap.add_argument("--taxel-stats", action="store_true",
                    help="also dump per-taxel activity/rest stats for the dead-taxel and "
                         "baseline decisions")
    args = ap.parse_args()

    import h5py

    os.makedirs(args.out, exist_ok=True)
    labels = load_labels(args.labels)
    prefixes = sorted({k.split("::")[0] for k in labels})
    stem = os.path.splitext(os.path.basename(args.shard))[0]
    idx = next_index(args.out)

    n_ok = n_miss = 0
    ts_all, tax_sum, tax_rest, n_frames = [], None, [], 0
    manifest = open(os.path.join(args.out, "manifest.jsonl"), "a")

    with h5py.File(args.shard, "r") as h5:
        root = h5["data"] if "data" in h5 else h5
        for group in sorted(root.keys()):
            g = root[group]
            if not hasattr(g, "keys") or PRESSURE_KEY not in g:
                continue
            press = np.asarray(g[PRESSURE_KEY], dtype=np.float32)      # (T,16,16)
            if press.ndim != 3 or press.shape[0] < 2:
                continue
            row, how = label_lookup(labels, stem, group, prefixes)
            if row is None:
                n_miss += 1
                continue

            st = moments(press)[:, None, :]                             # (T,1,6)
            np.save(os.path.join(args.out, f"state_{idx}.npy"), st.astype(np.float32))
            if not args.no_clips:
                np.save(os.path.join(args.out, f"clip_{idx}.npy"),
                        press[:, None, :, :].astype(np.float16))        # (T,1,16,16)

            pose = None
            for k in POSE_KEYS:
                if k in g:
                    pose = np.asarray(g[k], dtype=np.float16)
                    break
            if pose is not None:
                np.save(os.path.join(args.out, f"pose_{idx}.npy"), pose)

            ts = None
            for k in TS_KEYS:
                if k in g:
                    ts = np.asarray(g[k]).astype(np.float64).ravel()
                    break
            fps = None
            if ts is not None and ts.size > 1:
                d = np.diff(ts)
                d = d[d > 0]
                if d.size:
                    scale = 1e9 if np.median(d) > 1e6 else 1.0   # ns vs s
                    fps = float(scale / np.median(d))
                    ts_all.append(fps)

            manifest.write(json.dumps({
                "idx": idx, "shard": stem, "clip_id": f"{stem}::{group}", "join": how,
                "scene": row.get("clip_id", "").split("::")[0] or stem,
                "participant": (row.get("clip_id", "").split("::")[0] or stem),
                "action": row.get("action", ""), "grip_type": row.get("grip_type", ""),
                "object_name": row.get("object_name", ""),
                "object_category": row.get("object_category", ""),
                "environment": row.get("environment", ""),
                "peak_idx": row.get("peak_idx", ""), "onset_idx": row.get("onset_idx", ""),
                "post_idx": row.get("post_idx", ""),
                "T": int(press.shape[0]), "fps_est": fps,
                "has_clip": (not args.no_clips), "has_pose": pose is not None,
            }) + "\n")

            if args.taxel_stats:
                s = press.reshape(press.shape[0], -1)
                tax_sum = s.sum(0) if tax_sum is None else tax_sum + s.sum(0)
                tax_rest.append(np.percentile(s, 5, axis=0))
                n_frames += s.shape[0]
            idx += 1
            n_ok += 1

    manifest.close()
    print(f"[{stem}] extracted {n_ok} clips, {n_miss} label-miss, next idx {idx}")
    if ts_all:
        print(f"[{stem}] fps est: median {np.median(ts_all):.2f} "
              f"min {np.min(ts_all):.2f} max {np.max(ts_all):.2f}")
    if args.taxel_stats and tax_sum is not None:
        act = tax_sum / max(n_frames, 1)
        rest = np.median(np.stack(tax_rest), axis=0)
        dead = int((act <= 0).sum())
        np.save(os.path.join(args.out, f"taxelstats_{stem}.npy"),
                np.stack([act, rest]))
        print(f"[{stem}] taxels: {dead}/256 all-zero | mean-activity "
              f"p50 {np.median(act):.1f} | REST(p5) p50 {np.median(rest):.2f} "
              f"max {rest.max():.2f}  <- DC offset check")
    if n_miss:
        print(f"[{stem}] WARNING {n_miss} clips found no label row", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
