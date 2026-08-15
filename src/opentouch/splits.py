"""The OpenTouch train/val/test split — held out by LOCATION, not by clip.

Blocked since 2026-08-10 on a question this module deliberately does not answer: whether a
shard's "_p1"/"_p2" suffix names a participant or a recording session. The paper does not
define it (arXiv:2512.16842 describes 14 environments and 5-25 minute sessions, defines no
official split, and never mentions leakage), and the corpus carries no participant field.

So the unit of holdout is the LOCATION: every shard sharing a base name -- office_ml_p1 and
office_ml_p2, hardware_homedepot_p1..p5 -- goes to the same side. Under the "session"
reading no session is split across sides; under the "participant" reading no participant is
either. The question stops mattering instead of being guessed at. 26 shards collapse to 12
locations: hardware_homedepot 5, home_kitchen / grocery_target / fablab_ml 3 each,
sports_dicks / office_ml / office_csail / eat_ygf 2 each, and four singletons.

WHAT THIS DOES NOT GUARANTEE, and cannot: if one person recorded in several locations, they
appear on both sides. Nothing in the manifest identifies people, so no split built from it
can rule that out. Say so wherever these numbers are reported rather than implying a
participant-disjoint protocol.

WHY LOCATION AND NOT SCENE OR CLIP. Objects belong to places: the same jar, shelf and
counter recur across a location's clips, and a participant's habits recur within a session.
A clip-level or scene-level split puts near-duplicates on both sides and every score comes
out flattering. Coarseness is the price of not lying about generalization.

    python -m src.opentouch.splits                     # write splits.json, print the summary
    python -m src.opentouch.splits --seed 3 --dry-run
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re

from src.actionsense.eval_harness.config import load_config, Config
from .dataset import eligible_clips, group_keys, missing_groups

PART = re.compile(r"_p\d+$")


def location(shard: str) -> str:
    """'office_ml_p2' -> 'office_ml'. Shards without a suffix are their own location."""
    return PART.sub("", (shard or "").strip())


def by_location(clips) -> dict[str, list[int]]:
    out = collections.defaultdict(list)
    for r in clips:
        out[location(r.get("shard", ""))].append(r["idx"])
    return {k: sorted(v) for k, v in sorted(out.items())}


def assign(units: dict[str, list[int]], frac, seed: int) -> dict[str, str]:
    """Location -> bucket, greedily filling whichever bucket is furthest below quota.

    Sequential filling ("keep adding to train until 60% is reached") is what the throwaway
    driver did, and with 12 very unequal units it overshoots badly -- one 5-shard location
    can carry a whole bucket past its quota. Choosing the neediest bucket for each unit in
    turn keeps the proportions close even when unit sizes differ by 5x.
    """
    order = list(units)
    random.Random(seed).shuffle(order)                  # seed varies which peers go where
    order.sort(key=lambda u: -len(units[u]))            # but the awkward big units are
    #                                                    placed first, while there is room.
    # (Shuffling AFTER the sort would undo it -- the first version did exactly that and
    #  pushed 89% of the clips into train.)
    total = sum(len(v) for v in units.values())
    quota = dict(zip(("train", "val", "test"), (f * total for f in frac)))
    have = {k: 0 for k in quota}
    out = {}
    for u in order:
        b = max(quota, key=lambda k: quota[k] - have[k])
        out[u] = b
        have[b] += len(units[u])
    return out


def build(cfg: Config, frac=(0.6, 0.2, 0.2), seed: int = 0) -> dict:
    """-> {"train": [...], "val": [...], "test": [...], "locations": {...}, "seed": ...}

    Raises if the result would hand AR a group it never fit -- the check missing_groups()
    was written for. With TRAIN-relative grouping that can only happen when TRAIN itself
    has no "other" clips, which is worth failing loudly on rather than papering over.
    """
    clips = eligible_clips(cfg)
    units = by_location(clips)
    if len(units) < 3:
        raise ValueError(f"only {len(units)} locations; cannot make three splits")
    where = assign(units, frac, seed)

    ids = {b: sorted(i for u, bb in where.items() if bb == b for i in units[u])
           for b in ("train", "val", "test")}
    empty = [b for b, v in ids.items() if not v]
    if empty:
        raise ValueError(f"split {empty} came out empty at seed={seed}; try another seed")

    gtr = group_keys(cfg, ids["train"], ids["train"])
    for part in ("val", "test"):
        miss = missing_groups(gtr, group_keys(cfg, ids[part], ids["train"]))
        if miss:
            raise ValueError(
                f"{part} needs AR groups TRAIN never fits: {sorted(miss)}. With "
                f"TRAIN-relative grouping this means TRAIN has no clip in that group at "
                f"all (typically no 'other'); try another --seed.")

    return {**ids, "locations": {b: sorted(u for u, bb in where.items() if bb == b)
                                 for b in ("train", "val", "test")},
            "seed": seed, "frac": list(frac), "unit": "location"}


def save(cfg: Config, splits: dict, path: str | None = None) -> str:
    path = path or cfg.abspath("split_file")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)
    return path


def load(cfg: Config, path: str | None = None) -> dict:
    with open(path or cfg.abspath("split_file")) as f:
        return json.load(f)


def summarize(cfg: Config, splits: dict) -> str:
    n = {b: len(splits[b]) for b in ("train", "val", "test")}
    tot = sum(n.values())
    lines = [f"unit=location seed={splits['seed']} | clips {tot}"]
    for b in ("train", "val", "test"):
        lines.append(f"  {b:5s} {n[b]:5d} ({n[b] / tot:5.1%})  "
                     + ", ".join(splits["locations"][b]))
    gtr = group_keys(cfg, splits["train"], splits["train"])
    lines.append(f"  AR groups fitted on TRAIN: {len(set(gtr.values()))} "
                 f"(incl. 'other' = {sum(v == 'other' for v in gtr.values())} clips)")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/opentouch/eval_harness.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac", default="0.6,0.2,0.2")
    ap.add_argument("--out", help="default: paths.split_file from the config")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    sp = build(cfg, tuple(float(x) for x in a.frac.split(",")), a.seed)
    print(summarize(cfg, sp))
    if not a.dry_run:
        print("wrote", save(cfg, sp, a.out))


if __name__ == "__main__":
    raise SystemExit(main())
