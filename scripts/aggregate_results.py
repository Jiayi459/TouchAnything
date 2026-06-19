"""Aggregate CV results across runs/. Groups summary.json files by (model, scope,
protocol) and reports mean +/- std test skill vs persistence, plus per-horizon means.

    python scripts/aggregate_results.py [--runs runs]
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

DIR_RE = re.compile(r"^(?P<name>.+)_(?P<scope>grasp|full)_(?P<protocol>lto|loto)_f(?P<fold>\d+)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()

    groups = defaultdict(list)  # (name,scope,protocol) -> list of summary dicts
    for sj in sorted(glob.glob(os.path.join(args.runs, "*", "summary.json"))):
        m = DIR_RE.match(os.path.basename(os.path.dirname(sj)))
        if not m or not os.path.getsize(sj):
            continue
        s = json.load(open(sj))
        if "test" not in s:
            continue
        key = (m["name"], m["scope"], m["protocol"])
        groups[key].append((int(m["fold"]), s))

    if not groups:
        print(f"No completed runs with a test split under {args.runs}/")
        return

    for key in sorted(groups):
        runs = sorted(groups[key])
        name, scope, protocol = key
        skills = [s["test"]["mean_skill"]["model"] for _, s in runs]
        folds = [f for f, _ in runs]
        print(f"\n=== {name} | {scope} | {protocol} | folds {folds} (n={len(runs)}) ===")
        print(f"  mean-skill vs persistence: {np.mean(skills):+.4f} +/- {np.std(skills):.4f}  "
              f"(per-fold: {', '.join(f'{x:+.3f}' for x in skills)})")
        # per-horizon mean skill across folds
        horizons = sorted(int(h) for h in runs[0][1]["test"]["model"].keys())
        per_h = {h: np.mean([s["test"]["model"][str(h)]["skill"] for _, s in runs])
                 for h in horizons}
        print("  skill@h: " + ", ".join(f"h{h}={per_h[h]:+.3f}" for h in horizons))


if __name__ == "__main__":
    main()
