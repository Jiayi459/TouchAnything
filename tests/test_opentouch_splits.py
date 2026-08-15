"""Tests for src/opentouch/splits.py -- the location-level holdout.

Pure numpy/stdlib: no torch, so these run everywhere.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.actionsense.eval_harness.config import load_config          # noqa: E402
from src.opentouch import splits as S                                # noqa: E402
from src.opentouch.dataset import group_keys, missing_groups         # noqa: E402


def make_cfg(tmp_path, plan, min_group_size=30):
    """plan: [(shard, object_category, n_clips), ...] -> a config over a synthetic cache."""
    i = 0
    recs = []
    for shard, cat, n in plan:
        for _ in range(n):
            st = np.zeros((60, 1, 6), np.float32)
            st[:, 0, 0] = 50 + np.arange(60) * 0.1
            np.save(tmp_path / f"state_{i}.npy", st)
            recs.append({"idx": i, "shard": shard, "clip_id": f"c{i}",
                         "scene": shard, "action": "holding", "object_category": cat,
                         "environment": "e", "T": 60, "fps_est": 30.0,
                         "has_clip": False, "has_pose": False})
            i += 1
    (tmp_path / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in recs) + "\n")
    text = (open("configs/opentouch/eval_harness.yaml").read()
            .replace("states_root: data/opentouch_states", f"states_root: {tmp_path}")
            .replace("min_group_size: 30", f"min_group_size: {min_group_size}")
            .replace("split_file: data/opentouch_states/splits.json",
                     f"split_file: {tmp_path}/splits.json"))
    p = tmp_path / "harness.yaml"
    p.write_text(text)
    return load_config(str(p))


def test_location_strips_the_part_suffix():
    assert S.location("office_ml_p2") == "office_ml"
    assert S.location("hardware_homedepot_p5") == "hardware_homedepot"
    assert S.location("home_bedroom") == "home_bedroom"
    # only a trailing _pN is a part marker; digits elsewhere must survive
    assert S.location("grocery_p1_annex") == "grocery_p1_annex"


def test_shards_of_one_location_never_straddle_the_split(tmp_path):
    plan = [(f"{loc}_p{k}", cat, 12)
            for loc, cat, parts in [("office", "cup", 3), ("store", "box", 2),
                                    ("gym", "ball", 2), ("lab", "tool", 2),
                                    ("home", "pan", 2)]
            for k, cat in [(k, cat) for k in range(1, parts + 1)]]
    cfg = make_cfg(tmp_path, plan, min_group_size=5)
    sp = S.build(cfg, seed=0)

    rows = {r["idx"]: r for r in
            (json.loads(l) for l in open(tmp_path / "manifest.jsonl"))}
    seen = {}
    for bucket in ("train", "val", "test"):
        for i in sp[bucket]:
            loc = S.location(rows[i]["shard"])
            assert seen.setdefault(loc, bucket) == bucket, f"{loc} split across buckets"


def test_every_eligible_clip_lands_exactly_once(tmp_path):
    plan = [("a_p1", "cup", 10), ("a_p2", "cup", 10), ("b", "box", 10),
            ("c", "tool", 10), ("d", "pan", 10), ("e", "jar", 10)]
    cfg = make_cfg(tmp_path, plan, min_group_size=5)
    sp = S.build(cfg, seed=1)
    allids = sp["train"] + sp["val"] + sp["test"]
    assert len(allids) == len(set(allids)) == 60


def test_ar_never_sees_a_group_train_did_not_fit(tmp_path):
    """The KeyError('sports equipment') case: a category confined to one location."""
    plan = [("shop_p1", "cup", 40), ("shop_p2", "cup", 40), ("cafe", "mug", 40),
            ("lab", "tool", 40), ("gym", "sports", 40), ("home", "pan", 40)]
    cfg = make_cfg(tmp_path, plan, min_group_size=30)
    for seed in range(15):
        try:
            sp = S.build(cfg, seed=seed)
        except ValueError:
            continue                      # a seed may legitimately be rejected, loudly
        gtr = group_keys(cfg, sp["train"], sp["train"])
        for part in ("val", "test"):
            assert not missing_groups(gtr, group_keys(cfg, sp[part], sp["train"]))


def test_train_relative_grouping_is_what_makes_that_work(tmp_path):
    """Corpus-wide counting -- the old behaviour -- must still fail on the same split, or
    the previous test proves nothing about why it passes."""
    plan = [("shop_p1", "cup", 40), ("shop_p2", "cup", 40), ("cafe", "mug", 40),
            ("lab", "tool", 40), ("gym", "sports", 40), ("home", "pan", 40)]
    cfg = make_cfg(tmp_path, plan, min_group_size=30)
    sp = S.build(cfg, seed=0)
    old_tr = group_keys(cfg, sp["train"])                    # no train_idxs = corpus-wide
    old_missing = set()
    for part in ("val", "test"):
        old_missing |= missing_groups(old_tr, group_keys(cfg, sp[part]))
    assert old_missing, "expected the corpus-wide rule to leave an unfitted group here"


def test_determinism_and_seed_sensitivity(tmp_path):
    plan = [("a_p1", "cup", 10), ("a_p2", "cup", 10), ("b", "box", 10),
            ("c", "tool", 10), ("d", "pan", 10), ("e", "jar", 10)]
    cfg = make_cfg(tmp_path, plan, min_group_size=5)
    assert S.build(cfg, seed=2) == S.build(cfg, seed=2)
    assert S.build(cfg, seed=2)["test"] != S.build(cfg, seed=9)["test"]


def test_proportions_stay_near_the_quota(tmp_path):
    """Greedy neediest-bucket assignment, not sequential filling: with unit sizes differing
    5x, sequential filling overshoots the first bucket badly."""
    plan = [("big_p%d" % k, "cup", 20) for k in range(1, 6)] + \
           [("m1", "box", 12), ("m2", "tool", 12), ("m3", "pan", 12),
            ("s1", "jar", 6), ("s2", "lid", 6), ("s3", "tin", 6), ("s4", "pot", 6)]
    cfg = make_cfg(tmp_path, plan, min_group_size=5)
    tot = 100 + 36 + 24
    for seed in range(8):
        try:
            sp = S.build(cfg, seed=seed)
        except ValueError:
            continue
        # One indivisible 100-clip location out of 160 puts a hard floor under whichever
        # bucket holds it: ~62.5%. The quota cannot be met more closely than that, and a
        # split that pretends otherwise would have had to break the location apart.
        assert 0.50 <= len(sp["train"]) / tot <= 0.75
        assert len(sp["val"]) and len(sp["test"])


def test_save_and_load_round_trip(tmp_path):
    plan = [("a_p1", "cup", 10), ("a_p2", "cup", 10), ("b", "box", 10),
            ("c", "tool", 10), ("d", "pan", 10), ("e", "jar", 10)]
    cfg = make_cfg(tmp_path, plan, min_group_size=5)
    sp = S.build(cfg, seed=0)
    path = S.save(cfg, sp)
    assert S.load(cfg, path) == sp
    assert "location" == sp["unit"]


def test_too_few_locations_is_an_error(tmp_path):
    cfg = make_cfg(tmp_path, [("a", "cup", 10), ("b", "box", 10)], min_group_size=5)
    with pytest.raises(ValueError, match="locations"):
        S.build(cfg, seed=0)
