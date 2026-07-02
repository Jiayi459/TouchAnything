"""Canonical action-category taxonomy (single source of truth; pure stdlib, no torch).

Maps an EgoTouch task name (verb_object convention) to an action category by scanning
its tokens left-to-right for the first known action verb, and each category to a
temporal-pattern class (Axis B) used by the predictability study.

Imported by:
  - src/tactile_forecast/train.py   (--category filter for per-category forecasting)
  - scripts/categorize_actions.py   (re-exports; classification CLI)
  - scripts/predictability_by_category.py (via categorize_actions)

See docs/ACTION_CATEGORIES.md for the study and results.
"""
from __future__ import annotations

from collections import OrderedDict

# Ordered verb -> action category. First matching token (scanning left to right) wins.
VERB_CATEGORY = OrderedDict([
    # --- CORE GRASP (suitable for grasping an item) ---
    ("grasp", "Grasp/Hold/Lift"), ("grip", "Grasp/Hold/Lift"),
    ("hold", "Grasp/Hold/Lift"), ("lift", "Grasp/Hold/Lift"),
    ("pick", "Pick-up"),
    # --- other manipulations ---
    ("put", "Place/Put-down"), ("place", "Place/Put-down"),
    ("take", "Take/Retrieve"),
    ("push", "Push/Pull/Drag/Slide"), ("pull", "Push/Pull/Drag/Slide"),
    ("drag", "Push/Pull/Drag/Slide"), ("slide", "Push/Pull/Drag/Slide"),
    ("open", "Open/Close"), ("close", "Open/Close"), ("unfold", "Open/Close"),
    ("fold", "Fold/Cloth"), ("spread", "Fold/Cloth"), ("wring", "Fold/Cloth"),
    ("plug", "Plug/Unplug/Insert"), ("unplug", "Plug/Unplug/Insert"),
    ("insert", "Plug/Unplug/Insert"),
    ("squeeze", "Squeeze"),
    ("pinch", "Pinch"),
    ("twist", "Twist/Turn/Rotate"), ("turn", "Twist/Turn/Rotate"),
    ("rotate", "Twist/Turn/Rotate"),
    ("press", "Press/Click"), ("click", "Press/Click"),
    ("spray", "Spray"),
    ("swing", "Swing/Throw/Strike"), ("throw", "Swing/Throw/Strike"),
    ("bounce", "Swing/Throw/Strike"), ("hit", "Swing/Throw/Strike"),
    ("toss", "Swing/Throw/Strike"), ("practice", "Swing/Throw/Strike"),
    ("play", "Play (games/sports)"),
    ("wash", "Wash/Clean"), ("clean", "Wash/Clean"),
    ("buy", "Buy/Shop"), ("shop", "Buy/Shop"),
    ("cook", "Cook/Prepare"), ("brew", "Cook/Prepare"), ("mix", "Cook/Prepare"),
    ("make", "Cook/Prepare"), ("prepare", "Cook/Prepare"),
    ("boil", "Cook/Prepare"), ("collect", "Cook/Prepare"),
    ("organize", "Organize/Arrange"), ("sort", "Organize/Arrange"),
    ("arrange", "Organize/Arrange"), ("pack", "Organize/Arrange"),
    ("unpack", "Organize/Arrange"), ("assemble", "Organize/Arrange"),
    ("move", "Organize/Arrange"),
    ("cut", "Cut"),
    ("deflate", "Inflate/Deflate"), ("inflate", "Inflate/Deflate"),
    ("use", "Use tool/appliance"), ("work", "Use tool/appliance"),
    # --- leftover / misc verbs ---
    ("flip", "Other"), ("write", "Other"), ("change", "Other"),
    ("handle", "Other"), ("remove", "Other"), ("remote", "Other"),
    ("over", "Other"),
])

CORE_GRASP = {"Grasp/Hold/Lift", "Pick-up"}

# Axis B: temporal pattern class per verb category (a-priori hypothesis; see study).
# B1 periodic  B2 quasi-static  B3 ramp/slide  B4 one-shot transition  B5 long composite
TEMPORAL_PATTERN = {
    "Grasp/Hold/Lift":        "B2 quasi-static",
    "Pick-up":                "B4 transition",
    "Place/Put-down":         "B4 transition",
    "Take/Retrieve":          "B4 transition",
    "Push/Pull/Drag/Slide":   "B3 ramp/slide",
    "Open/Close":             "B4 transition",
    "Fold/Cloth":             "B1 periodic",
    "Plug/Unplug/Insert":     "B4 transition",
    "Squeeze":                "B3 ramp/slide",
    "Pinch":                  "B2 quasi-static",
    "Twist/Turn/Rotate":      "B1 periodic",
    "Press/Click":            "B4 transition",
    "Spray":                  "B1 periodic",
    "Swing/Throw/Strike":     "B4 transition",
    "Play (games/sports)":    "B5 composite",
    "Wash/Clean":             "B1 periodic",
    "Cook/Prepare":           "B5 composite",
    "Organize/Arrange":       "B5 composite",
    "Cut":                    "B1 periodic",
    "Inflate/Deflate":        "B3 ramp/slide",
    "Use tool/appliance":     "B5 composite",
    "Buy/Shop":               "B5 composite",
    "Other":                  "Other",
}


def categorize(task_name: str) -> str:
    """Return the action category for a task name (first known verb token wins)."""
    for tok in task_name.split("_"):
        if tok in VERB_CATEGORY:
            return VERB_CATEGORY[tok]
    return "Other"


def all_categories() -> list[str]:
    """Sorted list of distinct action categories."""
    return sorted(set(VERB_CATEGORY.values()))
