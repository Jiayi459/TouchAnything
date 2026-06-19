"""Categorize EgoTouch tasks/trajectories by hand action using a task-name verb taxonomy.

Classification only (no success forecasting). Assigns each task to one action
category by scanning its name tokens left-to-right for the first known action verb
(task names follow a verb_object convention). Core grasp categories are flagged.
"""
import os
from collections import defaultdict, OrderedDict

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "datasets", "EgoTouch")
SCENES = ["Home", "Office", "Outdoor", "Retail", "Workbench"]

# Ordered verb -> category. First matching token (scanning left to right) wins.
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


def categorize(task_name):
    for tok in task_name.split("_"):
        if tok in VERB_CATEGORY:
            return VERB_CATEGORY[tok]
    return "Other"


def main():
    rows = []  # (scene, task, category, n_traj)
    for sc in SCENES:
        sp = os.path.join(ROOT, sc)
        for t in sorted(os.listdir(sp)):
            tp = os.path.join(sp, t)
            if not os.path.isdir(tp) or t == "metadata":
                continue
            n = sum(1 for x in os.listdir(tp) if os.path.isdir(os.path.join(tp, x)))
            rows.append((sc, t, categorize(t), n))

    by_cat_tasks = defaultdict(list)
    by_cat_traj = defaultdict(int)
    for sc, t, cat, n in rows:
        by_cat_tasks[cat].append((sc, t, n))
        by_cat_traj[cat] += n

    order = sorted(by_cat_tasks, key=lambda c: (-by_cat_traj[c], c))
    tot_tasks = len(rows)
    tot_traj = sum(r[3] for r in rows)

    print(f"TOTAL: {tot_tasks} tasks, {tot_traj} trajectories\n")
    print(f"{'CATEGORY':<22} {'tasks':>6} {'traj':>6}  grasp?")
    print("-" * 50)
    for c in order:
        flag = "CORE-GRASP" if c in CORE_GRASP else ""
        print(f"{c:<22} {len(by_cat_tasks[c]):>6} {by_cat_traj[c]:>6}  {flag}")
    print("-" * 50)
    gt_tasks = sum(len(by_cat_tasks[c]) for c in CORE_GRASP if c in by_cat_tasks)
    gt_traj = sum(by_cat_traj[c] for c in CORE_GRASP if c in by_cat_traj)
    print(f"{'CORE-GRASP TOTAL':<22} {gt_tasks:>6} {gt_traj:>6}")

    # full task listing per category for review
    print("\n===== TASK ASSIGNMENTS PER CATEGORY =====")
    for c in order:
        print(f"\n[{c}]  ({len(by_cat_tasks[c])} tasks, {by_cat_traj[c]} traj)")
        for sc, t, n in by_cat_tasks[c]:
            print(f"    {sc}/{t}  ({n})")


if __name__ == "__main__":
    main()
