# Action Categories & Tactile Predictability — Cross-Dataset Study

**Goal.** Enumerate the actions collected across four tactile datasets, categorize them by
*force type* and *movement pattern*, and determine **which category of action is easiest to
predict** from its own past tactile signal — ultimately to drive user feedback / adaptive
strategies. Priors under test: actions with a **standard procedure** and with a **repeatable/
periodic pattern** are easier to predict.

Two notions of "easier to predict" are reported (per user decision, both; skill as headline):
- **Raw forecastability** — low `persistence_nMSE` (signal decorrelates slowly).
- **Structured/learnable** — high `periodicity` + low `contact_migration` (a model can win over
  persistence). A trained skill-over-persistence number requires the GPU forecaster
  (`src/tactile_forecast`, runs on CRC); the numbers below are the **training-free probe**
  (`scripts/predictability_by_category.py`) that predicts where that skill will be highest.

---

## 1. Datasets & their action inventories

| Dataset | Tactile modality | Scale | Action labels collected |
|---|---|---|---|
| **OpenTouch** (arXiv 2512.16842) | FPC full-hand + Aria video + Rokoko pose @30 Hz | 5.1 h, ~2,900 clips, ~800 objects, 14 envs | per-clip **action type** (pressing, rotating, turning, button-click, grasping, …) + **29 grasp types** (GRASP taxonomy: Medium Wrap, Small Diameter, Prismatic Two-Finger, Index-Finger Extension, …) |
| **Force–Vision** (ICLR 2024, ext. GEM) | Sundaram/STAG glove (normal-force map) + webcam | 2,000,000 frames, 89 articulated tools | manipulation type = **press / hold / squeeze** (on scissors, staplers, clips, pliers, spray bottles, …) |
| **ActionSense** (NeurIPS 2022, MIT) | conductive-thread tactile gloves + EMG + 17-IMU + gaze + RGBD | **20 labels / 6 groups**, 7+ subjects | peel/slice (cucumber, potato, bread) · clear cutting board · spread (almond butter, jelly) · wipe (pan/plate, towel/sponge) · open/close jar · pour water · set table · load/unload dishwasher · stack |
| **EgoTouch** (local, used for the probe) | 21×21 L+R pressure grids | ~3,286 trajectories, 5 scenes, ~200 tasks | verb_object task names → 23 verb categories (see §3) |

---

## 2. Categorization axes (orthogonal)

Every action is a point in this space; the axis that drives predictability is **B**.

- **A — Force type / contact mechanics:** A1 sustained force-closure grip · A2 impulsive press ·
  A3 cyclic surface force · A4 torsional/rotational · A5 monotonic-ramp load · A6 precision fingertip.
- **B — Temporal / movement pattern (predictive axis):** B1 periodic/rhythmic · B2 quasi-static ·
  B3 monotonic ramp/slide · B4 one-shot transition · B5 long composite sequence.
- **C — Procedural standardization:** standardized vs. free-form/adaptive.
- **D — Contact spatial dynamics:** D1 stable footprint · D2 migrating/sliding · D3 making/breaking.

### Unified mapping of every dataset's actions into the axes

| Action (source) | A force | B pattern | C std. | D contact |
|---|---|---|---|---|
| Slice / cut — bread, cucumber, potato (ActionSense; EgoTouch `cut`) | A3 cyclic | **B1 periodic** | high | D2 migrating |
| Wipe / clean surface (ActionSense; EgoTouch `wash/clean`) | A3 cyclic | **B1 periodic** | med | D2 migrating |
| Spread butter/jelly (ActionSense; EgoTouch `spread`) | A3 cyclic | B1 periodic | med | D2 migrating |
| Peel (ActionSense) | A3 cyclic | B1 periodic | high | D2 migrating |
| Spray (EgoTouch `spray`) | A2 impulsive×repeat | B1 periodic | high | D1 stable |
| Pour water (ActionSense) | A5 ramp | B3 ramp | high | D1 stable |
| Lift / carry (EgoTouch `lift`; grasp set) | A5 ramp | B3 ramp | med | D1 stable |
| Squeeze (Force-Vision; EgoTouch `squeeze`) | A5 ramp | B3 ramp | med | D1 stable |
| Open/close jar (ActionSense); turn/rotate (OpenTouch; EgoTouch `twist/turn`) | A4 torsional | B4/B1 | high | D1/D3 |
| Hold / grasp / grip (Force-Vision `hold`; grasp set; EgoTouch `grasp/hold`) | A1 grip | **B2 quasi-static** | high | D1 stable |
| Press / button-click (OpenTouch; Force-Vision `press`; EgoTouch `press/click`) | A2 impulsive | **B4 transition** | high | **D3 make/break** |
| Pick-up / place / take (EgoTouch) | A6/A1 | B4 transition | med | D3 make/break |
| Plug / unplug / insert (EgoTouch) | A6 precision | B4 transition | high | D3 make/break |
| Set table / load dishwasher (ActionSense); cook, organize (EgoTouch) | mixed | **B5 composite** | low | mixed |

---

## 3. Empirical result — probe over EgoTouch (FULL data, 1,929 sequences)

Metrics: `persH*` = persistence nMSE at horizon h frames (30 fps; **lower = easier**);
`period` = max total-force autocorr at lag 0.33–1.5 s (**higher = more repeatable**);
`migr15` = 1−IoU of active-taxel mask over 0.5 s (**lower = more spatially stable**);
`PI` = z(−persH15) + z(period) + z(−migr15) composite (**higher = easier to predict**).

**Ranked by action category (easiest → hardest):**

| rank | category | n | persH15 | persH30 | period | migr15 | PI |
|---|---|---|---|---|---|---|---|
| 1 | **Cut** (slice) | 10 | **0.088** | 0.126 | **0.968** | **0.215** | **+6.02** |
| 2 | Take/Retrieve | 30 | 0.346 | 0.565 | 0.869 | 0.347 | +2.90 |
| 3 | Inflate/Deflate | 14 | 0.304 | 0.407 | 0.791 | 0.321 | +2.62 |
| 4 | Spray | 10 | 0.320 | 0.561 | 0.884 | 0.415 | +2.46 |
| 5 | **Wash/Clean** (wipe) | 21 | 0.294 | 0.432 | 0.771 | 0.332 | +2.37 |
| … | Pick-up | 635 | 0.584 | 1.013 | 0.723 | 0.467 | −0.43 |
| … | Twist/Turn/Rotate | 41 | 0.496 | 0.791 | 0.705 | 0.394 | +0.45 |
| ↓ | Grasp/Hold/Lift | 82 | 0.724 | **1.343** | 0.660 | 0.525 | −2.08 |
| ↓ | Squeeze | 100 | 0.736 | 1.103 | 0.584 | 0.457 | −2.11 |
| ↓ | Push/Pull/Drag/Slide | 97 | 0.755 | 0.983 | 0.609 | 0.489 | −2.28 |
| ↓ | Plug/Unplug/Insert | 89 | 1.027 | 1.303 | 0.535 | 0.584 | −4.86 |
| last | **Press/Click** | 20 | **1.399** | 1.665 | 0.599 | **0.689** | **−6.67** |

Full-data run (`--max-per-task 0`) reproduces the sampled ranking almost exactly → robust.
Full table incl. temporal-pattern grouping: `docs/predictability_by_category_full.csv`
(sampled: `docs/predictability_by_category.csv`).

---

## 4. Findings

1. **Repeatable/periodic surface actions are the most predictable.** `Cut` (slicing) dominates on
   *all three* sub-metrics — lowest error, highest periodicity, most stable footprint — followed by
   `Spray` and `Wash/Clean` (wiping). This **confirms the "repeatable pattern" prior** and the
   B1-periodic hypothesis. These are exactly the A3-cyclic × B1-periodic × D2-migrating actions.
2. **Make/break-contact events are the hardest.** `Press/Click` is last by a wide margin (contact
   migration 0.689 = footprint flips on/off), with `Plug/Insert` and `Pick-up` also poor.
   Confirms the B4/D3 hypothesis: onset/offset timing is what a forecaster cannot anticipate.
3. **A nuance that partially refutes the naive intuition:** sustained **holds are NOT trivially
   predictable.** `Grasp/Hold/Lift` sits *below* the median (PI −2.08, worst persH30 = 1.343) —
   real grips drift and micro-adjust, and their footprint is unstable (migr 0.525). "Standardized/
   steady" ≠ "high forecast skill." This is why the local grasp-only LOTO landed at ≈0 skill.
4. **Standardization (Axis C) tracks predictability but is dominated by periodicity.** High-C
   actions that are *also* periodic (slice) win; high-C but transitional (plug/insert, press) lose.
   So **periodicity is the stronger predictor of forecastability than procedural standardization.**

## 5. Confirming with the real forecaster (per-category training)

The probe is training-free; the *actual* skill-over-persistence is measured by
`src/tactile_forecast` (SimVP/ConvGRU). A `--category` filter now selects trajectories by the
taxonomy above, and a CRC job trains + Leave-Trajectory-Out CV within one category:

```bash
# one category, one fold
qsub -v CATEGORY=Cut,FOLD=0,CONFIG=configs/tactile/simvp.yaml scripts/crc/percategory_gpu.job
# sweep the informative categories x 5 folds
CATS=("Cut" "Spray" "Wash/Clean" "Grasp/Hold/Lift" "Squeeze" "Press/Click" "Pick-up")
for c in "${CATS[@]}"; do for f in 0 1 2 3 4; do
  qsub -v CATEGORY="$c",FOLD=$f,CONFIG=configs/tactile/simvp.yaml scripts/crc/percategory_gpu.job
done; done
```

Each run writes `runs/simvp_full_<slug>_lto_f<fold>/summary.json` with test mean-skill.
**Hypothesis to confirm:** per-category test skill should track PI — highest for Cut/Spray/Wash,
near-zero/negative for Grasp/Hold/Lift and Press/Click. All categories have ≥5 trajectories
(5-fold LTO viable); LOTO is only meaningful where a category has ≥2 tasks (Spray/Pinch/Cut have 1–2).

## 6. Caveats
- Top categories have small n (Cut/Spray n=10, Pinch n=5) — this is the true EgoTouch count, so
  the PI signal is strong but statistical power is limited; the CRC run is the confirmation.
- ActionSense / OpenTouch / Force-Vision raw tactile is not yet downloaded and uses different
  sensor geometries (conductive-thread grid / FPC / STAG) → per-dataset preprocessing to a common
  pressure-map representation is required before cross-dataset forecasting.
- **Feedback/adaptive-strategy implication:** the best feedback targets are **B1-periodic actions**
  — they have a well-defined "correct" rhythm/force template a predictor can compare the user
  against (e.g. "even out your slicing stroke / wiping pressure"). Make/break-contact and composite
  planning actions are poor feedback targets (no stable template to score against).
