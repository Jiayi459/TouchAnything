# SESSION_LOG.md

Source-of-truth log of plans, modifications, analyses, questions/answers, and decisions.
Newest session at the bottom.

---

## Session 1 — 2026-06-17 — Environment setup, dataset download, working agreement, GitHub fork

### Context / platform
- Repo: TouchAnything (local at `c:\Users\haoji\TouchAnything`), Windows 11, PowerShell + Git Bash.
- `origin` remote = `https://github.com/Jianyi2004/TouchAnything` (the **original/upstream** repo).

### Work completed earlier this session
1. **Environment (no conda originally installed; `environment.yaml` is Linux-only).**
   - Created `.venv\` from system Python 3.10.11.
   - Later installed **Miniconda** at `C:\Users\haoji\miniconda3` (conda 26.3.2), initialized for PowerShell.
   - Built Windows-friendly conda env **`touchanything`** (Python 3.10.20, from `conda-forge` to avoid Anaconda commercial ToS).
   - Both envs hold identical **data-only** deps: huggingface_hub 1.5.0, hf_xet, numpy 1.24.3, h5py 3.15.1, opencv-python 4.8.1, pandas, scipy, tqdm, pillow, decord. **Not** the training stack (torch/lightning/triton/xformers/nvidia-* are Linux/GPU-only and not installable here).
2. **Dataset download** — `zhouzhoujy/EgoTouch` (HF), **metadata-only (no mp4)** via `scripts/download_egotouch.py` (`ignore_patterns=["*.mp4"]`).
   - Result: `datasets/EgoTouch/` = 14.91 GB, 11,258 json + 3,286 npz + `split.json` (full annotation coverage; videos skipped per user choice). Integrity spot-check passed (split.json dict with train/val/test_seen/test_unseen; pressure_grids.npz shapes (1652,21,21)).
   - Note: `run_convert_to_hdf5.sh` → `scripts/core/convert_to_hdf5.py` REQUIRES `chest/left/right.mp4` + `pressure_grids.npz` + `wilor_hands.json`, so HDF5 conversion needs a `--videos` re-download to run end-to-end. `wilor_hands.json` is JSON-Lines.

### This task: working agreement + GitHub fork

**Decisions/actions:**
- Created `CLAUDE.md` (Working Agreement) **verbatim** from user-provided text. ⚠️ See OPEN QUESTION 4 (project name reads "IntelligentCarpet"; references `compute_com.py` / "Session 2" from a different project).
- Created `.gitignore` excluding `datasets/`, `.venv/`, `.claude/`, logs, `__pycache__` — verified: only `CLAUDE.md`, `.gitignore`, `SESSION_LOG.md`, `scripts/download_egotouch.py` remain untracked. **Critical** so the 15 GB dataset / venv are never pushed.
- Created this `SESSION_LOG.md`.

**Goal:** "upload this repo to github as a fork from the original repo" = push current local state to a fork of `Jianyi2004/TouchAnything` under the **user's** GitHub account.

**Blockers found:** no `gh` CLI (either shell); `git user.name`/`user.email` unset; credential helper = `manager`; user's GitHub username unknown.

**Planned steps (PENDING user resolution of open questions — not yet executed):**
1. Set `git user.name` / `user.email`.
2. Create the fork of `Jianyi2004/TouchAnything` under the user's account (method TBD — see OQ2).
3. Re-point remotes: `origin` → user's fork, `upstream` → `Jianyi2004/TouchAnything`.
4. Commit the new local files (CLAUDE.md, .gitignore, SESSION_LOG.md, download script) — scope TBD (OQ3).
5. Push to the fork. (Outward-facing/publishing action — requires explicit go-ahead.)

### OPEN QUESTIONS — RESOLVED (2026-06-17)
1. **GitHub account/username** → to be obtained automatically from `gh auth status` after the user authenticates (see OQ2 answer).
2. **Fork creation method** → **Install `gh` + authenticate.** I install GitHub CLI; user runs `gh auth login` (interactive browser/device flow — I cannot do this for them); then `gh repo fork` + push.
3. **Commit scope** → **All 4 new files**: `CLAUDE.md`, `.gitignore`, `SESSION_LOG.md`, `scripts/download_egotouch.py`. Dataset + `.venv` stay excluded via `.gitignore`.
4. **CLAUDE.md content** → **Adapt to TouchAnything.** Done: "IntelligentCarpet" → "TouchAnything"; removed the `compute_com.py`/"Session 2" parenthetical in directive 5.

### Execution progress
- [x] CLAUDE.md adapted to TouchAnything.
- [x] Installed `gh` 2.94.0 at `C:\Program Files\GitHub CLI\gh.exe` (not on PATH in pre-existing shells; available in new terminals).
- [x] User authenticated `gh` as **Jiayi459** (scopes: repo, workflow, read:org, gist).
- [x] Set repo-local git identity: `Jiayi459 <jh9141@nyu.edu>`.
- [x] Created fork **`Jiayi459/TouchAnything`** (isFork=true, parent=Jianyi2004/TouchAnything).
- [x] Re-pointed remotes: `origin` → `Jiayi459/TouchAnything`, `upstream` → `Jianyi2004/TouchAnything`.
- [x] Committed 4 files (`1509fe9`) and pushed `main` to the fork. Remote HEAD verified = `1509fe9`. Dataset/.venv/.claude excluded (confirmed not staged).

### COMPLETED 2026-06-17. Fork live at https://github.com/Jiayi459/TouchAnything

### OPEN ITEM (not part of commit)
- `README.md` has an **accidental working-tree edit**: the string `& "C:\Program Files\GitHub CLI\gh.exe" auth login` was pasted into line 39 mid-sentence (likely a stray paste in the IDE). It was **not** staged/committed/pushed. Pending user decision: revert via `git restore README.md`, or keep/fix manually.

---

## Session 2 — 2026-06-18 — Dataset action categorization + grasp-success forecast (PLANNING)

### Request (verbatim intent)
Go through the EgoTouch dataset; (1) divide all data into categories based on **hand action**; (2) determine which category is **suitable for grasping an item**; (3) **forecast grasp success possibility**.

### Exploration completed (facts)
- Local dataset = metadata-only (no mp4): `datasets/EgoTouch/`. Structure: `Scene/task_name/trajectory_id/{pressure_grids.npz, wilor_hands.json, hamer_hands.json, rokoko_hands.json, vive_poses.json, manual_contact_annotation.json, masks.npz, jq_pressure.json}`.
- **212 real tasks** (213 folders minus `Home/metadata`), **1933 trajectories**. Scenes: Home 124, Office 13, Outdoor 25, Retail 19, Workbench 32 (task counts).
- Task names encode action verbs: grasp/grip/hold/lift, pick_up (largest), pull/push/drag, open/close, fold/spread/wring, plug_unplug, squeeze/pinch, twist/turn/rotate, play (games/sports), swing/throw/bounce/hit/toss, spray/press/click/slide, use/wash/buy/shop/take/put/move/organize/cut/assemble/write, etc.
- **Signals available** (metadata only): `pressure_grids.npz` = left/right tactile grids (T,21,21) normalized to [0,1] (attr `tactile_max`); hand pose (wilor/hamer); camera/wrist poses (vive/rokoko); masks.
- **No grasp success/failure labels exist.** `manual_contact_annotation.json` only has coarse per-traj `left_contact`/`right_contact` booleans, True in only ~5-6% of a 120-sample → NOT a usable success label.
- Pressure data quality: 120-traj sample had 0 all-NaN grids; some trajectories have partial-NaN frames (e.g., `Home/grasp_cola/20260320_090636_772` left grid). Must handle NaNs.

### Env feasibility
- Current `touchanything` conda env / `.venv`: numpy/scipy/h5py/opencv/pandas present. **No scikit-learn** (would need install) and **no deep-learning stack** (Linux/GPU-only, not installable on Windows). So: heuristic/statistical forecast = feasible now; classical ML (sklearn) = feasible after install; deep model = NOT feasible here.

### Proposed plan (PENDING user resolution — not yet implemented per CLAUDE.md directive 5)
1. Categorize all 1933 trajectories by hand-action type via task-name verb taxonomy (transparent, reproducible).
2. Mark grasp-suitable categories (core: grasp/grip/hold/lift/pick_up; partial: squeeze/pinch/take/twist_cap).
3. Define a grasp-success **proxy** from tactile pressure (no GT exists) and compute per-category success probability + per-trajectory scores; emit a markdown report + CSV.

### OPEN QUESTIONS — RESOLVED (2026-06-18)
1. **Categorization method** → **name/verb taxonomy.**
2. **"Suitable for grasp item"** → **core grasp verbs only** (grasp/grip/hold/lift + pick_up).
3. **Grasp-success forecast** → **DEFERRED.** "For now only classify, not yet for prediction success." Do not build any success metric/model yet.
4. **Deliverables** → **wait for user instruction after classification.** Classification done locally; nothing committed yet.

### Classification result (script: `scripts/categorize_actions.py`, NOT committed pending instruction)
- Method: assign each task to one action category by first known action verb in its name (verb_object convention).
- **212 tasks, 1930 trajectories** across 23 categories.
- **CORE GRASP (grasp-suitable) = `Grasp/Hold/Lift` (8 tasks, 82 traj) + `Pick-up` (64 tasks, 635 traj) = 72 tasks / 717 traj (~37% of all trajectories).**
- Largest non-grasp categories: Organize/Arrange (140), Use tool/appliance (131), Open/Close (130), Squeeze (100), Push/Pull/Drag/Slide (98), Plug/Unplug/Insert (89).
- Full per-category task assignment printed to console; reviewed edge cases (e.g. `pick_up_and_squeeze_power_bank`→Pick-up by leading verb; `grip_hand_dynamometer`→Grasp/Hold/Lift; `handle_*`,`remove_pen_cap`,`split_*`→Other).

### Pending user instruction
- Whether to commit `scripts/categorize_actions.py` and emit a report/CSV.
- When/if to proceed to the grasp-success forecast (and which success definition).

---

## Session 3 — 2026-06-18 — Grasp/Hold/Lift tactile subset + tactile→tactile forecasting plan

### Request
Download (prepare) the Grasp/Hold/Lift data; goal = train a model to predict **future tactile
from past tactile**. Produce an implementation plan: literature review, process, method choice,
validation, preprocessing. Be rigorous/constructive/precise/efficient.

### Key reasoning / decisions
- Task is **tactile→tactile** forecasting ⇒ **videos not needed**; `pressure_grids.npz` already
  on disk from the metadata pull. "Download" = prepare the subset, not re-fetch MP4s.
- Prepared subset `datasets/grasp_hold_lift_tactile/` (+ `manifest.csv`) via
  `scripts/prepare_grasp_tactile.py`. **82 traj, 31,577 frames @30fps.**

### EDA facts (ground the plan)
- Lengths skewed: min 71 / median 125 (~4.2s) / mean 385 / max 2206 ⇒ windowing required.
- Each hand 21×21 with **~50.8% structurally-NaN cells = fixed sensor mask** (~220 valid taxels);
  values in [0,1]; ~33% valid taxels active/frame (sparse).
- **Predictability probe** (`scripts/tactile_predictability_probe.py`): persistence nMSE
  0.04→0.13→0.23→0.47→0.72→1.34 at h=1/3/5/10/15/30; force autocorr 0.99→...→−0.17 at lag 30.
  ⇒ honest horizon **0.1–0.5s**; persistence is a strong baseline ⇒ judge models by **skill vs
  persistence**; **N=82 is the binding constraint**.

### Literature review (in plan)
Tactile prediction: ACTP/ACTVP (arXiv:2205.09430, Conv-LSTM, slip), DFPC strawberry
(2303.05393), Dream-Tac (2606.08737), Tactile diffusion policy (2510.13324). Backbones: OpenSTL,
ConvLSTM, PredRNN, **SimVP/TAU** (2206.05099, CVPR'22/23), PredFormer (2410.04733), survey
(2401.14718). Recommendation: **SimVP/TAU headline + ConvLSTM baseline**; transformer/generative
as extensions.

### Compute (UPDATED after user: "we have gpu, school CRC")
- Two-tier: local Windows (`touchanything` env, CPU torch) for dev/baselines; **CRC GPU cluster
  (Linux+CUDA, likely SLURM)** for training/CV/ablations. `environment.yaml` builds on cluster.
- GPU enables **pretrain on all 1,930 traj → fine-tune on 82 grasp clips** to fight small-N.

### Artifacts (NOT committed — pending approval per CLAUDE.md)
- `docs/TACTILE_PREDICTION_PLAN.md` (full plan), `scripts/prepare_grasp_tactile.py`,
  `scripts/tactile_predictability_probe.py`, `datasets/grasp_hold_lift_tactile/` (gitignored).

### DECISIONS — RESOLVED (2026-06-18, user)
1. Horizon = **0.5 s** (15 frames @30fps); report 1/3/5/10/15.
2. Hands = **both** (bimanual 2-ch primary, dominant-hand ablation).
3. Method = **ConvGRU primary** + ConvLSTM precedent baseline + SimVP/TAU headline CNN.
   GRU question answered: ConvGRU chosen for N=82 (fewer params/less overfit, keeps spatial
   structure); plain GRU rejected (flattening loses contact geometry).
4. Compute = **ND CRC**. CONFIRMED scheduler = **UGE/`qsub`** (NOT SLURM), GPU via
   `-q gpu -l gpu_card=1` (4-day limit). Conda: init once, `conda activate` in jobs.
5. **Pretrain on all 1,930 traj → fine-tune on 82 grasp clips.**
6. **Deterministic** next-frame prediction (generative deferred).
7. **CUDA** env (no local CPU torch).

### CRC env setup created (this turn; NOT committed)
- `scripts/crc/environment_tactile_cuda.yaml` (lean conda env), `scripts/crc/setup_crc_env.sh`
  (conda init + env + CUDA torch 2.5.1/cu124), `scripts/crc/train_gpu.job` (UGE GPU template),
  `scripts/crc/README.md` (rsync + setup + qsub workflow).
- Plan `docs/TACTILE_PREDICTION_PLAN.md` updated: §4 model lineup (ConvGRU), §6 compute (UGE),
  §10 decisions resolved.
- Refs: CRC GPU docs https://docs.crc.nd.edu/resources/gpu.html ; conda
  https://docs.crc.nd.edu/popular_modules/conda.html

### BUILD ("go", 2026-06-18) — `src/tactile_forecast/` package implemented
- `tactile_utils.py` (torch-free: mask/transform/window/splits/metrics) — **verified locally**
  on real data: mask 217 valid/hand (structural, 0 variance across trajs), 5,955 windows
  (Tin10/Tout15/stride5), LTO 5-fold (65/17), LOTO 8-fold, metric sanity OK.
- `data.py` (TactileWindows + trajectory-level split_train_val, mask-safe aug),
  `models/{conv_rnn(ConvGRU+ConvLSTM, scheduled sampling), simvp}`, `models.build_model`,
  `engine.py` (masked MSE + active-taxel weight, train/eval, SS schedule),
  `baselines.py` (persistence, last-velocity), `train.py`/`eval.py` (CLI: lto/loto, fold,
  grasp/full, pretrain & --pretrained finetune; outputs best.pt/train_log/test_metrics/summary).
- `configs/tactile/{convgru,convlstm,simvp}.yaml`. CRC: `scripts/crc/smoke_test.py` (synthetic
  e2e) + updated `train_gpu.job` (runs entrypoint via -v CONFIG/FOLD/SCOPE/PROTOCOL/PRETRAINED)
  + README run recipe.
- **All 11 modules byte-compile.** Torch path NOT run locally (no local torch per decision #7);
  to be smoke-tested on CRC (`python scripts/crc/smoke_test.py`).
- Headline metric = mean **skill vs persistence** (must be >0); honest horizon ≤0.5 s.

### CORRECTION (2026-06-18) — TAU not implemented
- User asked where the "TAU" training method is. **It is not in the code.** Built models =
  SimVP-lite, ConvGRU, ConvLSTM only. Plan said "SimVP/TAU" (family name) but only SimVP exists
  (`configs/tactile/simvp.yaml`; no `tau.yaml`). TAU = Temporal Attention Unit (Tan et al.,
  CVPR 2023, arXiv:2206.12126): SimVP skeleton + TAU translator (intra-frame statical + inter-frame
  dynamical attention, parallelizable) + Differential Divergence Regularization loss.
- ACTION PENDING user choice: either implement TAU (`models/tau.py` + DDR in engine + tau.yaml)
  OR edit plan wording "SimVP/TAU" → "SimVP" to match code.

### CRC RUN LOG
- 2026-06-19: Pushed full pipeline to fork (7e3bec0). User cloned on crcfe01.
- **Fix (fff1db7):** `setup_crc_env.sh` aborted on `set -u` — CRC `/etc/bashrc` has unbound
  `BASHRCSOURCED`. Removed `set -u`; now sources `$(conda info --base)/etc/profile.d/conda.sh`
  directly instead of `conda init`+`.bashrc`. Env `tactile` had not been created; gave user
  manual create commands + the fixed script.

### FIRST TRAINING RESULT + FIX (2026-06-19)
- Smoke test passed on GPU (A10, cuda True) after two harness fixes: smoke mask must be batched
  (B,C,H,W) [3a05619]; `horizon_metrics` mask cast to bool [65f1d64].
- First real run (ConvGRU, LTO fold0, grasp, no pretrain): pipeline OK end-to-end but **test
  mean-skill ≈ 0.0038 ≈ break-even with persistence** (skill@h ~0.01→0). Training loss *rose*
  over epochs under scheduled sampling. Diagnosis: models predict ABSOLUTE frames → easiest
  optimum is to copy last frame (= persistence). Persistence is very strong (probe: autocorr
  0.99@33ms).
- **Fix [390861c]: residual prediction** — models output Δ from last observed frame
  (`pred=clamp(last+Δ,0,1)`); persistence == zero delta, so skill comes from learned deviations.
  `residual` flag (default true) in ConvGRU/ConvLSTM/SimVP + configs. Also silenced torch.load
  weights_only warning. Awaiting rerun to confirm positive skill.

### RESIDUAL RESULT CONFIRMED (2026-06-19)
- ConvGRU, LTO fold0, grasp, no pretrain, residual=on: **test mean-skill = 0.174** vs persistence
  (last_vel −2.4). skill@h: h1=−0.04 (persistence near-unbeatable at 33ms), rising monotonically
  to h15=+0.25. => Future tactile IS predictable from past beyond persistence; gain grows with
  horizon over 0.5s. Caveat: single fold (test=17). Added `scripts/aggregate_results.py`
  [2e706c2] for mean±std across folds.

### LTO 5-FOLD CV RESULTS (grasp, no pretrain, residual on) — 2026-06-19
(2h runtime was recurrent models' Python time-loop, not a hang; completed fine.)
- ConvGRU : 0.138 ± 0.056 (h1=-0.090, h15=+0.207)
- ConvLSTM: 0.152 ± 0.031 (h1=+0.036, h15=+0.211)
- **SimVP : 0.192 ± 0.044 (h1=+0.065, h15=+0.235) — BEST at every horizon**
=> Conclusive: future tactile predictable from past beyond persistence (~19% error reduction
   over 0.5s). SimVP (non-recurrent) beats both recurrent models AND is far faster; recurrent
   models weak/negative at h1. **Promote SimVP to primary** (overturns prior ConvGRU pref).
   Fold 2 hardest for all (likely long grip_hand_dynamometer test split).

### LOTO RESULT (SimVP, grasp, no pretrain) — 2026-06-21
- **SimVP LOTO = +0.005 ± 0.111** (per-fold -0.131..+0.151). vs LTO +0.192.
- KEY FINDING: grasp-only learned tactile dynamics are **object-specific** — do NOT generalize
  to unseen objects (mean ≈ persistence; some folds worse). Motivates pretraining.

### PRETRAIN->FINETUNE SETUP [092ea63]
- train.py `--exclude-grasp`: drop the 8 grasp tasks during pretrain so LOTO held-out object is
  never seen (no leakage). download_egotouch.py `--pressure-only` (~1.7GB npz). env += hf_hub.
- Workflow: download --pressure-only -> pretrain SimVP --scope full --pretrain --exclude-grasp
  (~1848 traj) -> finetune --protocol loto --pretrained ... --out runs/simvp_ft_grasp_loto_fN.
- Compare `simvp_ft | grasp | loto` vs baseline `simvp | grasp | loto` (+0.005).

### PRETRAIN DONE (2026-06-22)
- First attempt ran on CPU front-end (device=cpu, 361k windows) -> stuck 13h; killed. Cause:
  ran on crcfe01 (no GPU). Fix: qsub GPU batch job + stride20/batch256 [60a49ab]; also fixed
  UGE inline-comment bug on `#$ -M` [6b93fb8].
- Pretrain (SimVP, scope full, --exclude-grasp = 1851 traj, 30 epochs, GPU) completed in ~5h
  (slow due to per-epoch val eval over ~62k windows). best.pt @ epoch23, val_skill 0.227 on
  held-out NON-grasp data => good general tactile predictor. -> runs/simvp_pretrain/best.pt

### HEADLINE RESULT (2026-06-23) — pretraining unlocks unseen-object prediction
- SimVP LOTO: scratch **+0.005 ± 0.111** -> pretrained->finetuned **+0.097 ± 0.122** (~18x,
  positive at all horizons; helped 6/8 held-out objects; fold5 stays ~-0.20 outlier).
- Full story: (1) tactile predictable from past (LTO +0.192); (2) doesn't generalize from few
  objects (LOTO scratch ~0); (3) broad multi-object pretrain enables it (LOTO +0.097).
- Documented in `docs/RESULTS.md` [aa7e49d]. CORE STUDY COMPLETE.

### OPTIONAL NEXT
- Visualization: predicted-vs-GT pressure GIFs (eval-based renderer).
- Ablations: LTO-finetune (does pretrain help seen-object too?); smaller SimVP (30.5M is
  over-param); investigate fold5 outlier object.
- Perf: speed up engine.evaluate (memory-heavy concat) if rerunning pretrain.
- rsync runs/ back to local for plots.
- Set up pretrain-on-full (1,930 traj) → finetune (needs full npz on CRC: rsync or add HF
  downloader). Minor: h1 slightly negative (model adds noise at easiest horizon) — possible
  later tweak (per-horizon loss weighting).
- Commit decision: Session 2/3 artifacts (categorize script, tactile prep/probe scripts, plan,
  src/tactile_forecast, configs, crc setup) — not yet committed/pushed to the fork.
- [ ] Set `git user.name`/`user.email` (name from gh login; email jh9141@nyu.edu).
- [ ] `gh repo fork Jianyi2004/TouchAnything` → re-point origin to fork, upstream to original.
- [ ] Commit 4 files, push to fork (publishing — proceed only after auth confirmed).

---

## Session 4 — 2026-07-01 — New direction: which ACTION CATEGORY is most predictable (cross-dataset)

### Goal (user request)
Read three tactile/force datasets, enumerate every action collected, categorize actions (by
force type / movement pattern / etc.), then run per-category prediction to find **which category
of action is easiest to predict** — or at least summarize the *traits* of an action series that
make it predictable. Priors given by user: (a) actions with a **standard procedure** are easier
to predict; (b) actions with a **repeatable/periodic pattern** are easier. Ultimate goal: use the
predictor to give the user **feedback / adaptive strategies** to improve performance.

This is a NEW research thread built on the existing tactile→tactile forecasting infra
(`src/tactile_forecast`, skill-over-persistence metric, LTO/LOTO protocols). Session 1-3
established: LTO seen-object +0.192; LOTO unseen ~0; pretrain→finetune LOTO +0.097.

### Datasets read (2026-07-01) — sources
1. **OpenTouch** (opentouch-tactile.github.io, arXiv 2512.16842). First in-the-wild egocentric
   FULL-HAND tactile dataset. Modalities @30Hz: FPC-based tactile sensor + Meta Aria egocentric
   video + Rokoko Smartgloves hand pose; 2ms sync. 5.1h recordings, ~2,900 curated clips,
   ~800 objects, 14 environments, 14 object categories. Labels per clip: object name, object
   category, environment, **action type**, **grasp type** (29 grasps from GRASP taxonomy:
   e.g. Medium Wrap, Small Diameter, Prismatic Two-Finger, Index-Finger Extension), NL caption.
   Action examples named in text: pressing, rotating, turning, button click, grasping; contact
   with chair/table/transparent objects. Full action list is in Supp. Mat. (not enumerated on
   arXiv HTML; would need supp PDF).
2. **Force–Vision / "Learning to Jointly Understand Visual and Tactile Signals"** (ICLR 2024,
   Li/Liu et al., extends GEM). Cross-modal force+vision on **articulated tools**. Sensor:
   Sundaram et al. STAG-style tactile glove (full-hand NORMAL force map) + webcam. Scale:
   **2,000,000 paired frames over 89 real object instances** (scissors, staplers, clips/clamps,
   pliers, spray bottles, ...). Manipulation types explicitly analyzed: **press, hold, squeeze**.
   Key finding they report: "press" clusters apart from "hold"/"squeeze" — press activates a
   *contiguous* hand region and is *one-directional*; squeeze/hold create *force closure* → the
   two force-closure actions are more similar to each other. (Directly relevant to our
   force-type axis.)
3. **ActionSense** (NeurIPS 2022 D&B, MIT CSAIL). Multimodal WEARABLE kitchen dataset.
   Modalities: custom conductive-thread tactile gloves + Myo EMG (forearm muscle) + 17-IMU Xsens
   body tracking + finger gloves + Pupil eye-tracking w/ first-person cam + 5 RGB + depth + 2 mic.
   **20 unique activity labels in 6 task categories** (Fig 2), ~7+ subjects. The 6 categories:
   (i) Peeling & slicing (cucumber/potato/bread; + auxiliary "clear cutting board"),
   (ii) Spreading (almond butter / jelly on bread w/ knife),
   (iii) Wiping (pan/plate w/ towel or sponge — periodic circular/linear strokes, force key),
   (iv) Open/close a jar (rotational, subtle, tactile+EMG key),
   (v) Pouring water (monotonically changing container weight; transparent liquid),
   (vi) High-level tableware sequences (set table; load/unload dishwasher; stacking).
   NOTE full 20 leaf labels live in Fig 2 / supp (not machine-readable from the main-text PDF).

### Existing repo datasets already in this taxonomy
- **EgoTouch** (21×21 pressure grid) — general in-the-wild tactile (used for pretraining).
- **grasp_hold_lift_tactile** — 8 tasks: grasp_body_lotion, grasp_cola, grasp_floral_water,
  grasp_power_adapter, grasp_sunscreen, grip_hand_dynamometer, hold_teapot, lift_towel.
  These are ALL sustained-grip/hold/lift = one corner of the proposed taxonomy (explains why
  LOTO≈0: near-static maps where persistence is already strong → low skill headroom).

### PROPOSED unified action taxonomy (draft — for user review)
Categorize along orthogonal axes; each concrete action = a point in this space.

**Axis A — Force type / contact mechanics**
- A1 Sustained force-closure grip/hold (grasp_*, hold_teapot, jar-hold, FV "hold"/"squeeze")
- A2 Impulsive one-directional press (button click, FV "press", stapler)
- A3 Cyclic surface force (wiping, spreading, slicing strokes, peeling strokes, scrubbing)
- A4 Torsional / rotational (open/close jar, turn knob, OpenTouch "rotating"/"turning")
- A5 Monotonic ramp load (pouring — weight ↓; lifting — load ↑; squeeze-to-close)
- A6 Precision fingertip (pinch, click, fine slice)

**Axis B — Movement / temporal pattern** (this is the predictability-driving axis)
- B1 Periodic / rhythmic-repeatable (wiping, slicing, peeling, spreading) → hypothesis HIGH skill
- B2 Quasi-static / near-constant (hold, grip, sustained press) → low MSE but LOW *skill* (persistence wins)
- B3 Monotonic ramp (pouring, lifting) → MEDIUM
- B4 Discrete one-shot transition (button click, jar-open "snap", pick-place onset) → LOW
- B5 Composite long-horizon sequence (set table, load dishwasher) → LOW (planning + many sub-actions)

**Axis C — Procedural standardization** (user prior)
- C-high: standardized (regular slicing strokes, standard jar twist, standard pour tilt)
- C-low: free-form/adaptive (wiping strategy, spreading adapts to substance, tableware planning)

**Axis D — Contact spatial dynamics of the tactile map**
- D1 Stable footprint (same taxels active) — grip/hold/press → spatially trivial
- D2 Migrating/sliding contact (wiping, slicing, peeling) — contact region translates
- D3 Making/breaking contact (pick-place, click) — onset/offset hardest

### Predictability hypothesis (to TEST, ranked most→least "skill-over-persistence")
1. B1 periodic surface actions (A3×B1×D2): structured motion persistence CAN'T capture → highest skill.
2. A4 rotational / B3 monotonic ramps: some learnable trend → medium.
3. A1/B2 sustained holds: low raw error but ~0 skill (persistence already near-perfect).
4. B4 discrete events / B5 long sequences: lowest (event timing / planning).
This predicts the user's priors partly REVERSE under a skill-over-persistence metric: "easy to
hold steady" ≠ "high forecasting skill." Must pin down the metric (OPEN Q1).

### OPEN QUESTIONS (must resolve before any implementation — plan-before-code)
- **Q1 — Definition of "easier to predict."** Raw accuracy (MSE/IoU/force-MAE) vs.
  **skill-over-persistence** (structured, learnable dynamics)? These rank categories differently
  (static holds win #1 on raw error but ~0 on skill). RECOMMEND: report both; headline on
  skill-over-persistence since that's where feedback/adaptive strategies have leverage.
- **Q2 — Which dataset(s) for this iteration?** ActionSense is the cleanest labeled *everyday-action*
  taxonomy with tactile time-series (best fit). OpenTouch adds action+grasp labels; force-vision
  adds press/hold/squeeze. Do we have download access? (EgoTouch was downloaded metadata-only;
  ActionSense/OpenTouch/FV not yet fetched, and their tactile sensor geometries differ from the
  21×21 EgoTouch grid → new preprocessing per dataset.)
- **Q3 — Prediction target / modality.** Continue tactile→tactile forecasting (reuse infra), or
  predict a force scalar, or cross-modal (vision/pose→tactile)? "Feedback to enhance performance"
  hints we may want to compare a user's applied force to a learned "ideal" template.
- **Q4 — Category granularity.** Use the proposed A/B/C/D axes (recommend B as primary grouping),
  or a flatter user-defined category set?
- **Q5 — Scope of THIS step.** Deliver categorization + study design only (await answers), or also
  stand up a first per-category forecasting run on whatever tactile data is already local?

### ANSWERS (user, 2026-07-01)
- Q1 = **Both, skill as headline** (report MSE/IoU/force-MAE + skill; rank by skill-over-persistence).
- Q2 = **All three datasets** — produce ONE unified categorization spanning ActionSense + OpenTouch
  + Force-Vision (+ local EgoTouch/grasp).
- Q3 = tactile / **physical-representation prediction** (predict the tactile physical signal;
  reuse tactile→tactile forecasting representation).
- Q5 = **Also prototype now.**

### CONSTRAINT REALITY CHECK
- torch NOT installable on this Windows box → cannot TRAIN here (training runs live on CRC GPU).
- Only EgoTouch (21×21 grids) + grasp_hold_lift tactile are downloaded locally; ActionSense /
  OpenTouch / Force-Vision raw data NOT local (different sensor geometries → per-dataset preprocessing later).
- ∴ "Prototype now" = a **training-free predictability probe** (numpy only) grouped BY CATEGORY over
  local EgoTouch. Directly measures "which category is most predictable" without a GPU. Existing
  `scripts/tactile_predictability_probe.py` (persistence nMSE, autocorr, smoothness) + 
  `scripts/categorize_actions.py` (verb→category) are the building blocks — MERGE them.

### PLAN (this step)
1. `scripts/predictability_by_category.py` — for every EgoTouch trajectory: categorize (verb map)
   + map to a **temporal-pattern class (Axis B)**, load pressure_grids.npz (L+R, nan→0), compute
   per-sequence: persistence nMSE (RAW hardness), constant-velocity nMSE, **velocity-skill vs
   persistence** (learnable first-order dynamics headroom = skill proxy), **periodicity score**
   (max total-force autocorr at lag 10–45 = rhythmic/repeatable evidence), **contact-migration**
   (1−IoU of active-taxel mask across h). Aggregate & RANK by category and by B-class. Write CSV.
2. `docs/ACTION_CATEGORIES.md` — unified cross-dataset taxonomy table mapping every ActionSense /
   OpenTouch / Force-Vision / EgoTouch action into Axes A(force)/B(temporal)/C(standardization)/
   D(contact-dynamics), with the per-category predictability numbers attached where measurable.
3. Interpret: does empirical velocity-skill/periodicity confirm the hypothesis (B1 periodic >
   ramps > holds > events)? Feed into the feedback/adaptive-strategy goal.

### IMPLEMENTATION (2026-07-01)
- Wrote `scripts/predictability_by_category.py` (numpy/venv only; imports `categorize` from
  `categorize_actions.py`). Per EgoTouch trajectory: persistence nMSE @h={1,5,15,30}, periodicity
  (max total-force autocorr, lag 10–45 frames), contact_migration (1−IoU active-taxel mask @h15).
  Composite `PI = z(−persH15)+z(periodicity)+z(−migr15)`. Groups by verb category AND temporal
  pattern (Axis B); ranks; writes `docs/predictability_by_category.csv`.
- DISCARDED a constant-velocity skill proxy: `h·velocity` extrapolation blows up on impulsive
  tactile spikes (velSk ≈ −15..−37), noise-dominated → not a valid training-free skill proxy.
  A real skill-over-persistence number needs the GPU forecaster. Documented in script docstring.
- Ran `--max-per-task 12` → 1,493 sequences.

### RESULTS (probe, EgoTouch, n=1493) — ranked easiest→hardest (PI)
- Easiest: **Cut/slice** PI+6.11 (persH15 0.088, periodicity 0.968, migr 0.215) >> Take +2.84 >
  Inflate +2.57 > **Spray** +2.38 > **Wash/Clean (wipe)** +2.31.
- Hardest: **Press/Click** −7.14 (persH15 1.399, migr 0.689) < Pinch −4.51 < Plug/Insert −3.82 <
  Fold −2.62 < Push/Pull −2.27 < Squeeze −1.98 < **Grasp/Hold/Lift −1.57**.
- FINDINGS: (1) periodic surface actions (cut/spray/wipe) most predictable — CONFIRMS "repeatable
  pattern" prior + B1 hypothesis. (2) make/break-contact events (press/click, plug) hardest —
  CONFIRMS B4/D3. (3) NUANCE refuting naive view: sustained HOLDS are NOT trivially predictable
  (Grasp/Hold/Lift below median, worst persH30=1.223) — grips drift + footprint unstable; explains
  grasp-only LOTO≈0. (4) periodicity predicts forecastability better than procedural standardization.
- Small-n caveat on top categories (Cut/Spray n=10). Next: full run (`--max-per-task 0`), then
  CONFIRM by running `src/tactile_forecast` per-category on CRC GPU (probe PI = the hypothesis).

### DELIVERABLES this session
- `docs/ACTION_CATEGORIES.md` — unified cross-dataset taxonomy (ActionSense+OpenTouch+Force-Vision
  +EgoTouch) mapped into Axes A/B/C/D + empirical predictability table + feedback-target implication
  (B1-periodic actions are the good feedback targets: they have a "correct rhythm/force template").
- `scripts/predictability_by_category.py`, `docs/predictability_by_category.csv`.
- Saved OpenTouch/ActionSense/Force-Vision paper PDFs were parsed via pypdf (pdftoppm/Read-PDF
  unavailable on Windows) to extract exact taxonomies.

### FOLLOW-UP (a) FULL-DATA PROBE + (b) PER-CATEGORY FORECASTER (2026-07-01/02)
User: "do a and b".
- (a) Ran probe `--max-per-task 0` → **1,929 sequences**. Ranking reproduces the sampled run
  almost exactly (Cut PI +6.02, Take +2.90, Inflate +2.62, Spray +2.46, Wash +2.37; bottom:
  Press/Click −6.67, Plug/Insert −4.86, Pinch −4.17, Fold −2.34, Push/Pull −2.28, Squeeze −2.11,
  Grasp/Hold/Lift −2.08). → ranking is ROBUST to sampling. Wrote `docs/predictability_by_category_full.csv`.
- (b) Wired the REAL forecaster for per-category confirmation:
  - NEW `src/tactile_forecast/categories.py` = single source of truth (VERB_CATEGORY, CORE_GRASP,
    categorize, TEMPORAL_PATTERN, all_categories). Pure stdlib (both `src/__init__.py` and
    `src/tactile_forecast/__init__.py` are torch-free, so local scripts can import it).
  - Refactored `scripts/categorize_actions.py` to import from that module (adds repo root to
    sys.path) — removes the duplicated verb map. Verified it still runs (212 tasks / 1930 traj).
  - `src/tactile_forecast/train.py`: added `--category NAME` (filters trajectories by
    categorize(task)); run-dir now carries a slug tag (`simvp_full_<slug>_lto_f<fold>`).
  - NEW `scripts/crc/percategory_gpu.job` (UGE) takes `-v CATEGORY,FOLD,CONFIG,PROTOCOL`; trains
    LTO within one category on `--scope full`. Header has the all-categories×5-folds submit loop.
  - Verified (torch-free): `--category` filter + slug over real data → every category has ≥5
    trajectories (5-fold LTO viable); slugs clean. py_compile passes on all edited files.
- Doc `docs/ACTION_CATEGORIES.md` updated with full-data table + §5 confirmation-run instructions.
- STATE: cannot train locally (no torch). CRC run is the remaining step to turn the probe
  HYPOTHESIS (PI ranking) into MEASURED per-category skill. All artifacts uncommitted (fork not set up).

### CRC STAGING + AGGREGATION (2026-07-02)
User: "stage the CRC commands". Also confirmed prediction methods = 3 architectures.
- METHODS (verified in `models/__init__.py` build_model): **ConvGRU**, **ConvLSTM** (both
  `ConvRNNSeq2Seq`, cell gru/lstm), **SimVP** (`simvp.py`, headline). Plus 2 non-learned baselines
  in eval (persistence, last_velocity). **TAU is NOT implemented** — this SimVP is "SimVP-lite"
  (Conv translator, n_trans=4), not the gated Temporal Attention Unit. TAU would be a translator
  swap if we want it; noted as optional.
- BUG FIXED: `scripts/aggregate_results.py` DIR_RE could not parse per-category run dirs
  (`simvp_full_cut_lto_f0`) → those runs were silently skipped. Rewrote regex to capture an
  optional slug `(?:_(?P<category>[^_]+(?:-[^_]+)*))?`; unit-tested on 7 dir names incl. the
  existing `simvp_ft_grasp_loto_f5`. Aggregator now groups by category and prints a
  **PER-CATEGORY RANKING** (mean test skill) — the study headline that confirms/breaks the probe PI.
- NEW `scripts/crc/run_percategory.sh` — one-command sweep (9 categories × 5 folds = 45 SimVP
  jobs; CONFIG/CATS/FOLDS overridable). CATS list uses only space-free category names (slashes ok).
- `scripts/crc/README.md` §6 — full staging walkthrough: (A) rsync working tree, (B) rsync ONLY
  `pressure_grids.npz` of full EgoTouch to /scratch365 + symlink, (C) env+smoke, (D) submit sweep,
  (E) rsync runs back + `aggregate_results.py`. Uses NETID placeholder (CRC netid = jhao3).
- Ready to run on CRC. I cannot submit (no CRC/SSH/torch here) — user launches it.

### SCOPE EXPANSION: all 4 datasets, probe-first, OpenTouch next (2026-07-02)
User: not grasp-focused (clarified: EgoTouch sweep already ranks ALL 23 categories, grasp is
just one). Wants which category is predictable ACROSS all 4 datasets. Decisions (AskUserQuestion):
next=OpenTouch, depth=probe-first (training-free, no GPU), where=CRC.
- DATA AVAILABILITY (checked): all 4 downloadable. OpenTouch = public Google-Drive via
  `scripts/download_data.sh` (26 HDF5 shards + `final_annotations` labels). ActionSense =
  public (delpreto/ActionNet, CC-BY-NC, HDF5). Force-Vision = public Google-Drive zip.
- CROSS-SENSOR CAVEAT: EgoTouch 21x21 2-hand / OpenTouch 16x16 1-hand / ActionSense
  conductive-thread / FV STAG differ → CANNOT compare raw skill across datasets. Design: rank
  WITHIN each dataset, then pool by TEMPORAL-PATTERN axis (B1..B5) to test if the same action
  KIND wins everywhere (sensor-agnostic answer). OpenTouch `action` free-text is mapped through
  the SAME categorize() verb taxonomy → lands in the same category/pattern space as EgoTouch.
- OPENTOUCH SCHEMA (verbatim from repo build_label_data.py): HDF5 `data/<clip_id>/right_pressure`
  = (T,16,16); labels in CSV/TSV keyed by clip id, cols object_name/object_category/environment/
  action/grip_type. 30 Hz. Pressure raw up to ~3072 (scale-invariant metrics → no normalization).
- BUILT:
  - NEW `src/tactile_forecast/predictability.py` — shared numpy metrics (seq_metrics, aggregate,
    add_predictability_index) for ANY (T,C,H,W) sensor. Sanity-tested (periodic→period 1.0,
    static→migr 0.0, event→period 0.0). EgoTouch probe left with its inline copy (don't disturb
    the committed/running script); shared module is go-forward, used by OpenTouch probe.
  - NEW `scripts/opentouch_predictability.py` — `--inspect` (dump HDF5 tree + label cols + join
    rate) and probe modes; groups by temporal-pattern / mapped-category / raw-action / grip_type;
    writes docs/predictability_opentouch.csv. Robust: auto-detect HDF5 clip groups + label key col.
  - Both compile; h5py 3.15.1 present locally.
- OPEN RISK — DISK: OpenTouch HDF5 bundles RGB (`rgb_images_jpeg`) → shards likely large; CRC
  home only 35G free and /scratch365/jhao3 not provisioned. PLAN: gauge size (download labels +
  1 shard, measure), then either (a) fits → download all + probe, (b) too big → build a
  streaming driver (gdown shard → per-shard probe → delete → next) or request scratch from
  crcsupport. Decide after the size gauge.
