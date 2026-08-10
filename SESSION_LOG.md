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

### OPENTOUCH VALIDATED ON CRC (2026-07-02)
- GAUGE: 1 shard (office_csail_p2.hdf5) = 561 MB → 26 shards ≈ 14.6 GB → FITS in 35 GB home.
  No streaming/scratch needed. Download-all approved.
- SCHEMA CONFIRMED live: HDF5 top keys {calibration, data, transform_slam_to_rgb}; `data/<clip>`
  has right_pressure (T,16,16) f32 max=3072, camera_poses, rgb_images_jpeg, hand_landmarks,
  timestamps, plus a per-clip `labels`=(0,0) index-pair (NOT the action). Labels come from
  final_annotation.zip → `final_annotations/<scene>_merged.csv`, key col `clip_id` =
  "<scene>::demo_NNN" (globally unique), cols incl. action (gerund), grip_type (GRASP taxonomy),
  object_category, environment, description, peak_idx. One row per clip.
- FIRST PROBE (1 shard, 111/113 usable, 0 unlabeled → join works): grip-type ranking sensible
  (Prismatic-3-Finger/Medium-Wrap easiest; Prismatic-4-Finger/Index-Extension hardest). Action
  vocab = gerunds: placing/adjusting/removing/pinching/picking up/holding/pulling/pushing/moving/
  pressing/turning. Category/pattern were all "Other" (gerund mismatch) — FIXED:
  - categories.py: added `categorize_phrase()` (inflection stemmer: pulling→pull, cutting→cut,
    placing→place, picking up→pick) + new verbs (slice/peel/chop→Cut, pour/scoop→Pour[new,B3],
    wipe/scrub→Wash/Clean, adjust→Organize). Left EgoTouch `categorize()` + the `spread`→Fold/Cloth
    mapping UNCHANGED (no silent shift to the running EgoTouch sweep).
  - opentouch_predictability.py now uses categorize_phrase; verified all 20 observed/likely verbs
    map to the right category+pattern (only "removing"→Other).
  - NEW scripts/crc/download_opentouch.sh (26 shard IDs + labels, verbatim from opentouch repo).
- NEXT: download all shards, run probe → full OpenTouch per-category + temporal-pattern ranking;
  compare to EgoTouch by the B-axis.

### OPENTOUCH FULL RESULT (2026-07-02) — 26 shards, 2496 usable / 2958 clips (457 unlabeled)
- Raw-action ranking (trustworthy): TOP pouring +4.4 / serving +3.6 / eating +3.4 / stirring +3.0
  / scooping +2.5 / flipping +2.4 / wiping +1.3; BOTTOM cutting(n4) -3.0 / moving -2.6 / turning
  -2.2 / pulling -1.8. Standouts have persH15 0.26-0.39 vs pack 0.7-0.9.
- contact_migration ≈ 0.005 for ALL categories (single-hand grasp footprint never breaks) →
  DEGENERATE metric here; PI driven by persH15 + periodicity.
- CROSS-DATASET SURPRISE: OpenTouch temporal-pattern ranks B4>B2>B1 — OPPOSITE of EgoTouch (B1 high).
  Cause: a-priori verb→pattern map breaks (turning-a-latch != rhythmic turn; many OT verbs unmapped
  → "Other" holds predictable food actions; Pour mislabeled B3). LESSON: assign Axis B per-action
  from data, not a-priori.
- DURABLE FINDING (answers user's "trait" goal): predictable = smooth continuous slowly-varying
  contact force (pour/stir/scoop/serve/wipe/slice); unpredictable = abrupt onset / make-break
  engagement (press/plug/pull/move/tap/stiff-turn). persH15 = sensor-agnostic predictor.
- Documented in docs/ACTION_CATEGORIES.md §3b. docs/predictability_opentouch.csv written on CRC.
- OPEN: (a) expand taxonomy to OT vocab (stir/serve/eat/flip/examine/carry/lower/align/type/touch/
  tighten/unscrew/tilt/tap/feel/inspect/switch/detach/attach/point/rest) + re-derive Axis B
  empirically; (b) then ActionSense + Force-Vision same recipe; (c) optional GPU forecasting to
  confirm probe on OpenTouch.

### ACTIONSENSE BUILT (2026-07-02) — dataset #3
- SCHEMA (from delpreto/ActionNet parsing_data): wearables HDF5 per subject-session,
  `<device>/<stream>/{data,time_s,time_str}`. Tactile = `tactile-glove-left`/`-right` ->
  `tactile_data/data` = (N,H,W) grids. Labels = `experiment-activities/activities/data` rows
  [Activity, Start/Stop, Valid, Notes] + time_s; pair Start->Stop for intervals, drop Valid in
  {Bad,Maybe}. 20 activity phrases (Peel/Slice/Spread/Open-close jar/Pour/Clean/Set/Stack/Load/
  Unload/Get/Clear). Subjects S00-S05 wore tactile; S06-S09 did NOT.
- Continuous recording -> SEGMENT by activity intervals, RESAMPLE each clip to 30 Hz (match
  EgoTouch/OpenTouch frame-based metrics), stack L+R -> (T,2,H,W), probe. (Resample = mild
  smoothing confound; acceptable for a within-dataset ranking; noted.)
- BUILT: scripts/crc/download_actionsense.sh (12 wearables URLs, S00-S05, curl, ~small);
  scripts/actionsense_predictability.py (--inspect + probe; segment/resample/stack; groups by
  raw activity / category / temporal-pattern). Taxonomy: added tableware verbs set/stack/load/
  unload/clear/get -> Organize/Arrange (B5). Kept spread->Fold/Cloth (EgoTouch spread_bed_sheet
  is genuinely cloth; ActionSense butter-spread mislabels but raw-activity label is unambiguous).
- All 20 labels map sanely (Peel/Slice->Cut B1, Pour->Pour B3, Clean->Wash/Clean B1, jar->
  Open/Close B4, tableware->Organize B5). compile + resample verified locally.
- NEXT (user on CRC): git pull; bash scripts/crc/download_actionsense.sh; probe --inspect (confirm
  tactile shape/Fs); then full probe. Then cross-dataset synthesis (EgoTouch+OpenTouch+ActionSense).

### ACTIONSENSE RESULT + THREE-DATASET SYNTHESIS (2026-07-03)
- DISK saga: ActionSense wearables HDF5 = 2-4 GB each (embed eye-video) → ~35 GB, exceeds 100 GB
  home (66 used) → repeated curl/truncation/ENOSPC. SOLVED via streaming driver
  scripts/crc/stream_actionsense.sh (download 1 file → probe → delete → next; --jsonl accumulate
  + --report-only aggregate). OpenTouch raw data got deleted along the way (kept its earlier CSV).
- ACTIONSENSE PROBE (299 clips, S00-S05, 32x32 2-hand, 6 Hz→30 Hz resample; persH1 & migration
  degenerate from upsampling — persH15/H30 + periodicity carry signal):
  - raw activity: Slice cucumber +2.7 / Pour +2.0 / Clear board +1.7 / Clean-plate-towel +1.7 /
    Peel +1.5 / Slice bread +1.4 ... bottom: Open/close jar -2.3/-2.9, Get/replace items -4.1.
  - category: Pour +2.6 > Cut +1.8 > Wash/Clean +0.1 > Fold/Cloth(spread) -0.5 > Organize -1.4 >
    Open/Close -2.6. temporal: B3 ramp +2.5 > B1 periodic +0.8 > B5 composite -1.2 > B4 trans -2.2.
  - Here a-priori Axis B WORKS (actions match canonical mechanics), unlike OpenTouch.
- SYNTHESIS (3 sensors) — TRAIT CONFIRMED: predictable = smooth/continuous/slowly-varying force
  (pour/slice/wipe/peel/stir/scoop); unpredictable = abrupt onset/make-break (open-close jar,
  press/click, plug, stiff turn). persH15 = sensor-agnostic predictor. REFINEMENT (from
  ActionSense): monotonic ramp (pour #1) > rhythmic cycle (slice) > sustained hold > transition —
  a cycle has force-reversal turning points; a pour doesn't. Category ranking is dataset-dependent;
  the TRAIT is stable → the durable answer + basis for user feedback (smooth actions have a
  scorable "correct" force profile). Written up in docs/ACTION_CATEGORIES.md §3c + §4.
- REMAINING: Force-Vision (4th dataset, press/hold/squeeze) optional; OpenTouch improved-taxonomy
  category view optional (needs 14 GB re-download); GPU per-category forecasting to confirm probe.

### NEW DIRECTION — GENERATIVE FORECASTER FOR SMOOTH ACTIONS (2026-07-03)
User directives: (1) DOCUMENT the study thoroughly → wrote docs/STUDY_SUMMARY.md. (2) DROP
EgoTouch going forward — usable hardware is the glove behind the 3 linked datasets (ActionSense/
OpenTouch/Force-Vision); EgoTouch = historical reference only. (3) TRAIN a GPU forecaster
(ConvLSTM family) on the predictable smooth-force actions (slice, wipe/clean, pour, peel) in
ActionSense. (4) BRAINSTORM the training framework in detail (generative framework? network? loss?
latent embedding? physical latent variables? why?).
- DESIGN DOC: docs/TACTILE_FORECAST_PLAN.md. Proposal = a PHYSICS-STRUCTURED LATENT WORLD MODEL:
  β-VAE encoder → low-dim latent with NAMED physical channels [total force F, center-of-pressure
  (x̄,ȳ), contact area A, patch orientation/eccentricity, motion phase (sinφ,cosφ), force-rate
  dF/dt] + small residual → ConvLSTM/GRU latent predictor (in LATENT space, reusing our ConvLSTM)
  → decoder. Phase 2: stochastic RSSM / latent diffusion. Rationale: small data (~300 clips) +
  interpretable latent needed for feedback. Loss = masked log-space recon + total-force + contact
  support(BCE/IoU) + physical-latent supervision + temporal smoothness(jerk) + spectral/phase
  (periodic subset) + β·KL. Small-data plan: shared model conditioned on action, self-supervised
  pretrain on the full continuous stream, heavy aug (flip/rotate/speed-warp), cross-glove transfer.
- OPEN QUESTIONS Q1-Q7 in the plan doc (latent form; deterministic-first vs RSSM; shared vs
  per-action; rate/horizon; use Xsens pose?; compute/caching; which actions in v1). AWAIT user
  input before implementing.
- TODO: cache segmented ActionSense smooth-action clips as small npz (avoid 30 GB re-download).

### v1 DECIDED + STATE EXTRACTOR BUILT (2026-07-03)
- Decisions (AskUserQuestion): actions = pour+slice; target = EXPLICIT physical state vector
  (Path A, user: "decide the latent variables like CoP/velocity, not learn them" → no VAE, no
  ConvLSTM); dynamics = GRU baseline THEN compare to structured ramp/oscillator; tactile-only.
- Clarified for user: "spatial residual map" = a learned feature grid for a neural decoder —
  dropped for v1 (we go fully explicit/analytic). ConvLSTM is for spatial grids, so a vector
  state ⇒ use GRU/Kalman, NOT ConvLSTM (explained the fork; user chose the vector path).
- BUILT: `src/tactile_forecast/physical_state.py` — analytic s(t): per-hand pressure moments
  [F,x̄,ȳ,sxx,syy,sxy] + derive() (area,θ,ecc,vx,vy,dF) + phase() (numpy Hilbert). Coords in
  [-1,1] (sensor-agnostic). UNIT-TESTED on synthetic: pour→F ramps/CoP fixed; slice→CoP
  oscillates, phase advances at exactly the injected freq.
- WIRED: `actionsense_predictability.py --extract-states DIR` saves state_N.npy (T,C,6) +
  manifest.jsonl (append across streamed files); `stream_actionsense.sh` now passes it so ONE
  re-stream produces the tiny state dataset (few MB) → rsync/commit, no more 30 GB re-downloads.
- NEXT: user re-streams once to build ~/actionsense/states/ → rsync to local → I build the GRU +
  structured forecasters (train on CPU locally). Then feedback demo.

### BASELINE-OFFSET BUG FOUND + FIXED (2026-07-03)
- First state dataset (299 clips, transferred to data/actionsense_states/) was DEGENERATE:
  real F ≈ 585,000 with ±0.5% wobble, CoP_x std ≈ 0.001 (no motion). CAUSE: ActionSense
  conductive-thread gloves have a large per-taxel DC baseline (~571/taxel, untared; that's the
  `tactile-calibration-scale` device's job) → total force dominated by offset, CoP pinned to
  center. (Also retro-explains ActionSense's tiny persH in the probe — static baseline inflates
  Var.) FIX: `physical_state.baseline_correct` subtracts per-taxel 5th-percentile-over-time before
  moments; verified on synthetic (recovers CoP std 0.354 = injected amplitude). clip_states
  baseline-corrects by default.
- Must RE-EXTRACT (saved moments can't be un-baselined). To make it the LAST CRC round, added
  `--save-clips-for "Pour,Slice"` → caches raw resampled (T,C,H,W) clips (float16) so all future
  preprocessing/forecasting is LOCAL. stream_actionsense.sh updated.
- ACTION: user re-streams once → transfer states/ (now ~200 MB incl. clip_N.npy) to
  data/actionsense_states/ → then build forecaster fully locally.
- DONE: corrected states transferred. Verified REAL signals: pour F ramps 950→9800 (was flat
  585k); slice CoP moves. 299 states + 70 clips local. clip_*.npy gitignored (200M); states kept.

### v1 FORECASTER BUILT + RESULT (2026-07-03) — runs fully local on CPU torch
- BUILT: `src/tactile_forecast/state_forecast.py` (data/windows/normalize + numpy baselines
  persistence/velocity/linfit + GRU seq2seq residual rollout) and
  `scripts/train_state_forecaster.py` (per-action, skill-vs-persistence per physical variable,
  all-12 vs core F+CoP summary, --downsample to native ~6 Hz).
- RESULT (pour, slice; downsample5→6Hz, t_in6/t_out12=1s→2s): GRU ≈ persistence, slightly BELOW
  on aggregate (all=-0.09..-1.1) and HIGH VARIANCE across seeds (pour 0_F once +0.40, once -0.13).
  velocity/linfit MUCH worse (-3..-19). Core F+CoP no robustly positive.
- INTERPRETATION (honest): this CONFIRMS the study's central principle at the physical-state level
  — smooth, slowly-varying signals are "predictable" precisely because PERSISTENCE predicts them
  well ⇒ little skill-over-persistence headroom. The smoothness that makes them predictable makes
  them hard to BEAT persistence on. (Same reason EgoTouch holds ≈ 0 skill.) Not a bug; a property.
- REFRAME (for the feedback GOAL): skill-vs-persistence is the WRONG target for smooth actions.
  Pivot to a NORMATIVE model: build the expected physical-state trajectory (mean±band, phase/DTW
  aligned across clips) per action; feedback = deviation of a user's F/CoP/dF-dt from the expert
  band ("force jerky", "rhythm irregular"). The extracted state is used DESCRIPTIVELY, not to beat
  persistence. DECISION PENDING with user: (a) pivot to normative feedback model; (b) push
  forecasting (pool pour+slice+peel+clean for 3-4x data, longer horizons, phase-explicit
  oscillator, regularization); (c) accept finding + write up.

### v2 SLOW/FAST + PROBABILISTIC — BREAKTHROUGH (2026-07-03)
User chose: separate slow+fast & model the FAST action component; probabilistic (mean+band).
- DIAGNOSIS confirmed: fast (high-pass) component of F/CoP decorrelates within ~2 s (autocorr
  ~0..-0.3) → persistence-of-fast is a WEAK baseline (headroom exists), and fast is 0.33-0.55 of
  slow amplitude (real signal). The slow grip/postural part is what made raw-state persistence
  unbeatable.
- BUILT `scripts/train_action_dynamics.py`: low/high-pass split (scipy butter, cut 0.4 Hz) of
  active-hand F,x,y → target = fast [F,x,y]; input += slow F + CoP velocity; action embedding;
  probabilistic GRU (mean+logvar), Gaussian NLL; 5-fold CV by trajectory; downsample 30→10 Hz,
  predict 0.5 s from 1 s.
- RESULT (5-fold): pooled(Pour,Slice,Peel,Clean) MEAN skill **+0.725**, band coverage@2sd **0.93**
  (ideal ~0.95 → well-calibrated!). Pour+Slice only +0.736. Per-target: F_fast +0.63-0.68,
  x_fast/y_fast +0.76-0.78. STABLE across folds (std 0.01-0.05) — v1's high variance gone.
- TAKEAWAYS: (1) the redesign (slow/fast + probabilistic), NOT extra data, drove the win (pooling
  ≈ pour+slice alone). (2) We now have a calibrated model of the expected fast action dynamics +
  uncertainty band → FEEDBACK-READY: score a user's fast F/CoP against the expert mean±band.
- NEXT OPTIONS: build the feedback/anomaly demo (deviation vs band); add phase/rhythm metric;
  per-hand (not just active); write up. Committed with the v2 code.

---

## COMPREHENSIVE SUMMARY (2026-07-06) — for explaining the work to other researchers

### A. Research question
Which *kind* of hand action is easiest to predict from its own past tactile signal, and — more
usefully — **what trait makes an action series predictable**, so a predictor can give a user
feedback to improve performance. Priors under test: standardized-procedure and repeatable/periodic
actions are easier.

### B. Datasets and what we did with each
Four tactile datasets; three processed with data, one from paper only.
1. **EgoTouch** (21×21 FPC grid, 2 hands, 30 Hz) — probed (23 verb categories) AND used to build the
   pixel forecaster (SimVP/ConvLSTM/ConvGRU). *Later deprecated* per user (not the target glove).
2. **OpenTouch** (arXiv 2512.16842; 16×16 FPC, 1 hand, 30 Hz) — probed 2,496 clips. Per-clip
   action + grip labels (GRASP taxonomy) in HDF5 + CSV.
3. **ActionSense** (NeurIPS'22; 32×32 conductive-thread, 2 hands, ~6 Hz) — probed 299 clips (S00-05)
   AND used for the physical-state forecaster (v1/v2). 20 kitchen activities as Start/Stop intervals.
4. **Force-Vision** (ICLR'24; STAG glove) — categorized from the paper only (press/hold/squeeze);
   NOT downloaded/probed.

### C. Processing pipeline per dataset
- **EgoTouch**: HF download (metadata + `pressure_grids.npz`, no video). Layout scene/task/traj.
  Tasks named `verb_object` → categorized by first-verb token. Pressure = (T,2,21,21), ~50% NaN
  structural sensor mask (zero-filled), log1p amplitude transform.
- **OpenTouch**: 26 HDF5 shards (~14 GB) + `final_annotations` CSVs via gdown. Each HDF5 =
  `data/<clip>/right_pressure` (T,16,16) + labels joined from per-scene CSV on `clip_id`
  ("<scene>::demo_N"). Free-text gerund actions normalized (pulling→pull) then verb-mapped.
- **ActionSense**: wearables HDF5 (2-4 GB each, embed EMG/Xsens/eye-video) → too big for the
  home quota, so a STREAMING driver downloads one file → processes → deletes → next.
  Tactile = `tactile-glove-{left,right}/tactile_data/data` (T,32,32). Activities from
  `experiment-activities/activities` rows [Activity,Start/Stop,Valid,Notes]; pair Start→Stop
  (drop Bad/Maybe) → intervals; slice tactile per interval; resample to 30 Hz; stack both gloves
  → (T,2,32,32). **Baseline correction** (per-taxel 5th-percentile subtraction) applied before any
  physical-state computation (see Problem P4).

### D. Scripts and their functions
- `scripts/categorize_actions.py` — assign each EgoTouch task to an action category (verb taxonomy);
  print per-category task/trajectory counts. (Classification only.)
- `scripts/predictability_by_category.py` — EgoTouch per-category **training-free probe**: load
  pressure, compute predictability metrics per trajectory, group by verb category AND
  temporal-pattern axis, rank by composite index; write CSV.
- `scripts/opentouch_predictability.py` — OpenTouch probe (HDF5 clips + CSV labels); `--inspect`
  schema mode + probe; group by temporal-pattern / mapped-category / raw-action / grip; CSV.
- `scripts/actionsense_predictability.py` — ActionSense probe: segment continuous tactile by
  activity intervals, resample, stack gloves, metrics; `--jsonl`/`--report-only` (streaming
  accumulate/aggregate); `--extract-states` (save physical-state trajectory per clip);
  `--save-clips-for` (cache raw clips).
- `scripts/crc/stream_actionsense.sh` — the download→probe→delete streaming driver (bounds disk to
  one file); accumulates per-clip records; final report + state extraction.
- `scripts/aggregate_results.py` — aggregate GPU pixel-forecaster runs by (model,scope,category);
  ranked per-category test skill.
- `src/tactile_forecast/train.py` — pixel forecaster trainer (SimVP/ConvLSTM/ConvGRU), LTO/LOTO CV,
  `--category` filter, skill-vs-persistence. (EgoTouch.)
- `scripts/train_state_forecaster.py` — **v1** physical-state forecaster (GRU vs baselines).
- `scripts/train_action_dynamics.py` — **v2** slow/fast probabilistic action-dynamics model.
Shared modules: `categories.py` (taxonomy), `predictability.py` (metrics), `physical_state.py`
(analytic state), `state_forecast.py` (v1 data/model).

### E. Algorithms and WHY we chose them
1. **Verb taxonomy categorization** (rule-based, first known verb token; gerund-normalized).
   *Why:* unify heterogeneous labels across datasets into ONE comparable category space + a
   temporal-pattern axis (B1 periodic … B5 composite), enabling cross-dataset comparison.
2. **Training-free predictability probe** — per clip: `persistence_nMSE@h` = MSE(y[t+h],y[t])/Var
   (decorrelation rate), `periodicity` = max total-force autocorr at lag 0.33-1.5 s,
   `contact_migration` = 1−IoU of active-taxel mask, composite `PI` = z(−persH15)+z(period)+z(−migr).
   *Why:* measure "how forecastable" WITHOUT training/GPU — fast, sensor-agnostic, and directly
   tests the periodicity/standardization priors. PI fuses the three physical axes.
3. **Pixel forecaster (SimVP/ConvLSTM/ConvGRU)**, skill vs persistence, LTO/LOTO.
   *Why:* standard tactile spatiotemporal forecasting; establishes REAL trained-model skill
   (not just the proxy) on EgoTouch, and a per-category comparison.
4. **Analytic physical-state extraction** — per hand per frame: 0th/1st/2nd pressure moments
   [F, CoP(x,y), spread(sxx,syy,sxy)]; derived area/orientation/velocity/dF-dt; Hilbert phase;
   per-taxel baseline subtraction; coords normalized to [-1,1].
   *Why:* a low-dimensional, fully interpretable state — data-efficient for small data AND directly
   usable for feedback (named physical variables a coach can talk about). User chose explicit
   variables over a learned latent.
5. **v1 GRU seq2seq on the raw state.** *Why:* a vector state calls for a vector sequence model
   (ConvLSTM is for spatial grids). RESULT: failed (≈ persistence).
6. **v2 slow/fast + probabilistic GRU** — low/high-pass split F/CoP; model the FAST action
   component; probabilistic head (mean+variance), Gaussian NLL; action embedding; k-fold CV.
   *Why:* the slow grip component is trivially persistent (killed v1); the fast component carries
   the stroke/pour dynamics and decorrelates within ~2 s (real headroom); probabilistic output
   yields the calibrated "expert band" feedback needs.

### F. Results
- **EgoTouch probe (1,929 clips):** easiest Cut(slice)+6.0, Take, Inflate, Spray, Wash/Clean;
  hardest Press/Click −6.7, Plug/Insert, Pinch, Grasp/Hold/Lift. Holds NOT trivially predictable.
- **EgoTouch pixel forecaster:** LTO (seen-object) +0.192 skill; LOTO (unseen) ≈ 0; broad
  pretraining lifts LOTO to +0.097.
- **OpenTouch probe (2,496):** easiest pour/serve/eat/stir/scoop/wipe; hardest turn(latch)/pull/
  move. `contact_migration≈0` (single-hand grasp never breaks contact) → degenerate there.
  A-priori temporal-pattern axis INVERTS vs EgoTouch (Problem P2).
- **ActionSense probe (299):** Pour +2.6 > Cut(slice/peel) +1.8 > Wash/Clean > Fold(spread) >
  Organize(tableware) > Open/Close(jar) −2.6. Cleanest confirmation; pattern axis works here.
- **HEADLINE (3 sensors):** predictable = **smooth, continuous, slowly-varying contact force**
  (pour/slice/wipe/peel/stir/scoop); unpredictable = **abrupt onset / make-or-break** (jar,
  press, plug, stiff turn). `persH15` is the sensor-agnostic predictor. Refinement: monotonic
  ramp (pour) > rhythmic cycle (slice) > hold > transition. Category ranking is dataset-dependent;
  the TRAIT is stable.
- **v1 forecaster (raw state):** GRU ≈ persistence (mean skill ~−0.1), HIGH variance. Confirms the
  "smooth ⇒ low skill-over-persistence headroom" principle at the state level.
- **v2 forecaster (slow/fast + probabilistic):** 5-fold CV mean skill **+0.725** (pooled) / **+0.736**
  (pour+slice) vs persistence-of-fast; per-target F +0.63-0.68, CoP +0.76-0.78; band coverage@2sd
  **0.93** (well-calibrated); STABLE across folds. Pooling ≈ pour+slice alone → the REPRESENTATION
  (slow/fast + probabilistic), not extra data, drove the win. → feedback-ready.

### G. Problems encountered (scientific/methodological; version-control issues excluded)
- **P1 Cross-sensor incomparability.** Four different glove geometries/rates → raw skill numbers
  are NOT comparable across datasets. *Fix:* rank WITHIN each dataset; compare across datasets only
  by the temporal-pattern axis and the qualitative trait.
- **P2 The a-priori temporal-pattern axis breaks across datasets.** The same verb behaves
  differently by context ("turning a stiff latch" in OpenTouch is an abrupt transition, not the
  rhythmic turn EgoTouch assumed), and many verbs were unmapped → dumped in "Other". *Fix:* expand
  the taxonomy; treat the pattern label as a-priori and let the measured periodicity decide;
  emphasize the trait over the category label.
- **P3 The "predictable" actions have little skill-over-persistence headroom.** The very smoothness
  that makes pour/slice predictable in absolute terms makes them near-perfectly predicted by
  persistence → a trained forecaster on the raw state can't beat it (v1 failed). This is the
  deepest finding, not a bug. *Fix/insight:* separate the persistent slow (grip) component and model
  the fast (action) component, which does have headroom (v2).
- **P4 Sensor DC baseline offset (ActionSense).** The conductive-thread glove is not tared: every
  taxel has a large resting value (~571/taxel) → total force ≈ constant (585,000 ± 0.5%) and CoP
  pinned to center — the first physical-state extraction was DEGENERATE (no motion visible). *Fix:*
  per-taxel 5th-percentile baseline subtraction before computing moments (validated on synthetic:
  recovers the true CoP oscillation). Also retro-explains the inflated (tiny) persH in the probe.
- **P5 Resampling artifact.** ActionSense native ~6 Hz upsampled to 30 Hz → adjacent frames are
  near-duplicates → persistence artificially strong at short horizons. *Fix:* forecast at native
  rate (downsample back to ~6-10 Hz).
- **P6 Noisy higher-order features dilute the metric.** The 2nd-moment shape terms (orientation/
  covariance) are unpredictable jitter and irrelevant to feedback, but equal-weighting dragged the
  mean skill negative. *Fix:* focus targets/metrics on the core feedback variables (F, CoP).
- **P7 Small data / high variance.** ~15-30 clips per activity, ~18 train trajectories per action →
  a single train/val split gave wildly unstable skill (+0.40 vs −0.13). *Fix:* k-fold CV
  (report mean±std); action pooling; probabilistic model; strong regularization.
- **P8 Disk/logistics (ActionSense).** Wearables files are 2-4 GB each (~35 GB total) vs a limited
  home quota → download failures/truncation. *Fix:* the streaming download→probe→delete driver +
  caching only the tiny states (and a small set of raw clips) locally so no dataset is re-downloaded.
- (Excluded per user: GitHub auth/divergent-branch/CRC-code-sync issues — real time sinks but
  not scientific.)

### H. One-paragraph narrative (for a researcher)
We built a sensor-agnostic, training-free probe to rank how forecastable each action's tactile
signal is, applied it across three tactile gloves, and found a stable, sensor-independent trait:
smooth continuous-force actions (pour, slice, wipe) are predictable, abrupt make/break actions
(jar, press) are not — but the very smoothness that makes them "predictable" means a naive
forecaster only matches persistence. Reducing each pressure field to interpretable physical
variables (force, center of pressure), then **separating the trivially-persistent grip from the
fast action component and modeling that component probabilistically**, yields a calibrated
forecaster (skill +0.73 over persistence, 93% band coverage) whose interpretable, bounded outputs
are exactly what is needed to give a user actionable feedback.

### PER-ACTION v2 COMPARISON → ACTION CHOICE (2026-07-06)
Ran v2 (slow/fast probabilistic, 5-fold) per single action. Fast-component skill / band coverage:
- Peel (n=30): +0.754 / 0.92 ; Slice (n=75): +0.738 / 0.92 ; Clean/wipe (n=60): +0.618 / 0.89 ;
  Pour (n=25): +0.610 / 0.81 (miscalibrated, small n).
RANKING: rhythmic repeated-stroke actions (peel, slice, wipe) > ramp (pour).
INSIGHT: this REVERSES the raw-signal trait ordering (which put pour/ramp top). Reason: for the
FAST component, rhythmic strokes have a clean oscillation (structured, predictable, calibratable),
whereas pour's fast part is unstructured tremor once the ramp is removed. Rhythmic actions also
have a well-defined "correct rhythm" → natural feedback template.
DECISION: target the REPETITIVE-STROKE family (slice + peel, then wipe) for the forecaster +
feedback demo; pour is the weakest fit despite being "most predictable" in the raw sense.
Horizon sweep (pooled): skill 0.74/0.68/0.62/0.58 at 0.5/1.0/1.5/2.0 s (gentle decay; 1 s a good
operating point). 0.73 "skill" is dimensionless (1-MSE/MSE_persistence); CoP in [-1,1] grid units
(not mm), force uncalibrated.

### REFACTOR: library / thin-CLI separation for the v2 forecaster (2026-07-07)
User: "plot scripts should only plot"; wanted clear structural repo. Done all 4:
1. NEW `src/tactile_forecast/action_dynamics.py` = LIBRARY (single source of truth): slow_fast,
   build_features, load_pooled, windows, split_train_test, Norm, ProbGRU, train(), evaluate(),
   forecast_clip(), save()/load() checkpoint. (Convention: src/ = importable nouns; scripts/ = verbs.)
2. `scripts/train_action_dynamics.py` slimmed to a CLI: k-fold CV report + train final on all clips
   + SAVE checkpoint to runs/ (gitignored). No model code. CV reproduces: Slice+Peel MEAN +0.770.
3. `scripts/plot_action_forecast.py` slimmed to plotting only: `--ckpt` loads a saved model (no
   training) OR default sweeps past-context (1/2/3/5/10s) training via the library. No model/train
   code; no more script->script import. Sweep skills reproduce (0.69/0.71/0.72...).
4. RENAMED probes to verbs: predictability_by_category.py->probe_egotouch.py,
   opentouch_predictability.py->probe_opentouch.py, actionsense_predictability.py->probe_actionsense.py
   (git mv, history kept). Fixed the one functional ref (stream_actionsense.sh PROBE path) and
   removed probe_egotouch's script->script import (now imports categorize from the package).
Verified: all compile; train CLI +0.770 + saves ckpt; plot --ckpt loads+plots (no train); plot
sweep reproduces. runs/ gitignored. Structure now: LIBRARY in src/tactile_forecast (action_dynamics,
predictability, physical_state, categories), thin CLIs in scripts/ (train_*, plot_*, probe_*, crc/).

### PAST-CONTEXT SWEEP: data-size confound + horizon note + TODO (2026-07-07)
- Training now SWEEPS past-context (scripts/train_action_dynamics.py --pasts 1,2,3,5,10; future
  1s = t_out 10). Full-quality result plateaus ~3s: 1s +0.69 / 2s +0.71 / 3s +0.72 / 5s +0.72 /
  10s +0.71 (reduced epochs25/folds3 demo). Saves a checkpoint per past (runs/ad_<acts>_p<p>s.pt).
- CONFOUND FOUND (user question): training data size is NOT equal across history lengths. In
  action_dynamics.windows(), `win = t_in + t_out` and the loop `range(0, T - win + 1, stride)` yields
  FEWER windows as t_in grows. Measured on Slice+Peel (t_out=10, stride=2): #windows 15,529 (1s) ->
  15,154 (2s) -> 14,779 (3s) -> 14,029 (5s) -> 12,154 (10s) — the 10s model trains on ~22% LESS data.
  Also clips with T < t_in+t_out are silently dropped (none here: min clip 114 >= 110 win, by luck).
  So the plateau/decline at long history is partly LESS DATA, not only decorrelation.
- TODO (fair comparison, LATER): shared-anchor mode — only forecast at positions a >= max(t_in),
  same anchors for every t_in, so all history lengths get IDENTICAL window count + identical future
  targets; only the depth of past differs. Re-run the sweep to see if the 3s sweet spot survives.
- HORIZON / RUN DIFFERENCES (clarification): the earlier CRC single run
  `train_action_dynamics.py --actions Slice,Peel` that gave MEAN +0.770 was the pre-sweep version =
  ONE config, 1s past, **0.5s** forecast (t_out=5), full epochs80/folds5. The new sweep default
  forecasts **1s** (t_out=10) -> harder -> the 1s-past row is +0.69, not +0.77. Same-machine reruns
  are reproducible (same code+data+seed); the local demo used reduced epochs25/folds3 (slightly
  different numbers, same trend) vs a full CRC/local run at epochs80/folds5.

### PLAN (approved 2026-07-08) — causal filter + raw-vs-highpass ablation + leakage checklist
User-approved changes (implementing now):
1. CAUSAL FILTER: action_dynamics.slow_fast filtfilt -> sosfilt (butter output='sos', forward-only).
   Rationale: filtfilt is non-causal (backward pass sees the future) -> the fast component leaks
   future info into both input and target for a forecasting task. sosfilt is causal. Also make
   velocity causal (np.gradient central-diff -> backward diff). Cost: startup transient -> cut
   first 5s (=50 frames @10Hz) per clip, in BOTH train and eval.
2. RAW-vs-HIGHPASS ablation (only the INPUT changes; target always fast [F,x,y]):
   input_mode='highpass' = [F_fast,x_fast,y_fast,F_slow,vx,vy] (current);
   input_mode='raw'      = [F,x,y,vx,vy] (no decomposition).
3. REPORT by every channel (F, CoP-x, CoP-y) x every history (1/2/3/5/10s) x per-forecast-step
   (+0.1..+1.0s) x each HAND (left=ch0, right=ch1, reported separately, not just active). Print
   history x channel tables per (input_mode, hand); write a full CSV with all breakdowns.
4. Pipeline order (confirmed, unchanged): raw field -> per-frame F + CoP -> causal high-pass ->
   z-score (train stats). z-score stays AFTER the filter.
5. LEAKAGE CHECKLIST: scripts/check_leakage.py (runnable, PASS/FAIL, run before every training) +
   docs/leakage_checklist.md. Six checks: (1) filter causal (impulse test), (2) norm stats
   train-only, (3) split by trajectory/no clip overlap, (4) input strictly before target,
   (5) baseline sees same past-only input, (6) pipeline order (CoP/force before filter; z-score
   train-only + consistent train/test).
FILES: action_dynamics.py (sosfilt, causal velocity, build_features input_mode+hand+warmup,
evaluate per-step), train_action_dynamics.py (--input-mode, --hands, per-step, CSV),
plot_action_forecast.py + plot_test_results.py (input_mode/hand passthrough), NEW check_leakage.py,
NEW docs/leakage_checklist.md. NO re-extraction needed (filter applied at load from committed states).
EXPERIMENT: rerun sweep raw vs highpass, both hands, on CRC -> compare.

### IMPLEMENTED: causal filter + ablation + leakage checklist — RESULTS (2026-07-08)
All 6 leakage checks PASS (scripts/check_leakage.py) after filtfilt->sosfilt + causal velocity.
Reduced demo (epochs15/folds3, pasts 1&3, both hands, raw&highpass) — full run needed to finalize:
- BIG FINDING: causal filter DROPS skill from ~+0.70 (old non-causal, epochs25) to ~+0.51-0.54
  (causal, epochs15). Part is fewer epochs, but the direction confirms filtfilt was LEAKING future
  info into the fast target and inflating skill. Full epochs80/folds5 run (CRC) will quantify the
  honest causal skill.
- RAW ~= HIGHPASS: raw-input vs highpass-input give nearly identical skill (left 0.510 vs 0.513;
  right 0.543 vs 0.540) -> the explicit slow/fast INPUT decomposition is UNNECESSARY; the model
  predicts the fast target equally well from raw signals. (Ablation answered.)
- HANDS: right hand slightly > left (~+0.54 vs ~+0.51) for Slice/Peel (right = dominant/tool hand).
- PER-STEP: skill RISES with forecast horizon (+0.11 @0.1s -> +0.59 @1.0s). Correct + expected:
  skill is vs persistence-of-fast, which is strong at 0.1s (autocorr ~0.8) but collapses by 1s
  (fast reverses), so the model's ADVANTAGE grows with horizon.
- Full per-(input_mode,hand,history,step,channel) breakdown -> docs/action_dynamics_results.csv.
NEXT: run full matrix on CRC (python scripts/check_leakage.py && python scripts/train_action_dynamics.py
--actions Slice,Peel) for the honest causal numbers; compare raw vs highpass definitively.

### FULL CAUSAL RESULT — Slice (CRC job 1169778, 2026-07-09)
Honest causal run (sosfilt, warmup 5s, all 6 leakage checks PASS). NOTE: ran SLICE ONLY (45 clips)
— the qsub `-v ACTIONS="Slice,Peel"` comma was split by UGE, dropping Peel; rerun `qsub
scripts/crc/train_state_gpu.job` (no -v, defaults to Slice,Peel) to add Peel. Results ->
docs/action_dynamics_results.csv (per input_mode x hand x history x forecast-step x channel).

Pooled MEAN skill vs persistence-of-fast (per history, from the .o tables):
             1s     2s     3s     5s    10s
raw/left   +0.402 +0.371 +0.313 +0.305 +0.259
raw/right  +0.401 +0.384 +0.385 +0.345 +0.294
hp /left   +0.376 +0.364 +0.335 +0.301 +0.232
hp /right  +0.392 +0.390 +0.371 +0.331 +0.314

FINDINGS (causal, Slice):
1. HONEST SKILL ~+0.40 at 1s history (was ~+0.70 with the leaky filtfilt) — the leak inflated by
   ~0.3. This is the number to report.
2. RAW ~= HIGHPASS confirmed at full scale: raw 0.40 vs highpass 0.38 (left), 0.40 vs 0.39 (right).
   The slow/fast INPUT decomposition does NOT help -> can simplify the model to raw input.
3. MORE HISTORY HURTS: skill DECLINES monotonically 1s->10s (e.g. raw/right 0.40->0.29). Opposite
   to the old non-causal (which plateaued). Confounded by fewer training windows for longer history
   (see fair-comparison TODO) — but honest pipeline shows no benefit from >1-2s of past.
4. RIGHT HAND > LEFT, esp. at long history (10s: right ~0.29-0.31 vs left ~0.23-0.26). Right =
   dominant/tool hand for slicing; its CoP-x (stroke) is the most predictable channel (x_skill ~0.45-0.48).
5. PER-STEP: skill starts NEGATIVE at +0.1s (persistence near-perfect that close; model slightly
   worse), crosses 0 by ~+0.2s, rises to ~+0.48 at +1.0s. Model only beats persistence at >0.2s lead.
6. CALIBRATION WORSE: coverage@2sd ~0.73-0.85 (was ~0.92 leaky), drops with history (0.83@1s->0.73@10s)
   -> bands overconfident on the honest (harder) task; a calibration fix / CRPS is warranted.

### FULL CAUSAL RESULT — Slice+Peel (CRC job 1170576, 2026-07-10)
Honest causal (sosfilt, warmup 5s, leakage checks pass), pooled Slice(45)+Peel(30)=75 clips.
docs/action_dynamics_results.csv (per input_mode x hand x history x step x channel).

avg-over-steps MEAN skill by (mode, hand, history):
             1s     2s     3s     5s    10s
raw/left   +0.365 +0.334 +0.317 +0.304 +0.245
raw/right  +0.404 +0.409 +0.399 +0.364 +0.322
hp /left   +0.374 +0.348 +0.343 +0.296 +0.257
hp /right  +0.427 +0.408 +0.394 +0.369 +0.328

FINDINGS (consistent with Slice-only, adding Peel):
1. Honest skill ~+0.40 (right hand); RAW ~= HIGHPASS again (hp marginally > raw on right; ~tie on
   left) -> slow/fast INPUT decomposition still unnecessary. Can simplify to raw input.
2. RIGHT HAND >> LEFT, and the RIGHT-HAND CoP_x (stroke direction) is by far the most predictable
   channel: +0.47 @1s and still +0.435 @10s (vs F +0.20-0.37, CoP_y +0.33-0.39). The knife-stroke
   left-right motion is the dominant, most-forecastable signal.
3. HISTORY: right hand ~flat 1-3s (+0.40) then declines; left declines from 1s. ~1-3s past optimal
   (data-size confound still applies; fair-comparison TODO).
4. PER-STEP: rises with horizon (raw/right 3s: 0.19@0.1s -> 0.50@1.0s); +0.1s now positive for the
   right hand (Peel helps short-horizon).
5. COVERAGE ~0.76-0.86 (overconfident, drops with history) -> calibration fix next (post-hoc sigma
   scaling to hit ~0.95).

### CALIBRATION FIX implemented (2026-07-10)
Post-hoc sigma-scaling to fix overconfident bands (coverage was ~0.76-0.86 << 0.95 ideal).
- action_dynamics.calibrate_sigma(model, norm, val_clips, t_in, t_out, target=0.95): s =
  percentile(|Y-mu|/sd, 95)/2 -> scaling sd by s makes +/-2sd contain ~95%. Fit on a VAL set.
- evaluate(..., sigma_scale) applies it to coverage; forecast_clip(..., sigma_scale) to the bands.
- train_action_dynamics CV: per fold hold out a VAL subset of TRAIN, calibrate on it, report BOTH
  covRaw and covCal on the test fold; CSV gains coverage_raw + coverage_cal. Final checkpoints train
  on 85%, calibrate sigma on 15%, store sigma_scale in meta; plots apply it.
- VERIFIED (reduced raw/right): covRaw 0.90-0.91 -> covCal 0.95-0.96. On the low-coverage configs
  (~0.76) the scale is larger and still lifts to ~0.95. Skill unchanged (calibration only rescales
  uncertainty, not the mean).
- NEXT: rerun full matrix on CRC (qsub) to get calibrated coverage in the CSV.

### RIGOROUS CODE + RESULTS REVIEW (2026-07-10, on request: "is F prediction too good?")

Scope: full review of action_dynamics pipeline + docs/action_dynamics_results.csv +
forecast_F/CoPx/CoPy figures. New diagnostic: scripts/tmp_diag_predictability.py (torch-free).

VERIFIED CORRECT (no leakage found):
- Causal filter (sosfilt) + backward-diff velocity; warmup trim (action_dynamics.py:39-51,77).
- Window construction: inputs strictly before targets, within-clip only (windows(), :100-113).
- Split by clip; norm stats from train clips only; sigma calibrated on val held out of train.
- forecast figures are honest: seeded from TRUE value at each 1 s anchor, then autoregressive
  (plot_forecast_overlay.py:37-41); test clips excluded from training (test_ids by clip, seed=1).
- check_leakage.py assertions are sound and match the code.

KEY NEW FINDING - THE SKILL BASELINE IS TOO WEAK; HEADLINE NUMBERS ARE INFLATED:
The fast target is ANTI-correlated with itself at 1 s lag (variance-weighted AC over the 75
Slice+Peel clips, both hands, ds=3/cut=0.4/warmup=5: F_fast rho(1.0s) = -0.19, x = -0.18,
y = -0.14). Persistence MSE at 1 s = 2(1-rho)*var ~ 2.4*var, so trivial baselines score high
skill-vs-persistence:
  - predict CONSTANT ZERO @1.0s: F +0.59, x +0.58, y +0.56  (>= GRU's ~0.50-0.54 in the CSV!)
  - damped persistence (best scalar a*last, a ~ -0.2): +0.57..+0.61 @1.0s
  - linear ridge (past 1 s of 3 channels): +0.62 @1.0s  (beats the GRU at the far step)
Per-step zero-baseline skill (mean-channel, all clips): -3.0 @0.1s, -0.38 @0.2s, +0.14 @0.3s,
+0.34 @0.4s, +0.44 @0.5s ... +0.57 @1.0s. The GRU beats BOTH trivial baselines only in the
~0.2-0.5 s band; by 0.6 s+ shrink-to-zero matches/exceeds it. The avg-over-steps "+0.40 honest
skill" headline mostly reflects (a) GRU ~ persistence at short steps where zero is terrible and
(b) GRU ~ shrink-to-zero at long steps where persistence is terrible. Conclusions like
"right-hand CoP-x most predictable, +0.47" must be re-checked against the zero/damped baselines
(x @1.0s zero-baseline ~ +0.58 > CSV +0.54).
ACTION: add zero + damped-persistence + linear-ridge baselines to evaluate() and report skill
vs the STRONGEST trivial baseline per step.

WHY F LOOKS "TOO GOOD" IN forecast_F.png (mechanism, not leakage):
1. Re-anchoring: every 1 s segment restarts from the TRUE last value; errors never compound
   beyond 10 frames. The eye reads 50 concatenated 1 s forecasts as one great long forecast.
2. F_fast is the smoothest, highest-SNR channel: F = sum over all taxels (spatial averaging),
   spectral centroid 0.56 Hz => typical period ~1.8 s, so a 1 s forecast is ~half a period of a
   smooth quasi-periodic stroke cycle seeded from truth. AC(0.1s)=0.91.
3. Causal-filter group delay near the 0.4 Hz cutoff leaks lagged SLOW trend into the "fast"
   target - an extra smooth predictable component (affects target definition, not causality).
4. MSE/NLL training => amplitude shrinkage toward 0 at uncertain times; since the target
   oscillates around 0, a shrunk forecast still LOOKS close. Visible in the figures (orange
   amplitude < black).
Why CoP looks worse: CoP = moment ratio (divide by F) => noise amplified at light contact,
heavy-tailed spikes that an MSE model rightly ignores; centroid 0.67-0.74 Hz (faster content).
So: F prediction is NOT suspiciously good - quantitatively it is no better than trivial
shrinkage at the 1 s step; the visual impression comes from 1-3 above.

OTHER ISSUES (secondary):
- ACAUSAL PREPROCESSING: physical_state.baseline_correct() subtracts the per-taxel 5th
  percentile over the WHOLE clip (physical_state.py:68) - uses future frames. Per-clip constant
  => negligible for the high-passed F target, but changes CoP nonlinearly and is not deployable.
  Fix: percentile over the first N seconds or a running percentile.
- hand="active" (action_dynamics.py:63-64) picks the hand from the WHOLE-clip mean force
  (future info). Default in plot_action_forecast.py only; CSV runs use explicit left/right. OK
  offline, flag for any online claim.
- STALE CSV: docs/action_dynamics_results.csv header has `coverage` (one col) but the current
  train_action_dynamics.py writes coverage_raw+coverage_cal - the CSV predates the calibration
  fix; regenerate.
- NO SUBJECT ID in the manifest (probe_actionsense.py:234-237): clip-level split mixes
  subjects/sessions => results = within-corpus generalization, not new-user.
- POOLED MSE in raw sensor units => skill dominated by high-amplitude clips (persistence MSE
  ~1.8e6 a.u.^2). Report per-clip skill median/IQR too.
- CSV writes fold MEANS only; cross_validate has per-fold arrays - add +/- std.
- First-step F skill negative (left): decoder gets y_last yet does worse than copying it =>
  parameterize the decoder as residual from y_last (predict delta).
- Fragile-but-correct: plot_forecast_overlay derives test_ids from the hand=left clip list and
  reuses them for hand=right; only safe because load_pooled ordering/length filter is
  hand-independent. Add an assert.

CONCLUSION for the user question: no implementation bug/leakage makes F "too good"; the figures
are honest but flattering (1 s re-anchoring). The real problem is the opposite direction: the
skill-vs-persistence metric OVERSTATES the model because persistence-of-fast is anti-correlated
at 1 s. The model has genuine (small) value only at ~0.2-0.5 s lead.

---

## Session (2026-07-13) — Method clarifications, horizon plot, two open design decisions

### User's 7 questions — answers (with code evidence)
1. **Prediction method / "0.1 s steps" / aggregation.** Clarified a conflation:
   - The 1/2/3/5/10 s are **five independent models** (different past-context), NOT aggregated.
     Panel (a) compares them; only one history is used per model.
   - The 0.1 s steps are how **one** model emits its 1 s: an **autoregressive** seq2seq GRU
     decoder rolls out t_out=10 steps, feeding its OWN prediction back
     (`action_dynamics.py:156-162`, key line 161 `inp = mu.unsqueeze(1)`).
   - User wants **one-shot / direct** multi-horizon (single forward pass emits all 10 frames,
     no feed-back). This is practical (swap decoder for a t_out*3 head), avoids rollout error
     compounding. **OPEN DECISION #1: switch to one-shot direct?** (would re-run sweep + figures).
2. **Skill.** `skill = 1 - MSE_model/MSE_persistence` (`action_dynamics.py:210-212`). Persistence =
   last observed value repeated over all 10 future steps. 0 = tie, +1 = perfect, <0 = worse.
3. **Raw WAS trained.** The sweep trains both input_modes every run (`train_action_dynamics.py`
   default `--input-modes raw,highpass`, driven by `train_state_gpu.job:34`). CSV has raw+hp rows;
   raw/highpass only changes the INPUT, both predict the same high-pass target.
4. **Horizon plot delivered** — `scripts/plot_horizon.py` -> `docs/horizon_highpass.png`.
   Calibrated highpass, right hand, 3 s history -> 1 s ahead on test clip 6. Per channel shows:
   history window consumed (grey), true future (black), forecast mean+calibrated +/-2sigma (blue),
   persistence (dashed). sigma_scale=1.966. Forecast bends toward truth (F, CoP-x); persistence flat.
   NOTE: uses the CURRENT autoregressive method; will be re-rendered if we adopt one-shot.
5. **Persistence baseline — meaningfulness.** For the zero-mean FAST target, persistence is a WEAK
   floor (last fast value ~0, easy to beat), which flatters skill. **OPEN DECISION #2: add stronger
   baselines** — zero/mean baseline and AR(1)/linear extrapolation — and report skill vs all three.
6. **Input = tactile-only.** Confirmed: only the tactile pressure map -> [F, CoP_x, CoP_y]
   (`action_dynamics.py:67`). NO EMG (Myo), NO Xsens IMU/motion, though ActionSense records them.
7. Full multi-day documentation pass to be done at session end (this entry is the running log).

### Open questions awaiting user
- **#1** Switch model from autoregressive rollout to one-shot direct 1 s forecast?
- **#2** Add zero and AR(1) baselines alongside persistence?

### Commits this session
- `195fc70` regenerate results_summary with calibrated coverage (dashed ~0.95 vs raw ~0.80)
- horizon plot + script (this session)

### PLAN (2026-07-13) — one-shot vs autoregressive + AR(1) baseline  [awaiting resolution]
User decisions: (1) build BOTH decoders and compare; (5) add AR(1)/linear-extrapolation baseline.

Planned changes (NOT yet implemented — rule 5):
1. `action_dynamics.py`: add a `decoder` mode to the model.
   - `autoregressive` (current): decoder GRU feeds its own prediction back (t_out steps).
   - `oneshot` (new): single forward pass emits all t_out*3 means + logvars directly
     (direct head from encoder hidden state; no feed-back -> no rollout error compounding).
2. `evaluate()`: add an AR(1) baseline forecast; report skill vs persistence AND vs AR(1).
3. `train_action_dynamics.py`: add `decoder` to the sweep (doubles configs); add `*_ar1` skill
   columns; keep persistence columns.
4. `plot_results_summary.py` / `plot_horizon.py`: show both decoders / both baselines.
5. Re-run sweep, regenerate figures.

OPEN QUESTIONS (need user answers before coding):
- Q1 One-shot head: direct MLP from encoder hidden (recommended) vs non-autoregressive GRU
  with a fixed input token?
- Q2 AR(1) coefficient phi: estimated per-channel on the TRAIN set (stable, recommended) vs
  per-window from each clip's own history (adaptive)? Or do you want plain linear least-squares
  slope extrapolation instead of AR(1)?
- Q3 Report skill vs BOTH persistence and AR(1) (recommended) vs REPLACE persistence with AR(1)?
- Q4 Re-run where: local (sweep now doubles) vs CRC batch job?

---

## Session (2026-07-14) — COLD-START ONBOARDING SNAPSHOT (read this to catch up fully)

Purpose: a self-contained state-of-the-project dump. A fresh Claude that reads this section
(plus the COMPREHENSIVE SUMMARY at ~line 687) should know essentially everything we know.

### 0. One-paragraph orientation
We forecast the near-future TACTILE dynamics of the hand during smooth manipulation, to later give
real-time feedback. Data = ActionSense conductive-thread gloves (32x32 taxels/hand, two hands).
We picked the smoothest, most continuous-force actions — SLICE (cutting) and PEEL — and train a
small probabilistic GRU to predict the next 1 s of the hand's tactile "physical state" from a few
seconds of history. Input is TACTILE ONLY (no EMG/Myo, no Xsens IMU). We report skill vs a
persistence baseline, with calibrated uncertainty bands.

### 1. Data inventory (Slice + Peel) — computed 2026-07-14 from data/actionsense_states/manifest.jsonl
RAW (full recording length, before warmup cut):
  Slice: 45 clips, 1705 s (~28.4 min), mean 37.9 s, min 11.3 s, max 220.1 s
  Peel : 30 clips, 1536 s (~25.6 min), mean 51.2 s, min 31.5 s, max  71.4 s
  TOTAL: 75 clips/trials, 3241 s (~54 min)
Trial breakdown (15 reps each): Slice cucumber/potato/bread; Peel cucumber/potato (5 dishes x 15).
USABLE after 5 s warmup cut/clip: Slice ~1480 s, Peel ~1386 s, ~2866 s (~48 min).
Each clip is used for BOTH hands separately -> 150 hand-trajectories. Downsample ds=3 -> 10 Hz
(changes sample count, not seconds). CAVEAT: shortest Slice clip is 11.3 s, so 10 s-history configs
drop the short clips -> the long-history skill rests on fewer/longer trials (training-size confound).

### 2. End-to-end forecasting pipeline (ORDER MATTERS)
(a) Upstream (already done, stored in state_N.npy): 32x32 taxel pressure map -> baseline-corrected
    (per-taxel 5th-percentile subtraction, fixes the untared-glove DC offset) -> per-hand physical
    MOMENTS [F (total force), CoP_x, CoP_y, sxx, syy, sxy].
(b) build_features (action_dynamics.py:54): read F/CoP moment channels FIRST, THEN causal high-pass
    them (slow_fast, butter+sosfilt, CAUSAL). Order = MOMENTS-THEN-HIGHPASS (NOT highpass taxels
    then moments — that would make CoP, a ratio, blow up when the fast denominator crosses 0).
    - target = fast [F_fast, x_fast, y_fast] (always).
    - input_mode highpass -> [F_fast,x_fast,y_fast,F_slow,vx,vy]; raw -> [F,x,y,vx,vy].
    - velocity vx,vy = causal backward difference. warmup_sec=5 s dropped (filter transient).
(c) windows (line 100): sliding X=feat[s:s+t_in], Y=targ[s+t_in:s+t_in+t_out]; input strictly
    before target (no leakage). fps=10 Hz -> t_in = history_s*10, t_out = 1 s = 10 frames.
(d) Model ProbGRU (line ~140): encoder GRU -> decoder GRU rolled out t_out steps, AUTOREGRESSIVE
    (line 161 inp = mu.unsqueeze(1) feeds its own prediction back), + action embedding, mu/logvar
    heads -> Gaussian per step. Loss = Gaussian NLL.
(e) evaluate/_predict (line 186): skill = 1 - MSE_model/MSE_persistence per channel & per step;
    coverage@2sd = fraction of truth inside mu +/- 2*sigma_scale*sd.
(f) calibrate_sigma (line 197): post-hoc scalar so coverage -> ~0.95 on a validation slice.

### 3. THREE code paths & the re-anchoring insight (critical for reading the figures)
There are three places a forecast is produced; only the OVERLAY PLOT re-anchors:
  Path A - the model (action_dynamics.py forward): rolls out 1 s from ONE anchor using its OWN
           predictions; never sees ground truth after the anchor.
  Path B - the skill metric (_predict): one model call per window, full 1 s rollout, NO mid-forecast
           refresh -> the reported skill numbers are HONEST 1-second-ahead.
  Path C - the overlay figure (plot_forecast_overlay.py:37-41): tiles the whole clip into
           consecutive 1 s blocks, RE-ANCHORS each block to a fresh GROUND-TRUTH window + true seed
           every 1 s. This makes forecast_F/CoPx/CoPy.png visually HUG the real curve (error can't
           accumulate past 1 s) — a flattering VISUAL only; it does NOT change the skill numbers.
  => docs/horizon_highpass.png (single anchor, Path A/B) is the HONEST picture; forecast_*.png is
     flattering. This resolved the "why do these plots look so different" question.

### 4. Results — skill per config (calibrated CSV docs/action_dynamics_results.csv)
Mean skill over the 10 forecast steps (0.1..1.0 s); skill = 1 - MSE_model/MSE_persistence:
  mode/hand/hist    F     CoP-x  CoP-y  mean   cov
  highpass/right/1s 0.394 0.455  0.390  0.413  0.948   <- best
  raw/right/1s      0.378 0.461  0.392  0.410  0.942   <- tied (raw ~= highpass)
  highpass/left/1s  0.355 0.350  0.362  0.356  0.949
  raw/left/1s       0.318 0.346  0.351  0.338  0.948
  ... skill DROPS monotonically 1s->10s history for every config (e.g. right/highpass 0.413->0.310).
Takeaways: (i) raw ~= highpass (input decomposition buys nothing); (ii) right hand > left (+0.05-0.07);
(iii) CoP-x on the right hand is the standout channel (0.42-0.46, barely decays with history — the
stroke direction); (iv) more history HURTS (causal filter: old context = noise); (v) coverage
~0.94-0.95 everywhere (calibrated). Per-step: skill RISES with lead time (persistence degrades faster
than the model). Full 200-row per-step table is in the CSV.
CAVEAT: skill is vs PERSISTENCE, a WEAK floor for the zero-mean fast target -> absolute levels are
flattered; the RANKING should survive a tougher baseline but magnitudes will drop. (-> AR(1) plan.)

### 5. Calibration status
Coverage@2sd was overconfident (~0.80) -> post-hoc sigma-scaling lifted it to ~0.947 across the
whole matrix, skill UNCHANGED (calibration rescales the band, not the mean). Two CSVs kept:
  docs/action_dynamics_results.csv        = calibrated (coverage_raw + coverage_cal)
  docs/action_dynamics_results_precal.csv = pre-calibration (recovered from git)
results_summary.png panel (d): solid=raw coverage (~0.80), dashed=calibrated (~0.95), red=ideal.

### 6. File map (what each script is)
  src/tactile_forecast/action_dynamics.py  = THE library (single source of truth): slow_fast,
      build_features, load_pooled, windows, Norm, ProbGRU, train, evaluate, calibrate_sigma,
      forecast_clip, save/load. Plot/CLI scripts import from here; they must not redefine logic.
  scripts/train_action_dynamics.py = sweep CLI (input_mode x hand x history), k-fold CV, writes CSV.
  scripts/check_leakage.py         = 6 leakage checks; run before every training (job aborts if fail).
  scripts/plot_results_summary.py  = 4-panel summary from the CSV (no training).
  scripts/plot_forecast_overlay.py = whole-clip rolling overlay (Path C, re-anchored; flattering).
  scripts/plot_horizon.py          = NEW: single-anchor honest view (history + 1 s forecast + band +
      persistence + truth), per channel -> docs/horizon_highpass.png.
  scripts/plot_test_results.py     = predicted-vs-true scatter per channel.
  scripts/probe_*.py               = per-dataset predictability probes (categorization phase).
  scripts/crc/train_state_gpu.job  = UGE batch job (git pull; check_leakage; run sweep; writes to
      runs/ which is gitignored to avoid pull collisions). CRC netid jhao3, conda env tactile.
  data/actionsense_states/         = state_N.npy (committed) + manifest.jsonl; clip_*.npy gitignored.

### 7. This session's Q&A (2026-07-13..14) — condensed
  - Prediction method: 5 independent history models (NOT aggregated); each emits 1 s via
    autoregressive rollout. (See Session 2026-07-13 entry #1.)
  - Skill definition: 1 - MSE_model/MSE_persistence (#2).
  - Raw WAS trained; raw/highpass only changes input, same target (#3).
  - Horizon plot delivered (#4). Persistence is a weak baseline (#5). Tactile-only input (#6).
  - "Why do forecast_*.png and horizon look so different?" -> the three-paths / re-anchoring insight
    (Section 3 above).
  - "F/CoP first or highpass first?" -> MOMENTS-THEN-HIGHPASS (Section 2b).
  - "How much Slice/Peel data?" -> Section 1.
  - "Skill per entry?" -> Section 4 table.

### 8. CONFIRMED decisions (this session) vs STILL-OPEN questions
CONFIRMED by user:
  - Build BOTH decoders (autoregressive + one-shot direct) and compare.
  - Add an AR(1)/linear-extrapolation baseline (stronger than persistence).
STILL OPEN (user has NOT answered; do not code until resolved — rule 5):
  - Q1 One-shot head design: direct MLP from encoder hidden (recommended) vs non-autoregressive GRU
    with a fixed input token.
  - Q2 AR(1) coefficient phi: per-channel on TRAIN (stable, recommended) vs per-window (adaptive);
    or plain linear least-squares slope extrapolation instead of AR(1).
  - Q3 Report skill vs BOTH persistence and AR(1) (recommended) vs REPLACE persistence.
  - Q4 Re-run location: CRC batch (recommended) vs local (sweep doubles with 2 decoders).
PLAN for the implementation is at ~line 1112 (PLAN 2026-07-13).

### 9. Previously-offered but NOT-yet-requested next steps (backlog)
  - Fair-comparison history run (equal window counts across history lengths).
  - Add calibrated +/-2sigma bands to the overlay figures.
  - Simplify model to raw input only (since raw ~= highpass).
  - Feedback demo on the calibrated right-hand CoP-x model.

### 10. Guardrails / gotchas learned (do not repeat)
  - CRC: git pull + grep coverage_cal BEFORE every qsub (else the job runs stale code).
  - Job writes CSV to runs/ (gitignored) to avoid overwriting the tracked docs/ CSV on pull.
  - filtfilt is NON-CAUSAL (leaks future) -> we use sosfilt (causal). Never reintroduce filtfilt.
  - Action matching uses label.startswith(action) (substring match wrongly pooled "bread slice").
  - Windows local shell = PowerShell (.venv\Scripts\python.exe); CRC = bash (plain python).

---

## Session (2026-07-16) — PLAN: frozen evaluation harness + classical baselines  [AWAITING RESOLUTION]

User task (also written into CLAUDE.md): build a FROZEN eval harness + 3 classical baselines
(persistence, seasonal-naive, AR) for tactile forecasting. Do NOT touch/retrain the probGRU.
Hard constraints: causal-only filtering; no cross-split leakage (fit on TRAIN, select on VAL, touch
TEST once); global train-derived normalization; CoP masking by force threshold; target-time indexing.
SE: single YAML config w/ hash, deterministic (identical tables across runs), pytest on synthetic
signals, modular (metrics/masking/baselines/evaluate/config/tests).

### Repo grounding (verified 2026-07-16) — 5 spec-vs-repo contradictions
A. RATE: spec "~15 Hz, 15 steps" but manifest fps=30 (all 299 clips), pipeline ds=3 -> 10 Hz
   effective -> 1 s = 10 steps. (Whether true hardware rate is 30 Hz is a separate open uncertainty.)
B. TARGET: spec "6-dim raw [F,CoPx,CoPy] x both hands"; probGRU actually predicts 3-dim HIGH-PASS
   (fast) [F_fast,x_fast,y_fast] for ONE hand (per-hand models). action_dynamics.py:28.
C. COLLISION: src/tactile_forecast/{baselines.py, eval.py} already exist but belong to the SEPARATE
   PIXEL-map forecaster (operate on (B,t_in,C,H,W) images; LTO/LOTO/grasp). Must NOT overwrite them.
D. DEPS: pytest MISSING, statsmodels MISSING (yaml/torch/scipy/numpy OK). Plan: pip install pytest;
   implement AR with numpy (Yule-Walker/OLS), no statsmodels dependency.
E. SPLIT: only 2-way split_train_test (seed=1, frac=0.25, by-clip). No VAL, no splits.json. Need a
   FROZEN 3-way split. Existing probGRU results used the 2-way test set (seed=1).
GOOD: causal-clean already (only filtfilt is a comment saying not to use it); Norm is global/
train-derived. Constraints 1 & 3 already satisfied upstream.

### Proposed design (NOT implemented yet)
- New package `src/tactile_forecast/eval_harness/` (avoids the pixel eval.py/baselines.py collision):
  metrics.py, masking.py, splits.py, baselines/{persistence,seasonal,ar}.py, evaluate.py, __init__.py
- `configs/eval_harness.yaml` (repo already uses configs/*.yaml): horizon, history, mask_threshold_pct,
  ar_orders, seasonal_period_range, ds/fps, paths, split_file. evaluate records sha256(config).
- `tests/` (new): pytest synthetic tests (sine seasonal, AR(2) recovery, masking, causality).
- Frozen split -> `data/actionsense_states/splits.json` (train/val/test lists of clip idx).
- Metrics indexed by TARGET time (t+h) with (t,h) metadata; documented in module docstring.
- Masking: one function; frame masked for CoP metric iff RAW total force < train per-hand 5th pct;
  force channels never masked.
- Determinism: fixed seeds; assert two runs produce byte-identical table.

### OPEN QUESTIONS (blocking — need answers before coding)
- Q1 TARGET: freeze harness on (a) 6-dim RAW [F,CoPx,CoPy]x2 hands [matches spec text; probGRU not
  scorable until re-scoped], (b) 3-dim FAST 1-hand [matches current probGRU], or (c) configurable?
- Q2 RATE/HORIZON: confirm 10 Hz -> 1 s = 10 steps (spec's "15" is wrong for this repo)?
- Q3 SPLIT: keep the existing seed=1 TEST set unchanged and carve VAL out of its TRAIN (past probGRU
  test numbers stay comparable), vs define a brand-new frozen 3-way split?
- Q4 PLACEMENT/DEPS: new `eval_harness/` package (don't touch pixel eval.py/baselines.py) + pip
  install pytest + AR via numpy (no statsmodels) — OK?
- Q5 (minor) MASK KEY: mask CoP by RAW total force even when the target is the fast component
  (recommended — masking reflects physical contact, not the zero-mean fast signal). OK?

### RESOLUTION + IMPLEMENTATION (2026-07-16) — frozen eval harness BUILT

User answers to the 4 open questions: Q1 target = 6-dim RAW both-hands; Q2 rate = 10 Hz / 10
steps; Q3 split = fresh 3-way 60/20/20 by recording, stratified by (action,object); Q4 = new
`eval_harness/` package + pip install pytest + AR via numpy (no statsmodels). ("What are steps"
and "how is the current split" were answered in-chat: a step = one predicted sample; 1 s horizon
= 10 samples at 10 Hz; old split was 2-way 75/25 seed=1, no val.)

BUILT (all imported from ONE place; harness is frozen):
- configs/eval_harness.yaml            single source of truth; sha256 stamped into results.
- src/tactile_forecast/eval_harness/
    config.py     Config + config_hash (sha256 of the yaml bytes).
    splits.py     stratified 60/20/20 by recording -> data/actionsense_states/splits.json
                  (COMMITTED; n=75 -> train 45 / val 15 / test 15). By recording => both hands
                  same split. Only partitions indices, never reads signals -> cannot leak.
    dataset.py    RAW 6-dim target [F_L,CoPx_L,CoPy_L,F_R,CoPx_R,CoPy_R] from state_N.npy moments
                  0..2 of each hand, downsampled x3 -> 10 Hz. Global TRAIN z-score (Norm). Per-hand
                  force thresholds = TRAIN 5th pct.
    masking.py    ONE CoP mask: a CoP target frame is dropped iff that hand's RAW force < TRAIN
                  threshold; force channels never masked. Keys off the TARGET-frame force.
    metrics.py    masked per-channel / per-horizon MSE, MAE, skill=1-MSE/MSE_persistence, nRMSE.
    baselines/    base.Baseline (predict(hist,H) reads only hist=past<=t) + predict_series
                  (rolling-origin, structurally causal). persistence, seasonal (causal
                  same-phase-k-periods-back; period ranked on TRAIN autocorr, selected on VAL),
                  ar (per-channel OLS AR(p) on normalized signal, fit TRAIN, order selected on
                  VAL, recursive causal forecast; numpy lstsq, no statsmodels).
    evaluate.py   fit(TRAIN)->select(VAL)->score(TEST once); determinism assert (two runs
                  identical); writes docs/harness_baselines.csv with config_hash.
- tests/test_harness.py (pytest, 6 tests, ALL PASS): seasonal exact on sine; seasonal selects
  true period; persistence MSE matches analytic 1-cos(2*pi*h/T); AR(2) coeff recovery + beats
  persistence; CoP masking excludes low-force frames from CoP but keeps force; causality (future
  corruption never changes a forecast issued at t) + non-vacuous past-dependence sanity.

RESULTS (docs/harness_baselines.csv, config_hash ccb0d9c5, TEST split, mean skill vs persistence):
  persistence  nRMSE 0.517  (reference, skill 0)
  seasonal(3)  nRMSE 0.556  skill -0.13..-0.18  -> WORSE than persistence: raw aggregate force/CoP
               is NOT cleanly periodic at a single global period; copying a period back loses to
               copying the last value. (Seasonal picked the min period 3, near-persistence.)
  ar(16)       nRMSE 0.470  skill +0.12..+0.23  -> BEATS persistence; right-hand CoP-x highest
               (+0.23), echoing the probGRU finding that right-hand CoP is most predictable.
Constraints honored: causal-only (no filtfilt; baselines structurally causal + tested); no
leakage (fit TRAIN / select VAL / TEST once); global TRAIN normalization; CoP masking; target-time
indexing. pytest + the harness both green. Deterministic (assert passes).

CONTRADICTIONS SURFACED (did not silently work around): (A) spec 15 Hz/15 steps vs repo 10 Hz/10
steps; (B) spec 6-dim raw both-hands vs probGRU 3-dim fast one-hand; (C) existing pixel
eval.py/baselines.py belong to a DIFFERENT stack (left untouched); (D) pytest+statsmodels missing;
(E) only a 2-way split existed. All resolved with the user before coding.

NOTE: this harness scores the RAW 6-dim target. The current probGRU predicts the 3-dim FAST 1-hand
target, so it is NOT directly scorable here yet (by design: task said don't touch it). To later
plug the probGRU in, it must be re-scoped to predict raw 6-dim both-hands, then call the same
metrics/masking/splits.

### Step 0 (2026-07-16) — exploration for the REFINED harness spec + contradictions

Explored existing preprocessing (do-not-duplicate; reuse loaders):
(a) SEGMENTATION: probe_actionsense.py cuts per-subject ActionSense HDF5 into per-activity clips
    by Start/Stop markers, resamples each to 30 Hz, -> PS.clip_states -> state_idx.npy + manifest.
    Files processed S00..S05 in order; idx runs across subjects but SUBJECT IS NOT RECORDED.
(b) SPLIT: old split_train_test (2-way 75/25 seed=1, no val); NEW frozen eval_harness/splits.py
    -> splits.json (3-way 60/20/20 by recording, stratified action x object). Treat as FROZEN.
(c) FILTERING: raw 6-dim target has NO temporal filter, BUT physical_state.baseline_correct
    subtracts a WHOLE-CLIP 5th-pct DC offset per taxel (physical_state.py:68, percentile over the
    full time axis incl. future) => NON-CAUSAL upstream offset. Reported per instructions.
(d) NORMALIZATION: upstream baseline_correct (DC); harness Norm = global per-channel z-score TRAIN
    only. No per-window normalization.

CONTRADICTIONS / BLOCKERS (surfaced, not worked around):
  1. "per subject x activity" impossible as written: manifest has activity, NO subject. Subject only
     recoverable by re-streaming 88GB HDF5 on CRC (upstream change + data regen). Available now:
     activity x object (5 groups). 
  2. baseline_correct NON-CAUSAL (whole-clip percentile). Making it causal needs re-streaming.
  3. Deps: statsmodels MISSING (spec wants AutoReg; escape clause allows numpy), pyarrow/fastparquet
     MISSING (parquet), pandas OK.

RETRAIN? No GRU retraining in any case (harness never touches it). Baseline fits are cheap/local.
ONLY heavy recompute = RE-STREAM states on CRC, needed ONLY if we choose subject-grouping (#1) or
causal baseline-correction (#2). Otherwise everything runs locally in seconds.

PROPOSED PLAN (deltas on existing eval_harness/, reuse loaders/splits/masking/metrics):
  seasonal -> per-group period from TRAIN autocorr peak (0.3-3 s config range), fallback persistence
    + warning, store T in results. AR -> statsmodels AutoReg (numpy fallback), orders {2,5,10,15,20,30},
    order on VAL, fit-scope config (default = chosen grouping). metrics -> skill vs persistence AND
    seasonal AND ar on identical masked frames. output -> tidy long CSV + parquet
    [model,channel,hand,horizon_step,metric,value,n_frames,config_hash]. plots -> per-channel
    full-horizon skill bars + per-step skill curves. evaluate -> score external model predictions in a
    standard indexed format. README usage section.

OPEN QUESTIONS (blocking; awaiting user):
  Q1 grouping for seasonal-T / AR-fit: activity x object (now) vs re-stream for subject vs global?
  Q2 non-causal baseline_correct: accept DC offset as-is vs make causal (needs re-stream)?
  Q3 deps: pip install statsmodels + pyarrow, or numpy AR + CSV-only?

### IMPLEMENTATION (2026-07-16) — refined harness delivered (statsmodels AR, per-group, tidy output)

Decisions: grouping = activity x object; baseline_correct accepted as-is (non-causal DC, documented);
installed statsmodels 0.14.6 + pyarrow 25. NO GRU retrain, NO re-stream.

Reworked eval_harness/ (reused loaders/splits/masking; changed baselines + output):
- config: fit_scope (group|global), ar_orders {2,5,10,15,20,30}, seasonal range in SECONDS (0.3-3s),
  seasonal_min_autocorr floor. dataset.group_keys -> 'action-object' (or 'ALL' global).
- baselines now GROUP-aware (fit(train,groups)/select(val,groups,H)/predict(hist,H,group)).
  * seasonal: period per group from TRAIN autocorrelation FUNDAMENTAL peak (smallest-lag local max
    within 95% of tallest, >= floor); no peak -> fallback persistence + warning; T stored.
  * ar: statsmodels AutoReg (trend='c', numpy OLS fallback) per group per channel, order selected
    per group on VAL by iterated H-step nMSE; recursive causal multi-step forecast.
- metrics: added masked_horizon_mae. evaluate: skill vs persistence AND seasonal AND ar on identical
  masked frames; tidy LONG table [model,channel,hand,horizon_step,metric,value,n_frames,config_hash]
  -> docs/harness_baselines.csv + .parquet (990 rows); sidecar _fitparams.csv (seasonal T + AR order
  per group). External-model scoring: --model-preds preds.npz (standard target-time-indexed format).
  Determinism asserted (two runs identical).
- scripts/plot_harness.py (plots only): docs/harness_skill_bars.png + harness_skill_curves.png.
- tests/test_harness.py: 7 pytest (added seasonal-fallback; group-aware). ALL PASS.
- src/tactile_forecast/eval_harness/README.md: how to score any future model.

RESULTS (TEST, config_hash b0194860): persistence nRMSE 0.517 (ref); ar nRMSE 0.467, skill vs
persistence +0.14..+0.25 (right-hand CoP-x best +0.25), AR order 20-30 per group; SEASONAL fell
back to persistence for ALL 5 groups (skill 0). Masking active: CoP n_frames 46647-47205 vs force
49670 (~2.5-3k low-force CoP frames removed).

FINDING / OPEN QUESTION: seasonal-naive is inert because RAW aggregate force/CoP has NO
autocorrelation peak in 0.3-3 s under its slow trend (autocorr monotonically decays -> no local
max). Options to make seasonal engage: estimate the period on a CAUSAL detrended signal (first
difference or causal high-pass; TRAIN-only, still causal per constraint 1). NOT added silently ->
awaiting user decision. Deliverables 1-5 complete; no retrain.

### Session (2026-07-20) — repo reorganization: file-by-file dataset categorization (NO move yet)
User asked to group TouchAnything-dataset files vs ActionSense files, but FIRST document what each
file does + its dataset, rigorously from code + SESSION_LOG. Wrote docs/REPO_ORGANIZATION.md.
FINDING: not a clean 2-way split — THREE dataset bodies + shared:
  A TouchAnything upstream (video+pose+tactile-pixel; DINOv2/MANO/WiLoR; EgoDex/EgoPressure)
  B ActionSense (ours: physical_state, state_forecast v1, action_dynamics v2 probGRU, eval_harness)
  C EgoTouch/OpenTouch (ours: predictability study + tactile-PIXEL forecaster; EgoTouch deprecated)
  D shared/infra/root.
COUPLING: B and C are interleaved in ONE package src/tactile_forecast/ (shared __init__/categories/
predictability) -> main import-risk for any physical move. Proposed target dirs touchanything/,
tactile_pixel/, actionsense/, shared/. OPEN QUESTIONS (awaiting user): Q1 2 vs 3 buckets (recommend
3); Q2 physically move (staged+tested, high-risk) vs adopt the doc as logical map; Q3 keep src/ root
vs top-level per-dataset dirs. NOTHING MOVED — plan-before-code.

### Reorg STAGE 1 DONE (2026-07-20) — ActionSense -> src/actionsense/  [verified]
git mv physical_state.py, state_forecast.py, action_dynamics.py, eval_harness/ from
src/tactile_forecast/ -> src/actionsense/ (+ new __init__). Rewrote imports in scripts
(check_leakage, train_action_dynamics, train_state_forecaster, probe_actionsense, plot_*),
tests/test_harness.py, and prose path refs (configs/eval_harness.yaml, docs). probe_actionsense
still imports categories/predictability from tactile_forecast (group C, moves in stage 2).
VERIFIED locally: pytest 7 pass; `python -m src.actionsense.eval_harness.evaluate` -> identical
config_hash b0194860 + determinism PASS; all B imports OK. src/tactile_forecast/ now = pure group C.
Fixed a `git add -A` slip that swept AGENTS.md + tmp_diag (untracked again, local kept).
REMAINING: stage 2 (C: rename src/tactile_forecast -> src/tactile_pixel), stage 3 (A: src/{data,
models,losses,utils,datasets,resources} -> src/touchanything/). Both are NOT runtime-testable locally
(training deps absent) -> will be static/compile + import-checked only. Plus scripts/configs/docs/data
grouping. Target: src/{touchanything,tactile_pixel,actionsense}/ + shared.

### Reorg STAGES 2-4a DONE (2026-07-20)
Stage 2: src/tactile_forecast -> src/tactile_pixel (group C: EgoTouch/OpenTouch pixel + predictability).
  Verified: C imports+compile; src.tactile_pixel.{train,eval} --help; probe_actionsense B+C cross-imports.
Stage 3: src/{data,models,losses,utils,datasets,resources} -> src/touchanything/ (group A upstream).
  A independent of B/C. Verified: 25 files py_compile; B+C+pytest still pass. (Full A runtime = CRC-only.)
Stage 4a: configs/ grouped -> configs/{actionsense,tactile_pixel,touchanything}/. Updated eval_harness
  DEFAULT_CONFIG + wilor mapping paths + doc/CRC refs. Harness re-verified (determinism PASS).
FINAL src/: actionsense/ tactile_pixel/ touchanything/ (+ __init__). configs/ grouped likewise.
DEFERRED (documented in docs/REPO_ORGANIZATION.md): scripts/ grouping (needs per-script sys.path depth
fix + invocation-ref updates in docs/CRC jobs; only ActionSense scripts testable locally); docs/ + data/
(referenced by path from the FROZEN harness config -> moving breaks outputs). Awaiting user go on those.
Commits: 3c13dfe (s1) + s2 + s3 + s4a. AGENTS.md/tmp_diag kept untracked throughout.

### OPEN QUESTIONS CLOSED (2026-07-21) — clean slate before new work
1. SEASONAL-NAIVE inert (raw aggregate force/CoP has no autocorrelation peak under its slow trend)
   -> RESOLVED: close as a documented finding. Leave seasonal as fallback-to-persistence. Rationale:
   causal detrending would fix period DETECTION but not ACCURACY (seasonal copies the RAW value one
   period back; for a slow-drifting signal "one period ago ~ now" -> stays ~persistence). AR is the
   strong baseline that beats persistence. NO code change.
2. probGRU one-shot-decoder comparison + AR(1) baseline (approved 2026-07-13, never built)
   -> RESOLVED: close as SUPERSEDED. The frozen eval_harness AR baseline replaces AR(1); the one-shot
   vs autoregressive probGRU comparison is moot because the probGRU predicts the OLD target (fast
   3-dim 1-hand) which the harness redefined (raw 6-dim) -> the probGRU would need re-scoping first.
3. Reorg scripts/docs/data grouping -> CLOSED earlier (user: "stop here"; src/ + configs/ grouped).
STATUS: no open questions remain. Ready to start new work.

### Session (2026-07-21) — PLAN APPROVED: forecast F/CoP from the tactile MAP (flatten vs CNN)

Full plan: C:\Users\haoji\.claude\plans\cheeky-meandering-wigderson.md. Summary:

GOAL: does the raw tactile pressure MAP carry extra spatially-structured signal for forecasting the
next 1 s of the 6-dim F/CoP target? Two encoders feeding an IDENTICAL GRU+one-shot head (only the
per-frame encoder differs): (a) FLATTEN (2x32x32 -> Linear -> d), (b) CNN (conv -> d). If CNN > flatten,
spatial structure of the contact patch contributes. Scored on the FROZEN eval harness (vs persistence/
seasonal/AR) via --model-preds.

LOCKED DECISIONS: target = future 6-dim F/CoP (harness, unchanged); RE-STREAM CRC for all 75 Slice/Peel
maps (only 45/75 local now); per-taxel baseline = FIRST-N-FRAMES (causal, replaces non-causal whole-clip
5th-pct); map amplitude = log1p + single GLOBAL train scale (same all taxels); time scale/split/target
MATCH the harness (ds=3 ->10Hz, horizon 10, origins min_history=40); SWEEP history t_in in {1,3,10 s}
(10/30/100 frames) for both encoders; left-pad early windows with zeros (post-baseline no-contact) so
predictions exist at every harness origin (fair-comparison caveat).

STEPS: (1) CRC re-stream probe_actionsense --save-clips-for "Slice,Peel"; verify state_N.npy identical
(preserves splits/target); scp clip_N.npy local. (2) new src/actionsense/tactile_map/data.py
preprocessing. (3) reuse eval_harness dataset.load_target / splits / baselines.origins for target+windows.
(4) models.py FlattenEncoder / CNNEncoder / shared Seq2Seq. (5) 2 enc x 3 hist = 6 runs -> export
preds_<enc>_<hist>.npz -> harness --model-preds -> compare (skill vs history, flatten vs cnn vs baselines).
FILES: src/actionsense/tactile_map/{__init__,data,models,train,export_preds}.py; configs/actionsense/
tactile_map.yaml; scripts/train_tactile_map.py; tests/test_tactile_map.py. REUSE: eval_harness/*,
tactile_pixel/tactile_utils.make_transform. VERIFY: unit tests (baseline causal, origin-alignment,
log1p invertible, train-only norm) + CPU smoke train + harness scoring + flatten-vs-cnn plot.
NOTE: target keeps whole-clip-baseline F/CoP (comparability); new causal baseline affects MAP INPUT only.
IMPLEMENTATION: build code + smoke-test on the 45 local maps first; full 75-map run after CRC re-stream.

### IMPLEMENTATION (2026-07-21) — tactile-map -> F/CoP forecaster BUILT + smoke-verified
New pkg src/actionsense/tactile_map/: data.py (causal first-N-frame per-taxel baseline + log1p +
global TRAIN scale + LAZY harness-aligned windows with zero left-pad), models.py (FlattenEncoder,
CNNEncoder, shared GRU + one-shot head -> (B,H,6)), train.py (fit TRAIN / early-stop VAL / export
TEST preds npz in RAW units). CLI scripts/train_tactile_map.py; configs/actionsense/tactile_map.yaml
(baseline_frames=10, alpha=10, d=64, hidden=64, epochs=60, sweep encoders x histories{1,3,10 s}).
Reuses eval_harness dataset.load_target / splits / baselines.origins / Norm. Deviation: inlined
log1p compress in data.py (avoid actionsense->tactile_pixel cross-group dep instead of make_transform).
tests/test_tactile_map.py: 8 pass (window causality, origin alignment, left-pad, log1p, baseline
first-N, model shapes, real-map load). 15/15 across both harness suites.

HARNESS BUG FIXED: baselines/__init__.py now exports `origins` (score_external used BL.origins; the
--model-preds path had never been exercised until now).

SMOKE (available maps, 1s hist, 2 epochs, 27 train / 9 test): train+export+HARNESS SCORING works
end-to-end. Skill vs persistence: flatten -1.82, cnn -3.03 (both < persistence) -- EXPECTED for a
2-epoch undertrained model (must infer current F/CoP level from the map while persistence gets the
last value free). Mechanics proven; real training will show the actual flatten-vs-cnn result.

DATA GAP FOUND + FIX: only 45/75 slice/peel had maps because stream_actionsense.sh used
`--save-clips-for "Pour,Slice"` (Peel never saved -> all 30 Peel missing). Fixed to
"Pour,Slice,Peel". RE-STREAM NEEDED on CRC to cache Peel maps (per approved plan). available_idxs()
checks file existence, so after copying clips locally the full sweep just runs (no manifest edit needed).

NEXT: user runs CRC re-stream (git pull -> bash scripts/crc/stream_actionsense.sh), verify state_0.npy
identical (idx mapping preserved), scp clip_*.npy to local, then run the full 6-run sweep + harness score.

### CRC LOGIN — reference (2026-07-21)  (source: scripts/crc/README.md §0 + session notes)
NetID = jhao3. UGE scheduler (qsub/qstat/qrsh). Front-end has NO GPU (torch.cuda False there) ->
GPU work goes through qsub batch jobs.
- On campus / ND VPN:      ssh -Y jhao3@crcfe01.crc.nd.edu     (crcfe02 also works)
- Off campus (no VPN):     ssh -Y jhao3@bastion.crc.nd.edu     then on bastion:  ssh crcfe01
- Auth: NetID password + 2FA (donGoogle Authenticator passcode, per account setup).
- Passwordless (optional): ssh-keygen -t ed25519 ; ssh-copy-id jhao3@crcfe01.crc.nd.edu (2FA may still apply).
- After login: echo $0 ; if not bash -> run `bash` ; then `conda activate tactile`.
- Repo on CRC: cd ~/TouchAnything && git pull   (fork: github.com/Jiayi459/TouchAnything).
- Pull results back to local: rsync -avz jhao3@crcfe01.crc.nd.edu:~/TouchAnything/runs/ ./runs/
  (CRC cannot push to GitHub without a PAT; use scp/rsync to move files).

### RESULT (2026-07-22) — tactile-map -> F/CoP: CNN beats flatten (spatial structure helps)
Full 75-recording map coverage (re-streamed Peel maps from CRC; state_0 identical -> frozen split
intact). Sweep: 2 encoders x history{1,3,10 s}, 40 epochs, scored on the frozen harness TEST (15/15).

CRITICAL FIX first: absolute-level map models MEAN-REVERTED (pred ~ train-mean, not current level)
-> deeply negative skill (-1.9..-2.2, worse than persistence). Diagnosed (corr(model,true)=0.78 so
maps carry signal, but net hedges to mean on a target where persistence is very strong). FIX =
predict RESIDUAL over persistence (delta vs last observed value; anchor added back at export). At
worst the model predicts delta=0 -> matches persistence. data.py target = Y[t+1:]-Y[t]; train.export
adds tnorm-space anchor tn[t]+delta. 9 unit tests pass.

RESIDUAL RESULT (mean skill vs persistence, harness TEST):
  persistence  0.000 (ref)      ar  +0.180  (still the ceiling; classical AR on aggregates)
  flatten  1s -0.044  3s -0.030  10s -0.020   (~= persistence; flattening loses the spatial info)
  cnn      1s +0.054  3s +0.064  10s +0.084   (POSITIVE; beats persistence; rises with history)
Headlines: (1) CNN > flatten at EVERY history -> the contact-patch SPATIAL STRUCTURE contributes to
predicting the CHANGE in F/CoP (answers the core question). (2) CNN improves with history (+0.05->+0.08).
(3) flatten stuck at persistence. (4) AR on aggregates (+0.18) STILL beats map+CNN (+0.08) -> spatial
structure helps vs flattening but hasn't beaten the strong aggregate baseline yet.
Artifacts: docs/tactile_map_results.csv (9 models, tidy), docs/tactile_map_skill_vs_history.png,
docs/tactile_map_skill_bars.png, scripts/plot_tactile_map.py. Ran on CPU locally (~25 min/sweep;
models tiny). NEXT ideas: give CNN more capacity / combine map+aggregate (hybrid) to try to beat AR;
probabilistic head; or CRC GPU for a bigger sweep.

### UPGRADE (2026-07-22) — tactile-map model now matches the probGRU protocol (5-fold CV + probabilistic)
Per user ("exactly what we did for F/CoP" -> add 5-fold CV + probabilistic head), ported the
action_dynamics protocol onto the tactile-map CNN:
- models.py Seq2Seq: now PROBABILISTIC one-shot head -> (mu, logvar) each (B,H,6); lv clamped [-6,4].
- train.py: Gaussian NLL loss on the RESIDUAL; _predict/evaluate/calibrate_sigma/cross_validate.
  5-fold CV by recording (norms+model fit on TRAIN; sigma calibrated on a VAL subset of TRAIN; skill
  vs persistence + coverage@2sd measured on the held-out TEST fold). Persistence == residual 0, so
  skill = 1 - MSE(mu)/MSE(resid). Uses ALL 75 map recordings (not the frozen 15-rec test).
- scripts/train_tactile_map.py: CV sweep (encoder x history x folds) -> docs/tactile_map_cv_results.csv
  [encoder,history_s,forecast_step_s, 6x <ch>_skill, mean_skill, coverage_raw, coverage_cal].
- scripts/plot_tactile_map.py: skill-vs-history + coverage(raw/cal) from the CV CSV.
- scripts/crc/train_tactile_map_gpu.job: symlinks ~/actionsense/states/clip_*.npy into data dir,
  runs the CV sweep on GPU (30 trainings = 2 enc x 3 hist x 5 folds; GPU now genuinely warranted).
- tests: 9 pass (model shape now (mu,lv) clamped). Local smoke (2 folds, 3 ep, 1s): flatten -0.049,
  cnn +0.032; coverage 0.90->0.95 after calibration (works).
SUPERSEDES the earlier deterministic frozen-split map result (docs/tactile_map_results.csv). Full CV
run to be done on CRC GPU (qsub train_tactile_map_gpu.job) with all data + 80 epochs, then pull the
CSV + plot locally.

### CRC LOGIN — UPDATED (2026-07-22): direct crcfe01 times out off-campus -> use bastion ProxyJump
Direct `ssh jhao3@crcfe01.crc.nd.edu` times out unless on campus/VPN (crcfe01 is firewalled). FIX:
always go through the bastion. Set up ONCE in ~/.ssh/config (done on this machine):
    Host crc
        HostName crcfe01.crc.nd.edu
        User jhao3
        ProxyJump jhao3@bastion.crc.nd.edu
        ForwardX11 yes
        ServerAliveInterval 60
Then simply:
    ssh crc                                   # front-end via bastion (2FA at bastion, then password)
    scp crc:~/TouchAnything/docs/x.csv docs/  # copy FROM crc to local (uses the config)
    scp local docs/x crc:~/TouchAnything/...  # copy TO crc
Manual equivalent (no config): ssh -Y -J jhao3@bastion.crc.nd.edu jhao3@crcfe01.crc.nd.edu
Auth: bastion asks for Google Authenticator code, then crcfe01 asks for the NetID password.
On CRC: conda activate tactile ; cd ~/TouchAnything && git pull.

### CRC LOGIN — AUTHORITATIVE (2026-07-22, per docs.crc.nd.edu/new_user/connecting_to_crc.html)
Destination is ALWAYS the front-end crcfe01/crcfe02.crc.nd.edu. Off-campus you MUST tunnel, two
OFFICIAL ways (docs quote: off-campus "first need to connect to the campus VPN"; alternative "use
the server bastion.crc.nd.edu as your login host"):
  A) ND VPN then direct: connect ND campus VPN (ND OIT / vpn.nd.edu) -> `ssh -Y jhao3@crcfe01.crc.nd.edu`
     (this is the plain "previous" command; it only works on-campus or on VPN).
  B) Bastion (no VPN): `ssh -Y -J jhao3@bastion.crc.nd.edu jhao3@crcfe01.crc.nd.edu`
     Automated by ~/.ssh/config Host `crc` (ProxyJump bastion) -> just `ssh crc`.
The earlier timeout was: off-campus WITHOUT VPN and using the direct command -> blocked; use A or B.
Auth: NetID password + Google Authenticator 2FA (Okta/authenticator) at the prompt(s).

### OVERFITTING CHECK (2026-07-22) — F/CoP probGRU overfits badly at 80 epochs (no early stopping)
scripts/plot_fcop_loss_curve.py: reuses action_dynamics (load_pooled/Norm/windows/ProbGRU + NLL),
splits clips 70/15/15, logs train/val/test NLL + MSE per epoch. Config raw/right/3s, 80 epochs.
FINDING (docs/fcop_loss_curve.png): classic overfitting on BOTH metrics.
  MSE (mean, drives skill): min-val @epoch 10 (0.729) -> rises to 0.937 by epoch 80 (train 0.430).
  NLL (mean+variance):      min-val @epoch 10 (0.220) -> rises to ~1.0 by epoch 80 (train -0.236).
  val & test track each other closely (no distribution mismatch); both diverge from train after ~ep 10.
IMPLICATION: action_dynamics.train runs 80 epochs and returns the FINAL model (NO early stopping) ->
the F/CoP sweep results (docs/action_dynamics_results.csv, the +0.40 skills) come from OVERFIT models;
early stopping at ~epoch 10 would improve skill AND calibration. The NEW tactile_map train.py already
early-stops (keeps best-val), so the CRC tactile-map CV run is NOT affected -- only the old probGRU is.
RECOMMENDATION: add early stopping (keep best-val weights) to action_dynamics.train + re-run the F/CoP
sweep; or lower epochs to ~15-20. (Login methods already documented above, commit e22231f/prior.)

### OVERFITTING FIXED (2026-07-22) — early stopping in action_dynamics.train
Added early stopping: train(..., val_clips=) keeps the lowest-VAL-NLL weights (over all epochs);
cross_validate + the final-model path pass the val subset. No leakage (split by clip; norm train-only).
DEMONSTRATION (raw/right/3s, same 70/15/15 clip split, 80 epochs, held-out TEST):
  early-stop (best-val):  TEST mean skill +0.546, coverage 0.93  (per-ch F/x/y 0.54/0.55/0.54)
  overfit (80ep final):   TEST mean skill +0.401, coverage 0.81  (per-ch 0.30/0.49/0.42)
=> early stopping lifts skill +0.40->+0.55 (~36% rel) AND calibration 0.81->0.93; the force channel
(most overfit) recovers most (0.30->0.54). The old F/CoP sweep numbers (docs/action_dynamics_results.csv)
were depressed by overfitting. TODO: re-run the full F/CoP sweep with early stopping to refresh the CSV.
Baselines reminder: persistence = last-value (skill-0 reference, per-split); linear AR (harness, raw
6-dim) = +0.18 mean skill (best right-CoP-x +0.25). Split confirmed leak-free (by clip, norm train-only).

ssh -Y -J jhao3@bastion.crc.nd.edu jhao3@crcfe01.crc.nd.edu
### RESULT (2026-07-22) — tactile-map CV on CRC GPU: CNN > flatten confirmed (5-fold, probabilistic)
Full CRC GPU run of the 5-fold probabilistic CV (train_tactile_map_gpu.job, all 75 map recordings,
80 epochs, early-stopped). Mean skill vs persistence + coverage (raw->calibrated):
  cnn      1s +0.052  3s +0.050  10s +0.063   cov 0.93-0.94 -> 0.95
  flatten  1s -0.040  3s -0.025  10s -0.026   cov 0.93      -> 0.95
=> CNN beats flatten at EVERY history (spatial contact-patch structure contributes), now under the
rigorous 5-fold + probabilistic + sigma-calibration protocol; bands calibrate cleanly to 0.95. CNN
best at 10s (+0.063); flatten stays just below persistence. Consistent with the earlier deterministic
frozen-split result (cnn +0.05..+0.08). Artifacts: docs/tactile_map_cv_results.csv, _skill_vs_history.png,
_coverage.png. (AR on aggregates +0.18 still the ceiling -- map+CNN helps vs flatten but not vs AR yet.)
Added scripts/plot_tactile_map_loss_curve.py (train/val/test NLL+MSE per epoch for both encoders) to
check whether the map model overfits like the F/CoP one (running).

### LOSS CURVES (2026-07-22) — tactile-map models overfit almost immediately (flatten ep1, cnn ep4)
scripts/plot_tactile_map_loss_curve.py (3s history, 60 ep, split by clip 52/11/12, train-only norm,
eval cap 2500). docs/tactile_map_loss_curve.png:
  flatten: min-val NLL @epoch 1; final NLL tr/va/te = -0.956/1.258/1.411 (memorizes instantly, val
           rises from epoch 1 -> NO generalization window).
  cnn:     min-val NLL @epoch 4; final NLL tr/va/te = -0.926/0.969/1.407 (val MSE dips/holds ~4 epochs
           before rising -> a real learning window). 
INTERPRETATION: both overfit MUCH faster than the F/CoP probGRU (ep 1-4 vs ep 10) -- the 2048-dim map
is over-parameterized for only 45 train recordings. This IS the mechanism behind CNN>flatten: the CNN's
spatial inductive bias extracts a little GENERALIZABLE signal before overfitting; flatten memorizes at
once. Modest CV skill (+0.05-0.06) reflects DATA SCARCITY, not encoder failure; early stopping (CV keeps
best-val) is essential (else val NLL ~1.3-1.4 by ep60). NEXT to widen the CNN lead: more data
(activities/subjects), regularization (dropout/weight-decay/smaller d), or glove augmentation.

### F/CoP EARLY-STOPPED SWEEP (2026-07-23, CRC) — cross-validated, big improvement + corrects a finding
Pulled runs/action_dynamics_results.csv -> docs/action_dynamics_results_earlystop.csv. Compared to the
OLD overfit docs/action_dynamics_results.csv (mean skill vs persistence-of-fast, 5-fold CV):
  Early stopping improved skill in EVERY config by +0.10..+0.23. Examples:
    raw/right 1s   +0.410 -> +0.513   highpass/right 3s +0.369 -> +0.519   raw/left 10s +0.219 -> +0.449
  New best ~+0.51-0.52 (right hand), ~+0.46 (left). Coverage stayed ~0.94-0.95 (calibration handled both).
CORRECTION: the earlier "more history HURTS" finding was an OVERFITTING ARTIFACT. Old: skill fell with
history (right 1s +0.41 -> 10s +0.31). New (early-stopped): skill is ~FLAT across history (right 1s +0.51
-> 10s +0.50); the gain is LARGEST at 10s (+0.18..+0.23), exactly where overfitting was worst. So longer
history was only losing because more input -> more overfitting without early stopping.
NOTE: these probGRU skills are vs persistence-of-fast on the FAST 3-dim 1-hand target -- NOT directly
comparable to the harness AR (+0.18, raw 6-dim target). docs/action_dynamics_results_earlystop.csv is the
honest result; docs/action_dynamics_results.csv (overfit) kept for the before/after diff.

### BEFORE/AFTER early-stopping loss figure (2026-07-23)
docs/fcop_earlystop_comparison.png (raw/right/3s): same loss curve, two DEPLOYED checkpoints marked.
AFTER early-stop (deploy min-val ep10): test NLL 0.217, test MSE 0.737. BEFORE (deploy final ep80):
test NLL 1.036, test MSE 0.951. Early stopping recovers +0.818 NLL / +0.215 MSE on the held-out test
by deploying the min-val checkpoint instead of the overfit final one. (Early stopping = checkpoint
selection; the training trajectory is identical -- one curve, different deployed point.)

### MSE SUMMARY (2026-07-23) — all models on one scale (raw/right/3s, normalized FAST target, same 70/15/15 split seed 0)
"var explained" = 1 - MSE/0.978 (0.978 = fast-signal variance = predict-mean MSE) = the HONEST R^2.
"skill vs pers" = 1 - MSE/1.624 (persistence-of-fast MSE) = the flattering metric.

  model                     | norm test-MSE | skill vs persistence | var explained (R^2)
  --------------------------|---------------|----------------------|--------------------
  probGRU early-stop (ep10) |     0.737     |        +0.55         |       +0.25
  AR (fast target, p=20)    |     0.763     |        +0.53         |       +0.22
  probGRU overfit (ep80)    |     0.951     |        +0.41         |       +0.03
  predict-mean (0)          |     0.978     |        +0.40         |        0.00 (= variance)
  persistence-of-fast       |     1.624     |         0.00         |       -0.66

KEY POINTS:
1. Persistence-of-fast (1.624) is a WEAK baseline -- WORSE than predicting the mean (0.978) -- because
   the high-pass component oscillates around 0, so "repeat last fast value" is anti-correlated. So
   skill-vs-persistence is inflated: even predict-mean scores +0.40.
2. HONEST metric = variance explained vs the mean: early-stop probGRU +0.25, AR +0.22, overfit +0.03.
3. The LINEAR AR nearly TIES the probGRU (0.763 vs 0.737, ~3%). The GRU's edge over a simple linear
   autoregression on the fast target is marginal.
4. The OVERFIT probGRU (0.951 ~= predict-mean 0.978) had ~ZERO real skill; its flashy +0.41 was entirely
   the weak-persistence illusion. Early stopping (0.737) gives genuine skill (+0.25 R^2).
SEPARATE CONTEXT: the harness AR on the RAW 6-dim both-hands target = +0.18 skill vs persistence -- a
DIFFERENT target/scale, not comparable to the fast-target MSEs above.

### DEFINITIVE COMPARISON (2026-07-23) — forecasting the raw 6-dim F/CoP, all same target/split/protocol
Added aggregate-F/CoP encoder (neural AR: GRU on the 6-dim history) scored by the same 5-fold
probabilistic CV as the map models. FOUR-WAY mean skill vs persistence (docs/forecaster_comparison.png):
  history | linear-AR | GRU-aggregate | CNN-map | flatten-map | persistence
     1s   |  +0.180   |   +0.120      | +0.052  |  -0.040     |   0
     3s   |  +0.180   |   +0.138      | +0.050  |  -0.025     |   0
    10s   |  +0.180   |   +0.142      | +0.063  |  -0.026     |   0
RANKING: linear AR > GRU-aggregate > CNN-map > flatten-map > persistence.
CONCLUSIONS:
 1. LINEAR AR IS BEST. Neither nonlinearity (GRU) nor a richer input (map) beats a simple per-channel
    linear autoregression on the raw F/CoP -- these dynamics are essentially linear-autoregressive.
 2. THE MAP IS AN INFERIOR INPUT to the aggregate for THIS target: map models (+0.05-0.06) << aggregate
    models (+0.12-0.14). Going through the pixel representation loses info -- the net must reconstruct
    the aggregate (F=sum, CoP=centroid) from pixels imperfectly.
 3. WITHIN the map, spatial structure still helps (CNN +0.05 > flatten -0.03) -- consistent with the
    earlier finding -- but not enough to reach the aggregate, let alone AR.
 4. GRU-aggregate improves slightly with history (+0.12->+0.14) then plateaus, still below AR.
Coverage ~0.94-0.95 (calibrated) for all learned models. Fast-component probGRU (action_dynamics)
kept separate/untouched. Artifacts: docs/forecaster_comparison.png, tactile_map_cv_results_aggregate.csv,
scripts/plot_forecaster_comparison.py.

### RIGOR CHECK (2026-07-23) — GRU-aggregate vs linear AR: same target/input, protocol now matched
Verified: BOTH use load_target = RAW 6-dim both-hands F/CoP (NOT the fast/high-pass component;
dataset.py:43-52). GRU-aggregate input=past of load_target, target=future of load_target
(train.py:143-144, AggWindows); AR operates on the same load_target. Same data (Slice+Peel, 75 recs),
same autoregressive input representation. THE ONE MISMATCH: the four-way plot's AR (+0.180) came from
the FROZEN split (harness fit_and_forecast on splits.json, test 15), while the GRU used 5-FOLD CV.
FIXED: ran AR on the IDENTICAL 5-fold folds (same recs order, same seed=0 fold_of, same val carve) ->
AR mean skill = +0.166 (per-fold 0.15-0.19, stable). So protocol-matched AR = +0.166 (vs +0.180 frozen).
Ranking UNCHANGED and now fully apples-to-apples: AR +0.166 > GRU-aggregate +0.12-0.14 > CNN-map
+0.05-0.06 > flatten-map -0.03 > persistence 0. Updated docs/forecaster_comparison.png (AR line 0.166).

---

## CONSOLIDATED RESULTS & METHODS (2026-07-24) — 1-second tactile forecasting on Slice/Peel

Rigorous write-up of the current forecasting results. Self-contained: a reader should be able to
reproduce and interpret everything from this section. (Supersedes the scattered incremental entries
above for the purpose of a clean summary; those remain as the running log.)

### 0. Common setup (shared by all raw-target experiments)
- **Data**: ActionSense wearables, conductive-thread gloves, 32x32 taxels/hand, 2 hands. Activities
  **Slice (45 recordings: cucumber/potato/bread x15) + Peel (30: cucumber/potato x15) = 75**. Each
  recording is one activity interval segmented by the HDF5 Start/Stop markers, resampled to 30 Hz,
  then **downsampled x3 -> 10 Hz** (`cfg.downsample=3`). Stored as `state_<idx>.npy` (T,2,6) physical
  moments (per-taxel 5th-pct DC baseline removed) + raw map `clip_<idx>.npy` (T,2,32,32).
- **Target (raw 6-dim)**: `eval_harness.dataset.load_target` -> [F_L, CoPx_L, CoPy_L, F_R, CoPx_R,
  CoPy_R] = moments 0..2 of each hand, at 10 Hz. This is the FULL raw signal (NOT high-pass filtered).
- **Horizon**: 1 s = **10 steps** (100 ms each). Forecast origins from `min_history=40`, stride 1;
  predictions indexed by TARGET time t+h.
- **Frozen split**: `data/actionsense_states/splits.json` = 60/20/20 by RECORDING, stratified by
  (activity, object): **train 45 / val 15 / test 15**. (Also used: 5-fold CV by recording, seed 0.)
- **Causality/leakage**: all filtering causal (sosfilt, no filtfilt); normalization fit on TRAIN only;
  split by clip so no window crosses splits (verified by `scripts/check_leakage.py`, 6 checks PASS).
- **Metric**: skill = 1 - MSE_model/MSE_persistence (0=tie persistence, 1=perfect, <0=worse), per
  channel/step + mean; CoP channels masked where that hand's raw force < TRAIN 5th-pct; coverage@2sd
  for probabilistic models (target 0.95 after sigma-calibration).

### 1. Classical baselines on the raw 6-dim target (frozen harness)
Method: `eval_harness/evaluate.py` fits on TRAIN, selects hyperparameters on VAL, scores TEST once.
- **persistence** y_hat(t+h)=y(t): the 0-skill reference. STRONG here (raw F/CoP drifts slowly).
- **seasonal-naive**: period estimated per (activity,object) from TRAIN autocorrelation. **Falls back
  to persistence for ALL groups** -> the raw aggregate has NO autocorrelation peak in 0.3-3 s (slow
  trend dominates). Documented finding, not a bug.
- **linear AR**: per-channel statsmodels AutoReg, trend='c', order p in {2,5,10,15,20,30} selected on
  VAL, per (activity,object). **Mean skill = +0.180** (frozen split); **+0.166** re-scored on 5-fold
  CV (protocol-matched, per-fold 0.15-0.19). Best channel right-hand CoP-x (+0.25). nRMSE 0.467 vs
  persistence 0.517.

### 2. Tactile-MAP forecasters (raw 6-dim target)  [scripts/train_tactile_map.py, src/actionsense/tactile_map/]
Question: does the raw pressure MAP carry extra 1-s predictive signal vs the aggregates?
Method (identical for both encoders; only the per-frame encoder differs):
- Input = past `t_in` frames of the map (2,32,32); preprocessing: causal per-taxel first-N-frame
  baseline (N=10), log1p compression (alpha=10), global TRAIN scale. History swept t_in in {1,3,10 s}.
- Encoders: **flatten** Linear(2048->64); **CNN** 3-conv -> global-avg-pool -> 64. Both -> shared GRU
  (hidden 64) -> one-shot PROBABILISTIC head (mu, logvar) -> (10,6).
- Target = RESIDUAL over persistence (predict change vs last value; at worst matches persistence).
- Training: Gaussian NLL, **5-fold CV by recording**, **early stopping** (best-VAL-NLL checkpoint),
  post-hoc sigma-calibration on VAL. (CRC GPU run for the map models.)
Results (mean skill vs persistence, 5-fold CV):
    history |  CNN(map) | flatten(map)
      1s    |  +0.052   |   -0.040
      3s    |  +0.050   |   -0.025
     10s    |  +0.063   |   -0.026     (coverage 0.93 raw -> 0.95 calibrated)
- **CNN > flatten at every history** -> the contact-patch SPATIAL structure contributes to predicting
  the change in F/CoP. Flatten sits at/below persistence.
- Loss curves (`plot_tactile_map_loss_curve.py`): both encoders **overfit almost immediately** (flatten
  min-val ep1, cnn ep4) -- the 2048-dim map is over-parameterized for 45 train recordings. CNN's brief
  generalization window (val holds ~4 epochs) IS the mechanism behind CNN>flatten. Early stopping is
  essential (else val NLL ~1.3-1.4 by ep60).

### 3. Aggregate-F/CoP GRU (neural AR, raw 6-dim target)  [encoder="aggregate"]
Method: identical protocol to sec.2 (5-fold CV, probabilistic, residual, early-stop, calibration) but
input = past `t_in` frames of the raw 6-dim F/CoP itself (autoregressive; the neural counterpart of AR).
Results (mean skill vs persistence, 5-fold CV): **1s +0.120, 3s +0.138, 10s +0.142** (coverage ~0.95).

### 4. FOUR-WAY comparison (raw 6-dim target, IDENTICAL 5-fold CV, same input/target/data)  [docs/forecaster_comparison.png]
    linear-AR +0.166  >  GRU-aggregate +0.12..+0.14  >  CNN-map +0.05..+0.06  >  flatten-map -0.03  >  persistence 0
RIGOR: verified GRU-aggregate and AR use the SAME target (load_target, raw 6-dim both hands, NOT fast),
SAME autoregressive input, SAME Slice/Peel data, and (after fixing) the SAME 5-fold folds (AR re-scored
5-fold = +0.166 vs +0.180 frozen-split). Caveat: AR is not swept over t_in (it selects its own order as
history) -> a single history-agnostic number.

### 5. Fast-component probGRU (SEPARATE experiment, different target)  [src/actionsense/action_dynamics.py]
This is the ORIGINAL v2 model on a DIFFERENT target: the **high-pass FAST component**, 3-dim, ONE hand
([F_fast,x_fast,y_fast]). NOT comparable to the raw-target results above (different target/baseline).
- **Overfitting fix**: `action_dynamics.train` originally ran 80 epochs and returned the FINAL model
  (no early stopping) -> badly overfit (train/val/test loss curve: val bottoms ~epoch 10 then rises;
  `docs/fcop_earlystop_comparison.png`). Added early stopping (keep best-VAL). Effect (5-fold CV):
  skill improved **+0.10..+0.23 per config**; best ~**+0.51** (right hand); and skill became **~flat
  across history** (the earlier "more history HURTS" finding was an OVERFITTING ARTIFACT -- longer
  history overfit more without early stopping). Coverage ~0.95 (calibration handled both).
- **MSE decomposition** (raw/right/3s, normalized fast target, one split): probGRU-earlystop 0.737,
  AR-on-fast 0.763, predict-mean 0.978 (=variance), probGRU-overfit 0.951, persistence-of-fast 1.624.
  Honest R^2 (=1-MSE/0.978): early-stop +0.25, AR +0.22, overfit +0.03. -> (a) persistence-of-fast is
  a WEAK baseline (worse than the mean) so skill-vs-persistence is inflated; (b) the honest signal is
  ~25% variance explained; (c) linear AR ~TIES the GRU on the fast target too.

### 6. ANALYSIS / CONCLUSIONS
1. **A linear autoregression is the best 1-s forecaster of the raw F/CoP.** Neither nonlinearity (GRU)
   nor a richer input (the tactile map) beats per-channel linear AR. The predictable part of this
   signal over 1 s is the (near-)linear autoregressive trend.
2. **The tactile MAP is an inferior input to the aggregate** for this target (map +0.05 << aggregate
   +0.14): passing through the pixel representation loses information -- the net must reconstruct
   F(=sum) and CoP(=centroid) from pixels, imperfectly, when those aggregates were available directly.
3. **Within the map, spatial structure still helps** (CNN +0.05 > flatten -0.03), robust across history
   and CV -- but not enough to reach the aggregate, let alone AR.
4. **Overfitting silently depressed the neural results** (both the fast-target probGRU and the map
   models). Early stopping is decisive: it turned a near-useless overfit fast-probGRU (R^2 +0.03) into
   a genuinely predictive one (R^2 +0.25), and it corrected the spurious "more history hurts" trend.
5. **Baseline choice matters for interpretation.** Skill-vs-persistence flatters models on the FAST
   target (persistence-of-fast is worse than the mean); variance-explained (R^2 vs mean) is the honest
   cross-target metric.

### 7. CAVEATS / LIMITATIONS
- **Data-scarce**: only 45 training recordings (Slice+Peel); the 2048-dim map models overfit by ~epoch
  1-4. Conclusions about "map doesn't help" are for THIS data regime; more data/regularization/augmentation
  could change the map's standing.
- **1-s horizon only**; linear AR dominates at 1 s -- longer horizons (where linear extrapolation breaks)
  were not tested and could favor the map/nonlinear models.
- **Two targets in play**: raw 6-dim both-hands (harness, sec.1-4) vs fast 3-dim one-hand (probGRU, sec.5).
  Their skill numbers are NOT comparable (different persistence baselines). R^2-vs-mean is comparable.
- AR uses its own order as history (not swept over t_in); it is a single number in the four-way plot.

### 8. ARTIFACTS
- Figures: `docs/forecaster_comparison.png` (four-way), `tactile_map_skill_vs_history.png`,
  `tactile_map_coverage.png`, `tactile_map_loss_curve.png`, `fcop_earlystop_comparison.png`,
  `fcop_loss_curve.png`, `harness_*curves.png`.
- Tables: `docs/tactile_map_cv_results.csv` (cnn/flatten), `tactile_map_cv_results_aggregate.csv`,
  `harness_baselines.csv` (persistence/seasonal/AR), `action_dynamics_results{,_earlystop}.csv` (fast).
- Code: `src/actionsense/{eval_harness, tactile_map, action_dynamics.py}`; `scripts/train_tactile_map.py`,
  `plot_forecaster_comparison.py`, `plot_tactile_map*.py`, `plot_fcop_loss_curve.py`, `check_leakage.py`.
- Tests: `tests/test_harness.py` (7), `tests/test_tactile_map.py` (10) -- all pass.

---

## PORTABILITY / CONTINUE ON ANOTHER COMPUTER (2026-07-24)
Repo: https://github.com/Jiayi459/TouchAnything (origin/main @ 0071021). Working tree CLEAN, nothing
unpushed. All ActionSense RAW TACTILE MAPS are now ON GITHUB (un-gitignored + committed).

### PUSHED to GitHub (a plain clone has all of this)
- All code: src/ (actionsense/{eval_harness,tactile_map,action_dynamics.py}, tactile_pixel/,
  touchanything/), scripts/, configs/, tests/. SESSION_LOG.md, CLAUDE.md, AGENTS.md.
- data/actionsense_states/: state_*.npy (299), manifest.jsonl, splits.json, AND all 100 clip_*.npy
  (raw tactile maps, ~401MB) -> covers all 75 Slice/Peel recordings + 25 others. (Un-gitignored 2026-07-24.)
- All docs/: every result CSV (harness_baselines, action_dynamics_results{,_earlystop},
  tactile_map_cv_results{,_aggregate}, ...) and every figure (forecaster_comparison.png, loss curves,
  skill plots, ...).
- scripts/tmp_diag_predictability.py.
=> Everything needed to REPRODUCE the forecasting work travels with the repo (states + maps + results).

### NOT pushed (gitignored) -- and why / how to get them
- datasets/ (~15GB): EgoTouch/OpenTouch/grasp_hold_lift_tactile = the OLDER PIXEL work. Too big for git,
  NOT needed for the current ActionSense forecasting. Re-download (scripts/download_egotouch.py,
  scripts/crc/download_opentouch.sh) only if returning to that thread.
- .venv/ (~1.3GB): Python env -> regenerate with pip (below).
- runs/ (~125MB): model npz + CRC logs -> regenerable; the results are already captured in docs/.

### SETUP on the new machine
  git clone https://github.com/Jiayi459/TouchAnything.git && cd TouchAnything
  python -m venv .venv
  .venv/bin/pip install numpy torch scipy pandas statsmodels pyarrow pytest matplotlib pyyaml   # (+ h5py only for CRC streaming)
  pytest tests/            # expect 17 passing (7 harness + 10 tactile_map)
Then you can immediately: view all results (docs/), re-run harness scoring, the aggregate-GRU, the
tactile-map CV (maps are present), and all plots. GPU training uses CRC (see CRC LOGIN section above:
`ssh crc` via bastion ProxyJump).

### SECOND COPY on CRC
~/TouchAnything (git repo; `git pull` to sync) + ~/actionsense/states/ (state_*.npy + clip_*.npy +
manifest.jsonl from the re-stream). The ActionSense HDF5 were deleted (streaming-delete); re-stream via
`bash scripts/crc/stream_actionsense.sh` (KEEP=1 to retain) only if regenerating states from scratch.

---

## Session (2026-08-05) — PROJECT CONCLUSIONS document

### Request
Read SESSION_LOG.md thoroughly; conclude all results so far as points, covering project detail with
high clarity/logic/completeness; write it as a separate file in the documents folder; refer to
specific code and documents throughout; list references at the end; ask questions if any arise.

### Work done
- Read all 1,927 lines of SESSION_LOG.md (Sessions 1-4 + COMPREHENSIVE SUMMARY + COLD-START SNAPSHOT
  + rigorous review + CONSOLIDATED RESULTS + PORTABILITY).
- **VERIFIED every quantitative claim against the committed artifacts** rather than trusting the log:
  recomputed mean skills directly from `docs/harness_baselines.csv` (AR +0.1886 over 66 rows;
  seasonal 0.0 == persistence, confirming the documented fallback), `docs/tactile_map_cv_results.csv`
  (cnn +0.052/+0.050/+0.063, flatten -0.040/-0.025/-0.026), `_aggregate.csv` (+0.120/+0.138/+0.142),
  and both `action_dynamics_results{,_earlystop}.csv` (per-config means + calibrated coverage). All
  match the log. Also confirmed `splits.json` = train 45 / val 15 / test 15, n=75, seed 0.
- Checked the harness config hash: `sha256(configs/actionsense/eval_harness.yaml)[:16]` =
  **8afc249f260894fd**, which MATCHES the hash stamped in `docs/harness_baselines.csv`. The
  `b0194860` cited in the 2026-07-16 log entry is stale — it predates commit `ee8d097` (reorg stage
  4a) which MOVED the yaml into `configs/actionsense/`, changing the file bytes and hence the hash.
  Results are current; only the log's hash string is historical. (No action needed; noted so a future
  reader isn't confused by the mismatch.)
- Verified the code line references cited in the new document actually exist (physical_state.py:68
  percentile baseline; action_dynamics.py slow_fast/build_features/windows/ProbGRU/train/
  calibrate_sigma/evaluate; tactile_map/data.py:51 log1p and :140 residual-over-persistence target;
  eval_harness/metrics.py:45 skill).

### Deliverable
**NEW `docs/PROJECT_CONCLUSIONS.md`** — the concluding document. Structure: (0) what the project
became; (1) the five phases; (2-6) per-phase results as points, each with code/doc citations;
(7) nine cross-cutting methodological conclusions; (8) caveats/limits; (9) status + open items
ranked by scientific value; (10) full reference list (log sections, project docs, code modules with
line anchors, scripts, result artifacts, external datasets/papers).

Content decisions (stated for review):
- Positioned as a CONCLUSION of the whole project, not a replacement for the existing
  docs/STUDY_SUMMARY.md (predictability study) or docs/RESULTS.md (EgoTouch pixel forecaster) —
  it links to both and does not duplicate their tables beyond the headline numbers.
- Deliberately foregrounded the SELF-CORRECTIONS (filtfilt leak -0.3 skill; early stopping turning
  R^2 +0.03 into +0.25 and retracting the "more history hurts" finding; the weak-baseline audit
  showing predict-zero scores +0.57) as first-class results, since they change how every earlier
  number must be read.
- Recorded the original grasp-success goal as formally unanswerable on EgoTouch (no success labels)
  — this is the reason the project pivoted, and it belongs in a conclusion.
- Flagged that the feedback/adaptive-strategy application (the ultimate stated goal) is still
  UNBUILT despite all its prerequisites now existing — open item #1.

### Open questions for the user (non-blocking; document delivered)
1. Audience: is this for a supervisor/collaborator handoff, or a draft skeleton toward a paper? A
   paper draft would want the four-way comparison and the metric audit promoted to the front and the
   EgoTouch phase compressed to a paragraph.
2. Should `docs/STUDY_SUMMARY.md` and `docs/RESULTS.md` now be marked as superseded-in-part by this
   document (they predate the causal-filter and early-stopping corrections), or left standing as
   phase-local records?
3. Should the honest R^2-vs-mean metric be recomputed for the raw-6-dim four-way comparison (§6.4)?
   Currently only the fast-target experiment (§5.5) has it; without it the +0.166 AR headline still
   rests on skill-vs-persistence, which the audit showed can flatter.

---

## Session (2026-08-06) — DATA AVAILABILITY AUDIT (ActionSense content/format; OpenTouch; Force-Vision)

### Request
Point out the ActionSense data and its content/format; and for OpenTouch
(opentouch-tactile.github.io) and the ICLR Force-Vision submission site, report whether they are
downloaded, how much, and where. Refer to SESSION_LOG for history.

### Method — verified on the filesystem, not from the log
Did not trust prior log statements; re-derived every number from the working tree
(`/Users/haojiayi/TouchAnything`, branch main, tree clean) with `du`, `numpy` loads of every
`.npy`, and a parse of `manifest.jsonl`. Also ran a `find ~ -maxdepth 4` sweep for
`*opentouch*`/`*force*vision*`/`*actionnet*` dirs and any `*.hdf5`/`*.h5` under `$HOME`.

### FINDING 1 — ActionSense: the ONLY dataset present on this machine (416 MB, git-tracked)
Location: `data/actionsense_states/` — 401 tracked files (`git ls-files data | wc -l` = 401),
416 MB, all committed to GitHub (un-gitignored 2026-07-24, see PORTABILITY section).
Two artifact kinds, disjoint index sets, plus two metadata files:
- **`state_*.npy` x 299** — physical-state time series, `float32`, shape **(T, 2, 6)**
  = (frames, {left,right} glove, `[F, xbar, ybar, sxx, syy, sxy]`), definitions in
  `src/actionsense/physical_state.py:26,37` (F = Σp; CoP = Σp·x/Σp; second moments). Resampled to
  **30 Hz** from the native ~6 Hz. T range 82 / median 763 / max 6602; **320,309 frames total**
  (per manifest). Derived AFTER per-taxel 5th-percentile baseline subtraction
  (`physical_state.py:64-68`).
- **`clip_*.npy` x 100** — raw tactile maps, `float16`, shape **(T, 2, 32, 32)** = (frames,
  {L,R}, 32x32 taxels), ~401 MB of the 416 MB, value range ~499-1016 (uncalibrated sensor units,
  DC-offset NOT removed at this stage). 422,696 frames across the 100 clips.
- **`manifest.jsonl`** (299 lines, one per state clip): `{idx, label, cat, fps, T, features,
  has_clip}`.
- **`splits.json`**: train 45 / val 15 / test 15 (n=75, seed 0) — the Slice/Peel forecasting subset.
- **STALE FIELD FOUND:** `manifest.jsonl` reports `has_clip=true` for only **70** clips, but **100**
  `clip_*.npy` exist on disk. The manifest was written before the second map-extraction pass; every
  clip id on disk does resolve to a manifest row, so nothing is orphaned — but code must not use
  `has_clip` as the source of truth for map availability. Glob the directory instead.
- **Which clips have raw maps** (all 100, by label): Pour water 25, Peel cucumber 15, Slice cucumber
  15, Peel potato 15, Slice potato 15, Slice bread 15. I.e. the 75 Slice/Peel recordings used by the
  tactile-map CV + 25 Pour. No maps exist for Clean/Spread/Organize/Open-close-jar.
- **Full label inventory (299 state clips, 22 raw labels / 6 categories)**: Clear cutting board 28,
  Pour water 25, Get items from fridge/cabinets/drawers 24, Peel cucumber 15, Slice cucumber 15,
  Peel potato 15, Slice potato 15, Slice bread 15, Spread almond butter 15, Spread jelly 15, Clean
  plate w/ sponge 15, Clean plate w/ towel 15, Clean pan w/ sponge 15, Clean pan w/ towel 15,
  Get/replace items 15, Open/close jar 9, Open jar 6, Get items from cabinets 6, Set table 6, Stack
  on table 5, Load dishwasher 5, Unload dishwasher 5. Categories: Cut / Pour / Wash-Clean /
  Fold-Cloth(spread) / Organize-Arrange / Open-Close.
- **NOT present locally:** the source ActionSense wearables HDF5 (~2-4 GB each x 12, ~35-88 GB) —
  deleted by the streaming driver `scripts/crc/stream_actionsense.sh` on CRC. Re-fetchable via
  `scripts/crc/download_actionsense.sh` (12 public URLs, data.csail.mit.edu/ActionNet, S00-S05).
  Consequence already noted in PROJECT_CONCLUSIONS §8: subject ids are unrecoverable without a
  ~88 GB re-stream.

### FINDING 2 — OpenTouch: downloaded once on CRC, since DELETED; nothing on this machine
- Was fully downloaded on CRC (2026-07-02): **26 HDF5 shards ~14 GB (561 MB/shard) + labels**, via
  `scripts/crc/download_opentouch.sh` (26 gdown file IDs + `final_annotation.zip` ID, copied
  verbatim from OpenTouch-MIT/opentouch), default dest `~/opentouch/data`. Probed successfully:
  2,496 usable of 2,958 clips.
- **Then deleted** during the ActionSense disk crisis (SESSION_LOG:571-573: "OpenTouch raw data got
  deleted along the way (kept its earlier CSV)"). Confirmed 0 bytes on this Mac: no `~/opentouch`,
  no `.hdf5`/`.h5` anywhere under `$HOME`, no `datasets/` dir in the repo.
- **Its derived result CSV is also not local:** `docs/predictability_opentouch.csv` was written on
  CRC and never pushed. `docs/` only holds `predictability_by_category{,_full}.csv`, which I
  verified are the **EgoTouch** probe outputs (grouping/group/n/pers_nmse_h1/h15/h30/periodicity/
  contact_migration/predictability_index; B5 composite n=379/411 — EgoTouch trajectory counts, not
  OpenTouch's 2,496). The OpenTouch numbers survive only as prose in
  `docs/ACTION_CATEGORIES.md` §3b and `docs/PROJECT_CONCLUSIONS.md`.
- Recovery cost if needed: `pip install gdown && bash scripts/crc/download_opentouch.sh` -> ~14 GB.

### FINDING 3 — Force-Vision (sites.google.com/view/iclr-submission-force-vision): NEVER downloaded
- 0 bytes, on any machine, at any point. Confirmed by log (SESSION_LOG:385, 588, 703) and by
  `docs/PROJECT_CONCLUSIONS.md:368` ("Force-Vision was never downloaded — its contribution is
  taxonomic only"), and by the disk sweep (no `*force*` dirs under `$HOME`).
- No download script exists for it (`scripts/crc/` has only `download_actionsense.sh` and
  `download_opentouch.sh`); `scripts/download_egotouch.py` covers EgoTouch.
- Availability was checked on 2026-07-02 (SESSION_LOG:483): **public Google-Drive zip**.
- Its entire contribution to the project is taxonomic, read from the paper: STAG glove
  (Sundaram et al.), 2,000,000 frames, 89 articulated tools, manipulation types press / hold /
  squeeze, mapped into Axes A/B/C/D in `docs/ACTION_CATEGORIES.md` §2. No tactile array from it has
  ever been loaded.

### Summary table (state as of 2026-08-06)
| Dataset | Downloaded? | How much | Where |
|---|---|---|---|
| ActionSense | Yes, derived form only | 416 MB (299 state + 100 map .npy + manifest/splits) | `data/actionsense_states/` (git-tracked, on GitHub); raw HDF5 deleted from CRC |
| OpenTouch | Yes once, now deleted | was ~14 GB / 26 shards; now 0 | ex-`~/opentouch/data` on CRC; nothing local; result CSV also only on CRC |
| Force-Vision | **No, never** | 0 | n/a — paper-only taxonomy |
| EgoTouch (4th, deprecated) | Yes, on CRC | part of gitignored `datasets/` ~15 GB | not local; deprecated 2026-07-03 |

### Conclusions / risks worth acting on
1. `manifest.jsonl:has_clip` is stale (70 vs 100). Low severity today because the forecasting code
   globs, but it is a live trap for any future consumer. Cheap fix: regenerate the field.
2. `docs/predictability_opentouch.csv` is the only quantitative OpenTouch artifact and it exists on
   exactly one machine (CRC) with no backup. If OpenTouch conclusions are to survive, that CSV
   should be pulled into `docs/` and committed — far cheaper than the 14 GB re-download.
3. Force-Vision remains the one clean way to add a 4th sensor geometry to the cross-dataset trait
   claim, and it is a single public Google-Drive zip. Still open, still cheap.

### No code changed this session (audit only).

---

## SESSION (2026-08-06) — PORT THE FORECASTER TO OPENTOUCH: dataset analysis + PLAN (AWAITING USER)

### Request
User: "look at https://opentouch-tactile.github.io/ in detail; check whether the predictability
algorithm could also be used on this dataset — right now it was only trained on ActionSense.
Analyze the dataset: format? actions? which actions should be downloaded? Reference SESSION_LOG
for how we downloaded before (I believe part of OpenTouch is already on CRC). Ask questions until
100% sure, then WAIT for permission before downloading."

### CLARIFICATION OF WHAT "THE PREDICTABILITY ALGORITHM" MEANS HERE
Two distinct artifacts exist in this project and only ONE of them is ActionSense-only:
1. **Training-free predictability probe** (persH1/15/30, periodicity, contact_migration, PI) —
   ALREADY RUN ON OPENTOUCH (2,496 usable of 2,958 clips, 2026-07-02; SESSION_LOG:530-543,
   docs/ACTION_CATEGORIES.md 3b). Nothing to port.
2. **Trained physical-state forecaster + frozen eval harness** (Phase D/E: F/CoP target, 1 s
   horizon, persistence / seasonal-naive / linear AR / GRU-aggregate / CNN-map / flatten-map,
   frozen 60/20/20 split, sigma-calibration) — ActionSense ONLY. **This is what the user means.**
   Porting it to OpenTouch = the first CROSS-SENSOR test of the four-way ranking
   (AR +0.166 > GRU +0.14 > CNN-map +0.05 > flatten-map -0.03 > persistence 0).
This is also PROJECT_CONCLUSIONS 9 open item #7 ("GPU per-category forecasting on OpenTouch to
convert the probe hypothesis into measured skill").

### DATA STATE (re-verified against the 2026-08-06 audit above)
- OpenTouch raw HDF5: downloaded once on CRC (~14 GB), **DELETED** during the ActionSense disk
  saga. Nothing on this Mac. `docs/predictability_opentouch.csv` also exists ONLY on CRC.
- => A full re-download is required. `scripts/crc/download_opentouch.sh` (26 shard IDs + labels)
  was re-verified today against the upstream `OpenTouch-MIT/opentouch/scripts/download_data.sh`:
  **27 IDs (26 shards + `1cM-816vcCnkgWVIGXZrR1o8TPsDvRVCZ` labels) — still current, unchanged.**

### DATASET ANALYSIS — OpenTouch (arXiv 2512.16842; opentouch-tactile.github.io)
**Capture**: Meta Aria glasses (egocentric RGB) + FPC full-hand tactile glove + Rokoko Smartgloves
(pose), synchronized at **30 Hz**, ~2 ms mean latency. 5.1 h total, ~3 h densely annotated,
14 environments, ~800 objects / 14 object categories. Single **RIGHT** hand.

**Distribution format**: Google Drive via `gdown`; **26 HDF5 shards** (~561 MB each, ~14.6 GB
total, one shard per scene+participant, e.g. `office_csail_p2.hdf5`, `fablab_ml_p1.hdf5`) plus
`final_annotation.zip` (small).

**HDF5 schema (confirmed live on CRC 2026-07-02, SESSION_LOG:507-523)**
```
<shard>.hdf5
  calibration, transform_slam_to_rgb
  data/<clip_id>/
      right_pressure     (T,16,16) float32, max 3072        <- THE ONLY THING WE NEED
      rgb_images_jpeg    <- the reason shards are 561 MB
      camera_poses, hand_landmarks, timestamps
      labels             (0,0) index-pair — NOT the action
```
**Labels**: `final_annotations/<scene>_merged.csv`, one row per clip, key `clip_id` =
`"<scene>::demo_NNN"` (globally unique), columns: `action` (free-text gerund), `grip_type`
(29-class GRASP taxonomy), `object_name`, `object_category`, `environment`, `description`,
`peak_idx`. Join rate verified 100% on the pilot shard; 457/2,958 clips carry no label row.

**Actions (observed vocabulary, gerunds)**: placing, adjusting, removing, pinching, picking up,
holding, pulling, pushing, moving, pressing, turning, pouring, serving, eating, stirring,
scooping, flipping, wiping, cutting, plus examining, carrying, lowering, aligning, typing,
touching, tightening, unscrewing, tilting, tapping, feeling, inspecting, switching, detaching,
attaching, pointing, resting. Mapped into our taxonomy by `categorize_phrase()`.

**Our own probe ranking on these actions (PI, easiest -> hardest)**:
`pouring +4.4 - serving +3.6 - eating +3.4 - stirring +3.0 - scooping +2.5 - flipping +2.4 -
wiping +1.3` ... `pulling -1.8 - turning -2.2 - moving -2.6 - cutting (n=4) -3.0`.
=> The forecasting experiment has a PRE-REGISTERED PREDICTION: skill should track this order.

### GAP ANALYSIS — what breaks when the ActionSense harness meets OpenTouch
| # | Gap | Detail | Proposed handling |
|---|---|---|---|
| G1 | **One hand** | target is 6-dim `[F,CoPx,CoPy] x {L,R}`; OpenTouch has right only | 3-dim target `[F_R,CoPx_R,CoPy_R]`; compare against the ActionSense RIGHT-hand columns only |
| G2 | **Clip length** | 5.1 h / 2,958 clips = **~6.2 s mean**. Harness `min_history: 40` @10 Hz = 4 s + 1 s horizon = 5 s floor -> most clips yield 0-12 origins | lower `min_history`; report retained-clip count; see OQ3 |
| G3 | **Rate** | OpenTouch is genuinely 30 Hz (ActionSense was 6 Hz *up*sampled then down to 10) | see OQ3 |
| G4 | **Baseline/DC** | FPC pressure is probably ~0 at rest (unlike the conductive thread's ~571/taxel DC that caused bug P4). But clips are segmented around `peak_idx`, so a causal first-N-frames baseline may already be IN contact | measure the rest-value distribution in `--inspect` BEFORE choosing; do not assume |
| G5 | **Force units** | 0..3072 FPC vs uncalibrated thread -> `F` is NOT comparable across corpora | zero-shot transfer would need per-corpus z-scoring; CoP (already [-1,1]) is the only natively transferable channel |
| G6 | **Split axis** | ActionSense manifest has **no subject id** -> its results are within-corpus only (open item #5) | OpenTouch `clip_id` encodes **scene + participant** (`_p1`/`_p2`) -> we CAN hold out environment/participant. **This is a genuine upgrade over ActionSense.** |
| G7 | **Fit grouping** | harness `fit_scope: group` = action x object | OpenTouch has `action` + `object_category` -> maps cleanly |
| G8 | **Map encoder** | flatten branch is `Linear(2048->64)` for 2x32x32 | 1x16x16 = 256 -> re-instantiate input dim (trivial); CNN branch is shape-agnostic |
| G9 | **Disk** | 14.6 GB of shards, of which we need only `right_pressure` (~276 MB fp16 for the WHOLE corpus) | stream-extract (download shard -> extract -> delete), same trick as `stream_actionsense.sh`; result is a permanent ~300 MB local cache, never re-download again |

### WHICH CLIPS/ACTIONS TO DOWNLOAD — the key point
Actions are **scattered across scenes**, so shards cannot be pre-filtered by action *blindly*.
BUT `final_annotation.zip` is small and independent -> **download labels FIRST (minutes), build the
action x scene contingency table, THEN decide which shards to pull.** This turns "what should be
downloaded" from a guess into a measurement. Recorded as the mandatory Step 0.

### PROPOSED PLAN (pending answers)
- **Step 0 (cheap, ~1 min, no commitment)**: `gdown 1cM-816vcCnkgWVIGXZrR1o8TPsDvRVCZ` -> unzip ->
  action x scene x participant x object_category counts + clip-duration histogram if derivable.
  Decide shard subset from real counts.
- **Step 1**: download the selected shards; stream-extract `right_pressure` + `timestamps` + label
  row -> `data/opentouch_states/{state_N.npy, clip_N.npy(fp16), manifest.jsonl}` mirroring the
  ActionSense layout so the harness loads it with a config swap, not a rewrite; delete each shard
  after extraction. Verify G4 (rest baseline) here.
- **Step 2**: `configs/opentouch/eval_harness.yaml` (3-dim target, rate/min_history per OQ3,
  split per OQ4) + minimal generalizations in `eval_harness/dataset.py` + `splits.py`.
- **Step 3**: classical baselines (persistence / seasonal-naive / linear AR) on the frozen split.
- **Step 4**: GRU-aggregate + CNN-map + flatten-map -> reproduce the four-way comparison figure.
- **Step 5**: per-action skill sweep -> test the probe's pre-registered ordering.
- Report BOTH skill-vs-persistence AND R^2-vs-mean (methodological lesson #4).

### OPEN QUESTIONS (asked of the user 2026-08-06; NOT resolved yet — no code, no download)
- **OQ1 — primary experiment**: in-domain refit of the four-way comparison (sensor-independence
  test) / zero-shot transfer of the ActionSense-fitted models / both / per-action skill sweep?
- **OQ2 — download scope**: all 26 shards, or a label-driven subset, or a 2-3 shard pilot first?
- **OQ3 — rate & history**: downsample 30->10 Hz (exactly matches the ActionSense harness:
  horizon = 10 steps, AR order grid 2..30) vs native 30 Hz (3x the windows, horizon = 30 steps,
  AR order grid must be rescaled) vs both. Couples to G2: at 10 Hz, `min_history: 40` = 4 s and
  most 6 s clips die; proposed default `min_history: 20` (2 s) with retention reported.
- **OQ4 — split axis**: hold out scene/participant (new-environment claim, impossible on
  ActionSense) vs clip-level stratified (apples-to-apples with the ActionSense protocol) vs both.
- **OQ5 — where**: CRC (download script already there, 14 GB fits) then rsync the ~300 MB cache
  local for CPU modelling — vs entirely local.
- **OQ6 — free rider**: while on CRC, also pull back `docs/predictability_opentouch.csv`
  (the only quantitative OpenTouch artifact, single-copy on CRC)? It is a file copy, not a rerun.
  *(Note: it is on CRC only if that file survived; if not, Step 1 regenerates it for free.)*

### STATUS: WAITING for user answers to OQ1-OQ6. Nothing downloaded, no code written.

### DECISIONS (user, 2026-08-06, via AskUserQuestion) — OQ1-OQ5 RESOLVED
| OQ | Decision | Consequence |
|---|---|---|
| OQ1 experiment | **Both**: in-domain refit of the four-way comparison **+** per-action skill sweep | One training pipeline; the sweep is a second reporting pass over the same TEST predictions, grouped by `action`. Pre-registered prediction = the probe order (pouring/serving/eating/stirring/scooping easiest; cutting/moving/turning/pulling hardest). |
| OQ2 download | **Labels first, then decide** | Step 0 = `gdown 1cM-816vcCnkgWVIGXZrR1o8TPsDvRVCZ` only. Build action x scene x participant x object_category counts. Shard selection is then EVIDENCE-DRIVEN, not guessed. No 14 GB commitment until the table exists. |
| OQ3 rate/history | **10 Hz (downsample 3), `min_history: 20`** | horizon = 10 steps = 1.0 s and AR order grid `[2,5,10,15,20,30]` stay IDENTICAL to the ActionSense harness -> numbers are directly comparable. min_history 40->20 (4 s -> 2 s) so ~6 s clips survive; retained-clip count is a REPORTED number, not a silent filter. Native 30 Hz deferred (not refused) as a later sensitivity check. |
| OQ4 split | **Both; scene/participant hold-out is the HEADLINE**, clip-level stratified reported alongside | Headline = a real new-environment/new-participant generalization claim, which ActionSense structurally could not make (no subject id; PROJECT_CONCLUSIONS 9 open #5). Clip-level = apples-to-apples with the existing ActionSense +0.166. Expect headline < clip-level; that gap is itself a result. |
| OQ5 where | **CRC download+extract, rsync ~300 MB cache local, model locally on CPU** | Mirrors the ActionSense workflow. User runs the CRC commands (I have no SSH); I do everything downstream. |
| OQ (map) | **DEFER the tactile-map branch.** First pass = persistence / seasonal-naive / linear AR / GRU-aggregate | The cache still stores raw 16x16 clips (fp16), so CNN-map/flatten-map can be added later WITHOUT re-downloading. **Recorded honestly: the four-way comparison is therefore NOT completed in this pass** — the refit tests AR vs GRU vs the two classical baselines only. G8 (Linear(2048->64) -> 256) stays open. |

### DEFAULTS I AM SETTING (stated, not asked — override any of these if you disagree)
- **D1 per-action sweep threshold**: report per-action skill only for actions with **n >= 30 clips**
  in the corpus; everything below is pooled into `other (n<30)` and named. Prevents a repeat of the
  probe's `cutting (n=4) -3.0` entry being read as a real ranking position.
- **D2 target**: 3-dim `[F_R, CoPx_R, CoPy_R]` (G1). When quoting ActionSense for comparison I will
  use its **right-hand** channels only, not the 6-dim mean.
- **D3 baseline/DC (G4)**: **measured, not assumed.** Step 1 dumps the per-taxel rest-value
  distribution and the value at frame 0 vs `peak_idx`. Only then do I pick between (a) no
  correction, (b) causal first-N-frames, (c) whole-clip percentile. If clips start already in
  contact, (b) is invalid and I will say so rather than apply it.
- **D4 metrics**: report BOTH skill-vs-persistence AND **R^2 vs the mean** (methodological lesson
  #4 — skill-vs-persistence is only meaningful when persistence is strong).
- **D5 clip retention**: every filter (unlabeled 457 clips, clips shorter than history+horizon,
  low-force CoP masking) reports its drop count in the results table.
- **D6 layout**: cache mirrors ActionSense (`data/opentouch_states/{state_N.npy, clip_N.npy,
  manifest.jsonl}`) + adds `scene`, `participant`, `action`, `object_category`, `grip_type` fields
  to the manifest so the harness loads it via a config swap rather than a rewrite.
- **D7 OQ6**: I will attempt to copy `docs/predictability_opentouch.csv` back from CRC opportunistically;
  if it is gone, Step 1 regenerates it for free from the same extracted cache. Not a blocker.

### STATUS: PLAN COMPLETE, AWAITING EXPLICIT GO. Nothing downloaded, no code written yet.
Next action on GO = Step 0 ONLY (labels zip, ~few MB) -> action x scene table -> return to user
with shard selection before any 14 GB transfer.

### FOLLOW-UP (2026-08-06): "why only right hand?" — VERIFIED FROM THE PAPER, + A PLAN-BREAKING FINDING
User challenged the single-hand claim. I re-verified against arXiv:2512.16842 (HTML) rather than
re-quoting the log, since my claim rested on ONE live shard inspection from 2026-07-02.

**Answer — it is a HARDWARE FACT, not a design choice of ours.** Paper, verbatim:
> "We instrument only the right dominant hand to simplify hardware and standardize annotations."
There is no `left_pressure` array in OpenTouch. Confirmed independently by our own live schema dump
(SESSION_LOG:507-523: `data/<clip>` = right_pressure / camera_poses / rgb_images_jpeg /
hand_landmarks / timestamps / labels — no left field) and by the upstream loader, which reads
`right_pressure` only. So D2 (3-dim target) is forced by the data, not chosen.
Mitigating fact: on ActionSense the RIGHT hand was already the BETTER hand (+0.05 skill over left;
right-hand `CoP_x` is the single most predictable channel, PROJECT_CONCLUSIONS 5.4) because it is
the dominant/tool hand. So we lose the weaker half of the ActionSense target, not the stronger.

**NEW FACT 1 — 169 taxels, not 256.** Paper: "a 16x16 electrode grid around a commercial
piezoresistive film, forming **169 taxels** that uniformly cover the fingers and palmar surface",
palmar side only. So the 16x16 array carries a **structural dead-taxel mask** (~87 dead of 256),
exactly like EgoTouch's 217-of-441. Consequence: dead taxels read ~0 and contribute zero weight to
the F/CoP moments, so the moments stay valid WITHOUT a mask — but any map-model normalization and
any per-taxel baseline must exclude them, or the dead cells will dominate the statistics.
Added to the Step-1 inspection list.

**NEW FACT 2 — CLIPS ARE ~1.9 s, NOT ~6.2 s. THIS BREAKS THE OQ3 DECISION.**
Paper: "Each recording session lasted 5 to 25 minutes, yielding short clips **averaging 57 frames**"
(= 1.9 s @30 Hz), sampled by pressure dynamics (lowest-pressure pre-peak / peak / lowest-pressure
post-peak). My earlier ~6.2 s figure was a bad inference (5.1 h / 2,958 clips) that wrongly treated
total RECORDING time as clip time — **corrected here.** (Residual tension: the paper also says
"2,900 clips (3 hours)" = 3.7 s/clip. Either way the true value is 1.9-3.7 s, not 6.2 s.)
Consistency check: our probe computed `persH30` on these clips, so they are >= 30 frames. OK.

**Why this breaks OQ3 (10 Hz, min_history 20, horizon 10):** a 57-frame clip downsampled 3x is
**19 frames total**. Required = min_history 20 + horizon 10 = 30 frames. **Every clip is discarded.**
Even a 3.7 s clip gives 37 frames = 7 origins. The 10 Hz / 1 s protocol that made OpenTouch
directly comparable to ActionSense is **not viable on this corpus.** Options, none yet chosen:
  (a) **native 30 Hz**, history 0.5 s (15 fr) + horizon 1 s (30 fr) = 45 fr floor -> ~12 origins on a
      57-frame clip. Keeps the 1 s horizon (the physically meaningful quantity); loses step-count
      comparability with the ActionSense AR order grid (rescale 2..30 -> 6..90).
  (b) **shorten the horizon to 0.5 s** at 10 Hz -> 20+5=25 fr floor; still kills a 19-frame clip.
      Only works combined with (a). => 0.5 s horizon @30 Hz = 15+15 = 30 fr floor, comfortable.
  (c) accept the loss and keep only the longest clips — must first MEASURE the length histogram.
**This is now the first thing Step 0/1 must measure**, and it is a cheap measurement: clip lengths
are derivable from the labels + timestamps without pulling all 26 shards.
=> OQ3 IS RE-OPENED. The 10 Hz answer stands only if the measured length histogram supports it,
which the paper says it will not. I will bring the histogram back before committing.

**Net effect on cost (favourable):** at ~57 frames/clip the extracted pressure cache is
2,958 x 57 x 256 x 2 B ~= **86 MB**, not the ~300 MB I estimated. The 14.6 GB of shards is almost
entirely `rgb_images_jpeg` (~87 KB/frame x ~169 k frames), which we discard.

### FOLLOW-UP 2 (2026-08-06): "why is history capped at 0.5 s when ActionSense swept 1/2/3/5/10 s?"
User is right to challenge. Answer, with the numbers measured rather than asserted.

**The cap is NOT a consequence of the 30 Hz rate. It is a consequence of CLIP DURATION.**
History is a quantity of *real time*. Sampling rate changes the frame COUNT inside a clip, never
its DURATION. Going 10 Hz -> 30 Hz triples the samples in a 1.9 s clip; it does not create a 4th
second that is not there. Usable history is capped at `clip_duration - horizon`, full stop.

**MEASURED — ActionSense recording durations** (299 local `state_*.npy`, 30 Hz manifest rate):
`min 2.7 s | p25 11.4 s | median 22.0 s | p75 45.9 s | max 220.1 s | mean 35.7 s | total 178 min`
=> a 10 s history costs `(10+1)*30 = 330` frames, affordable against a 660-frame median recording.
Even so it dropped 24% of recordings (227/299 fit). THAT is why the 1/2/3/5/10 s sweep existed.

**OpenTouch clips are ~10-20x shorter** (paper: mean 57 frames = 1.9 s; the alternative reading
2,900 clips / 3 h = 112 frames = 3.7 s). Window budget @30 Hz, 1 s horizon, stride 1:

| history | frames needed | clip = 57 fr (1.9 s) | clip = 112 fr (3.7 s) |
|---|--:|---|---|
| 0.5 s | 45 | 13 origins -> 38,454 total | 68 origins -> 201,144 total |
| 1.0 s | 60 | **DEAD** | 53 origins -> 156,774 total |
| 2.0 s | 90 | **DEAD** | 23 origins -> 68,034 total |
| 3.0 s | 120 | **DEAD** | **DEAD** |
| 5 / 10 s | 180 / 330 | **DEAD** | **DEAD** |

So 3/5/10 s history is **physically impossible on OpenTouch at any sampling rate** — it would
require 4-11 s clips from a corpus whose own paper reports 1.9-3.7 s. 0.5 s is safe under BOTH
readings; 1 s and 2 s are viable only under the longer reading. **Hence: sweep 0.5 / 1 / 2 s,
with 1 and 2 contingent on the measured length histogram.** I chose 0.5 s earlier as the only
value guaranteed to survive the pessimistic reading, not as a rate-imposed ceiling.

**Why this costs us almost nothing scientifically:** the ActionSense ablation already settled
history length as a non-variable — after early stopping, skill is FLAT across the sweep
(0.513 @1 s -> 0.502 @10 s, PROJECT_CONCLUSIONS 5.4), and the earlier "more history hurts" claim
was retracted as an overfitting artifact. 0.5/1/2 s spans the region where any real effect would
live. The OpenTouch run therefore loses the *dead* end of a sweep already known to be flat.

**THE ONE FACT THAT WOULD OVERTURN THIS** (must be checked in Step 0/1, cheap): the paper says
recording *sessions* ran 5-25 min. If the HDF5 shards contain those long sessions rather than only
the curated short clips, the full 1/2/3/5/10 s sweep becomes available and this whole constraint
evaporates. Current evidence says they do NOT: our live inspection found 113 clips in
`office_csail_p2.hdf5` and 2,958 across 26 shards, matching the paper's ~2,900 *curated* clips
(i.e. shard clips == curated clips). Not yet proven — Step 1 will dump the true length histogram
and settle it. If the long sessions ARE present, I will re-open the history sweep in full.

**Revised OQ3 recommendation** (pending the histogram): native **30 Hz**, horizon **1 s (30 steps)**,
history sweep **{0.5, 1, 2} s**, AR order grid rescaled 2..30 -> ~6..90 frames. Report the retained
clip count at each history length as a first-class number (D5).

---

## SESSION (2026-08-07) — STEP 0 EXECUTED: label reconnaissance. TWO PLAN-CHANGING FINDINGS.

### Pre-download decisions (user, AskUserQuestion)
| Q | Decision |
|---|---|
| What to extract | **pressure + Rokoko pose + timestamps + calibration**; discard ONLY `rgb_images_jpeg`. Rationale: pose costs ~15 MB now and a 14.6 GB re-download later; it is also an input channel ActionSense never had. |
| Shard pass | **stream-extract, delete as it goes** (peak disk ~1 GB, not 14.6 GB) |
| Drive quota | **log failures, continue, retry pass** (extraction is per-shard + append-only -> resumable) |
Also measured: **this Mac has only 4.4 GB free** -> the shards physically cannot land here.
CRC-for-shards / local-for-modelling is now forced, not merely preferred.

### STEP 0 DONE — labels only (459 KB, `final_annotation.zip`). No shards downloaded.
25 CSVs, **2,958 label rows**, 16 columns:
`clip_id, object_name, object_category, environment, action, grip_type, description,
ts_start, ts_end, model, onset_idx, onset_ts, peak_idx, peak_ts, post_idx, post_ts`.
Annotations are LLM-generated (`model` = `gpt-5`) then human-reviewed.
**`ts_start`/`ts_end` (ns) let us settle clip length WITHOUT downloading a single shard.**

### FINDING A — clip lengths are FINE. My pessimistic read was wrong; OQ3 resolves favourably.
| p0 | p5 | p25 | **p50** | p75 | p90 | p99 | max | mean | total |
|---|---|---|---|---|---|---|---|---|---|
| 0.53 s | 1.53 s | 2.03 s | **2.80 s** | 4.16 s | 6.33 s | 15.99 s | 45.99 s | 3.65 s | **3.00 h** |
3.00 h over 2,958 clips reproduces the paper's "2,900 clips (3 hours)" exactly => the paper's
"averaging 57 frames" refers to its own sampled-frame benchmark, NOT to clip length. My 1.9 s
figure was wrong; **the correct median is 2.80 s (84 frames @30 Hz)**.
**History budget @30 Hz, 1 s horizon, stride 1** (clips surviving / origins):
`0.5 s -> 2829 (96%), 191k · 1 s -> 2251 (76%), 153k · 2 s -> 1298 (44%), 101k ·
3 s -> 788 (27%), 70k · 5 s -> 328 (11%), 38k · 10 s -> 90 (3%), 12k`
=> **history sweep {0.5, 1, 2} s is comfortable; 3 s is viable (70k origins).** 5/10 s are not.
CAVEAT to carry: restricting to long clips SELECTS on action (long clips skew to eating/serving/
scooping), so the history sweep must report its action mix or the trend confounds with content.

### FINDING B — THE ACTION DISTRIBUTION BREAKS THE PER-ACTION SWEEP, AND IMPEACHES OUR OWN PRIOR RESULT
66 distinct actions, brutally long-tailed. **Only 14 have n >= 30.** The head is:
`picking up 974 · placing 253 · pulling 247 · pressing 237 · pushing 154 · holding 119 ·
grasping 111 · adjusting 89 · turning 84 · touching 82 · moving 78 · removing 57 · sliding 55 ·
inspecting 32`.
**Every one of those is in the ABRUPT / make-break class** — the hard end of our trait axis.
The actions our probe crowned as most predictable are near-empty:

| action | probe PI (2026-07-02) | **n clips** |
|---|--:|--:|
| pouring | **+4.4** | **7** |
| serving | +3.6 | 10 |
| eating | +3.4 | 8 |
| stirring | +3.0 | **5** |
| scooping | +2.5 | 8 |
| flipping | +2.4 | 12 |
| wiping | +1.3 | 20 |
| cutting | -3.0 | 4 |

**=> CORRECTION TO OUR OWN DOCS.** `docs/PROJECT_CONCLUSIONS.md` 4 and
`docs/ACTION_CATEGORIES.md` 3b quote "pouring +4.4 / serving +3.6 / eating +3.4 / stirring +3.0 /
scooping +2.5" as the OpenTouch headline supporting the cross-dataset trait claim. We flagged
`cutting (n=4)` but NOT the top of the ranking, which rests on **5-10 clips per action**.
OpenTouch's contribution to the trait claim is therefore **far weaker than documented** — it is
suggestive, not confirmatory. The trait claim still stands on EgoTouch (1,929 clips) and
ActionSense (299 clips); OpenTouch should be demoted to corroboration pending the measured rerun.
This must be fixed in both docs. (Not yet edited — flagged to user first.)

**Smooth-class total = 108 clips**, scattered across 21 of 25 scenes (office_ml_p1 21,
eat_ygf_p2 14, home_kitchen_p1 11, home_kitchen_p3 9, ...) => **no shard subset can capture them**;
a targeted download is impossible. This settles OQ2: **download all 26 shards.**

### Consequence for OQ1 — the per-action sweep must be REFRAMED (needs user call)
- The **in-domain refit** (persistence / seasonal / AR / GRU on all clips) is **UNAFFECTED** —
  it does not depend on the action mix. That half of the plan stands as approved.
- The **per-action sweep** cannot test the probe's ordering as designed: with D1 (n>=30) it would
  cover 14 actions that are ALL in the hard class = a restricted-range test with little spread.
  Options: (a) sweep the 14 n>=30 actions and report it honestly as a hard-class-only test;
  (b) pooled TRAIT contrast, smooth (108) vs abrupt (2,850), which tests the actual durable claim;
  (c) both. Recommendation: **(c)**, with (b) as the headline since it matches the real hypothesis.

### Other reconnaissance facts
- 26 shards vs 25 CSVs: `grocery_target_p3_p4_merged_by_ts` merges two shards into one annotation
  file. **This is almost certainly the source of the 457 "unlabeled" clips in the 2026-07-02
  probe** (its join used `<shard_stem>::<group>`, which cannot match the merged prefix).
  `extract_opentouch.py:label_lookup` handles it with a prefix-containment fallback ->
  recovers ~15% of the corpus that the old probe silently dropped.
- Clip-id prefixes: 25 distinct, all exactly matching their CSV stem (verified, no other surprises).
- `environment`: store 1107 · office 674 · kitchen 293 · workshop 232 · supermarket 229 · home 162
  · restaurant 108 · bedroom 55 · ... (21 values) -> supports the scene-level held-out split (OQ4).

### BUILT (compile-checked + unit-tested; nothing run on real shards yet)
- `scripts/extract_opentouch.py` — one shard -> cache (`state_N.npy` (T,1,6) moments,
  `clip_N.npy` (T,1,16,16) fp16, `pose_N.npy`, append-only `manifest.jsonl` carrying scene/action/
  grip/object/environment/peak_idx/T/fps_est). Hand axis kept at extent 1 so harness indexing
  matches ActionSense's (T,2,6). `--taxel-stats` dumps per-taxel activity + p5 rest level.
  **NO baseline correction applied** — deliberately deferred (D3): the ActionSense DC bug must not
  be blind-fixed on a different sensor when clips are segmented around a pressure peak.
- `scripts/crc/stream_opentouch.sh` — labels -> per shard {gdown -> extract -> `rm`} -> summary;
  `done_ids.txt` / `failed_ids.txt` make it resumable; `--taxel-stats` on shard 1 only.
- **Unit tests on synthetic (all pass)**: single hot taxel -> CoP equals that cell's coordinate
  exactly; empty frame -> F=0, CoP=0, all finite (no NaN); sliding blob -> CoPx strictly monotonic
  with CoPy constant (the stroke signature); 169-live-of-256 dead-taxel grid -> finite moments.

### STATUS
Step 0 complete. Extraction code ready and tested. **The 14.6 GB shard pass has NOT started** —
it must run on CRC (this Mac has 4.4 GB free) and I have no SSH, so the user runs one command.
Awaiting the user's call on the FINDING-B reframe (OQ1) and on fixing the two docs.

### CHALLENGE (user): "download all actions, even though the predictable ones are a small amount?"
Answered with measurement, and the measurement INVERTS the rationale.

**The predictable subset is not a dataset.** Trait-class totals (hist 0.5 s + 1 s horizon @30 Hz):
| class | clips | duration | windows |
|---|--:|--:|--:|
| SMOOTH / predictable | 108 | **8.2 min** | 9,847 |
| ABRUPT / hard | 2,850 | 171.8 min | 181,286 |
| ALL | 2,958 | 180.0 min | 191,133 |
| *(ActionSense for scale)* | *299* | *177.9 min* | — |
=> 8.2 min is **22x less** than ActionSense. A forecaster cannot be TRAINED on it. So
"download only the predictable actions" is not a cheaper experiment, it is **no experiment**.
The targeted download is also a poor trade on its own terms: the best 7 shards = 3.9 GB (27% of
the bytes) return only 68% of the smooth clips, because they are spread over 21 of 25 scenes.

**The correct rationale for taking all 26 shards — OpenTouch is the HARD-POLE COMPLEMENT.**
- ActionSense: 178 min, overwhelmingly SMOOTH (slice/peel/pour/clean).
- OpenTouch: 172 min, overwhelmingly ABRUPT (picking up/pressing/pulling/grasping).
- Near-identical duration, opposite poles => together they span the trait axis with ~175 min at
  EACH end. This is a much stronger position than "more smooth data", which OpenTouch can never
  supply. It also retires the earlier framing that OpenTouch's value is its predictable actions.

**NEW FALSIFIABLE PREDICTION (the payoff):** the trait hypothesis predicts OpenTouch's overall
1 s forecast skill should land **clearly below ActionSense's +0.166**, because 96% of its clips
are abrupt/make-break. If a 96%-abrupt corpus forecasts as well as a smooth one, **the trait claim
— the project's most portable finding — is falsified.** ActionSense could not run this test (its
hard actions, jar / get-replace, were a minority). Recorded BEFORE the data is seen.

**Design consequence (resolves the FINDING-B reframe):** train on the FULL corpus, then EVALUATE
stratified by trait class. 108 clips is too few to train a per-class model but 9,847 windows is
ample to evaluate on. Holding sensor, pipeline and model constant makes this a CLEANER trait test
than any cross-dataset comparison. => Option (c) from FINDING B, with the trait contrast as
headline and the 14-action n>=30 sweep as the secondary, restricted-range view.

**RISK, named in advance:** if OpenTouch skill lands ~0 across the board, the AR-vs-GRU ranking
test is underpowered (models all near zero are hard to order). Mitigation: 191k windows give power
at small effect sizes; report CONFIDENCE INTERVALS, not bare point estimates. If the CIs overlap,
the honest conclusion is "the ranking does not replicate at measurable resolution", NOT a re-rank.

**DECISION CONFIRMED: download all 26 shards.**

### CORRECTION (2026-08-09, prompted by user): "train on full corpus" IS CONFOUNDED FOR THE TRAIT TEST
User asked why training on the full corpus is reasonable and what stratified evaluation buys.
The first question exposes a genuine flaw in the design I logged on 2026-08-07. Recording it.

**THE FLAW.** I wrote "train on the FULL corpus, then EVALUATE stratified by trait class". With a
corpus that is 96% abrupt, a model trained pooled and then scored on the 108 smooth clips gives a
number with TWO indistinguishable causes: (i) smooth actions are intrinsically harder, or (ii) the
model hardly saw smooth actions in training. That is a TRANSFER measurement mislabelled as a TRAIT
measurement. The trait claim cannot be tested that way.

**THE FIX — the two experiments need DIFFERENT training designs. They were conflated.**
- **E1 sensor-independence** (does AR > GRU > CNN-map > persistence replicate on a 2nd sensor?):
  question is about MODEL FAMILIES, not action classes. Fit on the FULL corpus; the action mix is
  simply what OpenTouch is. **Pooled training is CORRECT here.** Unchanged from the approved plan.
- **E2 trait test** (are smooth actions more forecastable?): pooled training is the WRONG tool.
  Use only estimators with no cross-class contamination:
  1. **Training-free**: persistence nMSE @1 s and R^2-vs-mean, per class. No fitting => no confound.
     (This is exactly why the Phase-C probe result was clean, and it is the honest primary metric.)
  2. **Linear AR fit PER CLASS** (`fit_scope`): each class gets coefficients from its own data.
     Smooth = 9,847 windows, ample for AR(90) x 3 channels; abrupt = 181,286. Unconfounded, and AR
     is our best model anyway (it won the four-way comparison).
  3. **Pooled GRU is reported for E1 ONLY**, flagged as confounded by class imbalance for E2.
- SCOPE NOTE: the ActionSense config uses `fit_scope: group` = action x object, which is too fine
  for OpenTouch's rare actions (stirring n=5). **E2 fits at TRAIT-CLASS scope (2 groups)**; the
  n>=30 per-action sweep fits at action scope where the counts support it.

**WHY STRATIFY AT ALL (answering the second question).** The pooled average is structurally
incapable of expressing the finding. If the truth were smooth +0.22 / abrupt +0.06, the aggregate
is `0.96*0.06 + 0.04*0.22 = 0.067` — indistinguishable from the abrupt number alone. A real 3.7x
class difference vanishes into one digit because the 96% majority swamps it. Every claim in this
project is about DIFFERENCES BETWEEN KINDS OF ACTION, so on the trait question the stratification
is not post-hoc analysis layered on the result — **it IS the result**.

**Net effect on the download decision: NONE.** Both E1 (needs maximum data) and E2 (needs both
poles present, per-class fitting) want the full corpus. All 26 shards still stands.

### 2026-08-10 — FIRST EXTRACTION RUN PRODUCED A CORRUPT CACHE. TWO BUGS, BOTH MINE. FIXED.
User's scene table (cache vs labels) exposed it: `have` is an exact multiple of `want` for most
scenes — grocery_plant 140/70, office_ml_p1 662/331, sports_dicks 460/230 (2x);
hardware_homedepot_p5 552/184 (3x) — while home_bedroom 138/138 and office_csail_p2 113/113 are
correct. **Summing `have` = 4,511 manifest rows for a corpus of 2,958 clips.** More clips than
exist => definitive over-extraction. Meanwhile only 2,666 `state_*.npy` files exist on disk.

**BUG 1 — CONCURRENT RUNS (no lock).** The user legitimately started the script more than once
while working out screen/tmux/nohup. Consequences, all three of which the script permitted:
  (a) each instance read an empty/stale `done_ids.txt` -> re-extracted the same shards;
  (b) `next_index()` read the same manifest in both -> both wrote the SAME `state_N.npy`
      filenames -> **files overwrote each other, so the manifest row -> file mapping is broken**
      (4,511 rows vs 2,666 files). This is why no dedupe can repair the cache;
  (c) `rm -f "$SHARDS"/*.hdf5` at the top of each loop deleted the OTHER instance's in-flight
      download -> **the 19 "failures" were self-inflicted, not Drive quota.**
**BUG 2 — LABEL JOIN COLLISION (`p1`/`p2`).** The log shows shards `sports_dicks_p1` AND
`sports_dicks_p2`, but there is only ONE `sports_dicks` CSV. My prefix-containment fallback let
BOTH `p1::demo_000` and `p2::demo_000` claim the same annotation row, so one of them carried a
WRONG action/grip label. Independent of the concurrency bug, and worse: it is silent. My
2026-08-07 note claimed this fallback "recovers ~15% of the corpus" — it was also corrupting it.

**FIXES (committed; compile-checked + unit-tested).**
1. `stream_opentouch.sh`: atomic `mkdir` lock (`$WORK/.stream.lock`, NFS-safe) with EXIT/INT/TERM
   trap and an explicit stale-lock message; **per-instance shard dir `shards.$$`** so no run can
   delete another's download; `sort -u` on `done_ids.txt` to collapse earlier duplicates.
2. `extract_opentouch.py`: **join by TEMPORAL OVERLAP, not by name.** The CSV carries
   `ts_start`/`ts_end` (ns) and every clip carries `timestamps`, so the join is exact and
   name-independent — evidently how the authors merged p3/p4 (`..._merged_by_ts`). Rules: exact
   key accepted only if it genuinely overlaps (>=50% of the clip span); otherwise best-overlap row;
   **a label row may be claimed by at most one clip**; no overlap => reported MISS, never a
   silent wrong label. Manifest now records `label_cid` + `join` method + overlap fraction.
3. `extract_opentouch.py`: **idempotent** — `read_manifest()` returns the set of already-extracted
   `<shard>::<group>` keys and claimed label cids; re-running after an interrupt skips instead of
   duplicating.
**Unit tests (pass):** p1/p2 same group name at different times -> each gets its OWN correct row
(`pulling` / `pouring`); a claimed row is never handed out twice; a non-overlapping clip returns
MISS rather than a wrong label. Plus the earlier moment tests still hold.

**CONSEQUENCE: the cache must be REBUILT from scratch (~14.6 GB re-download).** Unavoidable —
(b) means row->file identity is lost, and (2) means an unknown subset of labels is wrong. Deleting
`~/opentouch/cache` and re-running is the only sound path. Cost is time, not data.
**Confirmed good so far:** `fps est` median **30.01 Hz** on every shard (min 30.00, max 30.01) =>
the native-30 Hz / 1 s-horizon plan stands. The `taxels:`/DC-offset line was truncated in the
terminal and is still OUTSTANDING.
**METHOD NOTE:** the scene-vs-label table is what caught this; a bare clip count (2,666 of 2,958)
looked merely incomplete. Any future cache build must be verified per-scene against the labels,
not by total count. Added as a standing check.
