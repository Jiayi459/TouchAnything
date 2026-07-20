# Training Plan — Generative Tactile Forecasting for Smooth-Force Actions

**Status: DESIGN / BRAINSTORM (plan-before-code). Nothing implemented yet — open questions at end.**

## 0. Objective & constraints
Train a model that **forecasts future tactile pressure fields** for the *predictable* actions we
identified — **slice, wipe/clean, pour, peel** (smooth, continuous, slowly-varying force) — all in
**ActionSense** (32×32 conductive-thread glove, 2 hands, native ~6 Hz). The forecaster's latent
must be **physically interpretable** so it can drive **user feedback** ("your pour force is
uneven / your slicing rhythm is irregular").

Hard constraints that shape every choice below:
- **Small data.** ActionSense usable clips: ~15–30 per activity, ~300 total (S00–S05). This is the
  dominant constraint → favors low-dimensional latents, physics priors, heavy augmentation,
  self-supervised pretraining, and *not* a data-hungry pure video-diffusion model from scratch.
- **Low, uneven sampling** (~6 Hz, timestamp-driven). Forecast horizons should be defined in
  **seconds**, not frames; resample to a fixed rate per model.
- **Sparse, heavy-tailed signal.** Most taxels ≈ 0; contact is a small moving patch. Losses and
  latents must respect this (don't let background zeros dominate).
- **Target = the glove hardware**, so the representation should transfer to OpenTouch/Force-Vision
  gloves later (a common resized grid + amplitude normalization).

## 1. Core design principle — a *physics-structured latent world model*
Rather than predict pixels directly, **encode the tactile field into a low-dimensional latent,
forecast the latent forward in time, and decode**. Why this fits the problem:
- The full 32×32×2×T tensor of a wipe/pour is *generated* by a handful of physical quantities
  (where contact is, how hard, and the phase of the motion). Modeling those directly is far more
  data-efficient than modeling every taxel.
- A smooth action ⇒ smooth latent trajectory ⇒ easy, stable forecasting (matches our finding that
  ramps/cycles are the predictable regime).
- An **interpretable** latent is exactly what feedback needs: deviations in named physical
  variables are actionable; deviations in an opaque 128-d vector are not.

## 2. Generative framework — options & recommendation
| option | fit | verdict |
|---|---|---|
| Deterministic frame regressor (SimVP/ConvLSTM) | simple, but blurs multimodal futures; no latent, no uncertainty | baseline only |
| **Variational latent dynamics / RSSM (Dreamer/PlaNet-style)** | latent forecasting + uncertainty + place for physics latents | **recommended core** |
| Sequence β-VAE + latent temporal model (ConvLSTM/GRU/transformer) | simpler RSSM; strong disentanglement pressure | **recommended first build** |
| Latent diffusion (diffuse the future latent) | best for uncertainty/sharpness | **phase 2**, once latent is stable |
| Pixel/video diffusion from scratch | data-hungry, uninterpretable | rejected (too little data) |

**Recommendation:** build a **conditional variational latent-dynamics model** and grow it:
1. **Phase 1** — β-VAE encoder → **physics-structured latent** `z` → **ConvLSTM/GRU latent
   predictor** → decoder. Deterministic latent rollout. (Uses the ConvLSTM we already have, now in
   *latent* space rather than pixel space.)
2. **Phase 2** — make the latent transition **stochastic** (RSSM: predict `p(z_{t+1}|z_t, a)`) for
   calibrated uncertainty, and/or a **latent diffusion** prior for multi-step sampling.
Condition on **action class** (slice/wipe/pour/peel) via an embedding so one model shares data
across actions (data efficiency) while specializing.

## 3. Network architecture (Phase 1)
- **Encoder** `E`: small CNN on the (2,32,32) frame (2 hands as channels) → per-frame features.
  32×32 is tiny, so 3–4 conv blocks + global pooling → latent params (μ, logσ). Optionally a short
  temporal conv / GRU over a few input frames so `z` captures velocity, not just a static frame.
- **Latent** `z ∈ R^d`, d ≈ 16–32, **partitioned** (see §4): interpretable physics channels +
  a small residual.
- **Temporal predictor** `T`: given `z_{t-k..t}` (+ action embedding), predict `z_{t+1..t+H}`.
  Start with **ConvLSTM if we keep a small spatial latent (e.g. 8×8×C)**, or GRU/Transformer if
  `z` is a vector. (Keeping a coarse spatial latent preserves the contact map's geometry; a pure
  vector is more compressive/interpretable. We'll likely keep BOTH: a physics vector + a small
  spatial residual map — see open Q.)
- **Decoder** `D`: latent → (2,32,32) pressure field (transpose-conv), trained to reconstruct and
  to render forecasts. Output through a softplus (non-negative pressure).

## 4. Physical latent variables — *what* and *why* (the creative core)
The tactile field of these actions is well-approximated by a few interpretable generative factors.
We **structure the latent to contain named physical channels**, computed directly from the data as
weak supervision (semi-supervised disentanglement), plus a residual for whatever is left:

| latent | definition (computed from the (2,32,32) field) | why it matters |
|---|---|---|
| **F — total normal force** | Σ pressure over valid taxels (per hand) | primary drive; slice/pour modulate F smoothly; feedback: "force too variable" |
| **(x̄, ȳ) — center of pressure (CoP)** | pressure-weighted centroid | the *motion*: wiping/slicing translate the CoP periodically; pour keeps it stable |
| **A — contact area / spread** | # active taxels, or 2nd moment (patch size) | grip vs press vs smear; changes with tool engagement |
| **(θ, ecc) — patch orientation & elongation** | eigen-decomp of the 2nd-moment matrix | knife-edge vs palm; stroke direction |
| **(sinφ, cosφ) — motion phase** | phase of the dominant periodic component of F(t)/CoP(t) | *the* variable for slice/wipe: forecasting = advancing φ; feedback: rhythm regularity |
| **dF/dt — force rate** | temporal derivative of F | ramps (pour) ≈ constant dF/dt; distinguishes ramp vs cycle |
| **residual z_r (≈8–16 d)** | learned, whatever the physics channels miss | fine spatial texture, sensor quirks |

Why these specifically: they are the **low-order moments of the pressure field + the temporal
phase** — a compact, near-complete generative basis for smooth contact. A wipe is essentially
"CoP orbits while F pulses at phase φ"; a pour is "F ramps while CoP holds." If the latent captures
these, (a) forecasting is a smooth, low-dimensional extrapolation, (b) every latent is a physical
quantity a coach could talk about. This is what turns prediction into *actionable feedback*.

(Kinematic option: ActionSense also has Xsens body + finger data — wrist velocity/pose is the
*cause* of the tactile. A later version can condition on / predict pose↔tactile jointly.)

## 5. Embedding tactile → low-dimensional latent
- **Weakly-supervised structuring:** compute F, CoP, A, orientation, φ, dF/dt analytically from
  each frame/window and train `E` so the corresponding latent channels *regress* to them (an
  auxiliary loss), while the VAE reconstruction forces the residual to capture the rest. This gives
  a disentangled, physical latent without hand-labeling.
- **β-VAE / information bottleneck** on the residual to keep it minimal and prevent it from
  absorbing the physical factors.
- **Self-supervised pretraining** of `E`/`D` on the *entire continuous ActionSense stream*
  (unlabeled, far more data than the 300 labeled clips) via masked-frame / next-frame
  autoencoding — then fine-tune the temporal predictor on the labeled smooth-action clips.

## 6. Loss function — tailored to this data
Composite, with the sparse/heavy-tailed/smooth nature baked in:
- **Masked, amplitude-transformed reconstruction:** MSE in `log1p(α·p)` space over *valid* taxels,
  with active taxels up-weighted (`active_weight`) so the small contact patch isn't drowned by
  background zeros. (We already do this in the pixel forecaster.)
- **Total-force term:** `|ΣF_pred − ΣF_true|` per hand — physically the most feedback-relevant
  scalar; keeps the global drive honest even when the map is slightly misplaced.
- **Contact-support term:** BCE / soft-IoU on the binarized active mask — *where* contact is.
- **Physical-latent supervision:** MSE between predicted latent physics channels and their
  analytic values (F, CoP, A, φ …) at forecast times — anchors the interpretable latent.
- **Temporal-smoothness prior:** penalize |Δ²| (jerk) of the predicted latent/force — encodes that
  these actions are smooth; discourages the model from hallucinating abrupt changes.
- **Spectral / phase term (periodic subset — slice/wipe):** match the power spectrum (or φ
  advance-rate) of predicted vs true total-force — rewards getting the *rhythm* right, not just
  the instantaneous value.
- **VAE terms:** weighted KL on the residual latent (β), + a small free-bits floor.
- **(Phase 2)** replace deterministic recon with a likelihood / diffusion loss for calibrated
  multi-step samples.

`L = w_rec·rec + w_F·force + w_c·support + w_phys·phys + w_sm·smooth + w_spec·spectral + β·KL`.
Weights tuned on a val split; ablate each term.

## 7. Data & training strategy (small-data)
- **Per-action, shared model:** one model conditioned on action-embedding, trained on all four
  smooth actions jointly (≈ pour 25 + slice 60 + peel 30 + clean/wipe 60 clips) → more data,
  action-specialization via conditioning. Optionally leave-one-subject-out for generalization.
- **Augmentation:** grid flips/rotations (respect L/R hand symmetry), amplitude jitter/scaling,
  temporal crops & speed-warp (crucial: teaches phase/rate invariance), small CoP translations.
- **Pretrain** encoder/decoder self-supervised on the full continuous stream; **fine-tune**
  temporal predictor on labeled clips.
- **Cross-dataset transfer (later):** resize OpenTouch 16×16 / Force-Vision grids to a common size
  + normalize → pretrain the encoder across all three gloves for a more robust tactile prior.

## 8. Evaluation & feedback
- **Forecast skill vs persistence** (per action, per horizon) — the metric we've used throughout.
- **Physical-variable error:** CoP-trajectory error, force-MAE, phase-rate error — interpretable.
- **Uncertainty calibration** (Phase 2): does predicted variance match error?
- **Feedback demo:** given a user's clip, encode → compare their physical-latent trajectory to the
  learned "expert" distribution for that action → surface the largest deviation as advice.

## 9. OPEN QUESTIONS / decisions before coding
- **Q1 Latent form:** pure interpretable **vector** `z` (max interpretability, may lose fine spatial
  detail) vs **vector physics + small spatial residual map** (keeps geometry, more params). Recommend
  the hybrid.
- **Q2 Framework depth for v1:** start with the **deterministic** latent β-VAE + ConvLSTM (simpler,
  fits data) and add stochastic RSSM / latent-diffusion in Phase 2? (Recommend yes.)
- **Q3 One shared model conditioned on action, or one model per action?** (Recommend shared — data.)
- **Q4 Horizon & rate:** what resample rate (e.g. 10–15 Hz) and forecast horizon (0.5–2 s) matter
  for the feedback use-case?
- **Q5 Use kinematics?** Condition on / co-predict Xsens wrist pose (the physical cause) now, or
  tactile-only first? (Recommend tactile-only v1, add pose later.)
- **Q6 Compute:** all on CRC GPU (torch env ready). Confirm data staging (stream ActionSense clips
  → cache the segmented (T,2,32,32) tensors as small npz so we don't re-download 30 GB each run).
- **Q7 Which actions in v1?** all four smooth actions, or start with pour+slice (the two clearest)?

## 10. v1 — DECIDED (2026-07-03)
User decisions: pour + slice; **explicit physical state** (Path A, not learned latent, not
ConvLSTM); dynamics = **GRU baseline then compare to a structured ramp/oscillator model**;
tactile-only.

**State `s(t)`** (analytic, per hand) = raw pressure moments [F, x̄, ȳ, sxx, syy, sxy]
(`src/actionsense/physical_state.py`, validated on synthetic pour/slice), with derived
series area, orientation θ, eccentricity, CoP velocity (ẋ,ẏ), dF/dt, and phase φ (Hilbert of
F(t)) computed at train time. Coordinates normalized to [-1,1] (sensor-size-agnostic).

**Pipeline:**
1. EXTRACT (done, code): `actionsense_predictability.py --extract-states DIR` saves per-clip
   `state_N.npy` (T,C,6) + `manifest.jsonl`; wired into `stream_actionsense.sh` so ONE re-stream
   yields the tiny state dataset (few MB) — rsync/commit it, never re-download 30 GB again.
2. FORECAST (next): dataset of (past s → future s) windows; **GRU** predictor (skill vs
   persistence on each physical variable, per horizon in seconds); then a **structured** model
   — pour = linear ramp in F (Kalman), slice = damped oscillator in (F, CoP phase) — compare.
   Small enough to train on CPU locally.
3. FEEDBACK: compare a clip's s-trajectory to the expert distribution per action → largest
   deviation = advice ("uneven pour force", "irregular slice rhythm").

Metrics: per-variable forecast skill vs persistence; phase-rate error (slice); force-ramp error
(pour). Renderer (optional, later): analytic blob from s for visualization.
