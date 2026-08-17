"""probGRU for OpenTouch — the ActionSense probabilistic GRU, architecture and loss verbatim.

WHY THIS EXISTS ALONGSIDE gru_aggregate.py. OQ-G (2026-08-11) made GRU-aggregate a
deterministic point forecaster and deleted the Gaussian head. The user overturned that on
2026-08-13: train OpenTouch with the SAME probGRU as ActionSense -- same architecture, same
loss. gru_aggregate.py is left untouched (it is the pre-registered deterministic arm); this
is a second, separate arm.

WHAT IS COPIED VERBATIM FROM src/actionsense/action_dynamics.py
  * ProbGRU: action embedding (8) -> encoder GRU -> AUTOREGRESSIVE decoder GRU seeded with the
    last observed target, mean and log-variance heads over [decoder state ; action embedding],
    logvar clamped to [-6, 4], the predicted mean fed back as the next decoder input.
  * The loss: 0.5 * (lv + (y - mu)^2 * exp(-lv)), i.e. Gaussian NLL up to a constant.
  * Early stopping on VAL NLL (the original notes the loss overfits badly after ~epoch 10).
  * Its hyperparameters: hidden 48, epochs 80, lr 3e-3, batch 64 (NOT gru_aggregate's 64/60 --
    those came from the tactile_map aggregate branch, a different model).

WHAT DIFFERS, AND WHY (each one is forced, not a preference)
  1. TARGET = RAW [F, CoPx, CoPy] (user decision 2026-08-13), not ActionSense's FAST. The
     frozen harness scores RAW, so this arm can be compared with persistence/seasonal/AR on
     one scale; a FAST-target model is not harness-scorable (SESSION_LOG 2026-08-06 note).
     Consequence to keep in mind when reading any number from it: F currently carries a large
     DC offset (D1 open), so much of the target is a constant.
  2. INPUT = all-frequency history [F, CoPx, CoPy, vx, vy] -- ActionSense's `raw` input mode,
     with the same CAUSAL backward-difference velocities. No low-pass anywhere in this file.
  3. NO WARMUP CUT. ActionSense drops the first 5 s of every clip solely because its causal
     filter has a startup transient. There is no filter here, and OpenTouch's median clip is
     2.80 s (84 frames), so a 5 s cut would empty the corpus rather than clean it.
  4. WINDOWS COME FROM THE HARNESS, not from ActionSense's stride-2 sampler: origins() defines
     both training and scoring windows, so what the model is trained on and what
     evaluate.score_external() reads are the same rolling origins. Stride is a sampling
     detail, not part of the architecture or the loss.
  5. TARGETS ARE NORMALIZED BY THE HARNESS'S TRAIN-FITTED Norm (the one the baselines use), so
     the GRU and the classical baselines share one normalization. Input features get their own
     TRAIN-fitted z-score because they include velocities, which the harness Norm does not
     cover -- this mirrors ActionSense's separate nx/ny normalizers.

ACTION VOCABULARY. The embedding needs an id per clip. OpenTouch's action field is long-tailed
(~50 values), so ids are built FROM TRAIN ONLY: any action with fewer than
`baselines.min_group_size` TRAIN clips collapses into "other" (id 0), which is also where an
action unseen in TRAIN lands at test time. The threshold is the same one the AR baselines use
to merge rare object categories, so the two arms treat rarity the same way.
"""
from __future__ import annotations

import collections
import time

import numpy as np
import torch
import torch.nn as nn

from src.actionsense.eval_harness.config import Config
from .baselines import origins
from .dataset import Norm, eligible_clips, load_target
from .gru_aggregate import configure_determinism

DEFAULT_HP = {"hidden": 48, "epochs": 80, "lr": 0.003, "batch": 64, "seed": 0,
              # Train NLL is a curve for the log, not a selection signal -- early stopping
              # reads VAL only -- so it does not need a full extra pass over TRAIN every
              # epoch. Evaluating it every 5th epoch keeps the curve readable and drops
              # roughly a fifth of the wall clock (an epoch is one fwd+bwd pass over TRAIN
              # plus one fwd over TRAIN plus one fwd over VAL; this removes 4/5 of the
              # middle term). Set to 1 to restore the old behaviour.
              "log_train_every": 5,
              # "raw" = ActionSense's five inputs verbatim; "raw+df" adds dF/dt as an
              # ablation. Recorded in the checkpoint, so a forecast can never be replayed
              # under the wrong feature set.
              "features": "raw",
              # Both OFF by default, so the arm stays input- and architecture-identical to
              # ActionSense unless a run says otherwise. They exist because the 2026-08-17
              # curves overfit from epoch 2 -- five times sooner than ActionSense's own note
              # -- and the first thing to try is the one with a mechanism behind it
              # (features="raw+df" removes the per-location DC level the model can memorise),
              # not a pile of regularisers applied at once.
              "weight_decay": 0.0, "dropout": 0.0}
OTHER = 0          # reserved embedding id: rare-in-TRAIN or unseen-at-TEST actions


def pick_device(spec: str | None = None) -> torch.device:
    """'cuda' when a GPU is actually present, else CPU. Note the determinism caveat that
    gru_aggregate.configure_determinism documents: cuDNN's RNN kernels are not guaranteed
    deterministic even under deterministic algorithms, so a CUDA run cannot claim bitwise
    reproducibility the way a CPU run can -- it must record that it ran on GPU."""
    if spec:
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------- data --
def causal_velocity(sig: np.ndarray, fps: float) -> np.ndarray:
    """v[t] = (sig[t] - sig[t-1]) * fps, v[0] = 0 -- ActionSense's _causal_diff. Backward,
    so no feature at time t sees a sample after t."""
    v = np.zeros_like(sig)
    v[1:] = np.diff(sig, axis=0) * fps
    return v


def features(Y: np.ndarray, fps: float, with_df: bool = False) -> np.ndarray:
    """(T,3) raw [F,CoPx,CoPy] -> (T,5) [F, CoPx, CoPy, vx, vy], ActionSense's `raw` mode.

    WITH_DF APPENDS dF/dt, AS AN ABLATION. ActionSense differences CoP and not force --
    FEATS_RAW is ("F","x","y","vx","vy") -- and no reason for the asymmetry is recorded;
    in its highpass mode force is already split into F_slow and F_fast, so the raw mode
    reads like an omission rather than a decision.

    It matters more here than it did there. With D1 declined the force channel is ~99.3%
    constant, and a difference is the one view of F that carries no DC at all: it removes
    the per-clip and per-session resting level that the model would otherwise have to
    cancel inside its hidden state first. The decoder is already seeded with the last
    observed target, so the LEVEL is supplied there; handing the encoder the RATE separates
    the two cleanly. Appended last so the first three columns stay the raw channels.
    """
    Y = np.asarray(Y, dtype=np.float64)
    v = causal_velocity(Y[:, 1:3], fps)
    cols = [Y, v]
    if with_df:
        cols.append(causal_velocity(Y[:, 0:1], fps))
    return np.concatenate(cols, axis=1)


class FeatNorm:
    """TRAIN-only z-score for the input features (the harness Norm covers 3 channels).

    Carries `with_df` so the feature layout travels with its normalizer: window_set and
    predict read it off this object instead of taking a parallel flag that could drift out
    of step with the statistics it was fitted against."""

    def __init__(self, mean, std, with_df: bool = False):
        self.mean, self.std, self.with_df = mean, std, with_df

    @staticmethod
    def from_train(cfg: Config, ids: list[int], with_df: bool = False) -> "FeatNorm":
        allx = np.concatenate(
            [features(load_target(cfg, i), cfg.fps, with_df) for i in ids], 0)
        sd = allx.std(0)
        sd[sd < 1e-8] = 1.0
        return FeatNorm(allx.mean(0), sd, with_df)

    def z(self, x):
        return (x - self.mean) / self.std


def action_vocab(cfg: Config, train_ids: list[int]) -> tuple[dict[str, int], dict[int, str]]:
    """-> (action -> embedding id, clip idx -> action). Built from TRAIN only: letting VAL/TEST
    actions define the vocabulary would leak the split into the model's input space."""
    by_idx = {r["idx"]: (r.get("action") or "").strip().lower()
              for r in eligible_clips(cfg, actions=())}
    counts = collections.Counter(by_idx.get(i, "") for i in train_ids)
    min_n = cfg.raw["baselines"].get("min_group_size", 1)
    vocab = {"other": OTHER}
    for a, n in sorted(counts.items()):
        if a and n >= min_n:
            vocab[a] = len(vocab)
    return vocab, by_idx


def _aid(vocab, by_idx, i: int) -> int:
    return vocab.get(by_idx.get(i, ""), OTHER)


def window_set(cfg: Config, ids: list[int], t_in: int, norm: Norm, fnorm: FeatNorm,
               vocab: dict[str, int], by_idx: dict[int, str]):
    """Harness-aligned windows -> (X (N,t_in,5), A (N,), Ylast (N,3), Y (N,H,3)) normalized.

    One item per rolling origin, exactly the origins score_external() will ask about. Early
    origins are LEFT-zero-padded rather than dropped, matching gru_aggregate."""
    H = cfg.horizon
    Xs, As, YL, Ys = [], [], [], []
    for i in ids:
        Y = load_target(cfg, i)
        f = fnorm.z(features(Y, cfg.fps, fnorm.with_df)).astype(np.float32)
        z = norm.z(np.asarray(Y, dtype=np.float64)).astype(np.float32)
        a = _aid(vocab, by_idx, i)
        for t in origins(len(z), cfg):
            w = f[max(t - t_in + 1, 0): t + 1]
            if w.shape[0] < t_in:
                w = np.concatenate([np.zeros((t_in - w.shape[0], f.shape[1]), np.float32), w], 0)
            Xs.append(w); As.append(a); YL.append(z[t]); Ys.append(z[t + 1: t + 1 + H])
    if not Xs:
        return (torch.zeros(0, t_in, 5), torch.zeros(0, dtype=torch.long),
                torch.zeros(0, 3), torch.zeros(0, H, 3))
    return (torch.from_numpy(np.stack(Xs)), torch.tensor(As, dtype=torch.long),
            torch.from_numpy(np.stack(YL)), torch.from_numpy(np.stack(Ys)))


# ------------------------------------------------------------------------------ model --
class ProbGRU(nn.Module):
    """Verbatim from src/actionsense/action_dynamics.py (only `n_out` is read from the data
    instead of hardcoded 3 -- the same channel-count fix the other OpenTouch forks made)."""

    def __init__(self, din: int, n_act: int, hid: int, n_out: int = 3, dropout: float = 0.0):
        super().__init__()
        self.emb = nn.Embedding(n_act, 8)
        self.enc = nn.GRU(din, hid, batch_first=True)
        self.dec = nn.GRU(n_out, hid, batch_first=True)
        self.mu = nn.Linear(hid + 8, n_out)
        self.lv = nn.Linear(hid + 8, n_out)
        # Identity at p=0, so the verbatim architecture is what runs unless asked otherwise.
        self.drop = nn.Dropout(dropout)

    def forward(self, x, aid, y_last, t_out):
        _, h = self.enc(x)
        e = self.emb(aid)
        inp = y_last.unsqueeze(1)
        mus, lvs = [], []
        for _ in range(t_out):
            o, h = self.dec(inp, h)
            oc = self.drop(torch.cat([o[:, -1], e], -1))
            mu = self.mu(oc); lv = self.lv(oc).clamp(-6, 4)
            mus.append(mu); lvs.append(lv)
            inp = mu.unsqueeze(1)                       # autoregressive: feed the mean back
        return torch.stack(mus, 1), torch.stack(lvs, 1)


def nll(mu: torch.Tensor, lv: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Gaussian NLL up to a constant -- ActionSense's loss, unchanged."""
    return 0.5 * (lv + (y - mu) ** 2 * torch.exp(-lv)).mean()


@torch.no_grad()
def _val_nll(m, X, A, YL, Y, H, batch, dev) -> float:
    m.eval()
    tot = n = 0.0
    for i in range(0, len(X), batch):
        mu, lv = m(X[i:i + batch].to(dev), A[i:i + batch].to(dev),
                   YL[i:i + batch].to(dev), H)
        tot += float(nll(mu, lv, Y[i:i + batch].to(dev))) * len(mu); n += len(mu)
    return tot / max(n, 1.0)


# ---------------------------------------------------------------------------- training --
def train(cfg: Config, train_ids: list[int], val_ids: list[int], t_in: int,
          hp: dict | None = None, norm: Norm | None = None, verbose: bool = True,
          device: str | None = None):
    """Fit on TRAIN, keep the lowest-VAL-NLL weights. TEST is never touched.

    `verbose` prints the window count up front and one line per epoch, flushed. The
    autoregressive decoder runs `horizon` sequential GRU steps per batch, so a CPU run is
    slow enough that a silent loop is indistinguishable from a hung one.

    -> (model, norm, fnorm, vocab, by_idx, history)"""
    hp = {**DEFAULT_HP, **(hp or {})}
    gen = configure_determinism(int(hp["seed"]))
    if norm is None:
        norm = Norm.from_train({i: load_target(cfg, i) for i in train_ids})
    fnorm = FeatNorm.from_train(cfg, train_ids, "df" in str(hp.get("features", "raw")))
    vocab, by_idx = action_vocab(cfg, train_ids)

    Xtr, Atr, Ltr, Ytr = window_set(cfg, train_ids, t_in, norm, fnorm, vocab, by_idx)
    Xva, Ava, Lva, Yva = window_set(cfg, val_ids, t_in, norm, fnorm, vocab, by_idx)
    if len(Xtr) == 0:
        raise ValueError("no TRAIN origins: clips too short for this history/horizon")

    H, bs = cfg.horizon, int(hp["batch"])
    n_ep = int(hp["epochs"])
    dev = pick_device(device)
    if verbose:
        print(f"    [t_in={t_in}] windows: train {len(Xtr)} val {len(Xva)} | "
              f"batches/epoch {-(-len(Xtr) // bs)} | vocab {len(vocab)} | device {dev}",
              flush=True)
    m = ProbGRU(Xtr.shape[-1], len(vocab), int(hp["hidden"]), n_out=len(cfg.channels),
                dropout=float(hp["dropout"])).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=hp["lr"],
                           weight_decay=float(hp["weight_decay"]))
    best, best_state, history = np.inf, None, {"train_nll": [], "val_nll": []}
    t0 = time.time()
    for ep in range(n_ep):
        m.train()
        perm = torch.randperm(len(Xtr), generator=gen)
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]
            mu, lv = m(Xtr[b].to(dev), Atr[b].to(dev), Ltr[b].to(dev), H)
            loss = nll(mu, lv, Ytr[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        every = max(1, int(hp["log_train_every"]))
        want_tr = (ep % every == 0) or (ep == n_ep - 1)
        tr = _val_nll(m, Xtr, Atr, Ltr, Ytr, H, bs, dev) if want_tr else float("nan")
        va = (_val_nll(m, Xva, Ava, Lva, Yva, H, bs, dev) if len(Xva)
              else (tr if want_tr else float("nan")))
        history["train_nll"].append(tr); history["val_nll"].append(va)
        improved = va < best
        if verbose:
            el = time.time() - t0
            trs = f"{tr:.5f}" if np.isfinite(tr) else "  --   "
            print(f"    [t_in={t_in}] epoch {ep + 1}/{n_ep} train {trs} val {va:.5f}"
                  f"{'  *best' if improved else ''} | {el:.0f}s elapsed, "
                  f"~{el / (ep + 1) * (n_ep - ep - 1):.0f}s left", flush=True)
        if improved:
            best, best_state = va, {k: v.detach().clone() for k, v in m.state_dict().items()}
    if best_state is not None:
        m.load_state_dict(best_state)
    m.eval()
    history["best_val_nll"] = float(best)
    history["selected_on"] = "val" if len(Xva) else "train"
    history["n_actions"] = len(vocab)
    history["device"] = str(dev)
    history["features"] = str(hp.get("features", "raw"))
    history["n_features"] = int(Xtr.shape[-1])
    history["weight_decay"] = float(hp["weight_decay"])
    history["dropout"] = float(hp["dropout"])
    return m, norm, fnorm, vocab, by_idx, history


def select_history(cfg: Config, train_ids: list[int], val_ids: list[int],
                   histories_s=(1.0, 2.0, 3.0), hp: dict | None = None,
                   device: str | None = None):
    """Input history chosen on VAL only, by NLL. -> (best_t_in, {t_in: best_val_nll})."""
    scores = {}
    for s in histories_s:
        t_in = max(1, int(round(s * cfg.fps)))
        print(f"  sweep: history {s} s -> t_in={t_in} frames", flush=True)
        *_, hist = train(cfg, train_ids, val_ids, t_in, hp, device=device)
        scores[t_in] = hist["best_val_nll"]
    return min(scores, key=scores.get), scores


# -------------------------------------------------------------------------- prediction --
@torch.no_grad()
def predict_clip(model, cfg: Config, norm: Norm, fnorm: FeatNorm, vocab, by_idx,
                 i: int, t_in: int) -> np.ndarray:
    """One clip's RAW-unit mean forecasts (n_origins, H, C), ordered like origins() -- the
    format evaluate.score_external() requires. The variance head is trained (it is in the
    loss) but not returned here: the frozen harness scores point error only."""
    model.eval()
    X, A, L, _ = window_set(cfg, [i], t_in, norm, fnorm, vocab, by_idx)
    if len(X) == 0:
        return np.zeros((0, cfg.horizon, len(cfg.channels)), dtype=np.float64)
    dev = next(model.parameters()).device
    mu, _ = model(X.to(dev), A.to(dev), L.to(dev), cfg.horizon)
    return norm.unz(mu.cpu().numpy().astype(np.float64))


def predict(model, cfg: Config, norm: Norm, fnorm: FeatNorm, vocab, by_idx,
            test_ids: list[int], t_in: int) -> dict[int, np.ndarray]:
    return {i: predict_clip(model, cfg, norm, fnorm, vocab, by_idx, i, t_in)
            for i in test_ids}


@torch.no_grad()
def predict_with_sigma(model, cfg: Config, norm: Norm, fnorm: FeatNorm, vocab, by_idx,
                       i: int, t_in: int) -> tuple[np.ndarray, np.ndarray]:
    """(mu, sigma) in RAW units, both (n_origins, H, C).

    The harness scores the mean only, so predict() drops the variance head; a forecast
    plot is the one place it should be visible, since a probabilistic model whose spread
    is never shown is indistinguishable from a point model. z-scoring is per channel and
    linear, so sigma_raw = exp(lv/2) * norm.std -- the mean shift cancels.
    """
    model.eval()
    X, A, L, _ = window_set(cfg, [i], t_in, norm, fnorm, vocab, by_idx)
    C = len(cfg.channels)
    if len(X) == 0:
        z = np.zeros((0, cfg.horizon, C))
        return z, z
    dev = next(model.parameters()).device
    mu, lv = model(X.to(dev), A.to(dev), L.to(dev), cfg.horizon)
    mu = mu.cpu().numpy().astype(np.float64)
    sd = np.exp(0.5 * lv.cpu().numpy().astype(np.float64)) * np.asarray(norm.std)
    return norm.unz(mu), sd
