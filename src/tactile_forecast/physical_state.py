"""Analytic physical-state extraction from a tactile pressure clip (numpy-only).

Turns a (T, C, H, W) pressure clip into a low-dimensional, fully interpretable physical
state trajectory — the target of the v1 forecaster (docs/TACTILE_FORECAST_PLAN.md). NO
learning: every quantity is computed directly from the pressure field.

Per hand (channel), each frame yields the 0th/1st/2nd pressure moments:
    F     total force                = Σ p
    xbar  center of pressure x       = Σ p·x / Σ p
    ybar  center of pressure y       = Σ p·y / Σ p
    sxx   spread (variance) in x      = Σ p·(x-xbar)² / Σ p
    syy   spread (variance) in y      = Σ p·(y-ybar)² / Σ p
    sxy   covariance                  = Σ p·(x-xbar)(y-ybar) / Σ p
From these, derived (in the dataset/loader, not here): contact area A = participation ratio,
orientation θ and eccentricity (eig of [[sxx,sxy],[sxy,syy]]), CoP velocity (ẋ,ẏ) and dF/dt
(finite differences), and motion phase φ (Hilbert transform of F(t) or CoP(t)).

Coordinates are normalized to [-1, 1] across the grid so features are sensor-size-agnostic
(comparable across the 16/21/32-wide gloves).
"""
from __future__ import annotations

import numpy as np

# per-hand raw feature names (order matters — this is the on-disk contract)
FEATURES = ("F", "xbar", "ybar", "sxx", "syy", "sxy")


def _grids(h, w):
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float64)
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float64)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return gx, gy


def frame_state(field: np.ndarray) -> np.ndarray:
    """field: (C, H, W) non-negative pressure -> (C, 6) [F, xbar, ybar, sxx, syy, sxy]."""
    field = np.clip(np.asarray(field, dtype=np.float64), 0.0, None)
    C, H, W = field.shape
    gx, gy = _grids(H, W)
    out = np.zeros((C, 6), dtype=np.float64)
    for c in range(C):
        p = field[c]
        F = p.sum()
        out[c, 0] = F
        if F <= 1e-9:
            continue
        xbar = (p * gx).sum() / F
        ybar = (p * gy).sum() / F
        dx, dy = gx - xbar, gy - ybar
        out[c, 1] = xbar
        out[c, 2] = ybar
        out[c, 3] = (p * dx * dx).sum() / F
        out[c, 4] = (p * dy * dy).sum() / F
        out[c, 5] = (p * dx * dy).sum() / F
    return out


def clip_states(clip: np.ndarray) -> np.ndarray:
    """clip: (T, C, H, W) -> (T, C, 6) raw physical-moment trajectory."""
    clip = np.asarray(clip)
    return np.stack([frame_state(clip[t]) for t in range(clip.shape[0])], axis=0)


# ---- derived features (used at train/analysis time, from the (T,C,6) moments) ---------- #
def derive(states: np.ndarray, fps: float) -> dict:
    """Expand (T,C,6) moments into interpretable per-hand series.

    Returns dict of (T, C) arrays: F, xbar, ybar, area, theta, ecc, vx, vy, dF.
    area  = participation-ratio-like effective size = sqrt(sxx*syy - sxy^2) (patch scale)
    theta = orientation of the contact patch (rad); ecc = elongation in [0,1]
    vx,vy = CoP velocity (grid-units/s); dF = dF/dt (force/s)
    """
    T, C, _ = states.shape
    F = states[..., 0]
    xbar, ybar = states[..., 1], states[..., 2]
    sxx, syy, sxy = states[..., 3], states[..., 4], states[..., 5]
    area = np.sqrt(np.clip(sxx * syy - sxy * sxy, 0, None))
    # eigenvalues of the 2x2 covariance for orientation + eccentricity
    tr = sxx + syy
    det = sxx * syy - sxy * sxy
    disc = np.sqrt(np.clip((tr * 0.5) ** 2 - det, 0, None))
    l1 = tr * 0.5 + disc
    l2 = tr * 0.5 - disc
    theta = 0.5 * np.arctan2(2 * sxy, sxx - syy)
    ecc = np.sqrt(np.clip(1.0 - l2 / (l1 + 1e-12), 0, 1))
    vx = np.gradient(xbar, axis=0) * fps
    vy = np.gradient(ybar, axis=0) * fps
    dF = np.gradient(F, axis=0) * fps
    return dict(F=F, xbar=xbar, ybar=ybar, area=area, theta=theta, ecc=ecc, vx=vx, vy=vy, dF=dF)


def phase(signal_1d: np.ndarray) -> np.ndarray:
    """Instantaneous phase of a (near-)periodic 1-D signal via the analytic signal.

    Returns (T,) phase in radians. For slice/wipe use F(t) or a CoP component. Uses a
    numpy FFT Hilbert transform (no scipy dependency)."""
    x = np.asarray(signal_1d, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    Xf = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2
    analytic = np.fft.ifft(Xf * h)
    return np.angle(analytic)
