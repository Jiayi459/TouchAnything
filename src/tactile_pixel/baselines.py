"""Non-learned baselines that any model must beat. Operate on torch tensors.

x: (B, t_in, C, H, W) -> prediction (B, t_out, C, H, W).
"""
import torch


def persistence(x, t_out):
    """y_hat[t+h] = last observed frame."""
    last = x[:, -1:]
    return last.repeat(1, t_out, 1, 1, 1)


def last_velocity(x, t_out):
    """Linear extrapolation from the last two frames, clipped to [0, 1]."""
    last = x[:, -1]
    vel = x[:, -1] - x[:, -2] if x.shape[1] >= 2 else torch.zeros_like(last)
    outs = [torch.clamp(last + (h + 1) * vel, 0.0, 1.0) for h in range(t_out)]
    return torch.stack(outs, 1)
