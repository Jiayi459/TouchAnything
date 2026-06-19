"""Synthetic end-to-end smoke test for the torch path (run on CRC after env setup):

    conda activate tactile
    python scripts/crc/smoke_test.py

Verifies every model forward/backward, masked loss, baselines, and metric shapes
without needing the dataset. Exits non-zero on failure.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from src.tactile_forecast import baselines, engine            # noqa: E402
from src.tactile_forecast import tactile_utils as U           # noqa: E402
from src.tactile_forecast.models import build_model           # noqa: E402

B, t_in, t_out, C, H, W = 4, 10, 15, 2, 21, 21
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, "| cuda:", torch.cuda.is_available())

mask_np = np.zeros((C, H, W), np.float32)
mask_np[:, 2:19, 2:19] = 1.0                                   # fake sensor mask (C,H,W)
mask_cw = torch.from_numpy(mask_np).to(dev)                    # (C,H,W) for metrics
mask_b = mask_cw.unsqueeze(0).expand(B, -1, -1, -1).contiguous()  # (B,C,H,W) as the dataloader yields
x = (torch.rand(B, t_in, C, H, W, device=dev)) * mask_cw
y = (torch.rand(B, t_out, C, H, W, device=dev)) * mask_cw
inv = (lambda a: a)

for name in ["convgru", "convlstm", "simvp"]:
    cfg = dict(name=name, in_ch=C, t_in=t_in, t_out=t_out, hid=16,
               layers=2, n_enc=2, n_trans=2, kernel=3)
    model = build_model(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    pred = model(x, y=y, ssprob=0.5)
    assert pred.shape == (B, t_out, C, H, W), (name, pred.shape)
    loss = engine.masked_mse(pred, y, mask_b, active_weight=3.0)
    loss.backward(); opt.step()
    assert torch.isfinite(loss), name
    assert (pred >= 0).all(), f"{name} produced negative pressure"
    n = sum(p.numel() for p in model.parameters())
    print(f"  {name:9s} OK  out={tuple(pred.shape)}  loss={loss.item():.4f}  params={n/1e6:.3f}M")

# baselines + metrics
pp = baselines.persistence(x.cpu(), t_out).numpy()
m = U.horizon_metrics(inv(pp), inv(y.cpu().numpy()), inv(x[:, -1].cpu().numpy()), mask_np)
assert set(m.keys()) == set(range(1, t_out + 1))
print(f"  baselines+metrics OK  (persistence h1 skill={m[1]['skill']:.3f})")
print("SMOKE TEST PASSED")
