"""Tactile->tactile forecasting on the EgoTouch Grasp/Hold/Lift subset.

See docs/TACTILE_PREDICTION_PLAN.md for the design. Entry points:
  python -m src.tactile_pixel.train --config configs/tactile/convgru.yaml --protocol lto --fold 0
  python -m src.tactile_pixel.eval  --ckpt <run>/best.pt
"""
