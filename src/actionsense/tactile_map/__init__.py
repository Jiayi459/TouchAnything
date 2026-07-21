"""Forecast the 6-dim F/CoP target from the raw tactile MAP (group B, ActionSense).

Predict the next 1 s of [F, CoP-x, CoP-y] x {left,right} from a history of pressure maps
(2 x 32 x 32), with two per-frame encoders feeding an identical GRU + one-shot head:
  - FlattenEncoder : flatten (2*32*32) -> linear -> embedding
  - CNNEncoder     : small conv stack -> embedding
If the CNN beats the flatten encoder, spatial structure of the contact patch contributes.

Everything is scored on the FROZEN eval harness (src/actionsense/eval_harness): same target,
split, rate, horizon, and origins -> predictions are exported as {idx: (n_origins, H, 6)} and
fed to `evaluate.py --model-preds`. See docs plan + SESSION_LOG (2026-07-21).
"""
