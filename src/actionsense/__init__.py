"""ActionSense physical-state forecasting (group B).

Aggregate force / center-of-pressure forecasting on the ActionSense conductive-thread gloves:
  - physical_state.py  analytic moment extraction (F, CoP, shear) from a pressure clip
  - state_forecast.py  v1 forecaster over the raw physical state (GRU vs structured baselines)
  - action_dynamics.py v2 slow/fast + probabilistic GRU (probGRU) library
  - eval_harness/       the FROZEN evaluation harness + classical baselines

Data lives in data/actionsense_states/. See docs/TACTILE_FORECAST_PLAN.md and docs/REPO_ORGANIZATION.md.
"""
