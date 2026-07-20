"""TouchAnything (upstream) — multi-view video -> hand-pose + tactile-pixel prediction (group A).

The original fork's model + data + training pipeline: DINOv2 vision encoders, a temporal
transformer, a 42-joint MANO hand-pose head, and a tactile-pixel head; trained on EgoDex video +
glove tactile (and EgoPressure). Subpackages:
  data/      EgoDex/HDF5/TouchAnything datasets + transforms + glove augmentation
  models/    vision/multi-view encoders, temporal transformer, pose enc/dec, fusion, top model
  losses/    pose + tactile losses
  utils/     config, logger, metrics, pressure-map + visualization helpers
  datasets/  EgoPressure loader
  resources/ MANO mesh + taxel<->position pressure maps
See docs/REPO_ORGANIZATION.md. (Distinct from src/actionsense/ and src/tactile_pixel/.)
"""
