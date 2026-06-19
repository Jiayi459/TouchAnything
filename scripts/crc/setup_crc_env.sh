#!/bin/bash
# One-time CUDA environment setup for tactile forecasting on ND CRC.
# Run from the repository root on any CRC node (front-end is fine), using bash.
# Docs: https://docs.crc.nd.edu/popular_modules/conda.html
#       https://docs.crc.nd.edu/resources/gpu.html
#
# NOTE: deliberately NOT using `set -u` — sourcing CRC's /etc/bashrc references
# unbound variables (BASHRCSOURCED) and would abort the script.
set -eo pipefail

ENV=tactile

# Make `conda` and `conda activate` available inside this non-interactive script
# by sourcing conda's profile directly (more robust than `conda init` + ~/.bashrc).
if ! command -v conda >/dev/null 2>&1; then
  module load conda
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create (or update) the lean env
if conda env list | awk '{print $1}' | grep -qx "${ENV}"; then
  echo "[setup] env '${ENV}' exists -> updating"
  conda env update -n "${ENV}" -f scripts/crc/environment_tactile_cuda.yaml
else
  echo "[setup] creating env '${ENV}'"
  conda env create -n "${ENV}" -f scripts/crc/environment_tactile_cuda.yaml
fi
conda activate "${ENV}"

# CUDA PyTorch (wheels bundle the CUDA runtime; needs a recent NVIDIA driver).
# Bump the cu124 pin if the cluster driver requires a newer CUDA.
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# Verify (cuda.is_available() is False on a front-end node; test on a gpu node)
python - <<'PY'
import torch
print("torch:", torch.__version__, "| compiled CUDA:", torch.version.cuda)
print("cuda available here:", torch.cuda.is_available(),
      "(False on a front-end node is normal; check inside a gpu job)")
PY

echo "Done. Use 'conda activate ${ENV}' in jobs (see scripts/crc/train_gpu.job)."
