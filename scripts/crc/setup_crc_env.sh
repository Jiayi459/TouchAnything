#!/bin/bash
# One-time CUDA environment setup for tactile forecasting on ND CRC.
# Run on a CRC front-end node, from the repository root, using bash.
# Docs: https://docs.crc.nd.edu/popular_modules/conda.html
#       https://docs.crc.nd.edu/resources/gpu.html
set -euo pipefail

ENV=tactile

# --- 1. One-time conda shell init (safe to re-run) ---
module load conda
conda init bash
# shellcheck disable=SC1090
source ~/.bashrc
module unload conda

# --- 2. Create / update the lean env ---
if conda env list | grep -qE "^\s*${ENV}\s"; then
  conda env update -n "${ENV}" -f scripts/crc/environment_tactile_cuda.yaml
else
  conda env create -n "${ENV}" -f scripts/crc/environment_tactile_cuda.yaml
fi
conda activate "${ENV}"

# --- 3. CUDA PyTorch (wheels bundle the CUDA runtime; needs a recent NVIDIA driver) ---
# Pinned to a known-good cu124 combo; bump if the cluster driver requires a newer CUDA.
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# --- 4. Verify (cuda.is_available() is False on a front-end node; test on a gpu node) ---
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("compiled CUDA:", torch.version.cuda)
print("cuda available here:", torch.cuda.is_available(), "(expected False on front-end; check inside a gpu job)")
PY

echo "Done. Use 'conda activate ${ENV}' in jobs. Verify GPU via scripts/crc/train_gpu.job."
