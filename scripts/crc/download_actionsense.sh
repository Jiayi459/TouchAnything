#!/bin/bash
# Download ActionSense wearables HDF5 (sensor-only, no video) for the tactile subjects.
# Subjects S00-S05 wore the tactile gloves; S06-S09 did NOT (skipped).
# Files are small (sensor streams only). Usage:
#   bash scripts/crc/download_actionsense.sh [DEST_DIR]      # default ~/actionsense
set -euo pipefail

DEST="${1:-$HOME/actionsense}"
mkdir -p "$DEST"
cd "$DEST"

URLS=(
  "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-07_experiment_S00/2022-06-07_17-18-17_actionNet-wearables_S00/2022-06-07_17-18-46_streamLog_actionNet-wearables_S00.hdf5"
  "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-07_experiment_S00/2022-06-07_18-10-55_actionNet-wearables_S00/2022-06-07_18-11-37_streamLog_actionNet-wearables_S00.hdf5"
  "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-13_experiment_S01_recordingStopped/2022-06-13_18-13-12_actionNet-wearables_S01/2022-06-13_18-14-59_streamLog_actionNet-wearables_S01.hdf5"
  "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-13_experiment_S02/2022-06-13_21-39-50_actionNet-wearables_S02/2022-06-13_21-40-16_streamLog_actionNet-wearables_S02.hdf5"
  "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-13_experiment_S02/2022-06-13_21-47-57_actionNet-wearables_S02/2022-06-13_21-48-24_streamLog_actionNet-wearables_S02.hdf5"
  "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-13_experiment_S02/2022-06-13_22-34-45_actionNet-wearables_S02/2022-06-13_22-35-11_streamLog_actionNet-wearables_S02.hdf5"
  "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-13_experiment_S02/2022-06-13_23-22-21_actionNet-wearables_S02/2022-06-13_23-22-44_streamLog_actionNet-wearables_S02.hdf5"
  "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-14_experiment_S03/2022-06-14_13-11-44_actionNet-wearables_S03/2022-06-14_13-12-07_streamLog_actionNet-wearables_S03.hdf5"
  "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-14_experiment_S03/2022-06-14_13-52-21_actionNet-wearables_S03/2022-06-14_13-52-57_streamLog_actionNet-wearables_S03.hdf5"
  "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-14_experiment_S04/2022-06-14_16-38-18_actionNet-wearables_S04/2022-06-14_16-38-43_streamLog_actionNet-wearables_S04.hdf5"
  "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-14_experiment_S05/2022-06-14_20-36-27_actionNet-wearables_S05/2022-06-14_20-36-54_streamLog_actionNet-wearables_S05.hdf5"
  "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-14_experiment_S05/2022-06-14_20-45-43_actionNet-wearables_S05/2022-06-14_20-46-12_streamLog_actionNet-wearables_S05.hdf5"
)

n=0
for URL in "${URLS[@]}"; do
  n=$((n + 1))
  out="$(basename "$URL")"
  echo "[$n/${#URLS[@]}] $out"
  if [ -s "$out" ]; then echo "  exists, skip"; continue; fi
  curl -fL --retry 3 -o "$out" "$URL" || echo "  WARN: failed $URL"
done
echo "done -> $DEST  ($(ls *.hdf5 2>/dev/null | wc -l) files)"
