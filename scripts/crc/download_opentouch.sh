#!/bin/bash
# Download the full OpenTouch dataset (26 HDF5 shards + labels) via gdown.
# Source: OpenTouch-MIT/opentouch scripts/download_data.sh (file IDs copied verbatim).
# ~14 GB total (561 MB/shard). Usage:
#   pip install gdown
#   bash scripts/crc/download_opentouch.sh [DEST_DIR]     # default ~/opentouch/data
set -euo pipefail

DEST="${1:-$HOME/opentouch/data}"
mkdir -p "$DEST"
cd "$DEST"

IDS=(
    "1EjMOzs45devBo0TqhuhZTT_Ll7HZ1lrW"
    "1fAmmieSr0yFm7ldhW7Smld7jUxBCw8fu"
    "1cUhgYbredkIRswanUiM5uDixiFLq4WCC"
    "1PCzWMJxtbD2HJLCl2WFzOIB-5RN3X81G"
    "1jFlYmCFb6GldbPJ-zLzJSCY-BKipdjPE"
    "1reSqa8v8RaY2kZXLw0_g7Amvq7lJl6Cu"
    "1atXpcctoHs4dbXhyAAO9EY88D2f1JYfT"
    "1Z3b-I6BMPgNlpiKw8gISkUi3VULUtLFN"
    "1u-6WGn3eMQJe3eh6lCFahlIcEVmkULna"
    "17wF0aBIH6RRtRGRaXeiI-Y4Lh5bnDFBL"
    "1KICpqtfmbnKhgHi-CIR9XAp24TE1945M"
    "1vkl6wat_dgF5NQs9QVDfCyJGyjEjd2FW"
    "1BbKU5vSH-wOrCnOjRWNe3H7niJP_uJrb"
    "1GCX4mAgCvOvmIQ0uXotqpoNdXgYzp4ki"
    "1rxsLWGw_diPvRnALxOYakCIweG90O28I"
    "1zAYfcMt2hqcG1bPtCOAkWu0zsd6lfvrX"
    "1tQh21z8KRxYHsh69dW6VcSw5Wux67R6_"
    "1jeA1bEit-tDQpfwt3NmTeC8iwM6I1qiE"
    "1UT5htydKCfBCO57On-mRJRz7mSi57K4u"
    "1h9Bl8CTGJWvU2XPr93fptBTpB2cwYgwq"
    "1SAbxWQZDEyTZ-ESVi9G5bxEc7ov-EO28"
    "1jKyVNsi7fsofSho_xoRi0Kgqem4zrk5F"
    "11LQ28c6jPhNfiu9fPDu5diruNUCa0bGM"
    "1dwlVYtBfyNUHg7Qxnxa_iYBYPCcn9VeX"
    "1X4-MS7Qodhtmn6zcY9a5cMq02eDLvOJq"
    "1VAKXJPO4j_40hpqslNJ4_WbgWfKaGLQC"
)
LABELS_ID="1cM-816vcCnkgWVIGXZrR1o8TPsDvRVCZ"

n=0
for ID in "${IDS[@]}"; do
    n=$((n + 1))
    echo "[$n/${#IDS[@]}] gdown $ID"
    gdown "$ID" || echo "  WARN: $ID failed (Drive quota?) — rerun later"
done
echo "labels: gdown $LABELS_ID"
gdown "$LABELS_ID" || true
if [ -f final_annotation.zip ]; then unzip -o final_annotation.zip; fi

echo "done -> $DEST  ($(ls *.hdf5 2>/dev/null | wc -l) HDF5 shards)"
