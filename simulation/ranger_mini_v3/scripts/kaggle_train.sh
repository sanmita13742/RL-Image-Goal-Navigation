#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

RUN_ID=${1:-"kaggle_run"}
DEVICE=${2:-"cuda"}
SMOKE=${3:-""}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/runs/$RUN_ID"

echo "======================================================"
echo "KAGGLE PIPELINE: TRAINING"
echo "Run ID: $RUN_ID"
echo "Device: $DEVICE"
echo "======================================================"

if [[ ! -d "$RUN_DIR" ]]; then
    echo "ERROR: Run directory $RUN_DIR not found. Did you run kaggle_explore.sh?"
    exit 1
fi

SMOKE_FLAG=""
if [[ "$SMOKE" == "--smoke" || "$SMOKE" == "smoke" || "$SMOKE" == "1" ]]; then
    SMOKE_FLAG="--smoke"
    echo "Running in SMOKE mode."
fi

if command -v python.exe &> /dev/null; then
    PYTHON_CMD="python.exe"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

echo "[1/2] Building Final Dataset (Hindsight & Embeddings)"
$PYTHON_CMD "$ROOT_DIR/scripts/build_final_dataset.py" \
    --config "runs/$RUN_ID/config_used.yaml" \
    --run-dir "$RUN_DIR" \
    $SMOKE_FLAG

echo "[2/2] Training TD3+BC"
$PYTHON_CMD "$ROOT_DIR/scripts/train_td3_bc.py" \
    --config "runs/$RUN_ID/config_used.yaml" \
    --run-dir "$RUN_DIR" \
    --device "$DEVICE" \
    $SMOKE_FLAG

echo "Training finished successfully."
