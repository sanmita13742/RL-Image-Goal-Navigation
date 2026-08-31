#!/usr/bin/env bash
set -euo pipefail

# Unbuffer python output to prevent Kaggle from thinking the kernel is dead
export PYTHONUNBUFFERED=1

# Arguments
RUN_ID=${1:-"kaggle_run"}
MAP=${2:-"maps/complex/scene.xml"}
SMOKE=${3:-""}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/runs/$RUN_ID"

echo "======================================================"
echo "KAGGLE PIPELINE: EXPLORATION"
echo "Run ID: $RUN_ID"
echo "Map: $MAP"
echo "======================================================"

mkdir -p "$RUN_DIR"
if [[ ! -f "$RUN_DIR/config_used.yaml" ]]; then
    cp "$ROOT_DIR/configs/pipeline.yaml" "$RUN_DIR/config_used.yaml"
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

$PYTHON_CMD "$ROOT_DIR/scripts/run_exploration.py" \
    --config "runs/$RUN_ID/config_used.yaml" \
    --run-dir "$RUN_DIR" \
    --map "$ROOT_DIR/$MAP" \
    $SMOKE_FLAG

echo "Exploration finished successfully."
