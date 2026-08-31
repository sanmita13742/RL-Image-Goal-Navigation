#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

RUN_ID=${1:-"kaggle_run"}
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/runs/$RUN_ID"
EXPLORE_DIR="$RUN_DIR/exploration"

echo "======================================================"
echo "KAGGLE PIPELINE: INTERMEDIATE ANALYSIS"
echo "Run ID: $RUN_ID"
echo "======================================================"

if [[ ! -d "$EXPLORE_DIR" ]]; then
    echo "ERROR: Exploration directory $EXPLORE_DIR not found. Did you run kaggle_explore.sh?"
    exit 1
fi

if command -v python.exe &> /dev/null; then
    PYTHON_CMD="python.exe"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

$PYTHON_CMD "$ROOT_DIR/analyze_dataset.py" \
    --session "$EXPLORE_DIR" \
    --outdir "$RUN_DIR/analysis_output"

echo "Analysis finished successfully. Output in $RUN_DIR/analysis_output"
