#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MAP=""
DURATION=120
DEVICE="auto"
RUN_ID=""
RESUME=""
SMOKE=0
CONFIG="configs/pipeline.yaml"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --map) MAP="$2"; shift ;;
        --duration-minutes) DURATION="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
        --run-id) RUN_ID="$2"; shift ;;
        --resume) RESUME="$2"; shift ;;
        --smoke) SMOKE=1 ;;
        --config) CONFIG="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [[ -z "$RESUME" && -z "$MAP" ]]; then
    echo "ERROR: --map <path> is required unless --resume is used."
    exit 1
fi

if [[ -z "$RUN_ID" && -n "$RESUME" ]]; then
    RUN_ID="$RESUME"
elif [[ -z "$RUN_ID" ]]; then
    RUN_ID="minav_$(date +%Y%m%d_%H%M%S)"
fi

echo "============================================================"
echo "MINav Reproduction Pipeline"
echo "============================================================"
echo "Run ID      : $RUN_ID"
echo "Map         : ${MAP:-N/A}"
echo "Device      : $DEVICE"
if [[ $SMOKE -eq 1 ]]; then
    echo "Mode        : SMOKE"
else
    echo "Mode        : FULL"
fi
echo "Start       : $(date)"
echo "Output      : runs/$RUN_ID/"
echo "============================================================"

CMD_ARGS="--config $CONFIG --device $DEVICE --run-id $RUN_ID"
if [[ -n "$MAP" ]]; then
    CMD_ARGS="$CMD_ARGS --map $MAP"
fi
if [[ -n "$DURATION" ]]; then
    CMD_ARGS="$CMD_ARGS --duration-minutes $DURATION"
fi
if [[ -n "$RESUME" ]]; then
    CMD_ARGS="$CMD_ARGS --resume $RESUME"
fi
if [[ $SMOKE -eq 1 ]]; then
    CMD_ARGS="$CMD_ARGS --smoke"
fi

if command -v python.exe &> /dev/null; then
    PYTHON_CMD="python.exe"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "ERROR: python not found"
    exit 1
fi

$PYTHON_CMD "scripts/pipeline.py" $CMD_ARGS

echo ""
echo "PIPELINE COMPLETE"
echo "Run ID      : $RUN_ID"
echo "Output      : runs/$RUN_ID/"
echo "End         : $(date)"
echo "Elapsed     : N/A" # Would need to track start time in bash to compute exactly, or pipeline.py can do it.

