#!/usr/bin/env bash
# run_linux.sh — Samuel Realtime Parrot launcher for Linux (PipeWire)
# Usage: ./scripts/run_linux.sh [--onnx models/samuel_controller.onnx] [--provider webgpu]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."
cd "${PROJECT_ROOT}"

# Activate venv if present
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
else
    echo "[warn] No .venv found — using system Python"
fi

# Default arguments
CHECKPOINT="hf:vvolhejn/samuel"
ONNX=""
PROVIDER="auto"
VAD_SILENCE="0.45"
LOG_LEVEL="INFO"
LIST_DEVICES=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --onnx) ONNX="$2"; shift 2 ;;
        --provider) PROVIDER="$2"; shift 2 ;;
        --vad-silence) VAD_SILENCE="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --list-devices) LIST_DEVICES=true; shift ;;
        --log-level) LOG_LEVEL="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=========================================="
echo "  Samuel Realtime Parrot — Linux Launcher"
echo "=========================================="
echo "Checkpoint: ${CHECKPOINT}"
[[ -n "${ONNX}" ]] && echo "ONNX: ${ONNX}"
echo "Provider: ${PROVIDER}"
echo "VAD Silence: ${VAD_SILENCE} sec"
echo "Log Level: ${LOG_LEVEL}"
echo ""

# Check virtual mic
if [[ -x "./scripts/setup_virtual_mic.sh" ]]; then
    echo "[info] Ensuring SamuelMic virtual sink exists..."
    ./scripts/setup_virtual_mic.sh
    echo ""
fi

echo "Starting pipeline... (Ctrl+C to stop)"
echo ""

if [[ "${LIST_DEVICES}" == true ]]; then
    uv run python -m samuel_realtime --list-devices --log-level "${LOG_LEVEL}"
else
    ONNX_ARG=()
    [[ -n "${ONNX}" ]] && ONNX_ARG=(--onnx "${ONNX}")
    uv run python -m samuel_realtime \
        --checkpoint "${CHECKPOINT}" \
        "${ONNX_ARG[@]}" \
        --provider "${PROVIDER}" \
        --vad-silence "${VAD_SILENCE}" \
        --log-level "${LOG_LEVEL}"
fi
