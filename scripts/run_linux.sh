#!/usr/bin/env bash
# run_linux.sh — Samuel Realtime Parrot launcher for Linux (PipeWire)
# Usage: ./scripts/run_linux.sh [--onnx models/samuel_custom_controller.onnx] [--provider webgpu]
# Env vars: SAMUEL_CHECKPOINT, SAMUEL_ONNX, SAMUEL_VAD_SILENCE, SAMUEL_INPUT_GAIN, SAMUEL_CONFIG

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

# Smart defaults: auto-detect fine-tuned checkpoint and ONNX model
CHECKPOINT="hf:vvolhejn/samuel"
if [[ -f "samuel_custom_last.pt" ]]; then
    CHECKPOINT="samuel_custom_last.pt"
fi

ONNX=""
if [[ -f "models/samuel_custom_controller.onnx" ]]; then
    ONNX="models/samuel_custom_controller.onnx"
elif [[ -f "models/samuel_controller.onnx" ]]; then
    ONNX="models/samuel_controller.onnx"
fi

PROVIDER="auto"
VAD_SILENCE="0.45"
INPUT_GAIN="1.0"
LOG_LEVEL="INFO"
LIST_DEVICES=false
CONFIG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --onnx) ONNX="$2"; shift 2 ;;
        --provider) PROVIDER="$2"; shift 2 ;;
        --vad-silence) VAD_SILENCE="$2"; shift 2 ;;
        --input-gain) INPUT_GAIN="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
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
echo "Input Gain: ${INPUT_GAIN}"
echo "Log Level: ${LOG_LEVEL}"
[[ -n "${CONFIG}" ]] && echo "Config: ${CONFIG}"
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
    ARGS=(--checkpoint "${CHECKPOINT}" --provider "${PROVIDER}" --vad-silence "${VAD_SILENCE}" --input-gain "${INPUT_GAIN}" --log-level "${LOG_LEVEL}")
    [[ -n "${ONNX}" ]] && ARGS+=(--onnx "${ONNX}")
    [[ -n "${CONFIG}" ]] && ARGS+=(--config "${CONFIG}")
    uv run python -m samuel_realtime "${ARGS[@]}"
fi