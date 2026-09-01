#!/bin/bash
# re_export_custom.sh — after Kaggle, re-export custom checkpoint to ONNX for Windows WebGPU
# Usage: ./scripts/re_export_custom.sh samuel_custom_last.pt [models/samuel_custom_controller.onnx]
set -e
CKPT="${1:-samuel_custom_last.pt}"
OUT="${2:-models/samuel_custom_controller.onnx}"

if [ ! -f "$CKPT" ]; then
    echo "[error] checkpoint not found: $CKPT"
    echo "  Download from Kaggle Output: /kaggle/working/samuel_custom_last.pt"
    echo "  Or from runs/kaggle_custom_voice_ft_*/checkpoints/last.pt"
    exit 1
fi

echo "[info] Re-exporting $CKPT -> $OUT"
uv run python scripts/export_onnx.py --checkpoint "$CKPT" --out "$OUT" --verify

echo "[ok] ONNX ready: $OUT ($(du -h "$OUT" | cut -f1))"
ls -lh "$OUT"*
echo ""
echo "Swap for realtime pipeline:"
echo "  SAMUEL_CHECKPOINT=$CKPT uv run python -m samuel_realtime --out-device Samuel_Virtual_Mic --onnx $OUT --provider webgpu"
echo "  # Windows:"
echo "  python -m samuel_realtime --out-device \"CABLE Input\" --onnx models\\samuel_custom_controller.onnx --provider webgpu --checkpoint $CKPT"
echo ""
echo "To push to HF for Windows download (no HIP):"
echo "  hf upload barbarabhb/samuel-realtime-parrot-custom $CKPT --repo-type model"
echo "  hf upload barbarabhb/samuel-realtime-parrot-custom ${CKPT%.pt}.json --repo-type model  # if you have custom_config.json"
