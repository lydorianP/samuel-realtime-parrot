#!/bin/bash
# kaggle/push.sh — automate Kaggle T4 x2 training + HF publishing for Samuel custom voice
# Usage: ./kaggle/push.sh
# Requires: uv, kaggle CLI (uv run kaggle), HF auth as barbarabhb locally, and Kaggle API token configured.
# You must manually set "T4 x2" in Kaggle UI after push (CLI cannot enforce it).
set -e

REPO_SLUG="barbarabhb/samuel-realtime-parrot-custom-train"
HF_REPO="barbarabhb/samuel-realtime-parrot-custom"
DATASET="barbarabhb/my-voice-wavs"

echo "=== Samuel Kaggle Push — T4 x2, never TPU ==="
echo "Kernel slug: $REPO_SLUG"
echo "HF target: $HF_REPO"
echo "Dataset: $DATASET (must exist on Kaggle and be attached in kernel-metadata.json)"
echo ""

# Check kaggle CLI
if ! uv run kaggle --version >/dev/null 2>&1; then
    echo "[error] kaggle CLI not found. Run: uv pip install kaggle"
    exit 1
fi
echo "[ok] kaggle CLI $(uv run kaggle --version)"

# Check auth
if ! uv run kaggle config view >/dev/null 2>&1 && ! uv run kaggle auth print-access-token >/dev/null 2>&1; then
    echo "[error] Kaggle API not authenticated."
    echo "  Run: uv run kaggle auth login"
    echo "  Or create https://www.kaggle.com/settings/api token and set:"
    echo "    export KAGGLE_API_TOKEN=your_token"
    echo "    or save to ~/.kaggle/access_token / ~/.kaggle/kaggle.json"
    exit 1
fi
echo "[ok] Kaggle auth"

# Check metadata
echo ""
echo "--- kernel-metadata.json ---"
cat kaggle/kernel-metadata.json
echo ""
python3 -c "
import json, sys
m=json.load(open('kaggle/kernel-metadata.json'))
assert m.get('enable_gpu')==True, 'enable_gpu must be true'
assert m.get('enable_tpu')==False, 'enable_tpu must be false (NO TPU)'
assert m.get('enable_internet')==True, 'enable_internet must be true'
assert 'barbarabhb/my-voice-wavs' in str(m.get('dataset_sources',[])) or 'my-voice-wavs' in str(m.get('dataset_sources',[])), 'dataset_sources must include my-voice-wavs'
print('[ok] metadata: enable_gpu true, enable_tpu false, enable_internet true, dataset attached')
"

# Check notebook HF handling
echo ""
echo "--- notebook HF_TOKEN handling ---"
if grep -q 'os.environ.get("HF_TOKEN")' kaggle/kaggle_train.ipynb && grep -q 'barbarabhb/samuel-realtime-parrot-custom' kaggle/kaggle_train.ipynb; then
    echo "[ok] notebook reads HF_TOKEN from env, pushes to $HF_REPO, never hardcoded"
else
    echo "[error] notebook missing HF_TOKEN env handling or wrong HF repo"
    grep -n "HF_TOKEN\|barbarabhb" kaggle/kaggle_train.ipynb | head -n 20
    exit 1
fi
if grep -q 'os.environ.get("GH_TOKEN")' kaggle/kaggle_train.ipynb; then
    echo "[ok] notebook reads GH_TOKEN from env"
else
    echo "[warn] GH_TOKEN not found in notebook"
fi

# Local HF auth check (fallback)
echo ""
echo "--- local HF auth (fallback push) ---"
if uv run hf auth whoami 2>&1 | grep -q "barbarabhb"; then
    echo "[ok] local HF auth as barbarabhb (for fallback push if Kaggle Secrets missing)"
    uv run hf auth whoami
else
    echo "[warn] local HF not as barbarabhb; fallback push may fail. Run: hf auth login"
    uv run hf auth list || true
fi

# Check local GH
echo ""
echo "--- local GH auth ---"
gh auth status 2>&1 | head -n 5 || echo "[warn] gh not authenticated"

# Push
echo ""
read -p "Push to Kaggle now? This will upload kaggle/kaggle_train.ipynb and start a run. [y/N] " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborted. Manually run: uv run kaggle kernels push -p kaggle/"
    exit 0
fi

echo ""
echo "Pushing kernel..."
uv run kaggle kernels push -p kaggle/
echo "[ok] pushed"

# CRITICAL: Kaggle CLI cannot set T4 x2 explicitly — must verify in UI
echo ""
echo "##########################################################################"
echo "!! CRITICAL: Verify accelerator is T4 x2 in Kaggle UI !!"
echo "   Open: https://www.kaggle.com/code/$(uv run kaggle kernels list --mine 2>&1 | grep $REPO_SLUG | head -n1 || echo $REPO_SLUG)"
echo "   Or go to https://www.kaggle.com/code -> Your kernel -> Settings (right pane) -> Accelerator: GPU T4 x2"
echo "   If it shows 'GPU P100' or 'TPU', CHANGE IT TO 'GPU T4 x2' and re-run or restart session."
echo "   The notebook will fail or be slow on P100 and crash on TPU (SEANet weight_norm XLA)."
echo "##########################################################################"
read -p "Have you verified T4 x2 in the UI? Press Enter to continue polling, or Ctrl+C to abort and fix manually. " _

# Poll status
echo ""
echo "Polling kernel status (every 30s, Ctrl+C to stop)..."
while true; do
    status=$(uv run kaggle kernels status "$REPO_SLUG" 2>&1 | head -n 5 || echo "unknown")
    echo "$(date '+%H:%M:%S') status: $status"
    # Kaggle status output contains "complete", "running", "error"
    if echo "$status" | grep -qi "complete"; then
        echo "[ok] Kernel completed"
        break
    fi
    if echo "$status" | grep -qi "error\|failed"; then
        echo "[error] Kernel failed — check logs:"
        uv run kaggle kernels logs "$REPO_SLUG" 2>&1 | tail -n 100
        exit 1
    fi
    sleep 30
done

# Retrieve artifacts
echo ""
echo "Downloading kernel output..."
mkdir -p kaggle/output
uv run kaggle kernels output "$REPO_SLUG" -p kaggle/output/
ls -lh kaggle/output/ | head -n 20
echo ""

# Verify downloaded .pt
if ls kaggle/output/samuel_custom_last.pt >/dev/null 2>&1; then
    size=$(stat -c%s kaggle/output/samuel_custom_last.pt 2>&1 | head -n1)
    mb=$(python3 -c "print(f'{int('$size')/1024/1024:.1f}')")
    echo "[ok] Downloaded kaggle/output/samuel_custom_last.pt $mb MB"
    if [ "$size" -lt 5000000 ]; then
        echo "[warn] File <5M — may be error log, not checkpoint. Check:"
        ls -lh kaggle/output/
        cat kaggle/output/*.log 2>&1 | tail -n 50 || true
    fi
else
    echo "[warn] samuel_custom_last.pt not in output — check kaggle/output/ and logs"
    ls -R kaggle/output 2>&1 | head -n 100
    uv run kaggle kernels logs "$REPO_SLUG" 2>&1 | tail -n 100
fi

if ls kaggle/output/custom_config.json >/dev/null 2>&1; then
    echo "[ok] custom_config.json present"
fi

# Check HF push success (from notebook logs)
echo ""
echo "Checking if notebook pushed to HF $HF_REPO..."
if uv run hf download "$HF_REPO" --help >/dev/null 2>&1; then
    # Try to list repo files via API
    uv run hf download "$HF_REPO" 2>&1 | head -n 20 || echo "HF repo may not exist yet or not public"
fi
# Fallback local push if notebook failed due to missing Secrets
echo ""
read -p "If HF push failed due to missing Kaggle Secrets, push locally now from kaggle/output/? [y/N] " do_push
if [[ "$do_push" == "y" || "$do_push" == "Y" ]]; then
    if [ -f "kaggle/output/samuel_custom_last.pt" ]; then
        echo "Pushing locally via huggingface_hub (barbarabhb)..."
        uv run python -c "
from huggingface_hub import create_repo, upload_file
import os
tok = open(os.path.expanduser('~/.cache/huggingface/token'), 'r').read().strip() if os.path.exists(os.path.expanduser('~/.cache/huggingface/token')) else None
# Use hf auth token
import subprocess
try:
    out = subprocess.check_output(['uv','run','hf','auth','token'], text=True).strip()
    tok = out
except: pass
repo='barbarabhb/samuel-realtime-parrot-custom'
create_repo(repo, repo_type='model', private=True, exist_ok=True, token=tok)
upload_file(path_or_fileobj='kaggle/output/samuel_custom_last.pt', path_in_repo='samuel_custom_last.pt', repo_id=repo, repo_type='model', token=tok)
upload_file(path_or_fileobj='kaggle/output/custom_config.json', path_in_repo='config.json', repo_id=repo, repo_type='model', token=tok)
print(f'Pushed to https://huggingface.co/{repo}')
"
    else
        echo "No kaggle/output/samuel_custom_last.pt to push"
    fi
fi

# Final URLs
echo ""
echo "=== DONE ==="
echo "Kaggle logs: https://www.kaggle.com/code/$REPO_SLUG"
uv run kaggle kernels logs "$REPO_SLUG" 2>&1 | tail -n 20 || true
echo "HF repo: https://huggingface.co/$HF_REPO"
echo "Local re-export ready: ./scripts/re_export_custom.sh kaggle/output/samuel_custom_last.pt models/samuel_custom_controller.onnx"
if [ -f "kaggle/output/samuel_custom_last.pt" ]; then
    echo "Then: uv run python -m samuel_realtime --checkpoint kaggle/output/samuel_custom_last.pt --onnx models/samuel_custom_controller.onnx --provider webgpu --out-device Samuel_Virtual_Mic"
fi
