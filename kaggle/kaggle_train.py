#!/usr/bin/env python3
"""
Kaggle Training Blueprint — paste cells sequentially into a Kaggle Notebook (GPU T4 x2).

This file is the canonical source; kaggle/kaggle_train.ipynb is auto-generated from it via `jupyter nbconvert`.
Keep this in sync with the notebook.

Director's spec: GPU T4 x2, never TPU, 2x T4 DDP, batch 16, SSL kept, 30-60min data.
"""

# ========== Cell 1: Environment & Repo Setup ==========
# %%bash
"""
# Install uv
!curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"
echo $PATH
uv --version

# Clone private repo — via GH_TOKEN secret if private
import os
GH_TOKEN = os.environ.get("GH_TOKEN", "")
REPO = "lydorianP/samuel-realtime-parrot"
if GH_TOKEN:
    !git clone https://$GH_TOKEN@github.com/$REPO.git
else:
    !git clone https://github.com/$REPO.git
%cd samuel-realtime-parrot
!ls -la

# Kaggle's image already has CUDA torch 2.x; we install our deps without overwriting torch
# Use uv sync --no-install-project? Instead, uv sync will keep Kaggle's torch if compatible
!uv python install 3.12
!uv sync

# Install vendor/samuel training deps (hydra, omegaconf, wandb, transformers, etc.)
# Kaggle's torch is CUDA, so vendor's torch==2.8.* will be satisfied
!uv pip install -e vendor/samuel 2>&1 | tail -n 20
# If above fails due to torch version pin, use --no-deps and rely on Kaggle's torch:
# !uv pip install --no-deps -e vendor/samuel

# Verify
!uv run python -c "import hydra, omegaconf, wandb, transformers; print('deps ok')"
"""

# ========== Cell 2: Dataset & Pitch Cache Preparation ==========
"""
# prepare_data.py is now scripts/prepare_custom_dataset.py in repo
# It correctly handles librosa.pyin 70-500, 4096/512, manifest + pitch cache header

import os
from pathlib import Path

# Ensure Kaggle dataset is attached as /kaggle/input/my-voice-wavs (or change WPA... path)
WAV_DIR = Path("/kaggle/input/my-voice-wavs")
if not WAV_DIR.exists():
    # Try common Kaggle input layout
    for p in Path("/kaggle/input").glob("*"):
        if list(p.glob("*.wav")) or list(p.rglob("*.wav")):
            WAV_DIR = p
            print(f"Found wavs at {WAV_DIR}")
            break
print(f"WAV_DIR={WAV_DIR} exists={WAV_DIR.exists()} wavs={len(list(WAV_DIR.glob('*.wav'))) if WAV_DIR.exists() else 0}")

!uv run python scripts/prepare_custom_dataset.py \
    --wav-dir /kaggle/input/my-voice-wavs \
    --manifest manifests/custom.jsonl \
    --pitch-cache manifests/pitch_cache/custom_spf512.npz \
    --sample-rate 44100 \
    --samples-per-frame 512

# Verify
import json, numpy as np
print(open("manifests/custom.jsonl").readline()[:200])
d = np.load("manifests/pitch_cache/custom_spf512.npz")
print("pitch cache keys:", list(d.files)[:6], "... n_files", d["n_files"], "sr", d["sample_rate"], "spf", d["samples_per_frame"])
"""

# ========== Cell 3: Distributed Training (DDP on 2x T4) ==========
# %%bash
"""
# Check GPUs
!nvidia-smi

# Launch DDP — Hydra overrides: batch 16 (T4 16GB each, 64 OOMs on waveguide), SSL kept
!torchrun --standalone --nproc_per_node=2 -m samuel.train \
    run.name=kaggle_custom_voice_ft \
    data.manifest_path=manifests/custom.jsonl \
    data.pitch_cache_path=manifests/pitch_cache/custom_spf512.npz \
    batch_size=16 \
    optim.max_steps=5000 \
    optim.warmup_steps=500 \
    log.eval_every=500 \
    log.ckpt_every=1000 \
    log.wandb_mode=offline

# If OOM, retry with batch_size=8:
# !torchrun --standalone --nproc_per_node=2 -m samuel.train run.name=kaggle_custom_voice_ft data.manifest_path=manifests/custom.jsonl data.pitch_cache_path=manifests/pitch_cache/custom_spf512.npz batch_size=8 optim.max_steps=5000 optim.warmup_steps=500 log.eval_every=500 log.ckpt_every=1000 log.wandb_mode=offline
"""

# ========== Cell 4: Extract & Download Checkpoint ==========
"""
import shutil
from pathlib import Path

ckpt_dir = Path("runs/kaggle_custom_voice_ft")
# Find latest run dir (timestamped)
candidates = sorted(ckpt_dir.parent.glob("kaggle_custom_voice_ft_*"))
if candidates:
    ckpt_dir = candidates[-1] / "checkpoints"
else:
    ckpt_dir = Path("runs/kaggle_custom_voice_ft/checkpoints")
print(f"ckpt_dir={ckpt_dir} exists={ckpt_dir.exists()}")
for p in sorted(ckpt_dir.glob("*.pt")):
    print(p, p.stat().st_size/1024/1024, "MB")

if ckpt_dir.exists():
    latest = max(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    dst = Path("/kaggle/working/samuel_custom_last.pt")
    shutil.copy(latest, dst)
    # Also copy config.json for local re-export
    src_cfg = latest.parent.parent / "config.json"
    if src_cfg.exists():
        shutil.copy(src_cfg, "/kaggle/working/custom_config.json")
    print(f"✅ Checkpoint ready: {dst} ({dst.stat().st_size/1024/1024:.1f} MB)")
    print(f"   Config: /kaggle/working/custom_config.json")
    # Also keep in repo for persistence
    import subprocess
    subprocess.run(["cp", str(latest), "samuel_custom_last.pt"])
    print("   Also copied to ./samuel_custom_last.pt (will persist if 'Save Output' ON)")
else:
    print("No checkpoint found — check runs/ and logs for errors")
    !ls -R runs | head -n 100
"""

# ========== Optional Cell 5: Push to HF (so Windows can download without HIP) ==========
"""
# After Cell 4, optionally push to HF for Windows WebGPU download
# Set HF_TOKEN as Kaggle Secret, then:
import os
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    !pip install -q huggingface_hub
    !huggingface-cli login --token $HF_TOKEN --add-to-git-credential
    !uv run hf upload lydorianP/samuel-custom /kaggle/working/samuel_custom_last.pt --repo-type model
    !uv run hf upload lydorianP/samuel-custom /kaggle/working/custom_config.json --repo-type model
    print("Uploaded to hf:lydorianP/samuel-custom")
else:
    print("Set HF_TOKEN secret to push")
"""
