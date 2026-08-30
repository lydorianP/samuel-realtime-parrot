# Kaggle Training — Samuel Custom Voice Fine-Tune (2x T4)

> **Accelerator: GPU T4 x2** — never TPU. SEANet `weight_norm` + causal `Conv1d` + dynamic `F.pad` → XLA recompiles every batch or fails.

## Quick Kaggle Setup

1. **Kaggle Notebook → Settings:**
   - Accelerator: `GPU T4 x2` (16GB each, 32GB total)
   - Internet: ON (for `facebook/wav2vec2-base-960h` download, ~360M)
   - Persistence: ON
   - Environment: `Docker` latest

2. **Add Secrets (optional, for private repo):**
   - Kaggle → Settings → Secrets → Add `GH_TOKEN` = GitHub PAT with `repo` scope
   - Used in Cell 1 to clone `lydorianP/samuel-realtime-parrot` (else public clone works)

3. **Attach Dataset:**
   - Upload folder of `.wav` (44100Hz or any rate, mono/stereo) as Kaggle Dataset `my-voice-wavs`
   - Or use small LibriSpeech subset (attach via Kaggle Datasets → `librispeech`)
   - Attach to notebook: `+ Add Input` → your dataset

4. **Copy `kaggle/kaggle_train.ipynb` to Kaggle** (or create new notebook and paste cells from `kaggle_train.py`)

## What the Notebook Does

- **Cell 1:** Installs `uv`, clones repo, `uv sync` (Kaggle's CUDA torch stays, our deps added). Handles private repo via `GH_TOKEN`.
- **Cell 2:** Runs `scripts/prepare_custom_dataset.py` → `manifests/custom.jsonl` + `manifests/pitch_cache/custom_spf512.npz` (correct `librosa.pyin` 70-500, 4096/512). Validates pitch cache header.
- **Cell 3:** `torchrun --standalone --nproc_per_node=2 -m samuel.train` with Hydra overrides: `batch_size 16` (T4 16GB OOM at 64 due to `pink_trombone_ola` waveguide memory), `max_steps 5000`, `warmup 500`, `eval_every 500`, `ckpt_every 1000`, `wandb_mode offline`. SSL loss `1.0` via `facebook/wav2vec2-base-960h` is **kept** (perceptual, prevents MFCC-only squeaks).
- **Cell 4:** Copies `runs/kaggle_custom_voice_ft/checkpoints/last.pt` → `/kaggle/working/samuel_custom_last.pt` for download (also `config.json`).

## After Kaggle

On local Arch Linux (ROCm host):

```bash
# 1. Download samuel_custom_last.pt from Kaggle Output
# 2. Verify
ls -lh samuel_custom_last.pt  # ~14M

# 3. Re-export ONNX for Windows WebGPU (no HIP)
uv run python scripts/export_onnx.py \
  --checkpoint samuel_custom_last.pt \
  --out models/samuel_custom_controller.onnx --verify

# 4. Swap checkpoint for realtime pipeline
# Option A: HF hub is still default; override via env:
SAMUEL_CHECKPOINT=/path/to/samuel_custom_last.pt uv run python -m samuel_realtime --out-device Samuel_Virtual_Mic

# Option B: Push to HF (so Windows machines can `hf download` without HIP):
#   rsync -a runs/kaggle_custom_voice_ft/ ... ; then `hf upload lydorianP/samuel-custom ...`
#   Then: uv run python -m samuel_realtime --checkpoint hf:lydorianP/samuel-custom --onnx models/samuel_custom_controller.onnx --provider webgpu
```

## Director's Notes (from spec)

- **Batch 16** not 64 — `pink_trombone_ola` memory heavy, 2x T4 OOM at 64. Drop to `8` if still OOM.
- **SSL 1.0** keep — MFCC-only collapses to squeaks.
- **30-60 min** varied speech minimum, else memorization ~500 steps.
- **Loss weights** `mfcc 1.0 entropy 0.1 smooth 0.3 accel 0.3 rest 0.01 ssl 1.0` are tuned; don't zero them.

## Local Test Without Kaggle

To validate `prepare_custom_dataset.py` locally on 2 fake wavs:

```bash
mkdir -p /tmp/my-voice-wavs
uv run python -c "import numpy as np, soundfile as sf; sr=44100; t=np.arange(sr*2)/sr; sf.write('/tmp/my-voice-wavs/a.wav', 0.3*np.sin(2*np.pi*220*t), sr); sf.write('/tmp/my-voice-wavs/b.wav', 0.3*np.sin(2*np.pi*180*t), sr)"
uv run python scripts/prepare_custom_dataset.py --wav-dir /tmp/my-voice-wavs --manifest manifests/test.jsonl --pitch-cache manifests/pitch_cache/test_spf512.npz --samples-per-frame 512
cat manifests/test.jsonl | head -n2
uv run python -c "import numpy as np; d=np.load('manifests/pitch_cache/test_spf512.npz'); print(list(d.files)[:5], d['n_files'], d['sample_rate'], d['samples_per_frame'])"
```

## Troubleshooting

- `pitch cache mismatch sample_rate/spf`: regenerate with same `--samples-per-frame` as training (512 for our checkpoint, 2048 for base).
- `CUDA OOM` on T4: `batch_size 8` or `batch_size 4` and `optim.max_steps 10000`.
- `facebook/wav2vec2-base-960h` download fail offline: ensure Kaggle Internet ON, or set `loss.ssl=0` (not recommended) or `log.wandb_mode=offline` still needs HF internet for first download.
- `wandb` hang: we set `offline`; upload later via `wandb sync`.

