# Samuel Realtime Parrot

Real-time phrase-by-phrase parroting pipeline using `vvolhejn/samuel` + Pink Trombone vocal tract simulator.

**Target:** Windows (VB-CABLE) primary, Linux (PipeWire Null-Sink) dev
**Inference:** ONNX Runtime WebGPU (Vulkan on Linux via Dawn, DirectML/DX12 on Windows) — no HIP on Windows. ROCm for training on AMD RX 9070 XT (gfx1201).
**VAD:** Silero, 450ms silence threshold (hard-cut barge-in)
**Env:** `uv` pinned Python 3.12

## Quickstart (Linux dev, Arch + ROCm 7.14)

```bash
# ROCm torch (gfx1201)
uv pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "torch[device-gfx1201]==2.12.0+rocm7.14.0"
uv pip install --no-deps julius  # avoids pypi torch pull

# Checkpoint (15M, cached)
hf download vvolhejn/samuel
# -> ~/.cache/huggingface/hub/models--vvolhejn--samuel/snapshots/77336e1e.../checkpoints/last.pt
```

## Hardware

- Dev: AMD Radeon RX 9070 XT (RADV GFX1201, Vulkan 1.4.354) — `ROCm 7.14.60850`, `torch 2.12.0+rocm7.14.0` verified `cuda.is_available() True`
- Training: Kaggle 2xT4 (no TPU — SEANet causal Conv1d poor on XLA)

## Repo Layout (Phase 0)

- `vendor/samuel` — git submodule `vvolhejn/samuel` (source for `pink_trombone.py` 1082 lines, `model.py`, `server.py`)
- `src/samuel_realtime_parrot/` — library skeleton (uv --lib)
- `pyproject.toml` / `uv.lock` — base deps (torch is external ROCm wheel, not in lock)
```

