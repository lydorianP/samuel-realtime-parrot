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

## Phase 1 — Audio Routing & I/O

### Linux (PipeWire/PulseAudio)

```bash
./scripts/setup_virtual_mic.sh          # creates null-sink SamuelMic (idempotent)
uv run python -m samuel_realtime.audio_io --list   # verify device 15 Samuel_Virtual_Mic
uv run python -m samuel_realtime.audio_io --test   # 440Hz tone to virtual sink (monitor via pavucontrol)
# or explicit
uv run python src/samuel_realtime/audio_io.py --list
```

Module: `src/samuel_realtime/audio_io.py` — cross-platform enumeration:
- Windows sigs: `CABLE Input` (playback) / `CABLE Output` (recording) via VB-Audio
- Linux sigs: `Samuel_Virtual_Mic` / `SamuelMic` null-sink, monitor source auto-detected
- Auto samplerate from `default_samplerate` (PipeWire sink 48000, model 44100 needs resample in Thread C)

Routing:
- Python `OutputStream(device="Samuel_Virtual_Mic")` → virtual speaker
- Discord/Zoom input → `"Monitor of Samuel_Virtual_Mic"`

Persistence: add `load-module module-null-sink sink_name=SamuelMic ...` to `~/.config/pulse/default.pa`

### Windows (VB-CABLE)

VB-CABLE requires admin install + reboot — not automatable. Verify:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\check_vbcable.ps1
# expects PnP "VB-Audio" + MMDevices "CABLE Input/Output"
# Install from https://vb-audio.com/Cable/ if missing
```

Wine note: checker shows no devices under Wine/vkd3d — expected, use native Windows for final.

Dependencies verified: `sounddevice 0.5.6`, `numpy 2.5.2`, `scipy 1.18.1` (already in `uv.lock`).

## Phase 2 — Inference Engine & ONNX Export

### Controller vs Synth split

- **Controller** (`PinkTromboneController` SEANet 512 spf, 345 frames @4s, 12 params): matmul-bound, exported to ONNX `opset 17` (18 fallback) for WebGPU/DirectML
- **Synth** (`pink_trombone_ola` waveguide, `ir_length 256`): stays PyTorch — runs on same device as controller (ROCm 37ms vs CPU 2663ms after warm, so GPU synth wins)

### Replicated server.py exactly

- `librosa.pyin fmin 70 fmax 500 frame 4096 hop=samples_per_frame` → `fill_unvoiced` clamp → trim/pad to `T_ctrl`
- `rms_normalize target_rms 0.05` (`config.json["data"]["target_rms"]`)
- `warm_up` 500ms 140Hz tone JITs `numba` — first real mimic without warm is ~2s, with warm ~150ms for 2s audio

```bash
uv run python scripts/export_onnx.py --out models/samuel_controller.onnx --verify
# -> models/samuel_controller.onnx 395KB + .data 14MB, opset 18 (17 requested, auto-convert failed for Pad)
# -> onnx.checker ok, ORT CPU median 103ms vs torch ROCm 88ms for 2s, max diff 0.0

uv run python scripts/bench_controller.py   # full benchmark (see numbers below)
```

`src/samuel_realtime/inference.py` — `SamuelEngine(checkpoint="hf:vvolhejn/samuel")` with `load()/warm_up()/mimic()/infer_controller()/synthesize()` + ONNX variants `infer_controller_onnx()`.

`src/samuel_realtime/providers.py` — factory `select_providers(request)` enforces **No HIP on Windows** (`Windows + MIGraphX → RuntimeError`), auto: `Linux → WebGPU→CPU`, `Windows → WebGPU→Dml→CPU`. `create_session()` wraps `ort.InferenceSession`.

Benchmark on `RX 9070 XT gfx1201 ROCm 7.14` (`torch 2.12`):

- 2s: pitch 77ms + controller ROCm 88ms + synth CUDA 37ms → full `mimic` **152ms median** (208ms first), 4s: ~2100ms total (345 frames) still realtime
- ONNX CPU: controller 103ms, synth 45ms, parity `max abs diff 0.0`, dynamic shapes `0.7s/3.3s` ok
- WebGPU not in `pip onnxruntime 1.29` (`['Azure','CPU']` only) — expected, needs plugin/`onnxruntime-directml` on Windows or `WebGPU` plugin build; CPU fallback is correct per spec.

## Phase 3 — VAD & 3-Thread Pipeline

### Silero VAD + Ring Buffer (`src/samuel_realtime/vad.py`)

- **Resample**: 44.1k mic → 16k via `soxr` (1.1.0, fast) for VAD; raw 44.1k kept in 5s ring for Samuel
- **State machine**: `speech>250ms` + `silence>450ms` → slice ring `max(1.2,min(speech+0.3,4.0))` + `250ms` pad → `audio_in_q` (bounded 2, drops oldest). Debounce via blocks (25 blocks ≈0.3s synthetic & realtime).
- **Interrupt**: speech start after silence → `interrupt_event.set()` (hard-cut)
- **Silero**: `torch.hub snakers4/silero-vad` on CPU, fallback energy `RMS>0.015` if `torchaudio` missing; `force_energy=True` for deterministic synthetic tests. `torchaudio 2.11+rocm7.14.0` via AMD wheel for Silero on ROCm host.

### 3-Thread Pipeline (`src/samuel_realtime/pipeline.py`)

- **Thread A (Capture)**: `sd.InputStream 44100 block 512` **blocking read** (not callback — avoids RT malloc with HIP/MIOpen) → `VADProcessor.process_block`
- **Thread B (Inference)**: `audio_in_q.get()` → `engine.mimic()` (or `mimic_onnx` with `providers.py`) → `synth_out_q` (44.1k PCM)
- **Thread C (Output)**: **Linux**: `pacat --playback --device=SamuelMic --rate=48000 --format=float32le --channels=1 --raw` (persistent subprocess, avoids PortAudio concurrent Input+Output `malloc` bug on PipeWire). **Windows**: `sd.OutputStream(device="CABLE Input" 48000)`. Both do `soxr 44.1→48k` resample, write `1024` blocks, check `interrupt_event` before each block → clear `synth_out_q`, write `512` zeros, break (hard-cut).

Why `pacat` on Linux? Concurrent `sd.InputStream` + `sd.OutputStream` on PipeWire triggers `malloc(): invalid size (unsorted)` in PortAudio (reproduced without torch). `pacat` via Pulse/PipeWire native avoids it; verified concurrent `InputStream 48k` + `pacat` OK.

### CLI (`src/samuel_realtime/__main__.py`)

```bash
uv run python -m samuel_realtime --help
uv run python -m samuel_realtime --list-devices          # 16 devices, 15 Samuel_Virtual_Mic 2ch 48000
uv run python -m samuel_realtime --out-device Samuel_Virtual_Mic --vad-silence 0.45
# Windows:
python -m samuel_realtime --out-device "CABLE Input" --onnx models\samuel_controller.onnx --provider webgpu
```

Flags: `--in-device/--out-device` (name/index, auto-detect), `--provider auto|webgpu|dml|migraphx|cpu` (enforces No HIP), `--vad-silence 0.45`, `--checkpoint hf:vvolhejn/samuel`, `--onnx`, `--log-level`.

Engine warm-up now multi-shape `0.5,1.0,1.5,2.0,4.0` to cache torch graphs (first `1.5s` after single-shape warm was `517ms` vs `109ms` cached).

### Verification (Phase 3)

```bash
uv run python scripts/test_vad.py          # synthetic 6.1s → 4 triggers @1.46,2.86,3.66,5.56 (energy forced)
uv run python scripts/test_resample.py     # 440Hz→48k preserves ~436Hz (chipmunk would be 478), hard-cut clears queue
uv run python scripts/test_pipeline_hello.py  # mimic 1.5s 109ms median after warm, CLI --help/list ok
bash scripts/wine_test.sh                  # native --help/list ok, Wine python manual install note, HIP block verified
timeout 20 uv run python -m samuel_realtime --out-device Samuel_Virtual_Mic
# → warm 7.3s, Silero loaded, InputStream opened (blocking read), pacat spawned, stable 20s no malloc
```

Resample check `soxr 44.1→48k` preserves duration `1.51s→1.51s`, no chipmunk. Barge-in verified via `interrupt_event` + `synth_out_q` clear.

Wine: `wine-11.16` native smoke ok, `python.exe` not in Wine prefix (manual install), DirectML/WebGPU expected CPU fallback under Wine.

Launch: `uv run python -m samuel_realtime --out-device Samuel_Virtual_Mic --vad-silence 0.45` → speak “Hello Samuel”, parrot after ~450ms.

```

