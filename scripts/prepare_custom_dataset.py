#!/usr/bin/env python3
"""Prepare custom voice dataset for Samuel fine-tuning — manifests + pitch cache.

Correctly replicates vendor/samuel/scripts/precompute_pitch.py logic:
  pyin 70-500Hz, frame 4096 hop=SPF, voiced_flag & isfinite, then fill_unvoiced is done at train time.
  Cache stores raw f0/voiced per file for train.py to slice per-chunk and fill.

Usage:
  uv run python scripts/prepare_custom_dataset.py \
    --wav-dir /path/to/wavs \
    --manifest manifests/custom.jsonl \
    --pitch-cache manifests/pitch_cache/custom_spf512.npz \
    --sample-rate 44100 --samples-per-frame 512

Input: folder of .wav (any rate, mono/stereo). Output: manifest + pitch cache.
Same as Kaggle Cell 2 but fixed: correct pitch cache shape, sample_rate handling, mono averaging.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


def compute_pitch(audio: np.ndarray, sr: int, spf: int, fmin=70.0, fmax=500.0, frame_length=4096):
    f0, voiced_flag, _ = librosa.pyin(audio, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame_length, hop_length=spf)
    voiced = voiced_flag & np.isfinite(f0)
    f0 = np.where(np.isfinite(f0), f0, 0.0).astype(np.float32)
    return f0, voiced.astype(bool)


def main():
    ap = argparse.ArgumentParser(description="Prepare custom dataset for Samuel (manifest + pitch cache)")
    ap.add_argument("--wav-dir", type=Path, required=True, help="Folder of .wav files")
    ap.add_argument("--manifest", type=Path, default=Path("manifests/custom.jsonl"))
    ap.add_argument("--pitch-cache", type=Path, default=Path("manifests/pitch_cache/custom_spf512.npz"))
    ap.add_argument("--sample-rate", type=int, default=44100)
    ap.add_argument("--samples-per-frame", type=int, default=512, help="Must match model param (512 for vvolhejn/samuel, 2048 for base)")
    ap.add_argument("--fmin", type=float, default=70.0)
    ap.add_argument("--fmax", type=float, default=500.0)
    ap.add_argument("--frame-length", type=int, default=4096)
    args = ap.parse_args()

    wav_paths = sorted(args.wav_dir.glob("*.wav")) + sorted(args.wav_dir.glob("*.WAV"))
    if not wav_paths:
        # also try nested
        wav_paths = sorted(args.wav_dir.rglob("*.wav"))
    if not wav_paths:
        raise SystemExit(f"No .wav found in {args.wav_dir}")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.pitch_cache.parent.mkdir(parents=True, exist_ok=True)

    out = {}
    manifest_entries = []
    for i, wav_path in enumerate(tqdm(wav_paths, desc="preparing")):
        try:
            audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        except Exception as e:
            print(f"skip {wav_path}: {e}")
            out[f"f0_{i}"] = np.zeros(0, dtype=np.float32)
            out[f"voiced_{i}"] = np.zeros(0, dtype=bool)
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != args.sample_rate:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=args.sample_rate).astype(np.float32)
        # Manifest entry — use absolute path for train.py's _load_resampled (it resolves relative to repo root)
        try:
            rel = wav_path.resolve().relative_to(Path.cwd())
            path_str = str(wav_path.resolve())
        except ValueError:
            path_str = str(wav_path.resolve())
        manifest_entries.append({"path": path_str, "duration": len(audio)/args.sample_rate, "sample_rate": args.sample_rate})

        f0, voiced = compute_pitch(audio, args.sample_rate, args.samples_per_frame, args.fmin, args.fmax, args.frame_length)
        out[f"f0_{i}"] = f0
        out[f"voiced_{i}"] = voiced

    # Manifest
    with open(args.manifest, "w") as f:
        for e in manifest_entries:
            f.write(json.dumps(e) + "\n")

    # Pitch cache header
    out["sample_rate"] = np.array(args.sample_rate)
    out["samples_per_frame"] = np.array(args.samples_per_frame)
    out["control_rate"] = np.array(args.sample_rate / args.samples_per_frame, dtype=np.float64)
    out["pyin_fmin"] = np.array(args.fmin, dtype=np.float64)
    out["pyin_fmax"] = np.array(args.fmax, dtype=np.float64)
    out["pyin_frame_length"] = np.array(args.frame_length)
    out["n_files"] = np.array(len(wav_paths))

    np.savez_compressed(args.pitch_cache, **out)
    print(f"saved {args.manifest} ({len(manifest_entries)} files)")
    print(f"saved {args.pitch_cache} (sr={args.sample_rate}, spf={args.samples_per_frame}, cr={args.sample_rate/args.samples_per_frame:.2f}Hz)")
    print(f"Upload both to Kaggle or use locally: --manifest {args.manifest} --pitch-cache {args.pitch_cache}")


if __name__ == "__main__":
    main()
