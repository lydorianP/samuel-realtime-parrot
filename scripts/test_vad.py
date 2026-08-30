#!/usr/bin/env python3
"""VAD unit test — feeds synthetic pauses via VADProcessor.feed_wav and prints trigger times.

We generate a wav: 1.0s 220Hz tone (speech proxy), 0.6s silence, 1.0s tone, 0.6s silence, etc.
Silero's energy fallback will treat tone as speech, silence as non-speech, so we can verify
450ms threshold without needing real mic.

Usage:
  uv run python scripts/test_vad.py
  uv run python scripts/test_vad.py --wav path/to/file.wav  # real file @44100
"""
import argparse
import queue
import threading
import time

import numpy as np
import soundfile as sf

from samuel_realtime.vad import VADProcessor


def make_synthetic(sr=44100):
    # Tone = speech proxy (RMS ~0.21), silence = zeros, pattern tests 450ms trigger
    def tone(sec, f=220):
        t = np.arange(int(sr*sec), dtype=np.float32)/sr
        return (0.3*np.sin(2*np.pi*f*t)).astype(np.float32)
    def silence(sec):
        return np.zeros(int(sr*sec), dtype=np.float32)
    # Build: speech 1.0, silence 0.6 (should trigger at ~0.45 into silence), speech 0.8, silence 0.5, speech 0.3 (too short? min 250ms, 0.3 should trigger), silence 0.7
    wav = np.concatenate([
        tone(1.0), silence(0.6),
        tone(0.8), silence(0.5),
        tone(0.3), silence(0.7),
        tone(1.2), silence(1.0),
    ])
    # Expected triggers roughly at: ~1.45s, ~2.85s, ~3.65s, ~5.55s (end of each speech+silence)
    return wav

def test_synthetic():
    print("=== Synthetic VAD test (450ms silence threshold) ===")
    q = queue.Queue(maxsize=2)
    ev = threading.Event()
    # Force energy for synthetic pure-tone (Silero misclassifies sine as non-speech)
    proc = VADProcessor(audio_in_q=q, interrupt_event=ev, silence_ms=450, min_speech_ms=250, pad_ms=250, force_energy=True)
    # Force energy fallback to be deterministic (avoid torch.hub download variability)
    # If silero loaded, it will use torch; tone may be classified differently but still speech.
    # We'll just use it as is.
    wav = make_synthetic()
    print(f"wav {len(wav)/44100:.2f}s, sr 44100, feeding blocksize 512")
    # Track triggers via queue size
    triggers = []
    # Feed and poll queue
    blocksize=512
    for i in range(0, len(wav), blocksize):
        proc.process_block(wav[i:i+blocksize])
        # Drain queue
        while not q.empty():
            try:
                chunk = q.get_nowait()
                q.task_done()
                # estimate timestamp at block end
                t = (i+blocksize)/44100
                triggers.append((t, len(chunk)))
                print(f"  trigger @ ~{t:.2f}s -> chunk {len(chunk)} samples ({len(chunk)/44100:.2f}s)")
                if ev.is_set():
                    print(f"    interrupt flag set at trigger")
                    ev.clear()
            except queue.Empty:
                break
        # Also check interrupt between speech restarts
        if ev.is_set():
            t = (i+blocksize)/44100
            print(f"  interrupt @ ~{t:.2f}s (speech restart)")
            ev.clear()
    print(f"Total triggers: {len(triggers)}")
    # Expect ~4 triggers for our synthetic (4 speech segments)
    if len(triggers) >= 3:
        print("[ok] VAD triggered as expected")
    else:
        print("[warn] Too few triggers — threshold may be too high or silero model classifying tone as non-speech")
    return triggers

def test_wav_file(path):
    print(f"=== File VAD test: {path} ===")
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 44100:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=44100).astype(np.float32)
        print(f"resampled {sr}->{44100}, now {len(wav)/44100:.2f}s")
    else:
        print(f"loaded {len(wav)/44100:.2f}s @44100")
    q = queue.Queue(maxsize=10)
    ev = threading.Event()
    proc = VADProcessor(audio_in_q=q, interrupt_event=ev, silence_ms=450, min_speech_ms=250)
    triggers=[]
    for i in range(0, len(wav), 512):
        proc.process_block(wav[i:i+512])
        while not q.empty():
            try:
                chunk = q.get_nowait()
                q.task_done()
                t=(i+512)/44100
                triggers.append(t)
                print(f"  trigger @ ~{t:.2f}s chunk {len(chunk)/44100:.2f}s interrupt={ev.is_set()}")
                ev.clear()
            except queue.Empty:
                break
    print(f"Triggers: {len(triggers)} at {triggers}")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--wav", type=str, default=None)
    args=p.parse_args()
    if args.wav:
        test_wav_file(args.wav)
    else:
        test_synthetic()
        # Also test interrupt: feed speech, silence, then speech while "playing"
        print("\n=== Interrupt test ===")
        q=queue.Queue(maxsize=2)
        ev=threading.Event()
        proc = VADProcessor(audio_in_q=q, interrupt_event=ev, silence_ms=450, min_speech_ms=250, force_energy=True)
        def tone(sec,f=200):
            t=np.arange(int(44100*sec),dtype=np.float32)/44100
            return (0.3*np.sin(2*np.pi*f*t)).astype(np.float32)
        wav = np.concatenate([tone(0.8), np.zeros(int(44100*0.6),dtype=np.float32), tone(0.5)])
        # Simulate: after first trigger, we are "playing" and new speech should set interrupt
        for i in range(0, len(wav), 512):
            proc.process_block(wav[i:i+512])
            if ev.is_set():
                print(f"  interrupt set at block {(i)/44100:.2f}s (hard-cut would fire)")
                ev.clear()
            if not q.empty():
                chunk=q.get_nowait()
                print(f"  queued {len(chunk)/44100:.2f}s at {i/44100:.2f}s")
                # Now while "playing" this chunk, next speech start should interrupt
                # Continue feeding
        print("interrupt test done")

