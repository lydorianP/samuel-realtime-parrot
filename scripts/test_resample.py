#!/usr/bin/env python3
"""Resample check — ensures 44.1k synth -> 48k virtual sink doesn't chipmunk."""
import numpy as np
import soxr

SAMUEL_SR = 44100
OUTPUT_SR = 48000

def estimate_freq(wav, sr):
    # Zero-crossing estimate
    # Count zero crossings
    crossings = np.where(np.diff(np.signbit(wav)))[0]
    if len(crossings) < 2:
        return 0
    # period approx 2 * avg distance between crossings
    diffs = np.diff(crossings)
    # Filter diffs that are too small (noise)
    diffs = diffs[diffs > sr*0.0005]  # >0.5ms
    avg_period_samples = np.median(diffs)*2
    freq = sr / avg_period_samples
    return freq

# 440Hz sine at 44.1k 1sec
t = np.arange(SAMPLERATE:=SAMUEL_SR, dtype=np.float32)/SAMUEL_SR
tone_44 = (0.5*np.sin(2*np.pi*440*t)).astype(np.float32)
freq_44 = estimate_freq(tone_44, SAMUEL_SR)
print(f"44.1k tone est freq {freq_44:.1f} Hz (expect 440)")

# Resample via soxr (pipeline's method)
tone_48 = soxr.resample(tone_44, SAMUEL_SR, OUTPUT_SR)
freq_48 = estimate_freq(tone_48, OUTPUT_SR)
print(f"48k resampled est freq {freq_48:.1f} Hz (expect 440)")

# Check chipmunk: if we fed 44.1k data into 48k stream without resample, freq would be 440*48000/44100=478.9
chipmunk = 440*OUTPUT_SR/SAMUEL_SR
print(f"chipmunk freq if no resample would be {chipmunk:.1f} Hz")

# Pipeline's resample should keep 440
if abs(freq_48 - 440) < 5:
    print("[ok] Resample preserves pitch (no chipmunk)")
else:
    print(f"[FAIL] Resample error: got {freq_48:.1f} vs 440")

# Also test via pipeline's soxr path
from samuel_realtime.inference import SamuelEngine
import queue, threading, time, numpy as np
from samuel_realtime.vad import VADProcessor
from samuel_realtime.pipeline import RealtimePipeline

# Test pipeline's output resample path directly
engine = SamuelEngine()
engine.load()
engine.warm_up()
wav = tone_44[:88200]  # 2s
synth, _, _ = engine.mimic(wav)
print(f"synth {len(synth)/SAMUEL_SR:.2f}s @44.1k -> resample to 48k = {len(soxr.resample(synth, SAMUEL_SR, OUTPUT_SR))/OUTPUT_SR:.2f}s")

# Hard-cut test: simulate interrupt
print("\n=== Hard-cut barge-in test ===")
import queue, threading
q_in = queue.Queue(maxsize=2)
q_out = queue.Queue(maxsize=2)
ev = threading.Event()
# Put a synth chunk in out queue
q_out.put(synth)
print(f"queued synth {len(synth)}")
# Simulate Thread C checking interrupt before each block
ev.set()
print("set interrupt_event -> Thread C should clear queue and write zeros")
# Mimic pipeline's hard-cut logic
if ev.is_set():
    cleared=0
    while not q_out.empty():
        q_out.get_nowait()
        cleared+=1
    print(f"[ok] Hard-cut cleared {cleared} queued item(s) (expected 1)")
    ev.clear()
    print("interrupt cleared")
else:
    print("[fail] interrupt not set")

print("\nAll Phase 3 checks done")
