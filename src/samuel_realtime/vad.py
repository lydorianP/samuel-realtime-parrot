"""Silero VAD processor with 5s 44.1k ring buffer, 450ms low-latency trigger, hard-cut interrupt.

Silero strict expects 16k mono; input is 44.1k blocks from sd.InputStream(44100, block 512).
We resample each block to 16k via soxr (fast) for VAD, but keep raw 44.1k in ring for Samuel.

State machine mirrors spec:
  speech_duration >250ms + silence_duration >450ms → slice ring (pad 250ms) → queue
  speech start after silence → set interrupt_event (hard-cut in Thread C)

Uses torch.hub Silero VAD (snakers4/silero-vad) with CPU. Falls back to energy VAD offline.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import soxr
import torch

logger = logging.getLogger(__name__)

# Samuel-native rate
SAMUEL_SR = 44100
VAD_SR = 16000

# Defaults per spec: 450ms silence trigger, 250ms min speech, 250ms pad
DEFAULT_SILENCE_MS = 450
DEFAULT_MIN_SPEECH_MS = 250
DEFAULT_PAD_MS = 250
RING_SECONDS = 5.0


class SileroVAD:
    """Thin wrapper around torch.hub Silero VAD, with energy fallback."""

    def __init__(self, threshold: float = 0.5, device: str = "cpu"):
        self.threshold = threshold
        self.device = torch.device(device)
        self.model = None
        self.use_torch = False
        self._load()

    def _load(self):
        # energy fallback params always available
        self.energy_thresh = 0.015
        try:
            # torch.hub will download ~2M model on first run (~1s), cached in ~/.cache/torch/hub
            self.model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
                trust_repo=True,
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            # utils contains get_speech_timestamps etc, not needed
            self.use_torch = True
            logger.info("Silero VAD loaded via torch.hub on %s", self.device)
        except Exception as e:
            logger.warning("Silero VAD torch.hub failed (%s) — falling back to energy VAD", e)
            self.use_torch = False

    @torch.no_grad()
    def is_speech(self, chunk16k: np.ndarray) -> tuple[bool, float]:
        """Return (is_speech, prob) for 16k mono chunk (numpy float32, ~512-1024 samples).

        Chunk should be ~30-100ms; shorter is ok but noisier.
        Silero expects torch float32 Tensor of shape [1, N] at 16k.
        """
        if chunk16k.size == 0:
            return False, 0.0
        # Ensure float32 normed [-1,1]
        x = chunk16k.astype(np.float32, copy=False)
        if self.use_torch and self.model is not None:
            try:
                t = torch.from_numpy(x).to(self.device)
                if t.ndim == 1:
                    t = t.unsqueeze(0)
                # Silero expects at least 512 samples; pad if tiny
                if t.shape[1] < 512:
                    pad = 512 - t.shape[1]
                    t = torch.nn.functional.pad(t, (0, pad))
                prob = self.model(t, VAD_SR).item()  # scalar 0..1
                return prob > self.threshold, float(prob)
            except Exception as e:
                logger.debug("Silero inference failed (%s) fallback energy", e)
                # fall through to energy

        # Energy fallback
        rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))
        prob = min(1.0, rms / 0.05)  # map 0.05 rms -> 1.0
        return rms > self.energy_thresh, prob


class VADProcessor:
    """Feeds 44.1k blocks, maintains ring, triggers queue on silence."""

    def __init__(
        self,
        audio_in_q: queue.Queue,
        interrupt_event: threading.Event,
        silence_ms: int = DEFAULT_SILENCE_MS,
        min_speech_ms: int = DEFAULT_MIN_SPEECH_MS,
        pad_ms: int = DEFAULT_PAD_MS,
        vad_threshold: float = 0.5,
        ring_seconds: float = RING_SECONDS,
        force_energy: bool = False,
    ):
        self.audio_in_q = audio_in_q
        self.interrupt_event = interrupt_event
        self.silence_thresh_s = silence_ms / 1000.0
        self.min_speech_s = min_speech_ms / 1000.0
        self.pad_samples = int(pad_ms / 1000.0 * SAMUEL_SR)
        self.ring_max = int(ring_seconds * SAMUEL_SR)

        self.vad = SileroVAD(threshold=vad_threshold)
        if force_energy:
            self.vad.use_torch = False
            logger.info("VAD force_energy=True — using energy fallback for deterministic tests")

        # Ring as deque of samples? For simplicity keep numpy growing and trim
        self.ring = np.zeros(0, dtype=np.float32)
        self.ring_lock = threading.Lock()

        # State
        self.speech_duration = 0.0
        self.silence_duration = 0.0
        self.in_speech = False
        self.last_trigger_block = -100000
        # For interrupt: track if we were in silence and now speech restarts
        self._prev_is_speech = False

        # Stats for debugging
        self.blocks_processed = 0

    def _resample_to_16k(self, block44k: np.ndarray) -> np.ndarray:
        # soxr resample 44100->16000 fast; block is 512 @44.1k -> ~186 @16k
        if block44k.size == 0:
            return block44k
        # soxr expects float32/64, sr in/out
        # Use soxr.resample for arbitrary rates, quality VQ (fast)
        try:
            out = soxr.resample(block44k, SAMUEL_SR, VAD_SR)
            return out.astype(np.float32, copy=False)
        except Exception:
            # fallback librosa (slower)
            import librosa

            return librosa.resample(block44k.astype(np.float32), orig_sr=SAMUEL_SR, target_sr=VAD_SR).astype(np.float32)

    def process_block(self, block44k: np.ndarray):
        """Call from Thread A's audio callback with raw 44.1k mono block (512)."""
        if block44k.ndim > 1:
            block44k = block44k.mean(axis=1)
        block44k = block44k.astype(np.float32, copy=False)

        # Append to ring (thread-safe)
        with self.ring_lock:
            self.ring = np.concatenate([self.ring, block44k])
            if len(self.ring) > self.ring_max:
                self.ring = self.ring[-self.ring_max :]

        # VAD on 16k version
        chunk16k = self._resample_to_16k(block44k)
        is_speech, prob = self.vad.is_speech(chunk16k)

        # Time of this block at 44.1k
        block_dur = len(block44k) / SAMUEL_SR  # ~0.0116s for 512
        self.blocks_processed += 1

        # State machine
        if is_speech:
            self.speech_duration += block_dur
            self.silence_duration = 0.0
            self.in_speech = True
            # Interrupt: speech just started after being in silence
            if not self._prev_is_speech and self.blocks_processed > 10:
                # Only signal interrupt if we've had at least ~100ms of prior silence history
                # Avoid spurious at startup
                self.interrupt_event.set()
                logger.debug("VAD speech start -> interrupt set (prob %.2f)", prob)
            self._prev_is_speech = True
        else:
            self.silence_duration += block_dur
            self._prev_is_speech = False
            # Stay in speech state until silence threshold crosses; don't reset speech_duration yet

        # Trigger condition: had enough speech, now enough silence
        if self.speech_duration >= self.min_speech_s and self.silence_duration >= self.silence_thresh_s:
            # Debounce based on blocks to work for both realtime (11.6ms/block) and synthetic fast-feed
            # 0.3s -> ~26 blocks at 512/44100; use 25 blocks
            if self.blocks_processed - self.last_trigger_block < 25:
                return
            self.last_trigger_block = self.blocks_processed

            # Slice ring: we want last (speech_duration + silence + pad) worth
            # Simpler: take last max(1.2s, speech_duration+0.3) + pad
            with self.ring_lock:
                # Estimate samples to keep: speech + 0.3s tail + pad*2
                keep_s = self.speech_duration + 0.3
                keep_s = max(1.2, min(keep_s, 4.0))  # clamp 1.2–4.0 as spec
                keep_samples = int(keep_s * SAMUEL_SR) + 2 * self.pad_samples
                keep_samples = min(keep_samples, len(self.ring))
                # Slice from ring end, with pad already included via keep
                start = len(self.ring) - keep_samples
                # Add pad at end already, ensure start with pad at beginning
                snippet = self.ring[start:].copy()
                # Trim leading/trailing silence slightly already via VAD, but keep pad

            # Normalize snippet peak to avoid clipping later? No, inference does rms_normalize, but we keep raw
            # Ensure at least 0.5s
            if len(snippet) < SAMUEL_SR * 0.5:
                logger.debug("VAD snippet too short %d — skipping", len(snippet))
                self.speech_duration = 0.0
                self.silence_duration = 0.0
                return

            # Push to queue non-blocking, drop oldest if full (bounded 2 per spec)
            try:
                self.audio_in_q.put_nowait(snippet)
                logger.info("VAD trigger: speech %.2fs silence %.2fs -> queued %d samples (%.2fs)", self.speech_duration, self.silence_duration, len(snippet), len(snippet)/SAMUEL_SR)
            except queue.Full:
                # Drop oldest and retry (hard-cut style — keep latest phrase)
                try:
                    self.audio_in_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.audio_in_q.put_nowait(snippet)
                    logger.warning("VAD queue full — dropped oldest, queued latest")
                except queue.Full:
                    logger.error("VAD queue still full after drop — discarding snippet")

            # Reset state for next phrase
            self.speech_duration = 0.0
            self.silence_duration = 0.0
            self.in_speech = False
            self._prev_is_speech = False

    def feed_wav(self, wav: np.ndarray, blocksize: int = 512):
        """Helper for unit tests: feed a whole wav through process_block chunked."""
        for i in range(0, len(wav), blocksize):
            self.process_block(wav[i : i + blocksize])
