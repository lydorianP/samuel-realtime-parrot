"""3-Thread pipeline: Capture+VAD -> Inference -> Output (hard-cut, 44.1->48k resample).

Thread A: sd.InputStream 44100 block 512 -> VADProcessor.process_block
Thread B: queue.get → SamuelEngine.mimic (or ONNX) → synth_out_q
Thread C: sd.OutputStream 48000 -> pull chunks, soxr 44.1k→48k, write 1024 blocks with interrupt check

All queues bounded=2 (phrase-level, not sample-level). Hard-cut clears synth queue and zeros 512 frames.
"""
from __future__ import annotations

import logging
import queue
import threading
import time

import numpy as np
import sounddevice as sd
import soxr

from .audio_io import get_virtual_input_name, get_virtual_output_name, list_devices, select_input_device, select_output_device
from .inference import SamuelEngine
from .vad import VADProcessor

logger = logging.getLogger(__name__)

SAMUEL_SR = 44100
OUTPUT_SR = 48000  # PipeWire/VB-CABLE default (auto-detected but we fix to 48000 for synth)


class RealtimePipeline:
    def __init__(
        self,
        engine: SamuelEngine,
        in_device: str | int | None = None,
        out_device: str | int | None = None,
        provider: str | None = None,  # "webgpu"/"dml"/"cpu"/None auto
        vad_silence_ms: int = 450,
        blocksize: int = 512,
        onnx_session=None,  # if provider is onnx, pass session
    ):
        self.engine = engine
        self.in_device = in_device
        self.out_device = out_device
        self.provider = provider
        self.onnx_session = onnx_session
        self.blocksize = blocksize
        self.vad_silence_ms = vad_silence_ms

        self.audio_in_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        self.synth_out_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        self.interrupt_event = threading.Event()

        self.vad_processor = VADProcessor(
            audio_in_q=self.audio_in_q,
            interrupt_event=self.interrupt_event,
            silence_ms=vad_silence_ms,
        )

        self.threads: list[threading.Thread] = []
        self.stop_event = threading.Event()

        # Resolve output device samplerate (virtual sinks are 48k)
        self.output_sr = OUTPUT_SR

    def _resolve_devices(self):
        # Input: smart selection (physical mic preferred over virtual monitor)
        in_dev = select_input_device(self.in_device, auto_physical=True)
        # Output: smart selection (virtual sink preferred)
        out_dev = select_output_device(self.out_device)
        if out_dev is None:
            logger.warning("No virtual output auto-detected — will use default output (may feed speakers)")
        else:
            logger.info("Auto virtual output: %s", out_dev)
        # Log enumeration
        for d in list_devices():
            logger.debug("dev [%d] %s in=%d out=%d sr=%.0f", d["index"], d["name"], d["max_input_channels"], d["max_output_channels"], d.get("default_samplerate", 0))
        return in_dev, out_dev

    # --- Thread A ---
    def _capture_thread(self, in_dev, out_dev):
        logger.info("Thread A (Capture+VAD) start: in=%s block=%d sr=%d silence=%dms", in_dev, self.blocksize, SAMUEL_SR, self.vad_silence_ms)
        try:
            with sd.InputStream(
                device=in_dev,
                samplerate=SAMUEL_SR,
                channels=1,
                blocksize=self.blocksize,
            ) as stream:
                logger.info("InputStream opened (blocking read, not callback — avoids RT malloc)")
                while not self.stop_event.is_set():
                    try:
                        data, overflowed = stream.read(self.blocksize)
                    except Exception as e:
                        logger.error("InputStream read failed: %s", e)
                        time.sleep(0.05)
                        continue
                    if overflowed:
                        logger.warning("Input overflowed: %s", overflowed)
                    block = data[:, 0] if data.ndim > 1 else data
                    try:
                        self.vad_processor.process_block(block.astype(np.float32, copy=False))
                    except Exception as e:
                        logger.error("VAD process failed: %s", e, exc_info=True)
        except Exception as e:
            logger.error("Capture thread crashed: %s", e, exc_info=True)
            self.stop_event.set()

    # --- Thread B ---
    def _inference_thread(self):
        logger.info("Thread B (Inference) start: provider %s onnx %s", self.provider, bool(self.onnx_session))
        while not self.stop_event.is_set():
            try:
                audio_chunk = self.audio_in_q.get(timeout=0.5)
            except queue.Empty:
                continue
            chunk_sec = len(audio_chunk) / SAMUEL_SR
            logger.info("Thread B got chunk %.2fs %d samples", chunk_sec, len(audio_chunk))
            start = time.perf_counter()
            try:
                if self.onnx_session is not None:
                    synth, params, voiced = self.engine.mimic_onnx(self.onnx_session, audio_chunk)
                else:
                    synth, params, voiced = self.engine.mimic(audio_chunk)
                ms = (time.perf_counter() - start) * 1000
                logger.info("Inference done: %.1f ms, params %s, synth %d samples (%.2fs)", ms, params.shape, len(synth), len(synth)/SAMUEL_SR)
                # Push to output queue (bounded)
                try:
                    self.synth_out_q.put(synth, timeout=1.0)
                except queue.Full:
                    logger.warning("synth_out_q full — dropping oldest")
                    try:
                        self.synth_out_q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self.synth_out_q.put(synth, timeout=0.5)
                    except queue.Full:
                        logger.error("synth_out_q still full — discarding synth")
            except Exception as e:
                logger.error("Inference failed: %s", e, exc_info=True)
            finally:
                self.audio_in_q.task_done()

    # --- Thread C ---
    def _output_thread(self, out_dev):
        import platform
        import shutil
        import subprocess

        logger.info("Thread C (Output) start: out=%s block 1024 sr %d→%d (resample)", out_dev, SAMUEL_SR, self.output_sr)

        # On Linux with Pulse/PipeWire, use pacat/pw-cat to avoid PortAudio concurrent Input+Output malloc bug.
        # On Windows, use sounddevice OutputStream to CABLE Input.
        use_pulse = platform.system() == "Linux" and shutil.which("pacat") is not None
        # Resolve sink name for pacat: pactl sink names are e.g. SamuelMic (not description)
        pulse_sink = None
        if use_pulse:
            # Try to map description to sink name via pactl; fallback to out_dev
            try:
                import subprocess as sp

                # out_dev may be "Samuel_Virtual_Mic" (description) or "SamuelMic" (name)
                # Try both; pacat accepts sink name
                candidates = [out_dev, "SamuelMic", "Samuel_Virtual_Mic"]
                # Query pactl sinks to find exact name
                try:
                    out = sp.check_output(["pactl", "list", "sinks", "short"], text=True)
                    for line in out.splitlines():
                        parts = line.split()
                        if len(parts) >= 2:
                            sink_name = parts[1]
                            if sink_name in candidates or any(c in line for c in candidates):
                                pulse_sink = sink_name
                                break
                    pulse_sink = pulse_sink or "SamuelMic"
                except Exception:
                    pulse_sink = "SamuelMic"
                logger.info("Using pacat pulse output to sink '%s' (from %s)", pulse_sink, out_dev)
            except Exception as e:
                logger.warning("Pulse sink resolve failed (%s), fallback to %s", e, out_dev)
                pulse_sink = out_dev or "SamuelMic"
        else:
            pulse_sink = out_dev

        if use_pulse:
            # Persistent pacat process
            def _open_pacat():
                args = [
                    "pacat",
                    "--playback",
                    f"--device={pulse_sink}",
                    f"--rate={self.output_sr}",
                    "--format=float32le",
                    "--channels=1",
                    "--raw",
                ]
                logger.info("Spawning pacat: %s", " ".join(args))
                try:
                    proc = subprocess.Popen(args, stdin=subprocess.PIPE, bufsize=0)
                    return proc
                except Exception as e:
                    logger.error("Failed to spawn pacat %s: %s", args, e)
                    return None

            pacat_proc = _open_pacat()
            if pacat_proc is None or pacat_proc.poll() is not None:
                logger.error("pacat failed to start, fallback to sounddevice")
                use_pulse = False
            else:
                try:
                    while not self.stop_event.is_set():
                        try:
                            synth = self.synth_out_q.get(timeout=0.5)
                        except queue.Empty:
                            if self.interrupt_event.is_set():
                                self.interrupt_event.clear()
                            continue

                        # Resample
                        if self.output_sr != SAMUEL_SR:
                            try:
                                synth_48 = soxr.resample(synth, SAMUEL_SR, self.output_sr)
                            except Exception:
                                import librosa

                                synth_48 = librosa.resample(synth.astype(np.float32), orig_sr=SAMUEL_SR, target_sr=self.output_sr).astype(np.float32)
                        else:
                            synth_48 = synth

                        logger.info("Playing synth %.2fs @%dHz (%d samples) via pacat", len(synth) / SAMUEL_SR, self.output_sr, len(synth_48))

                        # Hard-cut: check interrupt before each block write to pacat stdin
                        block = 1024
                        interrupted = False
                        for i in range(0, len(synth_48), block):
                            if self.interrupt_event.is_set():
                                logger.info("Hard-cut interrupt! clearing queue at %d/%d", i, len(synth_48))
                                while not self.synth_out_q.empty():
                                    try:
                                        self.synth_out_q.get_nowait()
                                        self.synth_out_q.task_done()
                                    except queue.Empty:
                                        break
                                # Write short zeros to flush pacat (10ms)
                                try:
                                    zeros = np.zeros(512, dtype=np.float32).tobytes()
                                    pacat_proc.stdin.write(zeros)
                                    pacat_proc.stdin.flush()
                                except Exception:
                                    pass
                                self.interrupt_event.clear()
                                interrupted = True
                                break
                            chunk = synth_48[i : i + block]
                            if len(chunk) < block:
                                # pad last
                                chunk = np.pad(chunk, (0, block - len(chunk)))
                            try:
                                pacat_proc.stdin.write(chunk.astype(np.float32).tobytes())
                                pacat_proc.stdin.flush()
                            except BrokenPipeError:
                                logger.warning("pacat pipe broken, restarting")
                                pacat_proc = _open_pacat()
                                if pacat_proc is None:
                                    interrupted = True
                                    break
                                try:
                                    pacat_proc.stdin.write(chunk.astype(np.float32).tobytes())
                                except Exception as e:
                                    logger.error("pacat write failed after restart: %s", e)
                                    interrupted = True
                                    break
                            except Exception as e:
                                logger.error("pacat write failed: %s", e)
                                interrupted = True
                                break
                            # Yield to allow interrupt
                            time.sleep(0.001)

                        self.synth_out_q.task_done()
                        if interrupted:
                            logger.info("Output interrupted (pacat), waiting next phrase")
                        else:
                            logger.info("Output finished phrase (pacat)")

                        # Check pacat health
                        if pacat_proc.poll() is not None:
                            logger.warning("pacat died (code %s), restarting", pacat_proc.returncode)
                            pacat_proc = _open_pacat()
                            if pacat_proc is None:
                                use_pulse = False
                                break
                    # Cleanup pacat
                    try:
                        if pacat_proc and pacat_proc.poll() is None:
                            pacat_proc.stdin.close()
                            pacat_proc.terminate()
                            pacat_proc.wait(timeout=1)
                    except Exception:
                        pass
                except Exception as e:
                    logger.error("Output thread (pacat) crashed: %s", e, exc_info=True)
                    self.stop_event.set()
                # If we fell back, continue to sounddevice path below
                if use_pulse:
                    return

        # Fallback / Windows path: sounddevice OutputStream
        try:
            with sd.OutputStream(
                device=out_dev,
                samplerate=self.output_sr,
                channels=1,
                blocksize=1024,
            ) as stream:
                logger.info("OutputStream (sounddevice) opened on %s", out_dev)
                while not self.stop_event.is_set():
                    try:
                        synth = self.synth_out_q.get(timeout=0.5)
                    except queue.Empty:
                        if self.interrupt_event.is_set():
                            self.interrupt_event.clear()
                        continue

                    if self.output_sr != SAMUEL_SR:
                        try:
                            synth_48 = soxr.resample(synth, SAMUEL_SR, self.output_sr)
                        except Exception:
                            import librosa

                            synth_48 = librosa.resample(synth.astype(np.float32), orig_sr=SAMUEL_SR, target_sr=self.output_sr).astype(np.float32)
                    else:
                        synth_48 = synth

                    logger.info("Playing synth %.2fs @%dHz (%d samples) via sounddevice", len(synth) / SAMUEL_SR, self.output_sr, len(synth_48))

                    block = 1024
                    interrupted = False
                    for i in range(0, len(synth_48), block):
                        if self.interrupt_event.is_set():
                            logger.info("Hard-cut interrupt! clearing queue at %d/%d", i, len(synth_48))
                            while not self.synth_out_q.empty():
                                try:
                                    self.synth_out_q.get_nowait()
                                    self.synth_out_q.task_done()
                                except queue.Empty:
                                    break
                            try:
                                zeros = np.zeros(512, dtype=np.float32)
                                stream.write(zeros)
                            except Exception:
                                pass
                            self.interrupt_event.clear()
                            interrupted = True
                            break
                        chunk = synth_48[i : i + block]
                        if len(chunk) < block:
                            pad = np.zeros(block - len(chunk), dtype=np.float32)
                            chunk = np.concatenate([chunk, pad])
                            stream.write(chunk)
                            break
                        stream.write(chunk)

                    self.synth_out_q.task_done()
                    if interrupted:
                        logger.info("Output interrupted, waiting next phrase")
                    else:
                        logger.info("Output finished phrase")
        except Exception as e:
            logger.error("Output thread crashed: %s", e, exc_info=True)
            self.stop_event.set()

    def start(self):
        in_dev, out_dev = self._resolve_devices()
        # Spawn threads
        tA = threading.Thread(target=self._capture_thread, args=(in_dev, out_dev), daemon=True, name="capture")
        tB = threading.Thread(target=self._inference_thread, daemon=True, name="inference")
        tC = threading.Thread(target=self._output_thread, args=(out_dev,), daemon=True, name="output")
        self.threads = [tA, tB, tC]
        for t in self.threads:
            t.start()
        logger.info("All threads started (capture/inference/output)")

    def stop(self):
        logger.info("Stopping pipeline...")
        self.stop_event.set()
        # Clear queues
        for q in [self.audio_in_q, self.synth_out_q]:
            while not q.empty():
                try:
                    q.get_nowait()
                    q.task_done()
                except queue.Empty:
                    break
        for t in self.threads:
            t.join(timeout=1.0)
        logger.info("Pipeline stopped")

    def run_forever(self):
        try:
            while not self.stop_event.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — stopping")
        finally:
            self.stop()
