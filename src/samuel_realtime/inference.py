"""Inference engine for Samuel realtime parrot — exact replica of server.py:_mimic.

Decouples Controller (SEANet + categorical head) from Synthesizer (pink_trombone_ola).
The Controller is ONNX-exportable; the Synth stays in PyTorch (waveguide heavy, CPU/ROCm).

Replicates server.py verbatim for fidelity:
  PYIN_FMIN=70, FMAX=500, FRAME_LENGTH=4096, hop=samples_per_frame
  fill_unvoiced interpolation, target_rms=0.05 normalization,
  warm-up tone 500ms @140Hz to JIT librosa.pyin (numba).

Usage:
  from samuel_realtime.inference import SamuelEngine
  engine = SamuelEngine(checkpoint="hf:vvolhejn/samuel")
  engine.load()
  engine.warm_up()  # ~2s on first call, then fast
  params = engine.infer_controller(wav_np)  # [T_ctrl, N_PARAMS]
  audio  = engine.synthesize(params)        # PCM float32 44100Hz
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import torch

# Ensure vendor/samuel/src is on path for imports from the submodule
_VENDOR_SRC = Path(__file__).resolve().parents[2] / "vendor" / "samuel" / "src"
if str(_VENDOR_SRC) not in sys.path:
    sys.path.insert(0, str(_VENDOR_SRC))

from samuel.model import PinkTromboneController, PinkTromboneControllerConfig  # noqa: E402
from samuel.pink_trombone import PARAM_NAMES, SAMPLE_RATE, pink_trombone_ola  # noqa: E402


def fill_unvoiced(f0: np.ndarray, voiced: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    """Linearly interpolate f0 across unvoiced runs — identical to samuel.data.fill_unvoiced."""
    n = f0.shape[0]
    if n == 0:
        return f0.astype(np.float32, copy=True)
    if not voiced.any():
        return np.full(n, 0.5 * (fmin + fmax), dtype=np.float32)
    idx = np.arange(n)
    out = np.interp(idx, idx[voiced], f0[voiced]).astype(np.float32)
    return np.clip(out, fmin, fmax)

logger = logging.getLogger(__name__)

# Constants — must match server.py exactly
PYIN_FMIN = 70.0
PYIN_FMAX = 500.0
PYIN_FRAME_LENGTH = 4096
IR_LENGTH = 256  # synth.ir_length from config (raw, not validated)

_HF_REPO_RE = re.compile(r"^hf:(?://)?(?P<repo_id>[^/@?#]+/[^/@?#]+)(?:@(?P<revision>[^/?#]+))?$")


def _resolve_hf_repo(ref: str) -> Path:
    from huggingface_hub import snapshot_download

    m = _HF_REPO_RE.match(ref)
    assert m is not None, ref
    repo_id, revision = m.group("repo_id"), m.group("revision")
    local_dir = Path(snapshot_download(repo_id, revision=revision, allow_patterns=["*.json", "*.pt"]))
    return local_dir / "checkpoints" / "last.pt"


def _resolve_config_path(checkpoint_path: Path) -> Path:
    override = os.environ.get("SAMUEL_RUN_CONFIG")
    if override:
        return Path(override)
    for cand in (
        checkpoint_path.parent.parent / "config.json",
        checkpoint_path.parent / "config.json",
    ):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"no config.json near {checkpoint_path}; set SAMUEL_RUN_CONFIG")


def pitch_track(audio: np.ndarray, samples_per_frame: int, t_ctrl: int) -> Tuple[np.ndarray, np.ndarray]:
    """Exact server.py:_pitch_track — pyin + fill_unvoiced + trim/pad to t_ctrl."""
    f0, voiced_flag, _prob = librosa.pyin(
        audio,
        fmin=PYIN_FMIN,
        fmax=PYIN_FMAX,
        sr=SAMPLE_RATE,
        frame_length=PYIN_FRAME_LENGTH,
        hop_length=samples_per_frame,
    )
    voiced = voiced_flag & np.isfinite(f0)
    f0 = np.where(np.isfinite(f0), f0, 0.0).astype(np.float32)
    if len(f0) < t_ctrl:
        pad = t_ctrl - len(f0)
        f0 = np.pad(f0, (0, pad))
        voiced = np.pad(voiced, (0, pad))
    f0, voiced = f0[:t_ctrl], voiced[:t_ctrl]
    return fill_unvoiced(f0, voiced, PYIN_FMIN, PYIN_FMAX), voiced


def rms_normalize(wav: np.ndarray, target_rms: float) -> np.ndarray:
    """server.py:_rms_normalize — scale to target_rms (0.05)."""
    rms = float(np.sqrt(np.clip((wav.astype(np.float64) ** 2).mean(), 1e-12, None)))
    return (wav * (target_rms / rms)).astype(np.float32)


class SamuelEngine:
    """Loads checkpoint, exposes controller + synth. Supports PyTorch and ONNX controllers.

    The ONNX controller is loaded via providers.get_onnx_session; the synth always
    runs as PyTorch (small, waveguide-bound, not worth ONNX). See providers.py
    for WebGPU/DirectML selection.
    """

    def __init__(self, checkpoint: str = "hf:vvolhejn/samuel", device: str | None = None):
        self.checkpoint_ref = checkpoint
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: PinkTromboneController | None = None
        self.run_cfg: dict | None = None
        self.target_rms: float = 0.05  # overwritten after config load
        self.fingerprint: str | None = None
        self.checkpoint_path: Path | None = None

    def load(self) -> PinkTromboneController:
        # Resolve checkpoint
        if _HF_REPO_RE.match(self.checkpoint_ref):
            self.checkpoint_path, _ = _resolve_hf_repo(self.checkpoint_ref), None
        elif Path(self.checkpoint_ref).exists():
            self.checkpoint_path = Path(self.checkpoint_ref)
        else:
            # Try as HF repo without prefix (fallback)
            try:
                self.checkpoint_path = _resolve_hf_repo(f"hf:{self.checkpoint_ref}")
            except Exception:
                self.checkpoint_path = Path(self.checkpoint_ref)

        cfg_path = _resolve_config_path(self.checkpoint_path)
        self.run_cfg = json.loads(cfg_path.read_text())
        model_cfg = PinkTromboneControllerConfig.model_validate(self.run_cfg["model"])
        self.target_rms = float(self.run_cfg["data"]["target_rms"])

        model = PinkTromboneController(model_cfg).to(self.device)
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        # checkpoint stores {"model": state_dict} — see server.py _load_model
        state = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state)
        model.eval()
        self.model = model
        logger.info("loaded %s config %s on %s frame_rate=%.3f target_rms=%.4f",
                    self.checkpoint_path, cfg_path, self.device, model_cfg.frame_rate, self.target_rms)
        return model

    def warm_up(self) -> None:
        """JIT-compile librosa.pyin + run one throwaway mimic so first real utterance is fast."""
        if self.model is None:
            raise RuntimeError("call load() before warm_up()")
        # 0.5s tone at 140 Hz, same as server.py:_warm_up
        t = np.arange(SAMPLE_RATE // 2, dtype=np.float32) / SAMPLE_RATE
        tone = 0.3 * np.sin(2 * np.pi * 140.0 * t, dtype=np.float32)
        try:
            self.mimic(tone)
        except Exception as e:
            logger.warning("warm-up failed (%s); first inference will be slow", e)
        else:
            logger.info("warm-up done")

    def infer_controller(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run SEANet controller on float32 audio @44100; returns (params [T_ctrl, N_PARAMS], voiced [T_ctrl])."""
        assert self.model is not None, "not loaded"
        if len(audio) < self.model.samples_per_frame:
            raise ValueError(f"audio too short: {len(audio)} < {self.model.samples_per_frame}")

        t_ctrl = self.model.t_ctrl_for(len(audio))
        f0, voiced = pitch_track(audio, self.model.samples_per_frame, t_ctrl)
        audio_in = rms_normalize(audio, self.target_rms)

        wav = torch.from_numpy(audio_in).to(self.device)[None, None, :]
        f0_t = torch.from_numpy(f0).to(self.device)[None, :]
        with torch.no_grad():
            params = self.model(wav, f0_t)  # [1, T, N_PARAMS]
        return params[0].cpu().numpy(), voiced

    def synthesize(self, params: np.ndarray) -> np.ndarray:
        """Run pink_trombone_ola synth on params [T_ctrl, N_PARAMS] -> PCM float32 @44100."""
        assert self.model is not None
        params_t = torch.from_numpy(params).to(self.device)[None, :, :]  # [1, T, P]
        with torch.no_grad():
            synth = pink_trombone_ola(
                params_t,
                seed=0,
                ir_length=IR_LENGTH,
                control_rate=self.model.config.frame_rate,
            )[0].cpu().numpy()
        return synth.astype(np.float32)

    def mimic(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Full pipeline: controller + synth. Returns (synth_audio, params, voiced).

        Mirrors server.py:_mimic but without b64 encoding — raw PCM for realtime queues.
        """
        params, voiced = self.infer_controller(audio)
        synth = self.synthesize(params)
        return synth, params, voiced

    # --- ONNX variants ---
    def infer_controller_onnx(self, session, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run controller via ONNX Runtime session (must have been exported with export_onnx.py).

        Session is expected to take (wav [1,1,S], f0 [1,T_ctrl]) and return [1, T_ctrl, N_PARAMS].
        We still compute f0 via pitch_track on CPU (librosa, not ONNX).
        """
        if self.model is None:
            raise RuntimeError("load() before infer")
        t_ctrl = self.model.t_ctrl_for(len(audio))
        f0, voiced = pitch_track(audio, self.model.samples_per_frame, t_ctrl)
        audio_in = rms_normalize(audio, self.target_rms)

        wav = audio_in[None, None, :].astype(np.float32)  # [1,1,S]
        f0_arr = f0[None, :].astype(np.float32)  # [1, T]

        # ONNX session expects float32
        ort_inputs = {"wav": wav, "f0": f0_arr}
        # Handle alternative input names (export uses wav/f0)
        # Try to map by session input names
        try:
            input_names = [i.name for i in session.get_inputs()]
            if len(input_names) == 2 and set(input_names) != {"wav", "f0"}:
                # Map positionally
                ort_inputs = {input_names[0]: wav, input_names[1]: f0_arr}
        except Exception:
            pass

        params = session.run(None, ort_inputs)[0]  # [1, T, 12]
        return params[0], voiced

    def mimic_onnx(self, session, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Full pipeline via ONNX controller + PyTorch synth."""
        params, voiced = self.infer_controller_onnx(session, audio)
        synth = self.synthesize(params)
        return synth, params, voiced
