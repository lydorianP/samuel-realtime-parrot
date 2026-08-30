"""ONNX Runtime provider factory — enforces No HIP on Windows, WebGPU preferred.

Policy:
  Linux dev (ROCm present):  prefer WebGPU (Vulkan via Dawn) -> CPU
                             optional MIGraphX if explicitly requested and not on Windows
  Windows (VB-CABLE target): WebGPU (Dawn → DirectML/DX12) -> DmlExecutionProvider -> CPU
                             NEVER ROCm / MIGraphX (HIP forbidden)

  Fallback to CPU is always required (per spec).

Provider strings (ORT 1.29+):
  WebGPUExecutionProvider  -> "WebGPU" (plugin, dawn)
  DmlExecutionProvider     -> "DmlExecutionProvider" (onnxruntime-directml)
  MIGraphXExecutionProvider -> "MIGraphXExecutionProvider" (rocm)
  CPUExecutionProvider     -> "CPUExecutionProvider"
"""
from __future__ import annotations

import logging
import platform

import onnxruntime as ort

logger = logging.getLogger(__name__)


def available_providers() -> list[str]:
    return ort.get_available_providers()


def select_providers(requested: str | None = None) -> list[str]:
    """Return ordered provider list per policy.

    requested: explicit override "webgpu" | "dml" | "migraphx" | "cpu" | None (auto).
    Auto respects OS and enforces No HIP on Windows.
    """
    system = platform.system()
    avail = available_providers()
    logger.info("ORT available providers: %s on %s", avail, system)

    def has(p: str) -> bool:
        return p in avail

    if requested:
        r = requested.lower()
        if r == "webgpu":
            # "WebGPU" is the canonic name; some builds expose "WebGPUExecutionProvider"
            for cand in ["WebGPUExecutionProvider", "WebGPU"]:
                if has(cand):
                    return [cand, "CPUExecutionProvider"]
            logger.warning("WebGPU requested but not available; falling back to CPU")
            return ["CPUExecutionProvider"]
        if r == "dml":
            if has("DmlExecutionProvider"):
                return ["DmlExecutionProvider", "CPUExecutionProvider"]
            logger.warning("DML requested but not available; fallback CPU")
            return ["CPUExecutionProvider"]
        if r == "migraphx":
            if system == "Windows":
                raise RuntimeError("MIGraphX/ROCm is forbidden on Windows per spec (NO HIP)")
            if has("MIGraphXExecutionProvider"):
                return ["MIGraphXExecutionProvider", "CPUExecutionProvider"]
            logger.warning("MIGraphX requested but not available; fallback CPU")
            return ["CPUExecutionProvider"]
        if r == "cpu":
            return ["CPUExecutionProvider"]
        raise ValueError(f"unknown provider request: {requested}")

    # Auto
    if system == "Windows":
        # Prefer WebGPU (which on Windows routes via DirectML internally), else DML, else CPU
        for cand in ["WebGPUExecutionProvider", "WebGPU"]:
            if has(cand):
                return [cand, "CPUExecutionProvider"]
        if has("DmlExecutionProvider"):
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
    else:
        # Linux
        for cand in ["WebGPUExecutionProvider", "WebGPU"]:
            if has(cand):
                return [cand, "CPUExecutionProvider"]
        # MIGraphX is allowed on Linux but not preferred by default (WebGPU already covers Vulkan)
        # Keep CPU as default fallback; user can request migraphx explicitly if they want ROCm graph
        return ["CPUExecutionProvider"]


def create_session(onnx_path: str, requested: str | None = None, **sess_options) -> ort.InferenceSession:
    """Create ORT session with selected providers, always with CPU fallback."""
    providers = select_providers(requested)
    logger.info("Creating ORT session %s with providers %s", onnx_path, providers)
    # Enforce No HIP on Windows at session creation time
    if platform.system() == "Windows" and any("MIGraphX" in p or "ROCm" in p for p in providers):
        raise RuntimeError("HIP/MIGraphX provider blocked on Windows")

    sess = ort.InferenceSession(str(onnx_path), providers=providers, **sess_options)
    active = sess.get_providers()
    logger.info("ORT session active providers: %s", active)
    return sess
