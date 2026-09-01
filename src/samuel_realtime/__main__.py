"""CLI entrypoint for Samuel realtime parrot pipeline.

Usage:
  uv run python -m samuel_realtime --help
  uv run python -m samuel_realtime --out-device "Samuel_Virtual_Mic" --vad-silence 0.45
  uv run python -m samuel_realtime --in-device default --out-device "CABLE Input" --provider webgpu
  SAMUEL_CHECKPOINT=path/to/model.pt uv run python -m samuel_realtime --onnx models/controller.onnx

Provider 'auto' uses providers.py auto logic (WebGPU→Dml→CPU on Windows, WebGPU→CPU on Linux).
--checkpoint defaults to hf:vvolhejn/samuel (15M, auto-cached).
Env vars override CLI args: SAMUEL_CHECKPOINT, SAMUEL_ONNX, SAMUEL_VAD_SILENCE, SAMUEL_INPUT_GAIN, SAMUEL_CONFIG.
"""
from __future__ import annotations

import argparse
import logging
import os
import platform
import sys
from pathlib import Path

from .audio_io import list_devices, select_input_device, select_output_device, get_pactl_sink_name
from .inference import SamuelEngine
from .pipeline import RealtimePipeline
from .providers import create_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="samuel_realtime",
        description="Real-time Samuel Pink Trombone phrase parrot (Windows VB-CABLE / Linux PipeWire)",
    )
    p.add_argument("--in-device", dest="in_device", default=None, help="Input device name/index or 'default' (auto-detect physical mic)")
    p.add_argument("--out-device", dest="out_device", default=None, help="Output device name/index (auto virtual mic: CABLE Input / Samuel_Virtual_Mic)")
    p.add_argument("--provider", default="auto", choices=["auto", "webgpu", "dml", "migraphx", "cpu"], help="ONNX provider (auto selects WebGPU/Dml/CPU, enforces No HIP on Windows)")
    p.add_argument("--vad-silence", type=float, default=0.45, help="Silence threshold seconds to trigger inference (0.45 low-latency)")
    p.add_argument("--input-gain", type=float, default=1.0, help="Input gain multiplier (e.g., 5.0 for quiet mics)")
    p.add_argument("--checkpoint", default="hf:vvolhejn/samuel", help="hf:vvolhejn/samuel or local last.pt (env: SAMUEL_CHECKPOINT)")
    p.add_argument("--onnx", type=Path, default=None, help="Path to samuel_controller.onnx (enables ONNX provider); if omitted uses PyTorch (env: SAMUEL_ONNX)")
    p.add_argument("--config", type=Path, default=None, help="Path to config.json (env: SAMUEL_CONFIG); overrides auto-detection near checkpoint")
    p.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def _resolve_checkpoint(raw: str) -> str:
    """Resolve checkpoint ref: add hf: prefix if it looks like an HF repo ID but lacks one."""
    if not raw or raw.startswith("hf:"):
        return raw
    if Path(raw).exists():
        return raw
    if "/" in raw and not raw.startswith("."):
        return f"hf:{raw}"
    return raw


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # Env var overrides (applied after parse so explicit CLI args win, but env fills defaults)
    if os.environ.get("SAMUEL_CHECKPOINT") and args.checkpoint == "hf:vvolhejn/samuel":
        args.checkpoint = os.environ["SAMUEL_CHECKPOINT"]
    if os.environ.get("SAMUEL_ONNX") and args.onnx is None:
        args.onnx = Path(os.environ["SAMUEL_ONNX"])
    if os.environ.get("SAMUEL_VAD_SILENCE") and args.vad_silence == 0.45:
        args.vad_silence = float(os.environ["SAMUEL_VAD_SILENCE"])
    if os.environ.get("SAMUEL_INPUT_GAIN") and args.input_gain == 1.0:
        args.input_gain = float(os.environ["SAMUEL_INPUT_GAIN"])
    if os.environ.get("SAMUEL_CONFIG") and args.config is None:
        args.config = Path(os.environ["SAMUEL_CONFIG"])

    # Resolve bare HF repo IDs
    args.checkpoint = _resolve_checkpoint(args.checkpoint)

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if args.list_devices:
        for d in list_devices():
            print(f"[{d['index']}] {d['name']}  in:{d['max_input_channels']} out:{d['max_output_channels']} sr:{d.get('default_samplerate')}")
        return 0

    # Resolve devices using smart selectors
    # --in-device: "default"/"auto"/None = auto-detect physical mic; explicit name/index = use that
    # --out-device: "default"/"auto"/None = auto-detect virtual sink; explicit name/index = use that

    # Try int conversion for index FIRST
    for attr in ["in_device", "out_device"]:
        val = getattr(args, attr)
        if val is not None:
            try:
                setattr(args, attr, int(val))
                continue
            except (ValueError, TypeError):
                pass

    # Now resolve with converted values
    in_dev = None if args.in_device in (None, "default", "auto") else args.in_device
    out_dev = None if args.out_device in (None, "default", "auto") else args.out_device

    # Apply smart selection (uses audio_io.select_* functions)
    try:
        in_dev = select_input_device(in_dev, auto_physical=True)
        out_dev = select_output_device(out_dev)
    except ValueError as e:
        logger.error("Device selection failed: %s", e)
        return 1

    # On Linux, map sounddevice output name to pactl sink name for pacat
    if out_dev and platform.system() == "Linux":
        pactl_name = get_pactl_sink_name(out_dev)
        if pactl_name:
            logger.info("Mapped output device '%s' to pactl sink '%s'", out_dev, pactl_name)
            out_dev = pactl_name
        else:
            logger.warning("Could not map '%s' to pactl sink; using as-is", out_dev)

    # Engine load + warmup
    logger.info("Loading SamuelEngine checkpoint %s", args.checkpoint)
    if args.config:
        os.environ["SAMUEL_RUN_CONFIG"] = str(args.config)
    engine = SamuelEngine(checkpoint=args.checkpoint)
    try:
        engine.load()
    except Exception as e:
        logger.error("Failed to load checkpoint: %s", e)
        return 1
    logger.info("Warming up (JIT pyin)...")
    engine.warm_up()
    logger.info("Warm-up done — engine ready (%s, spf %d, target_rms %.3f)", engine.device, engine.model.samples_per_frame, engine.target_rms)

    # ONNX session — explicit --onnx, env var, or auto-detect near checkpoint
    onnx_session = None
    provider_req = None if args.provider == "auto" else args.provider
    onnx_path = args.onnx
    if onnx_path is None and os.environ.get("SAMUEL_ONNX"):
        onnx_path = Path(os.environ["SAMUEL_ONNX"])
    if onnx_path is None:
        # Auto-detect: look for *.onnx in models/ dir relative to checkpoint
        ckpt_dir = Path(args.checkpoint).parent if Path(args.checkpoint).exists() else Path.cwd()
        for candidate in [ckpt_dir / "models" / "samuel_custom_controller.onnx", ckpt_dir / "models" / "samuel_controller.onnx"]:
            if candidate.exists():
                onnx_path = candidate
                logger.info("Auto-detected ONNX: %s", onnx_path)
                break
    if onnx_path:
        if not onnx_path.exists():
            logger.error("ONNX file not found: %s (run scripts/export_onnx.py)", onnx_path)
            return 1
        try:
            onnx_session = create_session(str(onnx_path), requested=provider_req)
            logger.info("ONNX session active: %s", onnx_session.get_providers())
        except Exception as e:
            logger.error("Failed to create ONNX session: %s", e)
            return 1
    else:
        if args.provider != "auto":
            logger.warning("--provider %s ignored without --onnx (PyTorch mode uses ROCm/CPU directly)", args.provider)

    # Build and run pipeline
    vad_ms = int(args.vad_silence * 1000)
    pipeline = RealtimePipeline(
        engine=engine,
        in_device=in_dev,
        out_device=out_dev,
        provider=args.provider,
        vad_silence_ms=vad_ms,
        onnx_session=onnx_session,
        input_gain=args.input_gain,
    )

    logger.info("Starting pipeline: in=%s out=%s provider=%s onnx=%s vad_silence=%.2fs",
                in_dev, out_dev, args.provider, args.onnx, args.vad_silence)
    logger.info("Speak now — pipeline will parrot after ~%.0fms silence. Ctrl+C to stop.", vad_ms)
    # Also log auto-detected devices
    try:
        from .audio_io import get_virtual_output_name
        virt = get_virtual_output_name()
        if virt:
            logger.info("Detected virtual output: %s", virt)
    except Exception:
        pass

    pipeline.start()
    pipeline.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
