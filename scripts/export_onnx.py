#!/usr/bin/env python3
"""Export PinkTromboneController (SEANet + head) to ONNX for WebGPU/DirectML.

Leaves the waveguide synth (pink_trombone_ola) in PyTorch — heavy but small;
controller is the matmul-bound part that benefits from GPU.

- eval mode so gumbel_softmax resolves to argmax + one_hot (exportable)
- dynamic axes on samples S and control frames T_ctrl
- opset 17 (required for recent ONNX Runtime WebGPU)
- verifies output shape [1, T_ctrl, 12]

Usage:
  uv run python scripts/export_onnx.py                          # default hf:vvolhejn/samuel -> samuel_controller.onnx
  uv run python scripts/export_onnx.py --checkpoint path/to/last.pt --out models/samuel.onnx
  uv run python scripts/export_onnx.py --verify                 # also runs a dummy forward check
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Ensure vendor path
_VENDOR_SRC = Path(__file__).resolve().parents[1] / "vendor" / "samuel" / "src"
if str(_VENDOR_SRC) not in sys.path:
    sys.path.insert(0, str(_VENDOR_SRC))

from samuel.model import PinkTromboneController, PinkTromboneControllerConfig

# Reuse resolve helpers from inference (avoid circular import)
import samuel_realtime.inference as inf_mod

DEFAULT_ONNX = Path("models") / "samuel_controller.onnx"


class ExportWrapper(torch.nn.Module):
    """Thin wrapper exposing controller with clear I/O names for ONNX.

    The real controller forward takes (wav [B,1,S], f0 [B,T_ctrl], tau, return_aux).
    We fix tau and drop aux for export; only the eval path (argmax) is exported.
    """

    def __init__(self, controller: PinkTromboneController):
        super().__init__()
        self.controller = controller

    def forward(self, wav: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        # controller expects eval mode -> no tau needed, but we pass 1.0 for signature
        return self.controller(wav, f0)


def export(checkpoint: str, out_path: Path, verify: bool = True, opset: int = 17) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    engine = inf_mod.SamuelEngine(checkpoint=checkpoint)
    model = engine.load()
    model.eval()

    wrapper = ExportWrapper(model).eval()

    # Dummy inputs: 2 seconds @44100 -> S=88200, spf=512 -> T_ctrl=ceil(88200/512)=173
    # Use 4s (training chunk) as default: 176400 samples -> 345 frames at spf512
    # Pick 2s for quicker export/verify
    spf = model.samples_per_frame
    for sec in [2.0, 0.5, 4.0]:
        S = int(44100 * sec)
        T = model.t_ctrl_for(S)
        wav = torch.randn(1, 1, S, dtype=torch.float32, device=engine.device)
        f0 = torch.full((1, T), 140.0, dtype=torch.float32, device=engine.device)

        # Trace with appropriate device — export on CPU for portability even if ROCm is available
        # Move wrapper to CPU for ONNX graph (weights are device-agnostic)
        wrapper_cpu = wrapper.to("cpu")
        wav_cpu = wav.cpu()
        f0_cpu = f0.cpu()

        print(f"[info] Export test: {sec}s -> S={S}, T_ctrl={T}, wav {tuple(wav_cpu.shape)}, f0 {tuple(f0_cpu.shape)}")
        with torch.no_grad():
            out = wrapper_cpu(wav_cpu, f0_cpu)
            print(f"  torch out shape {tuple(out.shape)} expected [1, {T}, 12]")

        # Now real export (only once — reuse last dummy)
        if sec == 2.0:
            dynamic_axes = {
                "wav": {2: "samples"},
                "f0": {1: "frames"},
                "params": {1: "frames"},
            }
            print(f"[info] Exporting to {out_path} opset={opset} ...")
            torch.onnx.export(
                wrapper_cpu,
                (wav_cpu, f0_cpu),
                str(out_path),
                input_names=["wav", "f0"],
                output_names=["params"],
                dynamic_axes=dynamic_axes,
                opset_version=opset,
                do_constant_folding=True,
            )
            print(f"[ok] ONNX written: {out_path} ({out_path.stat().st_size/1024/1024:.2f} MiB)")
            # Verify with onnx checker if available
            try:
                import onnx

                m = onnx.load(str(out_path))
                onnx.checker.check_model(m)
                print("[ok] onnx.checker passed")
            except Exception as e:
                print(f"[warn] onnx.checker failed: {e}")

            if verify:
                import onnxruntime as ort

                providers = ["CPUExecutionProvider"]
                sess = ort.InferenceSession(str(out_path), providers=providers)
                print(f"[info] ORT providers: {sess.get_providers()}")
                # Run session on same dummy
                wav_np = wav_cpu.numpy()
                f0_np = f0_cpu.numpy()
                # Input name mapping (export uses wav/f0)
                ort_out = sess.run(None, {"wav": wav_np, "f0": f0_np})[0]
                print(f"[ok] ORT output shape {ort_out.shape} (torch was {tuple(out.shape)})")
                # Compare close
                diff = (ort_out - out.cpu().numpy()).__abs__().max() if hasattr((ort_out - out.cpu().numpy()), "max") else float("inf")
                # numpy max
                import numpy as np

                max_abs = np.abs(ort_out - out.cpu().numpy()).max()
                print(f"[verify] max abs diff torch vs ORT: {max_abs:.6f}")
                if max_abs > 1e-3:
                    print("[warn] diff >1e-3 — check ops (interpolate/argmax) fidelity")
                else:
                    print("[ok] ORT matches torch within 1e-3")

            # Check different second shapes still run (dynamic)
            if verify:
                for sec2, S2 in [(0.7, int(44100 * 0.7)), (3.3, int(44100 * 3.3))]:
                    T2 = model.t_ctrl_for(S2)
                    wav2 = torch.randn(1, 1, S2, dtype=torch.float32).numpy()
                    f02 = (140.0 * torch.ones(1, T2)).numpy()
                    # Need session still
                    import onnxruntime as ort

                    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
                    out2 = sess.run(None, {"wav": wav2, "f0": f02})[0]
                    assert out2.shape == (1, T2, 12), f"dynamic shape failed: {out2.shape} vs {(1,T2,12)}"
                    print(f"[verify] dynamic {sec2}s OK shape {out2.shape}")

        # Break after first sec's export — other secs were just shape demos
        # Actually we already handled dynamic verify above; no need to re-export
        break

    return out_path


def main():
    p = argparse.ArgumentParser(description="Export Samuel controller to ONNX (opset 17, dynamic axes)")
    p.add_argument("--checkpoint", default="hf:vvolhejn/samuel", help="hf:vvolhejn/samuel or local last.pt")
    p.add_argument("--out", type=Path, default=DEFAULT_ONNX, help="output ONNX path")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    p.add_argument("--verify", action="store_true", help="also run torch vs ORT sanity check")
    p.add_argument("--no-verify", dest="verify", action="store_false", help="skip verify")
    p.set_defaults(verify=True)
    args = p.parse_args()
    export(args.checkpoint, args.out, verify=args.verify, opset=args.opset)


if __name__ == "__main__":
    main()
