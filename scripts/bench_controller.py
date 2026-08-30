#!/usr/bin/env python3
"""Benchmark Samuel controller: PyTorch (CPU vs ROCm) vs ONNX (CPU/WebGPU).

Outputs latency for 2s chunk (typical phrase) and verifies output parity.
"""
import time
import numpy as np
import torch
import sys
from pathlib import Path

_VENDOR_SRC = Path(__file__).resolve().parents[1] / "vendor" / "samuel" / "src"
if str(_VENDOR_SRC) not in sys.path:
    sys.path.insert(0, str(_VENDOR_SRC))

from samuel_realtime.inference import SamuelEngine
from samuel_realtime.providers import available_providers, select_providers

CHECKPOINT = "hf:vvolhejn/samuel"
ONNX_PATH = Path("models/samuel_controller.onnx")
SAMPLE_RATE = 44100

def make_dummy(sec=2.0):
    # 440Hz tone + noise, similar to warm-up
    t = np.arange(int(SAMPLE_RATE*sec), dtype=np.float32) / SAMPLE_RATE
    wav = 0.3 * np.sin(2*np.pi*180*t).astype(np.float32) + 0.02*np.random.randn(len(t)).astype(np.float32)
    return wav

def bench_torch(engine, wav, iters=5):
    times = []
    for i in range(iters):
        start = time.perf_counter()
        params, voiced = engine.infer_controller(wav)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times.append((time.perf_counter()-start)*1000)
    # Synth separate
    params,_ = engine.infer_controller(wav)
    t0 = time.perf_counter()
    synth = engine.synthesize(params)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    synth_ms = (time.perf_counter()-t0)*1000
    return times, synth_ms, params

def bench_onnx(engine, wav, session, iters=5):
    import samuel_realtime.inference as inf
    times=[]
    for i in range(iters):
        start=time.perf_counter()
        params, voiced = engine.infer_controller_onnx(session, wav)
        times.append((time.perf_counter()-start)*1000)
    params,_= engine.infer_controller_onnx(session, wav)
    t0=time.perf_counter()
    synth = engine.synthesize(params)
    synth_ms=(time.perf_counter()-t0)*1000
    return times, synth_ms, params

def main():
    print("=== Samuel Controller Benchmark ===")
    print(f"ORT available: {available_providers()}")
    print(f"torch {torch.__version__} hip {getattr(torch.version,'hip','n/a')} cuda_avail {torch.cuda.is_available()} dev {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"ROCm device: {torch.cuda.get_device_name(0)}")

    engine = SamuelEngine(checkpoint=CHECKPOINT)
    engine.load()
    # Warm up (also warms librosa)
    engine.warm_up()

    wav2 = make_dummy(2.0)
    wav4 = make_dummy(4.0)

    for sec, wav in [(2.0, wav2), (4.0, wav4)]:
        print(f"\n--- {sec}s chunk (samples {len(wav)}, spf {engine.model.samples_per_frame}, T_ctrl {engine.model.t_ctrl_for(len(wav))}) ---")
        # Torch
        torch_times, synth_ms, params_torch = bench_torch(engine, wav, iters=5)
        print(f"PyTorch controller (device {engine.device}): median {np.median(torch_times):.1f}ms mean {np.mean(torch_times):.1f}ms min {min(torch_times):.1f} max {max(torch_times):.1f} over 5")
        print(f"  synth (PyTorch) after: {synth_ms:.1f}ms  total controller+synth median {np.median(torch_times)+synth_ms:.1f}ms  output shape {params_torch.shape}")

        # ONNX CPU
        if ONNX_PATH.exists():
            import onnxruntime as ort
            prov = select_providers(None)
            print(f"  ONNX auto providers: {prov}")
            try:
                sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
                onnx_times, onnx_synth, params_onnx = bench_onnx(engine, wav, sess, iters=5)
                print(f"  ONNX CPU controller: median {np.median(onnx_times):.1f}ms mean {np.mean(onnx_times):.1f}ms  synth {onnx_synth:.1f}ms  shape {params_onnx.shape}")
                # Parity
                maxdiff = np.abs(params_torch - params_onnx).max()
                print(f"  max abs diff torch vs onnx: {maxdiff:.6f}")
                # Try WebGPU if avail
                if any("WebGPU" in p for p in available_providers()):
                    try:
                        prov_w = select_providers("webgpu")
                        sess_w = ort.InferenceSession(str(ONNX_PATH), providers=prov_w)
                        w_times, _, _ = bench_onnx(engine, wav, sess_w, iters=3)
                        print(f"  ONNX WebGPU controller: median {np.median(w_times):.1f}ms")
                    except Exception as e:
                        print(f"  WebGPU bench failed: {e}")
                else:
                    print("  WebGPU provider not installed (pip onnxruntime lacks WebGPU — needs plugin, expected, CPU fallback is correct)")
            except Exception as e:
                print(f"  ONNX bench failed: {e}")
        else:
            print("  ONNX not found, skipping")

    print("\n=== provider selection policy check ===")
    for req in [None, "cpu", "webgpu", "dml"]:
        try:
            print(f"  request {req!r} -> {select_providers(req)}")
        except Exception as e:
            print(f"  request {req!r} error {e}")
    print("  Windows HIP block test (simulate Windows):")
    import platform
    orig = platform.system
    try:
        platform.system = lambda: "Windows"
        try:
            select_providers("migraphx")
            print("    FAIL: HIP not blocked")
        except RuntimeError as e:
            print(f"    ok blocked: {e}")
    finally:
        platform.system = orig

if __name__=="__main__":
    main()
