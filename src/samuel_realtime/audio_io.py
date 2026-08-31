"""Cross-platform audio I/O helpers for Samuel realtime parrot.

Detects virtual audio devices:
  Windows: VB-CABLE -> playback "CABLE Input" / recording "CABLE Output"
  Linux:   PipeWire null-sink -> playback "Samuel_Virtual_Mic" or "SamuelMic"

Usage:
  uv run python src/samuel_realtime/audio_io.py          # list + test tone hint
  uv run python -m samuel_realtime.audio_io --list
  uv run python -m samuel_realtime.audio_io --test --device "Samuel_Virtual_Mic"
"""
from __future__ import annotations

import argparse
import platform
import sys

import numpy as np
import sounddevice as sd

# Signatures for virtual devices (case-insensitive substring match)
WINDOWS_OUTPUT_SIGS = ["CABLE Input", "CABLE Input (VB-Audio Virtual Cable)"]
WINDOWS_INPUT_SIGS = ["CABLE Output", "CABLE Output (VB-Audio Virtual Cable)"]
# PipeWire null-sink appears as both sink and monitor source; description may vary
LINUX_OUTPUT_SIGS = ["Samuel_Virtual_Mic", "SamuelMic"]
LINUX_INPUT_SIGS = ["Monitor of Samuel", "Monitor of Samuel_Virtual_Mic", "SamuelMic"]

# Fallback generic scan tokens
VIRTUAL_TOKENS_LINUX = ["samuel", "null-sink"]
VIRTUAL_TOKENS_WINDOWS = ["cable", "vb-audio"]


def list_devices() -> list[dict]:
    """Return sounddevice device list with indexes."""
    devices = sd.query_devices()
    # sd returns DeviceList (iterable of dicts), ensure list
    out = []
    for i, d in enumerate(devices):
        entry = dict(d)
        entry["index"] = i
        out.append(entry)
    return out


def _matches(name: str, sigs: list[str]) -> bool:
    n = name.lower()
    return any(s.lower() in n or n in s.lower() for s in sigs)


def find_virtual_devices() -> dict:
    """Scan devices and return detected virtual sink/monitor info.

    Returns dict with keys:
      platform, devices, virtual_output (device dict or None), virtual_input (device dict or None)
    """
    system = platform.system()
    devices = list_devices()

    # Choose signatures per OS, but also scan cross-platform for robustness (useful under Wine)
    if system == "Windows":
        out_sigs = WINDOWS_OUTPUT_SIGS
        in_sigs = WINDOWS_INPUT_SIGS
        tokens = VIRTUAL_TOKENS_WINDOWS
    else:
        out_sigs = LINUX_OUTPUT_SIGS
        in_sigs = LINUX_INPUT_SIGS
        tokens = VIRTUAL_TOKENS_LINUX + VIRTUAL_TOKENS_WINDOWS  # be permissive on Linux

    virtual_output = None
    virtual_input = None

    # First pass: exact signature matches
    for d in devices:
        name = d.get("name", "")
        if d.get("max_output_channels", 0) > 0 and virtual_output is None:
            if _matches(name, out_sigs):
                virtual_output = d
        if d.get("max_input_channels", 0) > 0 and virtual_input is None:
            if _matches(name, in_sigs):
                virtual_input = d

    # Second pass: token-based fuzzy match (for variants like "VB-Audio Virtual Cable")
    if virtual_output is None:
        for d in devices:
            if d.get("max_output_channels", 0) > 0:
                name = d.get("name", "").lower()
                if any(t in name for t in tokens):
                    virtual_output = d
                    break
    if virtual_input is None:
        for d in devices:
            if d.get("max_input_channels", 0) > 0:
                name = d.get("name", "").lower()
                if any(t in name for t in tokens):
                    virtual_input = d
                    break

    # Third pass (Linux): "Monitor of Samuel*" by token
    if virtual_input is None and system != "Windows":
        for d in devices:
            if d.get("max_input_channels", 0) > 0:
                name = d.get("name", "").lower()
                if "monitor" in name and "samuel" in name:
                    virtual_input = d
                    break

    return {
        "platform": system,
        "devices": devices,
        "virtual_output": virtual_output,
        "virtual_input": virtual_input,
    }


def get_virtual_output_name(prefer: list[str] | None = None) -> str | None:
    """Return best virtual output device name for this platform, or None."""
    info = find_virtual_devices()
    if info["virtual_output"]:
        return info["virtual_output"]["name"]
    # Allow caller preference fallback
    if prefer:
        for p in prefer:
            for d in info["devices"]:
                if p.lower() in d.get("name", "").lower() and d.get("max_output_channels", 0) > 0:
                    return d["name"]
    return None


def get_virtual_input_name(prefer: list[str] | None = None) -> str | None:
    """Return best virtual input (monitor) device name, or None."""
    info = find_virtual_devices()
    if info["virtual_input"]:
        return info["virtual_input"]["name"]
    if prefer:
        for p in prefer:
            for d in info["devices"]:
                if p.lower() in d.get("name", "").lower() and d.get("max_input_channels", 0) > 0:
                    return d["name"]
    return None


def select_input_device(
    preferred: str | int | None = None,
    auto_physical: bool = True,
) -> str | int | None:
    """Select input device with smart defaults.

    Priority:
    1. Explicit `preferred` (name or index)
    2. Auto-detect physical microphone (max_input_channels > 0, not virtual monitor)
    3. Virtual input (monitor of virtual sink) if no physical found
    4. None -> sounddevice default

    Args:
        preferred: Explicit device name substring or index
        auto_physical: If True and no preferred, try to find a physical mic

    Returns:
        Device name string, index, or None for default
    """
    devices = list_devices()
    system = platform.system()

    # 1. Explicit preferred
    if preferred is not None:
        if isinstance(preferred, int):
            if 0 <= preferred < len(devices):
                return preferred
        else:
            # Substring match
            for d in devices:
                if preferred.lower() in d.get("name", "").lower() and d.get("max_input_channels", 0) > 0:
                    return d["name"]
            raise ValueError(f"Input device '{preferred}' not found or has no input channels")

    # 2. Auto-detect physical microphone (exclude virtual monitors)
    if auto_physical:
        for d in devices:
            if d.get("max_input_channels", 0) > 0:
                name = d.get("name", "").lower()
                # Skip known virtual monitors
                if any(tok in name for tok in ["monitor", "cable", "vb-audio", "samuel", "null-sink"]):
                    continue
                # Prefer built-in/headset mics
                return d["name"]

    # 3. Fallback to virtual input (monitor)
    virt = get_virtual_input_name()
    if virt:
        return virt

    # 4. Default
    return None


def select_output_device(
    preferred: str | int | None = None,
) -> str | int | None:
    """Select output device with smart defaults.

    Priority:
    1. Explicit `preferred` (name or index)
    2. Auto-detect virtual sink (CABLE Input on Windows, Samuel_Virtual_Mic on Linux)
    3. None -> sounddevice default (may be speakers!)

    Args:
        preferred: Explicit device name substring or index

    Returns:
        Device name string, index, or None for default
    """
    devices = list_devices()

    # 1. Explicit preferred
    if preferred is not None:
        if isinstance(preferred, int):
            if 0 <= preferred < len(devices):
                return preferred
        else:
            for d in devices:
                if preferred.lower() in d.get("name", "").lower() and d.get("max_output_channels", 0) > 0:
                    return d["name"]
            raise ValueError(f"Output device '{preferred}' not found or has no output channels")

    # 2. Auto-detect virtual sink
    virt = get_virtual_output_name()
    if virt:
        return virt

    # 3. Default (warn: may be speakers)
    return None


def get_pactl_sink_name(sounddevice_name: str) -> str | None:
    """Map sounddevice output device name/description to pactl sink name.

    sounddevice uses PortAudio device descriptions (e.g., "CDS.KT USB Audio Analog Stereo")
    pactl uses PulseAudio sink names (e.g., "alsa_output.usb-KTMicro_CDS.KT_USB_Audio_2020-02-20-0000-0000-0000-00.analog-stereo")

    This function attempts to find the pactl sink name by matching keywords.
    """
    import subprocess
    try:
        out = subprocess.check_output(["pactl", "list", "sinks", "short"], text=True)
        # Extract keywords from sounddevice name (remove common prefixes/suffixes)
        keywords = sounddevice_name.lower().split()
        # Filter out generic words
        keywords = [k for k in keywords if k not in ("audio", "analog", "stereo", "digital", "output", "input", "device")]
        
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                sink_name = parts[1]
                line_lower = line.lower()
                # Match if all keywords appear in the pactl line
                if all(k in line_lower for k in keywords):
                    return sink_name
        return None
    except Exception:
        return None


def play_test_tone(
    device: str | int | None = None,
    duration: float = 1.5,
    freq: float = 440.0,
    samplerate: int | None = None,
    volume: float = 0.3,
) -> None:
    """Play a sine test tone to the given device.

    If device is None, auto-detects virtual output; falls back to default output.
    Samplerate is auto-selected from device default_samplerate (PipeWire null-sink is 48000),
    otherwise 44100. Pass samplerate explicitly to override.
    """
    resolved_device = device
    if resolved_device is None:
        resolved_device = get_virtual_output_name()
        if resolved_device is None:
            print("[info] No virtual output auto-detected — using default output device")
            resolved_device = None  # let sounddevice pick default
        else:
            print(f"[info] Auto-selected virtual output: {resolved_device}")

    # Resolve samplerate from device default if not specified
    effective_sr = samplerate
    if effective_sr is None:
        try:
            if isinstance(resolved_device, str):
                for d in list_devices():
                    if resolved_device.lower() in d.get("name", "").lower():
                        effective_sr = int(d.get("default_samplerate", 44100))
                        break
                effective_sr = effective_sr or 44100
            elif isinstance(resolved_device, int):
                d = sd.query_devices(resolved_device)
                effective_sr = int(d.get("default_samplerate", 44100))
            else:  # default device
                effective_sr = 48000  # PipeWire default; safe for SamuelMic
                # try query default
                try:
                    di = sd.default.device[1] if isinstance(sd.default.device, (list, tuple)) else None
                    if di is not None and di >= 0:
                        d = sd.query_devices(di)
                        effective_sr = int(d.get("default_samplerate", 48000))
                except Exception:
                    pass
        except Exception:
            effective_sr = 44100
        print(f"[info] Auto samplerate for device '{resolved_device}': {effective_sr}")

    samplerate = effective_sr

    # Resolve device index for nicer logging
    try:
        if isinstance(resolved_device, str):
            # Validate it exists
            devices = list_devices()
            matched = [d for d in devices if resolved_device.lower() in d.get("name", "").lower()]
            if not matched:
                print(f"[warn] Device '{resolved_device}' not found in query_devices — trying as literal name anyway")
        elif isinstance(resolved_device, int):
            print(f"[info] Using device index {resolved_device}")
    except Exception as e:
        print(f"[warn] device validation failed: {e}")

    t = np.linspace(0, duration, int(samplerate * duration), dtype=np.float32, endpoint=False)
    # Gentle Hann envelope to avoid clicks
    tone = (volume * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    envelope = np.hanning(len(tone)).astype(np.float32)
    # Keep middle 80% at full, fade 10% each side
    fade = int(len(tone) * 0.05)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    tone *= envelope

    print(f"[info] Playing {freq} Hz tone for {duration}s on device='{resolved_device}' sr={samplerate}")
    try:
        sd.play(tone, samplerate=samplerate, device=resolved_device, blocking=True)
        print("[ok] Playback finished")
    except Exception as e:
        print(f"[error] Playback failed on device '{resolved_device}': {e}")
        print("Hint: list devices with --list and try --device with index or exact name")
        raise


def main():
    parser = argparse.ArgumentParser(description="Samuel audio I/O helper — enumerate devices & test virtual mic")
    parser.add_argument("--list", action="store_true", help="List all audio devices")
    parser.add_argument("--test", action="store_true", help="Play test tone to virtual output")
    parser.add_argument("--device", type=str, default=None, help="Device name or index for --test")
    parser.add_argument("--freq", type=float, default=440.0, help="Tone frequency Hz")
    parser.add_argument("--duration", type=float, default=1.5, help="Tone duration seconds")
    args = parser.parse_args()

    # Always show detection summary unless --list alone is enough
    info = find_virtual_devices()
    print(f"Platform: {info['platform']}  Python: {platform.python_version()}  sounddevice: {sd.__version__ if hasattr(sd, '__version__') else 'unknown'}")
    print(f"Detected virtual OUTPUT: {info['virtual_output']['name'] if info['virtual_output'] else 'NOT FOUND'}")
    if info["virtual_output"]:
        d = info["virtual_output"]
        print(f"  -> index {d['index']}, out_ch {d['max_output_channels']}, sr {d.get('default_samplerate')}")
    print(f"Detected virtual INPUT (monitor): {info['virtual_input']['name'] if info['virtual_input'] else 'NOT FOUND'}")
    if info["virtual_input"]:
        d = info["virtual_input"]
        print(f"  -> index {d['index']}, in_ch {d['max_input_channels']}, sr {d.get('default_samplerate')}")

    if args.list or not args.test:
        print("\n--- All devices ---")
        for d in info["devices"]:
            print(f"[{d['index']}] {d['name']}  in:{d['max_input_channels']} out:{d['max_output_channels']} sr:{d.get('default_samplerate')}")

        if not args.list and not args.test:
            print("\nHints:")
            print("  uv run python -m samuel_realtime.audio_io --list          # enumerate")
            print("  uv run python -m samuel_realtime.audio_io --test          # tone to virtual sink")
            print("  ./scripts/setup_virtual_mic.sh                            # create SamuelMic on Linux")
            print('  powershell -File scripts\\check_vbcable.ps1                # check VB-CABLE on Windows')

    if args.test:
        # Allow integer device index
        dev = args.device
        if dev is not None:
            try:
                dev = int(dev)
            except ValueError:
                pass
        play_test_tone(device=dev, freq=args.freq, duration=args.duration)


if __name__ == "__main__":
    # Support both python src/samuel_realtime/audio_io.py and python -m samuel_realtime.audio_io
    try:
        main()
    except KeyboardInterrupt:
        print("\n[abort] interrupted")
        sys.exit(1)

