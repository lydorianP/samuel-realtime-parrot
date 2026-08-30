"""samuel_realtime — cross-platform audio I/O helpers for Samuel parrot pipeline."""

from .audio_io import (
    find_virtual_devices,
    get_virtual_input_name,
    get_virtual_output_name,
    list_devices,
    play_test_tone,
)

__all__ = [
    "list_devices",
    "find_virtual_devices",
    "get_virtual_output_name",
    "get_virtual_input_name",
    "play_test_tone",
]

# Re-export key classes for convenience
try:
    from .inference import SamuelEngine  # noqa: F401
    from .vad import SileroVAD, VADProcessor  # noqa: F401
    from .pipeline import RealtimePipeline  # noqa: F401
except Exception:
    pass
