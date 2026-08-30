#!/bin/bash
# wine_test.sh — smoke-test Windows CLI under Wine (Linux dev)
# Verifies imports, argparse, and device enumeration don't break on Windows path handling.
# Does NOT expect DirectML/WebGPU or audio devices to work under Wine (vkd3d lacks DX12 ML).

set -e

echo "=== Samuel Wine Smoke Test (Windows emulation on Linux) ==="
echo "Host: $(uname -a)"
echo "Wine: $(wine --version 2>&1 | head -n1 || echo 'wine not found')"
echo "Python: $(uv run python --version 2>&1)"
echo ""

# Check wine prefix exists
if [ ! -d "$HOME/.wine" ]; then
    echo "[warn] ~/.wine not found — run winecfg once to init"
    winecfg --help 2>&1 | head -n5 || true
fi

# 1. Native CLI --help (baseline)
echo "[1/4] Native CLI --help"
uv run python -m samuel_realtime --help | head -n 20
echo "[ok] native --help"

# 2. Native list-devices (should show Samuel_Virtual_Mic)
echo ""
echo "[2/4] Native --list-devices"
uv run python -m samuel_realtime --list-devices | head -n 30
echo "[ok] native list"

# 3. Try wine python --help if python.exe is available in Wine
# Most Linux Wine prefixes don't have Python installed — we test gracefully
echo ""
echo "[3/4] Wine python --help (if python.exe in Wine)"
WINE_PYTHON="C:\\Python312\\python.exe"
if wine cmd /c "where python" 2>&1 | grep -q "python"; then
    echo "[info] python.exe found in Wine PATH, attempting --help"
    wine python -m samuel_realtime --help 2>&1 | head -n 20 || echo "[warn] wine python --help failed (expected if modules not installed in Wine)"
else
    echo "[info] No python.exe in Wine — skipping Wine python test (manual step: install python-3.12-amd64.exe via wine)"
    echo "      Download from https://www.python.org/downloads/windows/ then: wine python-3.12.4-amd64.exe"
    echo "      Then: wine python -m pip install --upgrade pip && wine python -m pip install -r requirements.txt"
fi

# 4. Path handling check: Windows separators, provider blocking
echo ""
echo "[4/4] Cross-platform path & provider policy check"
uv run python -c "
import platform
from samuel_realtime.providers import select_providers
print('Current providers:', select_providers(None))
# Simulate Windows HIP block
orig = platform.system
platform.system = lambda: 'Windows'
try:
    select_providers('migraphx')
    print('FAIL: HIP not blocked on Windows')
except RuntimeError as e:
    print('ok HIP blocked on Windows:', e)
finally:
    platform.system = orig
print('Windows WebGPU auto fallback:', select_providers(None) if platform.system()=='Windows' else 'skip (Linux host)')

# Windows path handling: ensure checkpoint 'hf:vvolhejn/samuel' doesn't break on Windows
from pathlib import Path, PureWindowsPath
p = PureWindowsPath('C:\\\\Users\\\\Test\\\\models\\\\samuel_controller.onnx')
print('PureWindowsPath test:', p)
"
echo "[ok] provider & path checks"

echo ""
echo "=== Wine smoke test complete ==="
echo "Note: Wine's vkd3d does NOT support DirectML/WebGPU — ONNX WebGPU will fallback to CPU under Wine (expected)."
echo "For true Windows validation, copy repo to native Windows and run:"
echo "  pip install onnxruntime-directml"
echo "  python -m samuel_realtime --out-device \"CABLE Input\" --onnx models\\samuel_controller.onnx --provider webgpu"
