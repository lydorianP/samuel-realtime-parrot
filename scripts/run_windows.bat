@echo off
REM run_windows.bat — Samuel Realtime Parrot launcher for Windows (VB-CABLE)
REM Usage: scripts\run_windows.bat [--onnx models\samuel_custom_controller.onnx] [--provider webgpu]
REM Env vars: SAMUEL_CHECKPOINT, SAMUEL_ONNX, SAMUEL_VAD_SILENCE, SAMUEL_INPUT_GAIN, SAMUEL_CONFIG

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
set "VENV=%PROJECT_ROOT%\.venv"

if exist "%VENV%\Scripts\activate.bat" (
    call "%VENV%\Scripts\activate.bat"
) else (
    echo [warn] No .venv found — using system Python
)

set "CHECKPOINT="
set "ONNX="
set "PROVIDER=auto"
set "VAD_SILENCE=0.45"
set "INPUT_GAIN=1.0"
set "CONFIG="
set "LIST_DEVICES=0"
set "LOG_LEVEL=INFO"

:parse
if "%~1"=="" goto run
if "%~1"=="--onnx" set "ONNX=%~2" & shift & shift & goto parse
if "%~1"=="--provider" set "PROVIDER=%~2" & shift & shift & goto parse
if "%~1"=="--vad-silence" set "VAD_SILENCE=%~2" & shift & shift & goto parse
if "%~1"=="--input-gain" set "INPUT_GAIN=%~2" & shift & shift & goto parse
if "%~1"=="--checkpoint" set "CHECKPOINT=%~2" & shift & shift & goto parse
if "%~1"=="--config" set "CONFIG=%~2" & shift & shift & goto parse
if "%~1"=="--list-devices" set "LIST_DEVICES=1" & shift & goto parse
if "%~1"=="--log-level" set "LOG_LEVEL=%~2" & shift & shift & goto parse
shift & goto parse

:run
echo ==========================================
echo  Samuel Realtime Parrot — Windows Launcher
echo ==========================================

set "CKPT=%CHECKPOINT%"
if "%CKPT%"=="" (
    if exist "samuel_custom_last.pt" (
        set "CKPT=samuel_custom_last.pt"
        echo Checkpoint (auto): samuel_custom_last.pt
    ) else (
        set "CKPT=hf:vvolhejn/samuel"
        echo Checkpoint: hf:vvolhejn/samuel
    )
) else (
    echo Checkpoint: %CKPT%
)

if defined ONNX (
    echo ONNX: %ONNX%
) else (
    if exist "models\samuel_custom_controller.onnx" (
        echo ONNX (auto): models\samuel_custom_controller.onnx
    ) else if exist "models\samuel_controller.onnx" (
        echo ONNX (auto): models\samuel_controller.onnx
    ) else (
        echo ONNX: (PyTorch mode)
    )
)

echo Provider: %PROVIDER%
echo VAD Silence: %VAD_SILENCE% sec
echo Input Gain: %INPUT_GAIN%
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%\check_vbcable.ps1"

echo.
echo Starting pipeline... (Ctrl+C to stop)
echo.

if "%LIST_DEVICES%"=="1" (
    uv run python -m samuel_realtime --list-devices --log-level %LOG_LEVEL%
) else (
    set "CMD=uv run python -m samuel_realtime"
    if defined CKPT set "CMD=!CMD! --checkpoint "!CKPT!""
    if defined ONNX set "CMD=!CMD! --onnx "!ONNX!""
    if not "%PROVIDER%"=="auto" set "CMD=!CMD! --provider %PROVIDER%"
    if not "%VAD_SILENCE%"=="0.45" set "CMD=!CMD! --vad-silence %VAD_SILENCE%"
    if not "%INPUT_GAIN%"=="1.0" set "CMD=!CMD! --input-gain %INPUT_GAIN%"
    if defined CONFIG set "CMD=!CMD! --config "!CONFIG!""
    set "CMD=!CMD! --log-level %LOG_LEVEL%"
    echo !CMD!
    !CMD!
)

endlocal
pause