@echo off
REM run_windows.bat — Samuel Realtime Parrot launcher for Windows (VB-CABLE)
REM Usage: scripts\run_windows.bat [--onnx models/samuel_controller.onnx] [--provider webgpu]

set PYTHONPATH=%~dp0..\src
set VENV=%~dp0..\.venv

REM Check for virtual environment
if exist "%VENV%\Scripts\activate.bat" (
    call "%VENV%\Scripts\activate.bat"
) else (
    echo [warn] No .venv found — using system Python
)

REM Default arguments
set CHECKPOINT=hf:vvolhejn/samuel
set ONNX=
set PROVIDER=auto
set VAD_SILENCE=0.45

REM Parse arguments
:parse
if "%~1"=="" goto run
if "%~1"=="--onnx" set ONNX=%~2 & shift & shift & goto parse
if "%~1"=="--provider" set PROVIDER=%~2 & shift & shift & goto parse
if "%~1"=="--vad-silence" set VAD_SILENCE=%~2 & shift & shift & goto parse
if "%~1"=="--checkpoint" set CHECKPOINT=%~2 & shift & shift & goto parse
if "%~1"=="--list-devices" set LIST=1 & shift & goto parse
if "%~1"=="--log-level" set LOG_LEVEL=%~2 & shift & shift & goto parse
shift & goto parse

:run
echo ==========================================
echo  Samuel Realtime Parrot — Windows Launcher
echo ==========================================
echo Checkpoint: %CHECKPOINT%
if defined ONNX echo ONNX: %ONNX%
echo Provider: %PROVIDER%
echo VAD Silence: %VAD_SILENCE% sec
echo.

REM Auto-detect CABLE devices
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0check_vbcable.ps1"

echo.
echo Starting pipeline... (Ctrl+C to stop)
echo.

if defined LIST (
    uv run python -m samuel_realtime --list-devices --log-level %LOG_LEVEL%
) else (
    uv run python -m samuel_realtime ^
        --checkpoint "%CHECKPOINT%" ^
        %ONNX:--onnx=%ONNX% % ^
        --provider %PROVIDER% ^
        --vad-silence %VAD_SILENCE% ^
        --log-level %LOG_LEVEL%
)

pause
