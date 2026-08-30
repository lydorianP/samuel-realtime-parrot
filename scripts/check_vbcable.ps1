# check_vbcable.ps1 — Windows VB-CABLE verification for Samuel Realtime Parrot
# Usage: powershell -ExecutionPolicy Bypass -File scripts\check_vbcable.ps1
# Checks that VB-CABLE Virtual Audio Device is installed (reboot usually required after install).

$ErrorActionPreference = "Continue"

Write-Host "=== Samuel VB-CABLE Checker (Windows) ===" -ForegroundColor Cyan
Write-Host "Target devices: Playback = 'CABLE Input' | Recording = 'CABLE Output' (VB-Audio)" -ForegroundColor Gray
Write-Host ""

# 1. Enumerate PnP devices matching VB-Audio / CABLE
Write-Host "[1/3] Checking PnP devices for VB-Audio..." -ForegroundColor Yellow
try {
    $pnp = Get-PnpDevice -Class MEDIA -PresentOnly -ErrorAction SilentlyContinue |
           Where-Object { $_.FriendlyName -match "VB-Audio|CABLE|Virtual.*Audio" }
    if ($pnp) {
        $pnp | Format-Table Status, Class, FriendlyName, InstanceId -AutoSize
        Write-Host "[ok] VB-Audio PnP device found" -ForegroundColor Green
    } else {
        # Fallback: broader search
        $pnp2 = Get-CimInstance Win32_PnPEntity -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -match "VB-Audio|CABLE" }
        if ($pnp2) {
            $pnp2 | Select-Object Name, Status, DeviceID | Format-Table -AutoSize
            Write-Host "[ok] VB-Audio device found via CIM" -ForegroundColor Green
        } else {
            Write-Host "[warn] No VB-Audio PnP device found" -ForegroundColor Yellow
        }
    }
} catch {
    Write-Host "[warn] PnP check failed: $_" -ForegroundColor Yellow
}

# 2. Check audio endpoints via WMI/MMApi (works without extra modules)
Write-Host ""
Write-Host "[2/3] Checking audio endpoints (MMDevice)..." -ForegroundColor Yellow
try {
    $endpoints = Get-CimInstance Win32_SoundDevice -ErrorAction SilentlyContinue
    if ($endpoints) {
        $endpoints | Select-Object Name, Status, DeviceID | Format-Table -AutoSize
        $cable = $endpoints | Where-Object { $_.Name -match "CABLE" }
        if ($cable) {
            Write-Host "[ok] CABLE device present in Win32_SoundDevice" -ForegroundColor Green
        } else {
            Write-Host "[info] CABLE not in Win32_SoundDevice — checking via registry fallback" -ForegroundColor Gray
        }
    }
} catch {
    Write-Host "[warn] SoundDevice check failed: $_" -ForegroundColor Yellow
}

# Direct registry check for VB-CABLE (more reliable)
Write-Host ""
Write-Host "[3/3] Registry check for VB-CABLE driver..." -ForegroundColor Yellow
$found = $false
$paths = @(
    "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio\Render",
    "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio\Capture"
)
foreach ($base in $paths) {
    if (Test-Path $base) {
        Get-ChildItem $base -ErrorAction SilentlyContinue | ForEach-Object {
            $props = Get-ItemProperty "$($_.PSPath)\Properties" -ErrorAction SilentlyContinue
            if ($props) {
                $dump = $props | Out-String
                if ($dump -match "CABLE|VB-Audio") {
                    Write-Host "[ok] Found CABLE entry under $base\$($_.PSChildName)" -ForegroundColor Green
                    $found = $true
                }
            }
        }
    }
}
if (-not $found) {
    Write-Host "[info] Registry scan did not find CABLE — may still be ok if not yet rebooted" -ForegroundColor Gray
}

# Summary & guidance
Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Cyan
Write-Host "If no CABLE devices were found:" -ForegroundColor Yellow
Write-Host "  1. Download VB-CABLE from https://vb-audio.com/Cable/ (donation-ware, free)"
Write-Host "  2. Run installer as Administrator → Reboot (required for driver load)"
Write-Host "  3. After reboot, re-run this script. Expect:"
Write-Host "     - Playback device: 'CABLE Input'  (your Python script writes here)"
Write-Host "     - Recording device: 'CABLE Output' (Discord/Zoom listens here)"
Write-Host ""
Write-Host "Routing for Samuel:" -ForegroundColor Cyan
Write-Host '  Python OutputStream(device="CABLE Input", samplerate=44100, channels=1)'
Write-Host '  Discord/Zoom Input: "CABLE Output"'
Write-Host ""
Write-Host "Tip: Use mmsys.cpl (Sound control panel) or Settings > Sound to verify."
Write-Host "Wine note: This script will show no devices under Wine (vkd3d) — expected. Use native Windows for final verify."

