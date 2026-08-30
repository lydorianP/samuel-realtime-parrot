#!/bin/bash
# setup_virtual_mic.sh — Linux PipeWire / PulseAudio virtual mic for Samuel
# Creates a Null Sink "SamuelMic" whose Monitor acts as a virtual microphone.
# Primary target is Windows VB-CABLE; this script mirrors the routing on Linux dev (Arch).
set -e

SINK_NAME="SamuelMic"
SINK_DESC="Samuel_Virtual_Mic"

# PipeWire provides PulseAudio compat via pactl; works on pure PulseAudio too
if ! command -v pactl >/dev/null 2>&1; then
    echo "[error] pactl not found — install pulseaudio or pipewire-pulse"
    exit 1
fi

# Check if sink already exists
if pactl list sinks short | grep -q "$SINK_NAME"; then
    echo "[info] Sink '$SINK_NAME' already exists — skipping create"
    pactl list sinks short | grep "$SINK_NAME"
    echo "[info] Monitor source: Monitor of $SINK_NAME"
    pactl list sources short | grep "$SINK_NAME" || true
    exit 0
fi

echo "[info] Creating null sink: $SINK_NAME ($SINK_DESC)"
MODULE_ID=$(pactl load-module module-null-sink sink_name="$SINK_NAME" sink_properties=device.description="$SINK_DESC")
if [ -z "$MODULE_ID" ]; then
    echo "[error] Failed to load module-null-sink"
    exit 1
fi

echo "[ok] Loaded module-null-sink id=$MODULE_ID"
echo "[ok] Playback device: $SINK_DESC / $SINK_NAME"
echo "[ok] Recording device (virtual mic): Monitor of $SINK_DESC"

# Verify
sleep 0.5
pactl list sinks short | grep "$SINK_NAME"
pactl list sources short | grep "$SINK_NAME" || echo "[warn] Monitor not yet visible — try: pactl list sources short"

echo ""
echo "Routing:"
echo "  Your Python script → OutputStream(device=\"$SINK_DESC\" or \"$SINK_NAME\")"
echo "  Discord/Zoom input → \"Monitor of $SINK_DESC\""
echo "  Use pavucontrol (PulseAudio Volume Control) to verify visually."
echo ""
echo "To make persistent across reboot, add to /etc/pulse/default.pa or ~/.config/pulse/default.pa:"
echo "  load-module module-null-sink sink_name=$SINK_NAME sink_properties=device.description=$SINK_DESC"
echo ""
echo "To remove: pactl unload-module $MODULE_ID  (or pactl unload-module module-null-sink — removes all)"

