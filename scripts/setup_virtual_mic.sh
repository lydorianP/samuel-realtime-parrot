#!/bin/bash
# setup_virtual_mic.sh — Linux PipeWire / PulseAudio virtual mic for Samuel
# Creates a Null Sink "SamuelMic" + Virtual Source "SamuelMic.monitor"
# The virtual source appears as a proper Audio/Source node visible to all apps (pavucontrol, Audacity, Discord, OBS, etc.)
set -e

SINK_NAME="SamuelMic"
SINK_DESC="Samuel_Virtual_Mic"
SOURCE_NAME="SamuelMic.monitor"
SOURCE_DESC="Samuel_Virtual_Mic_Monitor"

# PipeWire provides PulseAudio compat via pactl; works on pure PulseAudio too
if ! command -v pactl >/dev/null 2>&1; then
    echo "[error] pactl not found — install pulseaudio or pipewire-pulse"
    exit 1
fi

# Check if sink already exists
if pactl list sinks short | grep -q "^[0-9]*\s$SINK_NAME\s"; then
    echo "[info] Sink '$SINK_NAME' already exists"
else
    echo "[info] Creating null sink: $SINK_NAME ($SINK_DESC)"
    pactl load-module module-null-sink sink_name="$SINK_NAME" sink_properties=device.description="$SINK_DESC" >/dev/null
    echo "[ok] Created sink"
fi

# Check if virtual source already exists
if pactl list sources short | grep -q "^[0-9]*\s$SOURCE_NAME\s"; then
    echo "[info] Virtual source '$SOURCE_NAME' already exists"
else
    echo "[info] Creating virtual source: $SOURCE_NAME ($SOURCE_DESC) monitoring $SINK_NAME.monitor"
    pactl load-module module-virtual-source source_name="$SOURCE_NAME" source_properties=device.description="$SOURCE_DESC" master="$SINK_NAME.monitor" >/dev/null
    echo "[ok] Created virtual source (Audio/Source)"
fi

sleep 0.5
echo ""
echo "=== Verification ==="
pactl list sinks short | grep "$SINK_NAME"
pactl list sources short | grep -E "$SINK_NAME|$SOURCE_NAME"
echo ""
echo "Routing:"
echo "  Your Python script → OutputStream(device=\"$SINK_DESC\" or \"$SINK_NAME\")"
echo "  Discord/Zoom/OBS/Audacity input → \"$SOURCE_DESC\" (or \"$SOURCE_NAME\")"
echo "  Use pavucontrol (Recording tab) to select \"$SOURCE_DESC\""
echo ""
echo "To make persistent across reboot, add to ~/.config/pulse/default.pa:"
echo "  load-module module-null-sink sink_name=$SINK_NAME sink_properties=device.description=$SINK_DESC"
echo "  load-module module-virtual-source source_name=$SOURCE_NAME source_properties=device.description=$SOURCE_DESC master=$SINK_NAME.monitor"
echo ""
echo "To remove: pactl unload-module module-virtual-source && pactl unload-module module-null-sink"

