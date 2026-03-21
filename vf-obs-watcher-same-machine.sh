#!/usr/bin/env bash
# vf-obs-watcher-same-machine.sh — Visual FaQtory OBS A/B Watcher
# ════════════════════════════════════════════════════════════════════════
# Watches the interpolated video output directory and triggers a prewarm
# A/B swap in OBS whenever a new mp4 appears.
#
# Prewarm timing:
#   Set OBS_PREWARM_SEC (seconds) to control how long obs-swap.py waits
#   for the target source to start playing before completing the swap.
#   Default: 0.8s. Tune this to your machine / file size.
#
# OBS WebSocket settings are read from environment variables by obs-swap.py:
#   OBS_HOST        (default: 127.0.0.1)
#   OBS_PORT        (default: 4455)
#   OBS_PASSWORD    (default: Setyup34!)
#   OBS_SCENE       (default: Ill Dynamics - Live on SkankOut)
#   OBS_PREWARM_SEC (default: 0.8)
#
# Note: With prewarm swap active, you can DISABLE "Close file when inactive"
#       on both OBS sources — media reload is handled via WebSocket.
#
# Part of QonQrete Visual FaQtory v0.5.8-beta
# ════════════════════════════════════════════════════════════════════════

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERP_DIR="$BASE_DIR/run/videos_interpolated"
OBS_DIR="$BASE_DIR/run/obs"
STATE_FILE="$OBS_DIR/.active_slot"

PYTHON_BIN="$BASE_DIR/.venv/bin/python"
OBS_SWAP_SCRIPT="$BASE_DIR/obs-swap.py"

# ── Prewarm tuning ─────────────────────────────────────────────────────
# Override here or via environment before launching this script.
# e.g.:  export OBS_PREWARM_SEC=1.2  &&  ./vf-obs-watcher-same-machine.sh
export OBS_PREWARM_SEC="${OBS_PREWARM_SEC:-0.8}"

mkdir -p "$OBS_DIR"

# ── Initialize state (start with A active) ─────────────────────────────
if [[ ! -f "$STATE_FILE" ]]; then
    echo "A" > "$STATE_FILE"
fi

get_active_slot() {
    cat "$STATE_FILE"
}

set_active_slot() {
    echo "$1" > "$STATE_FILE"
}

copy_atomic() {
    local src="$1"
    local dst="$2"

    cp "$src" "$dst.next"
    mv -f "$dst.next" "$dst"
}

process_file() {
    local file="$1"

    active="$(get_active_slot)"

    if [[ "$active" == "A" ]]; then
        target="B"
    else
        target="A"
    fi

    echo "--------------------------------------"
    echo "New file detected: $file"
    echo "Currently active: $active"
    echo "Copying to inactive slot: $target"

    copy_atomic "$file" "$OBS_DIR/current_${target}.mp4"

    echo "Triggering OBS prewarm swap to $target (prewarm=${OBS_PREWARM_SEC}s)"
    "$PYTHON_BIN" "$OBS_SWAP_SCRIPT" "$target" --prewarm "$OBS_PREWARM_SEC"

    set_active_slot "$target"

    echo "Now active: $target"
    echo "--------------------------------------"
}

echo "Watching $INTERP_DIR for new interpolated videos..."
echo "OBS prewarm: ${OBS_PREWARM_SEC}s (set OBS_PREWARM_SEC to override)"

# ── Monitor directory for new completed files ──────────────────────────
inotifywait -m -e close_write --format "%f" "$INTERP_DIR" | while read -r file; do
    if [[ "$file" == *.mp4 ]]; then
        process_file "$INTERP_DIR/$file"
    fi
done
