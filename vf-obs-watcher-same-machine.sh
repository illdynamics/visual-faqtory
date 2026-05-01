#!/usr/bin/env bash
# vf-obs-watcher-same-machine.sh — Visual FaQtory OBS A/B Watcher
# ════════════════════════════════════════════════════════════════════════
# Watches the interpolated video output directory and triggers an ALIGNED
# A/B swap in OBS whenever a new mp4 appears.
#
# v0.9.2+ behaviour:
#   The default obs-swap.py mode is now ALIGNED — it waits for the
#   currently visible clip to finish naturally before revealing the new
#   one. This keeps visuals on-beat with whatever's been queued up
#   musically, and avoids mid-clip jumps.
#
# Tuning knobs (env vars; override before launching):
#   OBS_HOST                   default: 127.0.0.1
#   OBS_PORT                   default: 4455
#   OBS_PASSWORD               default: Setyup34!
#   OBS_SCENE                  default: Ill Dynamics - Live on SkankOut
#   OBS_PREWARM_SEC            default: 0.8   (target-PLAYING wait)
#   OBS_MAX_WAIT_CURRENT_SEC   default: 180   (max wait for current end)
#   OBS_END_THRESHOLD_MS       default: 200   (effective-ended margin)
#   OBS_SWAP_MODE              default: aligned   ("aligned" | "immediate")
#
# With the aligned default, you can DISABLE "Close file when inactive" on
# both OBS sources — media lifecycle is controlled via WebSocket.
#
# Latest-wins note (TODO):
#   Today the watcher processes one swap at a time (serialized via flock)
#   and only updates .active_slot after obs-swap.py succeeds. That means
#   if many files arrive while we're waiting on the current clip to end,
#   each will be processed in arrival order — so a 30-clip burst will
#   take ~30 boundary waits. A future "latest-wins" mode could collapse
#   the backlog to "use the most recent file when the current ends".
#
# Part of Visual FaQtory v0.9.2-beta
# ════════════════════════════════════════════════════════════════════════

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERP_DIR="$BASE_DIR/run/videos"
OBS_DIR="$BASE_DIR/run/obs"
STATE_FILE="$OBS_DIR/.active_slot"
LOCK_FILE="$OBS_DIR/.swap.lock"

PYTHON_BIN="$BASE_DIR/.venv/bin/python"
OBS_SWAP_SCRIPT="$BASE_DIR/obs-swap.py"

# ── Tuning defaults (override via env before launching) ──────────────────
export OBS_PREWARM_SEC="${OBS_PREWARM_SEC:-0.8}"
export OBS_MAX_WAIT_CURRENT_SEC="${OBS_MAX_WAIT_CURRENT_SEC:-180}"
export OBS_END_THRESHOLD_MS="${OBS_END_THRESHOLD_MS:-200}"
export OBS_SWAP_MODE="${OBS_SWAP_MODE:-aligned}"

mkdir -p "$OBS_DIR"

# ── Initialize state (start with A active) ───────────────────────────────
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

# Resolve a python interpreter, preferring repo .venv but falling back
# gracefully if it's missing (lets manual `bash -n` and CI checks pass on
# fresh checkouts).
resolve_python() {
    if [[ -x "$PYTHON_BIN" ]]; then
        echo "$PYTHON_BIN"
        return 0
    fi
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return 0
    fi
    echo "python"
}

process_file() {
    local file="$1"

    local active target python_bin
    active="$(get_active_slot)"
    python_bin="$(resolve_python)"

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

    echo "Triggering OBS swap → $target  " \
         "(mode=$OBS_SWAP_MODE, prewarm=${OBS_PREWARM_SEC}s, " \
         "max_wait_current=${OBS_MAX_WAIT_CURRENT_SEC}s, " \
         "end_threshold=${OBS_END_THRESHOLD_MS}ms)"

    # Pass the new args explicitly so obs-swap.py picks them up regardless
    # of whether the env was inherited cleanly.
    if "$python_bin" "$OBS_SWAP_SCRIPT" "$target" \
        --prewarm "$OBS_PREWARM_SEC" \
        --max-wait-current "$OBS_MAX_WAIT_CURRENT_SEC" \
        --end-threshold-ms "$OBS_END_THRESHOLD_MS"; then
        # Only after the visible swap actually completed do we update
        # state. If obs-swap.py crashed or fail-opened with a non-zero
        # exit, .active_slot stays unchanged so the NEXT file still gets
        # routed at the (still-actually-active) inactive slot.
        set_active_slot "$target"
        echo "Now active: $target"
    else
        echo "ERROR: obs-swap.py failed for slot $target — keeping " \
             ".active_slot=$active so the next file still routes correctly." >&2
    fi
    echo "--------------------------------------"
}

# Wrapper that serialises swaps. flock is used in non-blocking mode
# (-n) so a swap that's already in progress causes the new event to be
# dropped rather than queueing forever — for the live-set use case we'd
# rather lose one stale prewarm than build a backlog. With aligned mode
# the running swap will already pick up the most recent inactive-slot
# file when the current clip ends, since copy_atomic happens BEFORE the
# lock is taken. (TODO future: latest-wins mode.)
process_file_locked() {
    local file="$1"
    # Use a subshell so flock's FD scoping is contained.
    (
        # Open lock fd 9 on the lockfile; -n = non-blocking, fail fast if held.
        if ! flock -n 9; then
            echo "swap already in progress — skipping event for $file" >&2
            exit 0
        fi
        process_file "$file"
    ) 9>"$LOCK_FILE"
}

# ── Manual test checklist (printed on startup) ───────────────────────────
print_manual_test_checklist() {
    cat <<'EOF'
Manual test checklist (live OBS required):
  1. Start OBS with both media sources pointing at run/obs/current_A.mp4
     and run/obs/current_B.mp4.
  2. Start this watcher.
  3. Let A play and loop.
  4. Drop a new mp4 into run/videos.
  5. Confirm A does NOT switch immediately.
  6. Confirm A's loop is disabled only after new video is parked.
  7. Confirm A plays until its end.
  8. Confirm B starts from frame 0 and becomes visible.
  9. Confirm B loops after becoming active.
 10. Drop another mp4; confirm B finishes before A becomes visible again.
EOF
}

echo "Watching $INTERP_DIR for new videos..."
echo "OBS swap mode:      ${OBS_SWAP_MODE}"
echo "OBS prewarm:        ${OBS_PREWARM_SEC}s"
echo "OBS max-wait curr:  ${OBS_MAX_WAIT_CURRENT_SEC}s"
echo "OBS end threshold:  ${OBS_END_THRESHOLD_MS}ms"
print_manual_test_checklist

# ── Monitor directory for new completed files ───────────────────────────
inotifywait -m -e close_write --format "%f" "$INTERP_DIR" | while read -r file; do
    if [[ "$file" == *.mp4 ]]; then
        process_file_locked "$INTERP_DIR/$file"
    fi
done
