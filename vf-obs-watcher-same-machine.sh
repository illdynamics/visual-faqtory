#!/usr/bin/env bash
# vf-obs-watcher-same-machine.sh — Visual FaQtory OBS A/B Watcher (HARDENED)
# ════════════════════════════════════════════════════════════════════════
# Watches the video output directory and triggers an ALIGNED A/B swap
# in OBS whenever a new mp4 appears.
#
# v0.9.3+ hardened behaviour:
#   - set -e REMOVED: subcommand failures do not crash the watcher.
#     The watcher MUST stay alive through transient OBS/network errors.
#   - File stability check: verifies the new mp4 isn't still growing
#     before copying it to the inactive slot.
#   - Play confirmation: after swap, verifies the target source is
#     actually playing. Retries play up to 3 times with 1s backoff.
#   - Last-known-good fallback: if the new source fails to play after
#     retries, reverts to the previous known-good active source.
#   - Diagnostics artifact: writes swap events, play results, and
#     fallback activations to run/obs/.watcher_diagnostics.jsonl.
#   - Structured timestamps on every log line.
#   - inotifywait restart loop: auto-recovers if the watch fails.
#
# Part of Visual FaQtory v0.9.3-beta
# ════════════════════════════════════════════════════════════════════════

set -uo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERP_DIR="$BASE_DIR/run/videos"
OBS_DIR="$BASE_DIR/run/obs"
STATE_FILE="$OBS_DIR/.active_slot"
LOCK_FILE="$OBS_DIR/.swap.lock"
DIAG_FILE="$OBS_DIR/.watcher_diagnostics.jsonl"
KNOWN_GOOD_FILE="$OBS_DIR/.last_known_good_video"

PYTHON_BIN="$BASE_DIR/.venv/bin/python"
OBS_SWAP_SCRIPT="$BASE_DIR/obs-swap.py"

export OBS_PREWARM_SEC="${OBS_PREWARM_SEC:-0.8}"
export OBS_MAX_WAIT_CURRENT_SEC="${OBS_MAX_WAIT_CURRENT_SEC:-180}"
export OBS_END_THRESHOLD_MS="${OBS_END_THRESHOLD_MS:-200}"
export OBS_SWAP_MODE="${OBS_SWAP_MODE:-aligned}"
OBS_PLAY_RETRIES="${OBS_PLAY_RETRIES:-3}"
OBS_PLAY_RETRY_DELAY_SEC="${OBS_PLAY_RETRY_DELAY_SEC:-1.0}"
OBS_FILE_STABLE_CHECKS="${OBS_FILE_STABLE_CHECKS:-2}"
OBS_FILE_STABLE_INTERVAL="${OBS_FILE_STABLE_INTERVAL:-0.3}"

mkdir -p "$OBS_DIR"

if [[ ! -f "$STATE_FILE" ]]; then
    echo "A" > "$STATE_FILE"
fi

# ── Timestamped logging ──────────────────────────────────────────────────
ts_log()  { echo "[$(date '+%H:%M:%S')] [watcher] $*"; }
ts_warn() { echo "[$(date '+%H:%M:%S')] [watcher] WARN: $*" >&2; }
ts_error(){ echo "[$(date '+%H:%M:%S')] [watcher] ERROR: $*" >&2; }

# ── Diagnostics artifact writer ───────────────────────────────────────────
write_diag() {
    local event="$1" slot="$2" extra="${3:-}"
    local ts
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date +%Y-%m-%dT%H:%M:%SZ)"
    printf '{"ts":"%s","event":"%s","slot":"%s","extra":"%s"}\n' \
        "$ts" "$event" "$slot" "$extra" >> "$DIAG_FILE"
}

get_active_slot() { cat "$STATE_FILE"; }
set_active_slot() { echo "$1" > "$STATE_FILE"; }

# ── File stability check ─────────────────────────────────────────────────
wait_file_stable() {
    local file="$1" checks="$2" interval="$3" last_size size stable
    last_size=-1; stable=0
    while (( stable < checks )); do
        if [[ ! -f "$file" ]]; then
            ts_warn "File vanished while waiting for stability: $file"
            return 1
        fi
        size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo 0)
        if [[ "$size" -gt 0 && "$size" == "$last_size" ]]; then
            stable=$((stable + 1))
        else
            stable=0
        fi
        last_size="$size"
        sleep "$interval"
    done
    if [[ ! -s "$file" ]]; then
        ts_error "File is empty after stability wait: $file"
        return 1
    fi
    if ! dd if="$file" bs=1 count=1 of=/dev/null status=none 2>/dev/null; then
        ts_error "File is not readable after stability wait: $file"
        return 1
    fi
    return 0
}

# ── Atomic copy ──────────────────────────────────────────────────────────
copy_atomic() {
    local src="$1" dst="$2"
    cp "$src" "$dst.next" || { ts_error "copy_atomic: cp failed $src -> $dst.next"; return 1; }
    mv -f "$dst.next" "$dst" || { ts_error "copy_atomic: mv failed $dst.next -> $dst"; return 1; }
    return 0
}

resolve_python() {
    if [[ -x "$PYTHON_BIN" ]]; then echo "$PYTHON_BIN"; return 0; fi
    if command -v python3 >/dev/null 2>&1; then command -v python3; return 0; fi
    echo "python"
}

# ── Known-good video tracking ────────────────────────────────────────────
mark_known_good() {
    local slot="$1"
    local src="$OBS_DIR/current_${slot}.mp4"
    if [[ -f "$src" && -s "$src" ]]; then
        cp "$src" "$KNOWN_GOOD_FILE" 2>/dev/null || true
        ts_log "Known-good marked: slot=$slot"
        write_diag "known_good_marked" "$slot" "file=$src"
    fi
}

restore_known_good() {
    local slot="$1"
    local dst="$OBS_DIR/current_${slot}.mp4"
    if [[ -f "$KNOWN_GOOD_FILE" && -s "$KNOWN_GOOD_FILE" ]]; then
        cp "$KNOWN_GOOD_FILE" "$dst" 2>/dev/null || return 1
        ts_warn "Restored known-good video to slot $slot"
        write_diag "known_good_restored" "$slot" ""
        return 0
    fi
    ts_error "No known-good video available to restore!"
    return 1
}

# ── Play confirmation with retries ───────────────────────────────────────
confirm_source_playing() {
    local slot="$1" retries="$2" delay="$3"
    local python_bin source_name attempt state media_state cursor
    python_bin="$(resolve_python)"
    source_name="Live-Visuals-${slot}"

    for (( attempt=1; attempt <= retries; attempt++ )); do
        state=$("$python_bin" -c "
import os, sys
try:
    from obsws_python import ReqClient
    host = os.environ.get('OBS_HOST', '127.0.0.1')
    port = int(os.environ.get('OBS_PORT', '4455'))
    password = os.environ.get('OBS_PASSWORD', 'Setyup34!')
    cl = ReqClient(host=host, port=port, password=password)
    resp = cl.get_media_input_status('${source_name}')
    ms = getattr(resp, 'media_state', getattr(resp, 'mediaState', 'UNKNOWN'))
    cur = getattr(resp, 'media_cursor', getattr(resp, 'mediaCursor', None))
    print(f'{ms}|{cur}')
except Exception as e:
    print(f'ERROR|{e}')
" 2>/dev/null) || state="ERROR|connection_failed"

        media_state="${state%%|*}"; cursor="${state#*|}"
        ts_log "Play confirm $attempt/$retries slot=$slot: $media_state cursor=$cursor"
        write_diag "play_confirm" "$slot" "attempt=$attempt state=$media_state cursor=$cursor"

        if [[ "$media_state" == "OBS_MEDIA_STATE_PLAYING" ]]; then
            ts_log "✓ Slot $slot confirmed PLAYING (cursor=$cursor)"
            write_diag "play_confirmed" "$slot" "cursor=$cursor"
            return 0
        fi

        if (( attempt < retries )); then
            ts_log "Retrying play for slot $slot in ${delay}s..."
            "$python_bin" -c "
import os, sys, time
try:
    from obsws_python import ReqClient
    host = os.environ.get('OBS_HOST', '127.0.0.1')
    port = int(os.environ.get('OBS_PORT', '4455'))
    password = os.environ.get('OBS_PASSWORD', 'Setyup34!')
    cl = ReqClient(host=host, port=port, password=password)
    cl.trigger_media_input_action('${source_name}', 'OBS_WEBSOCKET_MEDIA_INPUT_ACTION_PLAY')
    time.sleep(0.2)
    cl.trigger_media_input_action('${source_name}', 'OBS_WEBSOCKET_MEDIA_INPUT_ACTION_RESTART')
except Exception as e:
    print(f'Play retry error: {e}', file=sys.stderr)
" 2>/dev/null || true
            sleep "$delay"
        fi
    done

    ts_error "Slot $slot FAILED to confirm PLAYING after $retries attempts"
    write_diag "play_failed" "$slot" "attempts=$retries"
    return 1
}

# ── Main swap orchestrator ───────────────────────────────────────────────
process_file() {
    local file="$1" active target python_bin swap_rc inactive_file
    active="$(get_active_slot)"
    python_bin="$(resolve_python)"

    if [[ "$active" == "A" ]]; then target="B"; else target="A"; fi

    ts_log "──────────────────────────────────────"
    ts_log "New file: $file  |  active=$active  |  target=$target"

    # File stability
    ts_log "Checking file stability (checks=${OBS_FILE_STABLE_CHECKS})..."
    if ! wait_file_stable "$file" "$OBS_FILE_STABLE_CHECKS" "$OBS_FILE_STABLE_INTERVAL"; then
        ts_error "File stability FAILED — SKIPPING $file"
        write_diag "file_stability_failed" "$target" "file=$file"
        return 1
    fi
    ts_log "File stable ✓"

    # Copy
    inactive_file="$OBS_DIR/current_${target}.mp4"
    if ! copy_atomic "$file" "$inactive_file"; then
        ts_error "Copy FAILED — SKIPPING"
        write_diag "copy_failed" "$target" "file=$file"
        return 1
    fi
    local fsize
    fsize=$(stat -c%s "$inactive_file" 2>/dev/null || stat -f%z "$inactive_file" 2>/dev/null || echo 0)
    write_diag "copy_complete" "$target" "file=$file size=$fsize"

    # Swap
    ts_log "Triggering OBS swap → $target (mode=${OBS_SWAP_MODE})"
    write_diag "swap_triggered" "$target" "mode=${OBS_SWAP_MODE}"
    "$python_bin" "$OBS_SWAP_SCRIPT" "$target" \
        --prewarm "$OBS_PREWARM_SEC" \
        --max-wait-current "$OBS_MAX_WAIT_CURRENT_SEC" \
        --end-threshold-ms "$OBS_END_THRESHOLD_MS" || true
    swap_rc=$?

    if [[ $swap_rc -ne 0 ]]; then
        ts_warn "obs-swap.py exited with code $swap_rc — continuing with play confirmation"
        write_diag "swap_nonzero_exit" "$target" "exit_code=$swap_rc"
    else
        write_diag "swap_complete" "$target" ""
    fi

    # Play confirmation with retries
    if confirm_source_playing "$target" "$OBS_PLAY_RETRIES" "$OBS_PLAY_RETRY_DELAY_SEC"; then
        set_active_slot "$target"
        mark_known_good "$target"
        ts_log "✓ Now active: $target"
        write_diag "active_changed" "$target" "from=$active"
    else
        ts_error "Play confirmation FAILED for $target — triggering fallback"
        write_diag "fallback_triggered" "$target" "reason=play_confirmation_failed"

        if restore_known_good "$target"; then
            ts_warn "Fallback: restored known-good video to $target, retrying swap"
            "$python_bin" "$OBS_SWAP_SCRIPT" "$target" \
                --prewarm "$OBS_PREWARM_SEC" \
                --max-wait-current "$OBS_MAX_WAIT_CURRENT_SEC" \
                --end-threshold-ms "$OBS_END_THRESHOLD_MS" || true

            if confirm_source_playing "$target" 2 "$OBS_PLAY_RETRY_DELAY_SEC"; then
                set_active_slot "$target"
                ts_warn "Fallback SUCCESS: $target now playing known-good video"
                write_diag "fallback_success" "$target" "source=known_good"
            else
                ts_error "CRITICAL: Fallback also failed. Keeping active=$active unchanged."
                write_diag "fallback_failed" "$active" "target=$target"
            fi
        else
            ts_error "CRITICAL: No known-good video. Keeping active=$active unchanged."
            write_diag "critical_no_fallback" "$active" "target=$target"
        fi
    fi
    ts_log "──────────────────────────────────────"
}

process_file_locked() {
    local file="$1"
    (
        if ! flock -n 9; then
            ts_warn "Swap already in progress — skipping event for $file"
            exit 0
        fi
        process_file "$file"
    ) 9>"$LOCK_FILE"
}

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

seed_known_good() {
    local active active_file
    active="$(get_active_slot)"
    active_file="$OBS_DIR/current_${active}.mp4"
    if [[ -f "$active_file" && -s "$active_file" ]]; then
        cp "$active_file" "$KNOWN_GOOD_FILE" 2>/dev/null || true
        ts_log "Seeded known-good from active slot $active"
    fi
}

# ── Startup ──────────────────────────────────────────────────────────────
ts_log "=== Visual FaQtory OBS A/B Watcher STARTING (v0.9.3-hardened) ==="
ts_log "Watch dir:         $INTERP_DIR"
ts_log "OBS swap mode:     ${OBS_SWAP_MODE}"
ts_log "Play retries:      ${OBS_PLAY_RETRIES}"
ts_log "File stable checks:${OBS_FILE_STABLE_CHECKS}"
print_manual_test_checklist
seed_known_good

# ── Enforce loop ON on active source at startup ──────────────────────────
enforce_active_loop() {
    local active slot python_bin source_name
    active="$(get_active_slot)"
    python_bin="$(resolve_python)"
    source_name="Live-Visuals-${active}"
    ts_log "Ensuring loop ON for active source ${source_name}..."
    "$python_bin" -c "
import os
try:
    from obsws_python import ReqClient
    host = os.environ.get('OBS_HOST', '127.0.0.1')
    port = int(os.environ.get('OBS_PORT', '4455'))
    password = os.environ.get('OBS_PASSWORD', 'Setyup34!')
    cl = ReqClient(host=host, port=port, password=password)
    cl.set_input_settings('${source_name}', {'looping': True}, overlay=True)
    print(f'Loop ON for ${source_name}')
except Exception as e:
    print(f'Could not set loop on ${source_name}: {e}')
" 2>/dev/null || ts_warn "Failed to enforce loop on ${source_name}"
}
enforce_active_loop

# inotifywait restart loop
while true; do
    ts_log "Starting inotifywait on $INTERP_DIR..."
    inotifywait -m -e close_write -e moved_to --format "%f" "$INTERP_DIR" 2>/dev/null | while read -r file; do
        if [[ "$file" == *.mp4 ]] && [[ "$file" != *_venice.mp4 ]]; then
            process_file_locked "$INTERP_DIR/$file"
        fi
    done
    ts_warn "inotifywait exited — restarting in 3s..."
    sleep 3
done
