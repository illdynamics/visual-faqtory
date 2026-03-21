#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# vf-obs-watcher-srt-endpoints.sh — Visual FaQtory External SRT A/B Playout
# ═══════════════════════════════════════════════════════════════════════════════
#
# Watches for new generated videos and exposes them as two SRT listener
# endpoints (A/B warm-swap). A remote OBS instance pulls these as media inputs.
#
# All values are env-overridable. See vf-obs-watcher-srt-endpoints.env.example.
#
# Part of QonQrete Visual FaQtory v0.5.9-beta
# ═══════════════════════════════════════════════════════════════════════════════

set -u

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─────────────────────────────────────────
# Load optional env file
# ─────────────────────────────────────────
VF_SRT_ENV="${VF_SRT_ENV:-${BASE_DIR}/vf-obs-watcher-srt-endpoints.env}"
if [[ -f "$VF_SRT_ENV" ]]; then
  # shellcheck disable=SC1090
  set -a; source "$VF_SRT_ENV"; set +a
  echo "[SRT] Loaded env from $VF_SRT_ENV"
fi

# ─────────────────────────────────────────
# OBS WebSocket (only used if OBS_AUTOSWAP=1)
# ─────────────────────────────────────────
OBS_AUTOSWAP="${OBS_AUTOSWAP:-1}"
OBS_WS_HOST="${OBS_WS_HOST:-127.0.0.1}"
OBS_WS_PORT="${OBS_WS_PORT:-4455}"
OBS_WS_PASSWORD="${OBS_WS_PASSWORD:-}"
OBS_SCENE_NAME="${OBS_SCENE_NAME:-Live Visuals}"
OBS_SOURCE_A="${OBS_SOURCE_A:-Live-Visuals-A}"
OBS_SOURCE_B="${OBS_SOURCE_B:-Live-Visuals-B}"

# ─────────────────────────────────────────
# SRT endpoints
# ─────────────────────────────────────────
SRT_BIND_IP="${SRT_BIND_IP:-0.0.0.0}"
SRT_PUBLIC_IP="${SRT_PUBLIC_IP:-${SRT_BIND_IP}}"
SRT_PORT_A="${SRT_PORT_A:-9998}"
SRT_PORT_B="${SRT_PORT_B:-9999}"
SRT_LATENCY="${SRT_LATENCY:-20}"

# ─────────────────────────────────────────
# Encoding
# ─────────────────────────────────────────
OUT_WIDTH="${OUT_WIDTH:-1280}"
OUT_HEIGHT="${OUT_HEIGHT:-720}"
OUT_FPS="${OUT_FPS:-30}"
ENC_PRESET="${ENC_PRESET:-veryfast}"
ENC_BITRATE="${ENC_BITRATE:-6000k}"
ENC_BUFSIZE="${ENC_BUFSIZE:-12000k}"

# ─────────────────────────────────────────
# Timing
# ─────────────────────────────────────────
WARMUP_SEC="${WARMUP_SEC:-1.2}"

# ─────────────────────────────────────────
# Paths
# ─────────────────────────────────────────
PLAYOUT_DIR="${VF_PLAYOUT_DIR:-${BASE_DIR}/run/obs}"
PYTHON_BIN="${VF_PYTHON_BIN:-${BASE_DIR}/.venv/bin/python}"
FFMPEG_BIN="${VF_FFMPEG_BIN:-ffmpeg}"
INOTIFYWAIT_BIN="${VF_INOTIFYWAIT_BIN:-inotifywait}"

# ─────────────────────────────────────────
# Auto-detect watch dir from config
# ─────────────────────────────────────────
_auto_detect_watch_dir() {
  local cfg=""
  if [[ -f "${BASE_DIR}/worqspace/config-live.yaml" ]]; then
    cfg="${BASE_DIR}/worqspace/config-live.yaml"
  elif [[ -f "${BASE_DIR}/worqspace/config.yaml" ]]; then
    cfg="${BASE_DIR}/worqspace/config.yaml"
  fi
  if [[ -n "$cfg" ]]; then
    # Check if per_cycle_interpolation is true
    if grep -qE '^\s*per_cycle_interpolation:\s*true' "$cfg" 2>/dev/null; then
      echo "${BASE_DIR}/run/videos_interpolated"
      return
    fi
  fi
  echo "${BASE_DIR}/run/videos"
}

VF_WATCH_DIR="${VF_WATCH_DIR:-$(_auto_detect_watch_dir)}"

STATE_FILE="$PLAYOUT_DIR/.active_slot"

# ─────────────────────────────────────────
# Internal
# ─────────────────────────────────────────
CHILD_PIDS=()

mkdir -p "$PLAYOUT_DIR"

slot_file() { echo "$PLAYOUT_DIR/current_$1.mp4"; }
pid_file()  { echo "$PLAYOUT_DIR/.ffmpeg_$1.pid"; }

log() { echo "[SRT] $(date '+%H:%M:%S') $*"; }

get_active_slot() {
  [[ -f "$STATE_FILE" ]] && cat "$STATE_FILE" || echo "A"
}

set_active_slot() {
  echo "$1" > "$STATE_FILE"
}

other_slot() {
  [[ "$1" == "A" ]] && echo "B" || echo "A"
}

# ─────────────────────────────────────────
# Cleanup on exit
# ─────────────────────────────────────────
cleanup() {
  log "Shutting down — killing child processes..."
  for s in A B; do
    stop_ffmpeg "$s"
  done
  for pid in "${CHILD_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  log "Cleanup complete."
  exit 0
}

trap cleanup EXIT INT TERM HUP

# ─────────────────────────────────────────
# FFmpeg management
# ─────────────────────────────────────────
stop_ffmpeg() {
  local pf
  pf="$(pid_file "$1")"
  [[ -f "$pf" ]] || return
  local pid
  pid=$(cat "$pf")
  kill "$pid" 2>/dev/null || true
  sleep 0.2
  kill -9 "$pid" 2>/dev/null || true
  rm -f "$pf"
}

start_ffmpeg() {
  local slot="$1"
  local port="$SRT_PORT_A"
  [[ "$slot" == "B" ]] && port="$SRT_PORT_B"

  local gop=$((OUT_FPS * 1))

  log "Starting ffmpeg slot $slot on ${SRT_BIND_IP}:${port}"

  "$FFMPEG_BIN" -hide_banner -loglevel warning \
    -re -stream_loop -1 -i "$(slot_file "$slot")" \
    -an \
    -vf "scale=${OUT_WIDTH}:${OUT_HEIGHT},fps=${OUT_FPS},format=yuv420p" \
    -c:v libx264 -preset "$ENC_PRESET" -tune zerolatency \
    -b:v "$ENC_BITRATE" -maxrate "$ENC_BITRATE" -bufsize "$ENC_BUFSIZE" \
    -g "$gop" -keyint_min "$gop" -sc_threshold 0 \
    -muxdelay 0 -muxpreload 0 \
    -f mpegts \
    "srt://${SRT_BIND_IP}:${port}?mode=listener&latency=${SRT_LATENCY}" \
    >/dev/null 2>&1 &

  local pid=$!
  echo "$pid" > "$(pid_file "$slot")"
  CHILD_PIDS+=("$pid")
}

restart_ffmpeg() {
  stop_ffmpeg "$1"
  start_ffmpeg "$1"
}

# ─────────────────────────────────────────
# OBS WebSocket helpers
# ─────────────────────────────────────────
obs_enable() {
  [[ "$OBS_AUTOSWAP" == "1" ]] || return 0
"$PYTHON_BIN" - <<PY 2>/dev/null || true
from obsws_python import ReqClient
cl = ReqClient(host="$OBS_WS_HOST", port=int("$OBS_WS_PORT"), password="$OBS_WS_PASSWORD")
scene="$OBS_SCENE_NAME"
source="$1"
sid=cl.get_scene_item_id(scene, source).scene_item_id
cl.set_scene_item_enabled(scene, sid, True)
PY
}

obs_disable() {
  [[ "$OBS_AUTOSWAP" == "1" ]] || return 0
"$PYTHON_BIN" - <<PY 2>/dev/null || true
from obsws_python import ReqClient
cl = ReqClient(host="$OBS_WS_HOST", port=int("$OBS_WS_PORT"), password="$OBS_WS_PASSWORD")
scene="$OBS_SCENE_NAME"
source="$1"
sid=cl.get_scene_item_id(scene, source).scene_item_id
cl.set_scene_item_enabled(scene, sid, False)
PY
}

# ─────────────────────────────────────────
# Video swap logic
# ─────────────────────────────────────────
process_video() {
  local file="$1"
  log "New completed file: $file"

  local active
  active="$(get_active_slot)"
  local target
  target="$(other_slot "$active")"

  local tmp
  tmp="$(slot_file "$target").tmp"
  cp "$file" "$tmp" || return

  mv -f "$tmp" "$(slot_file "$target")"

  restart_ffmpeg "$target"

  # Prewarm target BEFORE disabling active
  local target_src="$OBS_SOURCE_A"
  local other_src="$OBS_SOURCE_B"
  [[ "$target" == "B" ]] && target_src="$OBS_SOURCE_B" && other_src="$OBS_SOURCE_A"

  log "Prewarming $target_src for ${WARMUP_SEC}s"
  obs_enable "$target_src"

  sleep "$WARMUP_SEC"

  log "Disabling $other_src"
  obs_disable "$other_src"

  set_active_slot "$target"
  log "Switched to slot $target"
}

# ═══════════════════════════════════════════
# INIT
# ═══════════════════════════════════════════

command -v "$INOTIFYWAIT_BIN" >/dev/null 2>&1 || { echo "[SRT] ERROR: inotifywait required (apt install inotify-tools)"; exit 1; }
command -v "$FFMPEG_BIN" >/dev/null 2>&1 || { echo "[SRT] ERROR: ffmpeg not found"; exit 1; }

[[ -f "$STATE_FILE" ]] || set_active_slot "A"

# Seed black placeholders if no media yet
for s in A B; do
  if [[ ! -f "$(slot_file "$s")" ]]; then
    log "Creating black placeholder for slot $s"
    "$FFMPEG_BIN" -hide_banner -loglevel error \
      -f lavfi -i "color=c=black:s=${OUT_WIDTH}x${OUT_HEIGHT}:r=${OUT_FPS}" \
      -t 4 -c:v libx264 -preset ultrafast -tune zerolatency \
      -pix_fmt yuv420p "$(slot_file "$s")"
  fi
done

restart_ffmpeg "A"
restart_ffmpeg "B"

# Initial OBS state
if [[ "$OBS_AUTOSWAP" == "1" ]]; then
  local_initial="$(get_active_slot)"
  if [[ "$local_initial" == "A" ]]; then
    obs_enable "$OBS_SOURCE_A"
    obs_disable "$OBS_SOURCE_B"
  else
    obs_enable "$OBS_SOURCE_B"
    obs_disable "$OBS_SOURCE_A"
  fi
  log "OBS autoswap: ENABLED"
else
  log "OBS autoswap: DISABLED — SRT endpoints active, manual switching required"
fi

# Ensure watch dir exists
mkdir -p "$VF_WATCH_DIR"

# ─────────────────────────────────────────
# Startup banner
# ─────────────────────────────────────────
echo ""
echo "  ┌──────────────────────────────────────────────────────────────┐"
echo "  │           VF SRT WATCHER v0.5.9-beta                        │"
echo "  ├──────────────────────────────────────────────────────────────┤"
echo "  │  Slot A : srt://${SRT_PUBLIC_IP}:${SRT_PORT_A}?mode=caller&latency=${SRT_LATENCY}"
echo "  │  Slot B : srt://${SRT_PUBLIC_IP}:${SRT_PORT_B}?mode=caller&latency=${SRT_LATENCY}"
echo "  │  Bind   : ${SRT_BIND_IP}"
echo "  │  Public : ${SRT_PUBLIC_IP}"
echo "  │  Encode : ${OUT_WIDTH}x${OUT_HEIGHT}@${OUT_FPS}fps ${ENC_BITRATE} ${ENC_PRESET}"
echo "  │  Warmup : ${WARMUP_SEC}s"
echo "  │  Watch  : ${VF_WATCH_DIR}"
echo "  │  OBS    : autoswap=$([ "$OBS_AUTOSWAP" == "1" ] && echo "ON" || echo "OFF")"
echo "  └──────────────────────────────────────────────────────────────┘"
echo ""

log "Watching with inotify: $VF_WATCH_DIR"

"$INOTIFYWAIT_BIN" -m -e close_write --format "%f" "$VF_WATCH_DIR" | while read -r f; do
  [[ "$f" == *.mp4 ]] || continue
  process_video "$VF_WATCH_DIR/$f"
done
