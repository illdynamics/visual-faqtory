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
# Part of Visual FaQtory (version sourced from VERSION)
# ═══════════════════════════════════════════════════════════════════════════════

set -u
set -o pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION_FILE="${BASE_DIR}/VERSION"
SCRIPT_VERSION="$(cat "$VERSION_FILE" 2>/dev/null || echo "v0.7.9-beta")"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--smoke-check|--dry-run] [--status [--verbose]] [--process-file /path/to/clip.mp4] [--validate-file /path/to/clip.mp4] [--reseed-slots] [--help]

  --smoke-check, --dry-run  Validate config/tools, print status, then exit.
  --status                  Print scriptable key=value status and exit.
  --verbose                 Add extra key=value lines for --status.
  --process-file PATH       One-shot load of a completed .mp4 into the inactive slot, then exit.
  --validate-file PATH      Validate a candidate .mp4 and exit 0/1.
  --reseed-slots            Re-seed A/B slot files from the newest valid watched clip, or placeholders.
  --help                    Show this help.

Useful env overrides:
  VF_SRT_ENV=/path/to/vf-obs-watcher-srt-endpoints.env
  VF_CONFIG_FILE=/path/to/worqspace/config.yaml
  VF_WATCH_DIR=/path/to/run/videos
USAGE
}

SMOKE_CHECK="${VF_DRY_RUN:-0}"
STATUS_ONLY="${VF_STATUS_ONLY:-0}"
STATUS_VERBOSE="${VF_STATUS_VERBOSE:-0}"
PROCESS_FILE=""
VALIDATE_FILE=""
RESEED_SLOTS_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke-check|--dry-run)
      SMOKE_CHECK=1 ;;
    --status)
      STATUS_ONLY=1 ;;
    --verbose)
      STATUS_VERBOSE=1 ;;
    --process-file)
      shift
      [[ $# -gt 0 ]] || { echo "[SRT] ERROR: --process-file requires a path" >&2; usage >&2; exit 2; }
      PROCESS_FILE="$1" ;;
    --validate-file)
      shift
      [[ $# -gt 0 ]] || { echo "[SRT] ERROR: --validate-file requires a path" >&2; usage >&2; exit 2; }
      VALIDATE_FILE="$1" ;;
    --reseed-slots)
      RESEED_SLOTS_ONLY=1 ;;
    --help|-h)
      usage; exit 0 ;;
    *)
      echo "[SRT] ERROR: Unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
  shift
done

log() { echo "[SRT] $(date '+%H:%M:%S') $*"; }
warn() { echo "[SRT] WARN: $*" >&2; }
fatal() { echo "[SRT] ERROR: $*" >&2; exit 1; }

VF_SRT_ENV="${VF_SRT_ENV:-${BASE_DIR}/vf-obs-watcher-srt-endpoints.env}"
if [[ -f "$VF_SRT_ENV" ]]; then
  set -a; source "$VF_SRT_ENV"; set +a
  log "Loaded env from $VF_SRT_ENV"
fi

OBS_AUTOSWAP="${OBS_AUTOSWAP:-1}"
OBS_WS_HOST="${OBS_WS_HOST:-127.0.0.1}"
OBS_WS_PORT="${OBS_WS_PORT:-4455}"
OBS_WS_PASSWORD="${OBS_WS_PASSWORD:-}"
OBS_SCENE_NAME="${OBS_SCENE_NAME:-Live Visuals}"
OBS_SOURCE_A="${OBS_SOURCE_A:-Live-Visuals-A}"
OBS_SOURCE_B="${OBS_SOURCE_B:-Live-Visuals-B}"

SRT_BIND_IP="${SRT_BIND_IP:-0.0.0.0}"
SRT_PUBLIC_IP="${SRT_PUBLIC_IP:-${SRT_BIND_IP}}"
SRT_PORT_A="${SRT_PORT_A:-9998}"
SRT_PORT_B="${SRT_PORT_B:-9999}"
SRT_LATENCY="${SRT_LATENCY:-20}"

OUT_WIDTH="${OUT_WIDTH:-1280}"
OUT_HEIGHT="${OUT_HEIGHT:-720}"
OUT_FPS="${OUT_FPS:-30}"
ENC_PRESET="${ENC_PRESET:-veryfast}"
ENC_BITRATE="${ENC_BITRATE:-6000k}"
ENC_BUFSIZE="${ENC_BUFSIZE:-12000k}"

WARMUP_SEC="${WARMUP_SEC:-1.2}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-60}"
READY_POLL_INTERVAL_SEC="${READY_POLL_INTERVAL_SEC:-1}"
READY_STABLE_POLLS="${READY_STABLE_POLLS:-2}"
PRELOAD_EXISTING_ON_START="${PRELOAD_EXISTING_ON_START:-1}"
DEDUP_WINDOW_SEC="${DEDUP_WINDOW_SEC:-2}"
HEALTHCHECK_INTERVAL_SEC="${HEALTHCHECK_INTERVAL_SEC:-5}"
WATCH_RETRY_SEC="${WATCH_RETRY_SEC:-2}"

PLAYOUT_DIR="${VF_PLAYOUT_DIR:-${BASE_DIR}/run/obs}"
PYTHON_BIN="${VF_PYTHON_BIN:-${BASE_DIR}/.venv/bin/python}"
CONFIG_PYTHON_BIN="${VF_CONFIG_PYTHON_BIN:-}"
FFMPEG_BIN="${VF_FFMPEG_BIN:-ffmpeg}"
FFPROBE_BIN="${VF_FFPROBE_BIN:-ffprobe}"
INOTIFYWAIT_BIN="${VF_INOTIFYWAIT_BIN:-inotifywait}"
VF_CONFIG_FILE="${VF_CONFIG_FILE:-}"
VF_WATCH_DIR="${VF_WATCH_DIR:-}"

STATE_FILE="$PLAYOUT_DIR/.active_slot"
SRT_URL_A="srt://${SRT_PUBLIC_IP}:${SRT_PORT_A}?mode=caller&latency=${SRT_LATENCY}"
SRT_URL_B="srt://${SRT_PUBLIC_IP}:${SRT_PORT_B}?mode=caller&latency=${SRT_LATENCY}"

CHILD_PIDS=()
LAST_PROCESSED_FILE=""
LAST_PROCESSED_TS=0
CONFIG_FILE=""
STARTUP_PRELOAD_FILE=""
FFPROBE_AVAILABLE=0
WATCH_MODE="auto"
RUNTIME_PYTHON_BIN=""
RUNTIME_PYTHON_AVAILABLE=0
OBS_RUNTIME_READY=0
OBS_WS_REACHABLE=0
CLEANUP_ACTIVE=0
HEALTH_MONITOR_PID=""

slot_file() { echo "$PLAYOUT_DIR/current_$1.mp4"; }
pid_file() { echo "$PLAYOUT_DIR/.ffmpeg_$1.pid"; }

pick_config_python_bin() {
  if [[ -n "$CONFIG_PYTHON_BIN" ]]; then echo "$CONFIG_PYTHON_BIN"; return 0; fi
  if [[ -x "$PYTHON_BIN" ]]; then echo "$PYTHON_BIN"; return 0; fi
  if command -v python3 >/dev/null 2>&1; then command -v python3; return 0; fi
  if command -v python >/dev/null 2>&1; then command -v python; return 0; fi
  return 1
}

pick_runtime_python_bin() {
  if [[ -x "$PYTHON_BIN" ]]; then echo "$PYTHON_BIN"; return 0; fi
  if command -v "$PYTHON_BIN" >/dev/null 2>&1; then command -v "$PYTHON_BIN"; return 0; fi
  if command -v python3 >/dev/null 2>&1; then command -v python3; return 0; fi
  if command -v python >/dev/null 2>&1; then command -v python; return 0; fi
  return 1
}

resolve_config_file() {
  if [[ -n "$VF_CONFIG_FILE" ]]; then
    [[ -f "$VF_CONFIG_FILE" ]] || fatal "VF_CONFIG_FILE does not exist: $VF_CONFIG_FILE"
    echo "$VF_CONFIG_FILE"; return 0
  fi
  if [[ -f "${BASE_DIR}/worqspace/config.yaml" ]]; then echo "${BASE_DIR}/worqspace/config.yaml"; return 0; fi
  if [[ -f "${BASE_DIR}/worqspace/config-live.yaml" ]]; then echo "${BASE_DIR}/worqspace/config-live.yaml"; return 0; fi
  return 1
}

resolve_path_from_base() {
  local raw="$1"
  if [[ -z "$raw" ]]; then echo "$BASE_DIR/run"; return 0; fi
  if [[ "$raw" = /* ]]; then echo "$raw"; else echo "${BASE_DIR}/${raw#./}"; fi
}

read_config_value_lines() {
  local cfg="$1"
  local pybin=""
  pybin="$(pick_config_python_bin)" || return 1
  "$pybin" - "$cfg" <<'PY'
import sys
try:
    import yaml
except Exception:
    raise SystemExit(1)
cfg_path = sys.argv[1]
with open(cfg_path, 'r', encoding='utf-8') as handle:
    data = yaml.safe_load(handle) or {}
if not isinstance(data, dict):
    data = {}
run_cfg = data.get('run') or {}
finalizer_cfg = data.get('finalizer') or {}
out = run_cfg.get('output_dir') or './run'
interp = bool(finalizer_cfg.get('per_cycle_interpolation', False))
print(str(out))
print('true' if interp else 'false')
PY
}

resolve_watch_mode() {
  local per_cycle="false"
  local cfg_lines=()
  if [[ -n "$VF_WATCH_DIR" ]]; then echo "override"; return 0; fi
  if [[ -n "$CONFIG_FILE" ]]; then
    mapfile -t cfg_lines < <(read_config_value_lines "$CONFIG_FILE" 2>/dev/null || true)
    if [[ -n "${cfg_lines[1]:-}" ]]; then per_cycle="${cfg_lines[1]}"; fi
  fi
  if [[ "$per_cycle" == "true" ]]; then echo "videos_interpolated"; else echo "videos"; fi
}

resolve_watch_dir() {
  local output_dir="${BASE_DIR}/run"
  local cfg_lines=()
  if [[ "$WATCH_MODE" == "override" ]]; then echo "$(resolve_path_from_base "$VF_WATCH_DIR")"; return 0; fi
  if [[ -n "$CONFIG_FILE" ]]; then
    mapfile -t cfg_lines < <(read_config_value_lines "$CONFIG_FILE" 2>/dev/null || true)
    if [[ -n "${cfg_lines[0]:-}" ]]; then output_dir="$(resolve_path_from_base "${cfg_lines[0]}")"; fi
  fi
  if [[ "$WATCH_MODE" == "videos_interpolated" ]]; then echo "${output_dir}/videos_interpolated"; else echo "${output_dir}/videos"; fi
}

get_active_slot() { [[ -f "$STATE_FILE" ]] && cat "$STATE_FILE" || echo "A"; }
set_active_slot() { echo "$1" > "$STATE_FILE"; }
other_slot() { [[ "$1" == "A" ]] && echo "B" || echo "A"; }

copy_to_slot() {
  local source_file="$1" slot="$2" tmp_file
  tmp_file="$(slot_file "$slot").tmp"
  cp "$source_file" "$tmp_file" || return 1
  mv -f "$tmp_file" "$(slot_file "$slot")"
}

find_newest_existing_mp4() {
  [[ -d "$VF_WATCH_DIR" ]] || return 1
  find "$VF_WATCH_DIR" -maxdepth 1 -type f -name '*.mp4' -printf '%T@|%p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d'|' -f2-
}

is_duplicate_event() {
  local file="$1" now
  now=$(date +%s)
  if [[ "$file" == "$LAST_PROCESSED_FILE" ]] && (( now - LAST_PROCESSED_TS < DEDUP_WINDOW_SEC )); then return 0; fi
  LAST_PROCESSED_FILE="$file"
  LAST_PROCESSED_TS=$now
  return 1
}

validate_media_file() {
  local file="$1" size
  [[ -f "$file" ]] || return 1
  size=$(stat -c%s "$file" 2>/dev/null || echo 0)
  [[ "$size" -gt 0 ]] || return 1
  if (( FFPROBE_AVAILABLE == 1 )); then
    "$FFPROBE_BIN" -v error -select_streams v:0 -show_entries stream=codec_type -of default=nokey=1:noprint_wrappers=1 "$file" 2>/dev/null | grep -q '^video$'
  else
    return 0
  fi
}

wait_for_ready_file() {
  local file="$1" deadline stable last_size size
  deadline=$(( $(date +%s) + READY_TIMEOUT_SEC ))
  stable=0
  last_size="-1"
  size="0"
  while (( $(date +%s) <= deadline )); do
    if [[ ! -f "$file" ]]; then sleep "$READY_POLL_INTERVAL_SEC"; continue; fi
    size=$(stat -c%s "$file" 2>/dev/null || echo 0)
    if [[ "$size" -gt 0 && "$size" == "$last_size" ]]; then stable=$((stable + 1)); else stable=0; fi
    last_size="$size"
    if (( stable >= READY_STABLE_POLLS )); then
      if validate_media_file "$file"; then return 0; fi
    fi
    sleep "$READY_POLL_INTERVAL_SEC"
  done
  return 1
}

ffmpeg_pid_alive() {
  local slot="$1" pf pid
  pf="$(pid_file "$slot")"
  [[ -f "$pf" ]] || return 1
  pid=$(cat "$pf" 2>/dev/null || true)
  [[ -n "$pid" ]] || { rm -f "$pf"; return 1; }
  if kill -0 "$pid" 2>/dev/null; then return 0; fi
  warn "Removing stale pid file for slot $slot: $pf"
  rm -f "$pf"
  return 1
}

cleanup() {
  local status="${1:-$?}"
  trap - EXIT INT TERM HUP
  if [[ "${CLEANUP_ACTIVE:-0}" != "1" ]]; then exit "$status"; fi
  log "Shutting down — killing child processes..."
  if [[ -n "${HEALTH_MONITOR_PID:-}" ]]; then kill "$HEALTH_MONITOR_PID" 2>/dev/null || true; fi
  for s in A B; do stop_ffmpeg "$s"; done
  for pid in "${CHILD_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  log "Cleanup complete."
  exit "$status"
}
trap 'cleanup $?' EXIT
trap 'cleanup 130' INT
trap 'cleanup 143' TERM HUP

stop_ffmpeg() {
  local slot="$1" pf pid
  pf="$(pid_file "$slot")"
  [[ -f "$pf" ]] || return 0
  pid=$(cat "$pf" 2>/dev/null || true)
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    sleep 0.2
    kill -9 "$pid" 2>/dev/null || true
  else
    warn "Ignoring stale ffmpeg pid for slot $slot"
  fi
  rm -f "$pf"
}

seed_slot_placeholder() {
  local slot="$1"
  log "Creating black placeholder for slot $slot"
  "$FFMPEG_BIN" -hide_banner -loglevel error -f lavfi -i "color=c=black:s=${OUT_WIDTH}x${OUT_HEIGHT}:r=${OUT_FPS}" -t 4 -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p "$(slot_file "$slot")"
}

ensure_slot_file() {
  local slot="$1" other other_path
  if validate_media_file "$(slot_file "$slot")"; then return 0; fi
  other="$(other_slot "$slot")"
  other_path="$(slot_file "$other")"
  if validate_media_file "$other_path"; then
    log "Recovering missing/corrupt slot $slot from slot $other"
    copy_to_slot "$other_path" "$slot"
    return $?
  fi
  warn "Slot $slot is missing or invalid; reseeding placeholder"
  seed_slot_placeholder "$slot"
}

start_ffmpeg() {
  local slot="$1" port gop pid
  port="$SRT_PORT_A"; [[ "$slot" == "B" ]] && port="$SRT_PORT_B"
  ensure_slot_file "$slot" || return 1
  gop=$((OUT_FPS * 1))
  log "Starting ffmpeg slot $slot on ${SRT_BIND_IP}:${port} using $(slot_file "$slot")"
  "$FFMPEG_BIN" -hide_banner -loglevel warning -re -stream_loop -1 -i "$(slot_file "$slot")" -an -vf "scale=${OUT_WIDTH}:${OUT_HEIGHT},fps=${OUT_FPS},format=yuv420p" -c:v libx264 -preset "$ENC_PRESET" -tune zerolatency -b:v "$ENC_BITRATE" -maxrate "$ENC_BITRATE" -bufsize "$ENC_BUFSIZE" -g "$gop" -keyint_min "$gop" -sc_threshold 0 -muxdelay 0 -muxpreload 0 -f mpegts "srt://${SRT_BIND_IP}:${port}?mode=listener&latency=${SRT_LATENCY}" >/dev/null 2>&1 &
  pid=$!
  echo "$pid" > "$(pid_file "$slot")"
  CHILD_PIDS+=("$pid")
  sleep 0.2
  if ! kill -0 "$pid" 2>/dev/null; then warn "ffmpeg slot $slot exited immediately — check codec/input/network settings"; rm -f "$(pid_file "$slot")"; return 1; fi
  return 0
}

restart_ffmpeg() {
  local slot="$1"
  stop_ffmpeg "$slot"
  if start_ffmpeg "$slot"; then return 0; fi
  warn "Retrying ffmpeg slot $slot once after reseeding placeholder"
  seed_slot_placeholder "$slot" || return 1
  start_ffmpeg "$slot"
}

seed_slots_from_startup_clip() {
  [[ "$PRELOAD_EXISTING_ON_START" == "1" ]] || return 0
  STARTUP_PRELOAD_FILE="$(find_newest_existing_mp4 || true)"
  if [[ -z "$STARTUP_PRELOAD_FILE" ]]; then log "No existing .mp4 clip found for startup preload in $VF_WATCH_DIR"; return 0; fi
  log "Startup preload candidate: $STARTUP_PRELOAD_FILE"
  if ! wait_for_ready_file "$STARTUP_PRELOAD_FILE"; then warn "Startup preload candidate did not become ready in time: $STARTUP_PRELOAD_FILE"; STARTUP_PRELOAD_FILE=""; return 0; fi
  for s in A B; do
    if copy_to_slot "$STARTUP_PRELOAD_FILE" "$s"; then log "Seeded slot $s with startup clip $(basename "$STARTUP_PRELOAD_FILE")"; else warn "Failed to seed slot $s from startup clip: $STARTUP_PRELOAD_FILE"; return 1; fi
  done
  return 0
}

reseed_slots_from_watch_dir() {
  local newest
  newest="$(find_newest_existing_mp4 || true)"
  if [[ -n "$newest" ]] && wait_for_ready_file "$newest"; then
    for s in A B; do copy_to_slot "$newest" "$s" || return 1; done
    log "Re-seeded both slots from $(basename "$newest")"
    STARTUP_PRELOAD_FILE="$newest"
    return 0
  fi
  warn "No valid watched clip available for reseed; falling back to placeholders"
  seed_slot_placeholder A || return 1
  seed_slot_placeholder B || return 1
  STARTUP_PRELOAD_FILE=""
  return 0
}

obs_enable() {
  [[ "$OBS_AUTOSWAP" == "1" ]] || return 0
  [[ "$OBS_RUNTIME_READY" == "1" ]] || return 0
  "$RUNTIME_PYTHON_BIN" - <<PY 2>/dev/null || true
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
  [[ "$OBS_RUNTIME_READY" == "1" ]] || return 0
  "$RUNTIME_PYTHON_BIN" - <<PY 2>/dev/null || true
from obsws_python import ReqClient
cl = ReqClient(host="$OBS_WS_HOST", port=int("$OBS_WS_PORT"), password="$OBS_WS_PASSWORD")
scene="$OBS_SCENE_NAME"
source="$1"
sid=cl.get_scene_item_id(scene, source).scene_item_id
cl.set_scene_item_enabled(scene, sid, False)
PY
}

process_video() {
  local file="$1" reason="${2:-watch-event}" active target target_src other_src
  [[ -f "$file" ]] || { warn "Ignoring missing file ($reason): $file"; return 0; }
  [[ "$file" == *.mp4 ]] || { warn "Ignoring non-mp4 file ($reason): $file"; return 0; }
  if is_duplicate_event "$file"; then log "Ignoring duplicate event for $file"; return 0; fi
  if ! wait_for_ready_file "$file"; then warn "Timed out waiting for ready video ($reason): $file"; return 0; fi
  log "Loading completed file ($reason): $file"
  active="$(get_active_slot)"
  target="$(other_slot "$active")"
  if ! copy_to_slot "$file" "$target"; then warn "Failed to copy $file into slot $target"; return 1; fi
  restart_ffmpeg "$target" || warn "Restart for slot $target reported an error"
  target_src="$OBS_SOURCE_A"; other_src="$OBS_SOURCE_B"
  [[ "$target" == "B" ]] && target_src="$OBS_SOURCE_B" && other_src="$OBS_SOURCE_A"
  log "Prewarming $target_src for ${WARMUP_SEC}s"
  obs_enable "$target_src"
  sleep "$WARMUP_SEC"
  log "Disabling $other_src"
  obs_disable "$other_src"
  set_active_slot "$target"
  log "Switched to slot $target"
}

slot_pid_value() { local slot="$1" pf; pf="$(pid_file "$slot")"; [[ -f "$pf" ]] && cat "$pf" || true; }

print_status() {
  cat <<STATUS
CONFIG_FILE=${CONFIG_FILE}
WATCH_DIR=${VF_WATCH_DIR}
WATCH_MODE=${WATCH_MODE}
WATCH_DIR_EXISTS=$([[ -d "$VF_WATCH_DIR" ]] && echo 1 || echo 0)
PLAYOUT_DIR=${PLAYOUT_DIR}
ACTIVE_SLOT=$(get_active_slot)
PRELOAD_FILE=${STARTUP_PRELOAD_FILE}
OBS_AUTOSWAP=${OBS_AUTOSWAP}
OBS_RUNTIME_READY=${OBS_RUNTIME_READY}
SRT_URL_A=${SRT_URL_A}
SRT_URL_B=${SRT_URL_B}
STATUS
  if [[ "$STATUS_VERBOSE" == "1" ]]; then
    cat <<STATUSV
SLOT_A_FILE=$(slot_file A)
SLOT_B_FILE=$(slot_file B)
SLOT_A_PRESENT=$([[ -f "$(slot_file A)" ]] && echo 1 || echo 0)
SLOT_B_PRESENT=$([[ -f "$(slot_file B)" ]] && echo 1 || echo 0)
SLOT_A_VALID=$(validate_media_file "$(slot_file A)" >/dev/null 2>&1 && echo 1 || echo 0)
SLOT_B_VALID=$(validate_media_file "$(slot_file B)" >/dev/null 2>&1 && echo 1 || echo 0)
SLOT_A_PID=$(slot_pid_value A)
SLOT_B_PID=$(slot_pid_value B)
SLOT_A_PID_RUNNING=$(ffmpeg_pid_alive A >/dev/null 2>&1 && echo 1 || echo 0)
SLOT_B_PID_RUNNING=$(ffmpeg_pid_alive B >/dev/null 2>&1 && echo 1 || echo 0)
FFPROBE_AVAILABLE=${FFPROBE_AVAILABLE}
RUNTIME_PYTHON_AVAILABLE=${RUNTIME_PYTHON_AVAILABLE}
RUNTIME_PYTHON_BIN=${RUNTIME_PYTHON_BIN}
OBS_WS_REACHABLE=${OBS_WS_REACHABLE}
STATUSV
  fi
}

print_banner() {
  echo ""
  echo "  ┌──────────────────────────────────────────────────────────────┐"
  printf "  │           VF SRT WATCHER %-29s│\n" "$SCRIPT_VERSION"
  echo "  ├──────────────────────────────────────────────────────────────┤"
  echo "  │  Slot A : ${SRT_URL_A}"
  echo "  │  Slot B : ${SRT_URL_B}"
  echo "  │  Bind   : ${SRT_BIND_IP}"
  echo "  │  Public : ${SRT_PUBLIC_IP}"
  echo "  │  Encode : ${OUT_WIDTH}x${OUT_HEIGHT}@${OUT_FPS}fps ${ENC_BITRATE} ${ENC_PRESET}"
  echo "  │  Warmup : ${WARMUP_SEC}s"
  echo "  │  Watch  : ${VF_WATCH_DIR} (${WATCH_MODE})"
  echo "  │  Config : ${CONFIG_FILE:-<default>}"
  echo "  │  OBS    : autoswap=$([ "$OBS_AUTOSWAP" == "1" ] && echo "ON" || echo "OFF")"
  echo "  └──────────────────────────────────────────────────────────────┘"
  echo ""
}

probe_obs_websocket_reachability() {
  local pybin="$1"
  [[ -n "$pybin" ]] || return 1
  "$pybin" - <<PY >/dev/null 2>&1
import socket
try:
    sock = socket.create_connection(("$OBS_WS_HOST", int("$OBS_WS_PORT")), timeout=0.3)
except OSError:
    raise SystemExit(1)
else:
    sock.close()
    raise SystemExit(0)
PY
}

detect_runtime_capabilities() {
  local silent="${1:-0}"
  local require_core_tools="${2:-0}"
  if [[ "$require_core_tools" == "1" ]]; then
    command -v "$INOTIFYWAIT_BIN" >/dev/null 2>&1 || fatal "inotifywait required (install inotify-tools or set VF_INOTIFYWAIT_BIN)"
    command -v "$FFMPEG_BIN" >/dev/null 2>&1 || fatal "ffmpeg not found (set VF_FFMPEG_BIN if needed)"
  fi

  if command -v "$FFPROBE_BIN" >/dev/null 2>&1; then
    FFPROBE_AVAILABLE=1
  else
    FFPROBE_AVAILABLE=0
    if [[ "$silent" != "1" ]]; then warn "ffprobe not found — file readiness will use stable-size checks only"; fi
  fi

  RUNTIME_PYTHON_BIN="$(pick_runtime_python_bin || true)"
  if [[ -n "$RUNTIME_PYTHON_BIN" ]]; then
    RUNTIME_PYTHON_AVAILABLE=1
  else
    RUNTIME_PYTHON_AVAILABLE=0
  fi

  OBS_RUNTIME_READY=0
  OBS_WS_REACHABLE=0
  if [[ "$OBS_AUTOSWAP" == "1" ]]; then
    if [[ "$RUNTIME_PYTHON_AVAILABLE" != "1" ]]; then
      [[ "$silent" == "1" ]] || warn "OBS_AUTOSWAP=1 but no usable python interpreter was found"
    elif ! "$RUNTIME_PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import importlib
importlib.import_module('obsws_python')
PY
    then
      [[ "$silent" == "1" ]] || warn "OBS_AUTOSWAP=1 but obsws_python is not importable via $RUNTIME_PYTHON_BIN"
    else
      OBS_RUNTIME_READY=1
      if probe_obs_websocket_reachability "$RUNTIME_PYTHON_BIN"; then
        OBS_WS_REACHABLE=1
      elif [[ "$silent" != "1" ]]; then
        warn "OBS websocket is not reachable at ${OBS_WS_HOST}:${OBS_WS_PORT}; autoswap may stay manual until OBS is up"
      fi
    fi
  fi
}

validate_runtime() {
  detect_runtime_capabilities 0 1
}

health_monitor_loop() {
  while true; do
    sleep "$HEALTHCHECK_INTERVAL_SEC"
    if [[ ! -d "$VF_WATCH_DIR" ]]; then warn "Watch dir missing, recreating: $VF_WATCH_DIR"; mkdir -p "$VF_WATCH_DIR"; fi
    [[ -f "$STATE_FILE" ]] || set_active_slot A
    ensure_slot_file A || warn "Failed to recover slot A file"
    ensure_slot_file B || warn "Failed to recover slot B file"
    for slot in A B; do
      if ! ffmpeg_pid_alive "$slot"; then warn "ffmpeg slot $slot is not running; restarting"; restart_ffmpeg "$slot" || warn "Restart failed for slot $slot"; fi
    done
  done
}

validate_file_mode() {
  local file="$1"
  if wait_for_ready_file "$file"; then log "Validated media file: $file"; return 0; fi
  warn "Media file is not ready/valid: $file"
  return 1
}

watch_loop() {
  while true; do
    mkdir -p "$VF_WATCH_DIR"
    log "Watching with inotify (close_write,moved_to): $VF_WATCH_DIR"
    "$INOTIFYWAIT_BIN" -m -e close_write,moved_to --format "%e|%f" "$VF_WATCH_DIR" 2>/dev/null | while IFS='|' read -r events file_name; do
      [[ "$file_name" == *.mp4 ]] || continue
      process_video "$VF_WATCH_DIR/$file_name" "$events"
    done
    local watch_status=${PIPESTATUS[0]}
    warn "inotifywait exited with status ${watch_status}; retrying in ${WATCH_RETRY_SEC}s"
    sleep "$WATCH_RETRY_SEC"
  done
}

CONFIG_FILE="$(resolve_config_file || true)"
WATCH_MODE="$(resolve_watch_mode)"
VF_WATCH_DIR="$(resolve_watch_dir)"
mkdir -p "$PLAYOUT_DIR" "$VF_WATCH_DIR"
[[ -f "$STATE_FILE" ]] || set_active_slot A
if [[ "$PRELOAD_EXISTING_ON_START" == "1" ]]; then STARTUP_PRELOAD_FILE="$(find_newest_existing_mp4 || true)"; fi

if [[ "$STATUS_ONLY" == "1" ]]; then detect_runtime_capabilities 1 0; print_status; exit 0; fi
if [[ -n "$VALIDATE_FILE" ]]; then
  if command -v "$FFPROBE_BIN" >/dev/null 2>&1; then FFPROBE_AVAILABLE=1; else FFPROBE_AVAILABLE=0; fi
  validate_file_mode "$VALIDATE_FILE"; exit $?;
fi

validate_runtime
if [[ "$SMOKE_CHECK" == "1" ]]; then print_banner; print_status; log "Smoke-check OK — runtime dependencies and watcher config look sane."; exit 0; fi
if [[ "$RESEED_SLOTS_ONLY" == "1" ]]; then reseed_slots_from_watch_dir; exit $?; fi

CLEANUP_ACTIVE=1
if ! seed_slots_from_startup_clip; then warn "Startup preload failed; continuing with placeholders as needed"; fi
for s in A B; do [[ -f "$(slot_file "$s")" ]] || seed_slot_placeholder "$s"; done
restart_ffmpeg A || warn "Initial ffmpeg start for slot A reported an error"
restart_ffmpeg B || warn "Initial ffmpeg start for slot B reported an error"
health_monitor_loop &
HEALTH_MONITOR_PID=$!
CHILD_PIDS+=("$HEALTH_MONITOR_PID")

if [[ "$OBS_AUTOSWAP" == "1" ]]; then
  initial_slot="$(get_active_slot)"
  if [[ "$initial_slot" == "A" ]]; then obs_enable "$OBS_SOURCE_A"; obs_disable "$OBS_SOURCE_B"; else obs_enable "$OBS_SOURCE_B"; obs_disable "$OBS_SOURCE_A"; fi
  if [[ "$OBS_RUNTIME_READY" == "1" ]]; then log "OBS autoswap: ENABLED"; else warn "OBS autoswap requested but runtime integration is not ready; leaving endpoint switching manual"; fi
else
  log "OBS autoswap: DISABLED — SRT endpoints active, manual switching required"
fi
print_banner
if [[ -n "$PROCESS_FILE" ]]; then log "One-shot processing requested: $PROCESS_FILE"; process_video "$PROCESS_FILE" manual; exit $?; fi
watch_loop
