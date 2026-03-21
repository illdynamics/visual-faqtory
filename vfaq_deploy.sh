#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# vfaq_deploy.sh — Visual FaQtory GPU Server Deploy Script
# ═══════════════════════════════════════════════════════════════════════════════
#
# Spins up the full Visual FaQtory pipeline on a fresh GPU server:
#   - ComfyUI backend
#   - Visual FaQtory generator
#   - ZeroTier mesh network
#   - Crowd Control server (prompt queue + QR + overlay)
#   - SRT A/B playout watcher (external live visuals over ZeroTier)
#
# Part of QonQrete Visual FaQtory v0.5.9-beta
# ═══════════════════════════════════════════════════════════════════════════════

set -e

ARCHIVE_URL="https://wonq.tv/vfaq.zip"
ARCHIVE_NAME="vfaq.zip"
RUN_NAME=$(date +"%Y-%m-%d_%H%M%S")

COMFY_DIR="ComfyUI"
VFAQ_DIR="visual-faqtory"

# ── ZeroTier ─────────────────────────────────────────────────────────────────
ZT_NETWORK="0cccb752f72ecf52"
ZT_PREFIX="10.11.12"          # Expected IP prefix on this network
ZT_WAIT_MAX=60                # Max seconds to wait for ZT IP

# ── Crowd Control ────────────────────────────────────────────────────────────
# Override these via env or edit here
VF_CROWD_TOKEN="${VF_CROWD_TOKEN:-CHANGE_ME_LONG_RANDOM}"
VF_CROWD_PORT="${VF_CROWD_PORT:-8808}"
VF_CROWD_PUBLIC_URL="${VF_CROWD_PUBLIC_URL:-https://wonq.tv/visuals}"

# ── SRT ──────────────────────────────────────────────────────────────────────
SRT_PORT_A="${SRT_PORT_A:-9998}"
SRT_PORT_B="${SRT_PORT_B:-9999}"
SRT_LATENCY="${SRT_LATENCY:-20}"
OBS_AUTOSWAP="${OBS_AUTOSWAP:-0}"   # Default OFF — manual switch from OBS side

# ═════════════════════════════════════════════════════════════════════════════
# Functions
# ═════════════════════════════════════════════════════════════════════════════

function check_nvenc() {
  echo "==> Checking NVENC support..."

  ffmpeg -f lavfi -i testsrc=duration=1:size=1280x720:rate=30 \
    -c:v h264_nvenc -preset p5 -pix_fmt yuv420p -y /tmp/nvenc_test.mp4 >/dev/null 2>&1

  NVENC_STATUS=$?
  rm -f /tmp/nvenc_test.mp4

  if [ $NVENC_STATUS -ne 0 ]; then
    echo -e "\n\033[1;31m=============================================="
    echo " GPU SERVER INVALID FOR THIS WORKLOAD"
    echo " NVENC ENCODER NOT AVAILABLE"
    echo " This pod likely hides NVIDIA video engines."
    echo " Choose another GPU server."
    echo "==============================================\033[0m\n"
    exit 1
  fi

  echo "NVENC OK ✓"
}

function download_and_unpack() {
  echo "==> Starting background download..."
  wget -q --show-progress -O "${ARCHIVE_NAME}" "${ARCHIVE_URL}" &
  WGET_PID=$!

  echo "==> Installing system dependencies while downloading..."
  apt update -qq
  apt install -y -qq python3.11 python3.11-venv python3.11-dev python3-pip \
    git curl build-essential ffmpeg screen wget unzip inotify-tools

  echo "==> Waiting for archive download to complete..."
  wait $WGET_PID

  echo "==> Extracting archive..."
  unzip -qo "${ARCHIVE_NAME}"
}

function install_zerotier() {
  echo "==> Installing ZeroTier..."

  if command -v zerotier-cli >/dev/null 2>&1; then
    echo "ZeroTier already installed ✓"
  else
    curl -s https://install.zerotier.com | bash
  fi

  # Start the service
  systemctl enable zerotier-one 2>/dev/null || true
  systemctl start zerotier-one 2>/dev/null || zerotier-one -d 2>/dev/null || true

  # Brief pause for daemon startup
  sleep 2

  echo "==> Joining ZeroTier network ${ZT_NETWORK}..."
  zerotier-cli join "${ZT_NETWORK}"

  echo ""
  echo -e "\033[1;33m══════════════════════════════════════════════════════════\033[0m"
  echo -e "\033[1;33m  ZeroTier joined network: ${ZT_NETWORK}\033[0m"
  echo -e "\033[1;33m  IMPORTANT: Authorize this node in ZeroTier Central!\033[0m"
  echo -e "\033[1;33m  Node ID: $(zerotier-cli info | awk '{print $3}')\033[0m"
  echo -e "\033[1;33m══════════════════════════════════════════════════════════\033[0m"
  echo ""
}

function wait_for_zt_ip() {
  echo "==> Waiting for ZeroTier IP (${ZT_PREFIX}.x)..."

  local elapsed=0
  ZT_IP=""

  while [ $elapsed -lt $ZT_WAIT_MAX ]; do
    # Grab the first IP matching our prefix from any zt interface
    ZT_IP=$(ip -4 addr show | grep -oP "${ZT_PREFIX}\.\d+" | head -1)

    if [ -n "$ZT_IP" ]; then
      echo "ZeroTier IP acquired: ${ZT_IP} ✓"
      return 0
    fi

    sleep 2
    elapsed=$((elapsed + 2))
    echo "  ... waiting (${elapsed}s / ${ZT_WAIT_MAX}s)"
  done

  echo -e "\n\033[1;31m=============================================="
  echo " ZeroTier IP not acquired within ${ZT_WAIT_MAX}s"
  echo " Make sure you authorized this node in ZeroTier Central!"
  echo " Node ID: $(zerotier-cli info | awk '{print $3}')"
  echo " You can re-run:  ./vfaq_deploy.sh services"
  echo "==============================================\033[0m\n"
  return 1
}

function setup_comfy() {
  echo "==> Setting up ComfyUI..."
  cd "${COMFY_DIR}"
  pip install --upgrade pip
  pip install -r requirements.txt
  if [ -f "${HOME}/${COMFY_DIR}/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt" ]; then
    pip install -r "${HOME}/${COMFY_DIR}/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt"
  fi
  if [ -f "${HOME}/${COMFY_DIR}/models/svd/svd_xt_1_1.safetensors" ] && \
     [ ! -e "${HOME}/${COMFY_DIR}/models/checkpoints/svd_xt_1_1.safetensors" ]; then
    mkdir -p "${HOME}/${COMFY_DIR}/models/checkpoints"
    ln -sf "${HOME}/${COMFY_DIR}/models/svd/svd_xt_1_1.safetensors" \
           "${HOME}/${COMFY_DIR}/models/checkpoints/svd_xt_1_1.safetensors"
  fi
  cd ..
}

function setup_vfaq() {
  echo "==> Setting up visual-faqtory..."
  cd "${VFAQ_DIR}"
  pip install --upgrade pip
  pip install -r requirements.txt
  cd ..
}

function start_comfy_background() {
  echo "==> Starting ComfyUI..."

  # Kill existing if re-running
  screen -S comfyui -X quit 2>/dev/null || true

  cd "${COMFY_DIR}"
  screen -S comfyui -dm bash -c "python main.py --listen 0.0.0.0 --port 8188"
  cd ..
  echo "ComfyUI started in screen 'comfyui' ✓"
}

function start_crowd_control() {
  echo "==> Starting Crowd Control server..."

  # Kill existing if re-running
  screen -S crowd -X quit 2>/dev/null || true

  cd "${VFAQ_DIR}"
  screen -S crowd -dm bash -c \
    "python vfaq_cli.py crowd \
      --token '${VF_CROWD_TOKEN}' \
      --port ${VF_CROWD_PORT} \
      --public-url '${VF_CROWD_PUBLIC_URL}' \
      2>&1 | tee crowd-control.log"
  cd ..

  echo "Crowd Control started in screen 'crowd' ✓"
  echo "  Submit page : http://${ZT_IP:-localhost}:${VF_CROWD_PORT}/visuals/"
  echo "  OBS overlay : http://${ZT_IP:-localhost}:${VF_CROWD_PORT}/visuals/overlay"
  echo "  Status JSON : http://${ZT_IP:-localhost}:${VF_CROWD_PORT}/visuals/api/status"
  echo "  QR code     : http://${ZT_IP:-localhost}:${VF_CROWD_PORT}/visuals/qr.png"
}

function start_srt_watcher() {
  echo "==> Starting SRT A/B watcher..."

  if [ -z "$ZT_IP" ]; then
    # Try to grab it if already assigned
    ZT_IP=$(ip -4 addr show | grep -oP "${ZT_PREFIX}\.\d+" | head -1)
  fi

  if [ -z "$ZT_IP" ]; then
    echo -e "\033[1;31m  No ZeroTier IP found — SRT watcher will bind 0.0.0.0 only.\033[0m"
    echo "  Run './vfaq_deploy.sh services' after ZT authorization to restart with proper IP."
    ZT_IP="0.0.0.0"
  fi

  # Kill existing if re-running
  screen -S srtwatcher -X quit 2>/dev/null || true

  cd "${VFAQ_DIR}"
  screen -S srtwatcher -dm bash -c \
    "SRT_BIND_IP=0.0.0.0 \
     SRT_PUBLIC_IP=${ZT_IP} \
     SRT_PORT_A=${SRT_PORT_A} \
     SRT_PORT_B=${SRT_PORT_B} \
     SRT_LATENCY=${SRT_LATENCY} \
     OBS_AUTOSWAP=${OBS_AUTOSWAP} \
     bash vf-obs-watcher-srt-endpoints.sh \
     2>&1 | tee srt-watcher.log"
  cd ..

  echo "SRT watcher started in screen 'srtwatcher' ✓"
  echo "  Slot A : srt://${ZT_IP}:${SRT_PORT_A}?mode=caller&latency=${SRT_LATENCY}"
  echo "  Slot B : srt://${ZT_IP}:${SRT_PORT_B}?mode=caller&latency=${SRT_LATENCY}"
}

function start_services() {
  echo ""
  echo "==> Starting all services..."

  # Refresh ZT IP
  ZT_IP=$(ip -4 addr show | grep -oP "${ZT_PREFIX}\.\d+" | head -1)

  start_crowd_control
  echo ""
  start_srt_watcher
}

function run_vfaq() {
  echo "==> Running visual-faqtory with run name: ${RUN_NAME}"
  cd "${VFAQ_DIR}"
  python vfaq_cli.py run -n "${RUN_NAME}"
  cd ..
}

function show_status() {
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "  Visual FaQtory v0.5.9-beta — Service Status"
  echo "═══════════════════════════════════════════════════════════════"

  ZT_IP=$(ip -4 addr show | grep -oP "${ZT_PREFIX}\.\d+" | head -1)
  echo "  ZeroTier IP  : ${ZT_IP:-NOT ASSIGNED}"

  for s in comfyui crowd srtwatcher; do
    if screen -list | grep -q "\.${s}"; then
      echo "  screen/${s} : RUNNING"
    else
      echo "  screen/${s} : NOT RUNNING"
    fi
  done

  if [ -n "$ZT_IP" ]; then
    echo ""
    echo "  ── Crowd Control ──"
    echo "  Submit   : http://${ZT_IP}:${VF_CROWD_PORT}/visuals/"
    echo "  Overlay  : http://${ZT_IP}:${VF_CROWD_PORT}/visuals/overlay"
    echo "  Status   : http://${ZT_IP}:${VF_CROWD_PORT}/visuals/api/status"
    echo "  QR       : http://${ZT_IP}:${VF_CROWD_PORT}/visuals/qr.png"
    echo ""
    echo "  ── SRT Endpoints (use in OBS as Media Source → caller) ──"
    echo "  Slot A   : srt://${ZT_IP}:${SRT_PORT_A}?mode=caller&latency=${SRT_LATENCY}"
    echo "  Slot B   : srt://${ZT_IP}:${SRT_PORT_B}?mode=caller&latency=${SRT_LATENCY}"
  fi

  echo ""
  echo "  screen -r comfyui     # attach ComfyUI"
  echo "  screen -r crowd       # attach Crowd Control"
  echo "  screen -r srtwatcher  # attach SRT watcher"
  echo "═══════════════════════════════════════════════════════════════"
  echo ""
}

function save_workspace() {
  echo "==> Archiving workspace..."
  tar -czvf /workspace/saved-runs.tar.gz "${VFAQ_DIR}/worqspace"
  echo "Saved at /workspace/saved-runs.tar.gz"
}

function clean_all() {
  echo "==> Cleaning environment..."

  # Stop all screens
  for s in comfyui crowd srtwatcher; do
    screen -S "$s" -X quit 2>/dev/null || true
  done

  rm -rf "${COMFY_DIR}"
  rm -rf "${VFAQ_DIR}"
  rm -f "${ARCHIVE_NAME}"

  if [ -f /workspace/saved-runs.tar.gz ]; then
    rm -f /workspace/saved-runs.tar.gz
    echo "Removed /workspace/saved-runs.tar.gz"
  fi

  rm -- "$0"

  echo "Full cleanup complete."
}

# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

case "$1" in
  initrun)
    download_and_unpack
    check_nvenc
    install_zerotier
    setup_comfy
    setup_vfaq
    start_comfy_background
    echo "==> Waiting 10s for ComfyUI startup..."
    sleep 10
    wait_for_zt_ip || true
    start_services
    echo ""
    show_status
    echo "==> Starting generation pipeline..."
    run_vfaq
    ;;
  run)
    start_comfy_background
    sleep 5
    run_vfaq
    ;;
  services)
    # (Re)start crowd control + SRT watcher — use after ZT authorization
    wait_for_zt_ip
    start_services
    show_status
    ;;
  status)
    show_status
    ;;
  save)
    save_workspace
    ;;
  clean)
    clean_all
    ;;
  *)
    echo ""
    echo "Visual FaQtory v0.5.9-beta — GPU Server Deploy"
    echo ""
    echo "Usage:"
    echo "  ./vfaq_deploy.sh initrun    # Full setup: download, install, ZeroTier, start everything, run"
    echo "  ./vfaq_deploy.sh run        # Start ComfyUI + run generator (no setup)"
    echo "  ./vfaq_deploy.sh services   # (Re)start Crowd Control + SRT watcher (after ZT auth)"
    echo "  ./vfaq_deploy.sh status     # Show all service status + URLs"
    echo "  ./vfaq_deploy.sh save       # Archive workspace"
    echo "  ./vfaq_deploy.sh clean      # Nuke everything"
    echo ""
    exit 1
    ;;
esac
