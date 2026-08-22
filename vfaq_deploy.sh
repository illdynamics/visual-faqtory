#!/usr/bin/env bash

set -e

ARCHIVE_URL="https://live.wonq.tv/vfaq.tar.gz"
ARCHIVE_NAME="vfaq.tar.gz"
RUN_NAME=$(date +"%Y-%m-%d_%H%M%S")

COMFY_DIR="ComfyUI"
VFAQ_DIR="visual-faqtory"

function install_system() {
  echo "==> Installing system dependencies..."
   apt update
   apt install -y python3.11 python3.11-venv python3.11-dev python3-pip \
                      git curl build-essential ffmpeg screen wget
}

function download_and_unpack() {
  echo "==> Downloading vfaq archive..."
  wget -O ${ARCHIVE_NAME} ${ARCHIVE_URL}
  echo "==> Extracting..."
  tar -xzf ${ARCHIVE_NAME}
}

function setup_comfy() {
  echo "==> Setting up ComfyUI..."
  cd ${COMFY_DIR}
  pip install --upgrade pip
  pip install -r requirements.txt
  pip install -r ${HOME}/${COMFY_DIR}/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt
  ln -s ${HOME}/${COMFY_DIR}/models/svd/svd_xt_1_1.safetensors ${HOME}/${COMFY_DIR}/models/checkpoints/svd_xt_1_1.safetensors
  cd ..
}

function setup_vfaq() {
  echo "==> Setting up visual-faqtory..."
  cd ${VFAQ_DIR}
  pip install --upgrade pip
  pip install -r requirements.txt
  cd ..
}

function start_comfy_background() {
  echo "==> Starting ComfyUI..."
  cd ${COMFY_DIR}
  screen -S comfyui -dm bash -c "python main.py --listen 0.0.0.0 --port 8188"
  cd ..
}

function run_vfaq() {
  echo "==> Running visual-faqtory with run name: ${RUN_NAME}"
  cd ${VFAQ_DIR}
  python vfaq_cli.py run -n ${RUN_NAME}
  cd ..
}

function save_workspace() {
  echo "==> Archiving workspace..."
  tar -czvf /workspace/saved-runs.tar.gz ${VFAQ_DIR}/worqspace
  echo "Saved at /workspace/saved-runs.tar.gz"
}

function clean_all() {
  echo "==> Cleaning environment..."

  rm -rf ${COMFY_DIR}
  rm -rf ${VFAQ_DIR}
  rm -f ${ARCHIVE_NAME}

  if [ -f /workspace/saved-runs.tar.gz ]; then
    rm -f /workspace/saved-runs.tar.gz
    echo "Removed /workspace/saved-runs.tar.gz"
  fi

  rm -- "$0"

  echo "Full cleanup complete."
}

case "$1" in
  initrun)
    install_system
    download_and_unpack
    setup_comfy
    setup_vfaq
    start_comfy_background
    sleep 10
    run_vfaq
    ;;
  run)
    start_comfy_background
    sleep 5
    run_vfaq
    ;;
  save)
    save_workspace
    ;;
  clean)
    clean_all
    ;;
  *)
    echo "Usage:"
    echo "  ./vfaq_deploy.sh initrun"
    echo "  ./vfaq_deploy.sh run"
    echo "  ./vfaq_deploy.sh save"
    echo "  ./vfaq_deploy.sh clean"
    exit 1
    ;;
esac
