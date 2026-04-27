#!/usr/bin/env bash
set -euo pipefail
usage() {
  cat <<USAGE
Usage: $(basename "$0") --watch-dir DIR --source FILE [--mode direct-write|atomic-move] [--name output.mp4]
USAGE
}
WATCH_DIR=""; SOURCE_FILE=""; MODE="direct-write"; OUT_NAME=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --watch-dir) shift; WATCH_DIR="${1:-}" ;;
    --source) shift; SOURCE_FILE="${1:-}" ;;
    --mode) shift; MODE="${1:-}" ;;
    --name) shift; OUT_NAME="${1:-}" ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
[[ -n "$WATCH_DIR" && -n "$SOURCE_FILE" ]] || { usage >&2; exit 2; }
[[ -f "$SOURCE_FILE" ]] || { echo "Source file missing: $SOURCE_FILE" >&2; exit 1; }
[[ "$SOURCE_FILE" == *.mp4 ]] || { echo "Source file must be an .mp4" >&2; exit 1; }
mkdir -p "$WATCH_DIR"
OUT_NAME="${OUT_NAME:-$(basename "$SOURCE_FILE")}"
TARGET="$WATCH_DIR/$OUT_NAME"
case "$MODE" in
  direct-write)
    cp "$SOURCE_FILE" "$TARGET"
    echo "mode=direct-write"
    echo "target=$TARGET" ;;
  atomic-move)
    tmpdir="$(mktemp -d)"
    trap 'rm -rf "$tmpdir"' EXIT
    cp "$SOURCE_FILE" "$tmpdir/$OUT_NAME"
    mv -f "$tmpdir/$OUT_NAME" "$TARGET"
    echo "mode=atomic-move"
    echo "target=$TARGET" ;;
  *) echo "Invalid mode: $MODE" >&2; exit 2 ;;
esac
