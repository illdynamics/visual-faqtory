# Visual FaQtory v0.7.6-beta — SRT endpoint validation report

This pass validates and hardens the two-endpoint SRT playout watcher after the earlier implementation patch.

## Scope checked

### 1) Startup with empty watch dir
Validated by inspection and focused tests:
- the watcher still creates / keeps slot state in `run/obs`
- if no existing watched clip is available, it falls back to black placeholders
- `--status` remains quiet and scriptable when no runtime processes were started

### 2) Startup with pre-existing cycle clips
Validated by focused tests and direct script inspection:
- the watcher scans the resolved watch folder for the newest `.mp4`
- when `PRELOAD_EXISTING_ON_START=1`, that newest ready clip is seeded into both slot files before the live loop starts
- `PRELOAD_FILE=` is exposed via `--status`

### 3) Watch behavior for direct writes and atomic moves
Validated by script inspection and tests:
- inotify watches `close_write,moved_to`
- direct file writes into the watch dir are picked up
- atomic renames / moves into the watch dir are picked up
- non-`.mp4` files are ignored cleanly

### 4) Slot A/B swap logic
Validated by a new one-shot helper path and tests:
- the inactive slot is chosen as the target on each new clip
- the new clip is copied into the inactive slot first
- ffmpeg for that slot is restarted
- OBS prewarm / disable ordering remains target-first, then old-source-off
- state flips only after the target slot has been loaded

### 5) OBS autoswap on/off behavior
Validated by inspection:
- `OBS_AUTOSWAP=0` leaves SRT endpoints running and skips scene-item toggling
- `OBS_AUTOSWAP=1` now resolves a usable Python interpreter more defensively before trying `obsws_python`
- missing OBS Python tooling still warns clearly instead of crashing the watcher

### 6) Env overrides and watch-dir logic
Validated by tests:
- `VF_CONFIG_FILE` overrides config discovery
- default config discovery prefers `worqspace/config.yaml` and only falls back to `worqspace/config-live.yaml`
- `run.output_dir` is honored
- exact watch-dir logic is now explicit via `WATCH_MODE=`:
  - `WATCH_MODE=videos` → watch `<output_dir>/videos`
  - `WATCH_MODE=videos_interpolated` → watch `<output_dir>/videos_interpolated`
  - `WATCH_MODE=override` → `VF_WATCH_DIR` won
- relative `VF_WATCH_DIR` overrides are resolved from repo base

### 7) Systemd example correctness
Validated by inspection and patched:
- install notes explicitly include `inotify-tools`
- the unit now runs `ExecStartPre=... --smoke-check`
- `TimeoutStopSec=15` was added for cleaner service shutdown behavior

### 8) Documentation accuracy / operator usability
Validated and tightened:
- docs now explain `WATCH_MODE`
- docs include an operator checklist for first live deployment
- the env example documents override behavior more clearly
- `--status` is clean enough for scriptable checks

## Bugs fixed in this pass

1. **EXIT trap masked failure modes and polluted status output**
   The watcher cleanup trap previously exited `0` unconditionally and also printed shutdown logs for `--status`. This pass preserves the real exit status and only runs noisy cleanup when runtime streaming actually started.

2. **`WATCH_MODE` was being lost via command-substitution scoping**
   The previous implementation set watch mode inside a subshell-style command substitution. This pass derives `WATCH_MODE` explicitly before resolving `WATCH_DIR`, so status output now matches the actual watch path.

3. **OBS Python runtime selection was too brittle**
   The watcher previously warned about alternate Python interpreters but still tried to execute the default missing `.venv` path in autoswap helpers. It now resolves a real runtime interpreter up front.

4. **Manual swap validation was awkward**
   Added `--process-file /path/to/clip.mp4` as a lightweight one-shot helper to load a completed clip into the inactive slot and exit. This makes smoke-checking swap behavior less janky.

## Automated validation status

### Statically inspected
- `vf-obs-watcher-srt-endpoints.sh`
- `vf-obs-watcher-srt-endpoints.env.example`
- `EXTERNAL-LIVE-VISUALS-SETUP.md`
- `ops/systemd/vf-srt-watcher.service.example`

### Unit tested
Focused watcher tests cover:
- config preference for `config.yaml`
- `VF_CONFIG_FILE` override behavior
- exact `WATCH_MODE` / interpolation directory selection
- preload discovery of the newest existing clip
- smoke-check dependency validation
- nonzero exit behavior when required tools are missing
- one-shot file processing and slot state flip without OBS autoswap
- watcher event subscription includes both `close_write` and `moved_to`
- quiet `--status` output without cleanup chatter

### Smoke tested
Repository tests executed per test module after this pass:
- `tests/test_animatediff_backend.py` → 7 passed
- `tests/test_animatediff_validation.py` → 4 passed
- `tests/test_backend_routing.py` → 6 passed
- `tests/test_qwen_hybrid_validation.py` → 13 passed
- `tests/test_resume_and_loop_closure.py` → 3 passed
- `tests/test_split_mock_smoke.py` → 1 passed
- `tests/test_srt_watcher.py` → 9 passed
- `tests/test_venice_backend.py` → 12 passed
- `tests/test_visual_faqtory_config.py` → 3 passed

Total: **58 passed**

## Known limitations

- This pass validates watcher logic, docs, and shell behavior, but it does **not** live-test SRT connectivity against a real remote OBS instance in-session.
- File readiness still relies on stable-size polling plus optional `ffprobe`; it cannot prove application-level playback smoothness without a real encoder / network / OBS loop.
- The watcher is intentionally `.mp4`-only right now.

## Bottom line

The two-endpoint SRT watcher is now in much healthier shape for real-world use:
- startup preload works
- direct writes and atomic moves are watched
- watch-dir resolution is explicit and inspectable
- status/smoke-check are cleaner
- slot swapping is easier to validate manually
- service docs are less likely to troll the operator
