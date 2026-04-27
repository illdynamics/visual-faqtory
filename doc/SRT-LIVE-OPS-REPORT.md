# SRT Live Ops Report

## Scope

This report covers the two-endpoint SRT A/B watcher implemented by `vf-obs-watcher-srt-endpoints.sh`, its env example, systemd unit example, and the external live visuals setup documentation.

## Validation status

| Area | Statically verified | Unit tested | Smoke tested locally | Live tested with real OBS caller / real network |
|---|---:|---:|---:|---:|
| Watch-dir resolution / config selection | Yes | Yes | Yes | No |
| Startup with empty watch dir | Yes | Yes | Yes | No |
| Startup preload from existing clips | Yes | Yes | Yes | No |
| Direct-write pickup | Yes | Yes | Via helper/manual drill only | No |
| Atomic-move pickup | Yes | Yes | Via helper/manual drill only | No |
| A/B slot swap logic | Yes | Yes | Yes | No |
| OBS autoswap on/off logic | Yes | Yes | Yes | No |
| File validation / corrupt mp4 rejection | Yes | Yes | Yes | No |
| Stale pid / ffmpeg crash recovery | Yes | Yes | Yes | No |
| Watch-dir disappearance / remount recovery | Yes | Partially | Yes | No |
| systemd restart behavior | Yes | Example reviewed | Example smoke-checked | No |

## What was actually exercised offline

- Shell-script control paths were inspected directly.
- `tests/test_srt_watcher.py` covers status output, config resolution, preload discovery, smoke-check behavior, one-shot swap processing, helper modes, and key watcher-content assertions.
- `--smoke-check`, `--status`, `--status --verbose`, `--validate-file`, and `--reseed-slots` were designed to be usable without a running OBS instance.

## What was not live-tested in this pass

No real OBS caller, no real SRT network path, and no real long-running systemd service were exercised in-session here. This report therefore does not claim live proof for remote OBS connections, long-session stability under real network churn, or real boot-time service restarts on a real host.

## Operational hardening added in this pass

- stale ffmpeg pid files are detected and cleaned up
- slot files are re-seeded if missing or invalid
- ffmpeg slots are restarted by a background health monitor if they die after startup
- watch dir is recreated if it disappears and the inotify loop is restarted
- corrupt / incomplete mp4 files are rejected when `ffprobe` is available
- `--validate-file` provides a fast operator check for a candidate clip
- `--reseed-slots` re-seeds both slots from the newest valid watched clip or placeholders
- `--status --verbose` exposes slot file / pid / validity state

## Remaining real-world risks

- SRT network issues, firewall issues, MTU weirdness, or ZeroTier quirks still require real network testing.
- If `ffprobe` is unavailable, validation falls back to stable-size checks only; that is weaker against structurally corrupt files.
- OBS autoswap still depends on `obsws_python`, working WebSocket credentials, and correct scene/source names.
- A watcher restart can recover slot files and ffmpeg processes, but it cannot repair an upstream generator that stopped producing clips.
