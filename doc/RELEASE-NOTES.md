# Release Notes

## v0.9.0-beta

### Major: Backend cleanup, log consolidation, branding update

**Branding**
- Removed all "QonQrete" prefixes from product name — now simply "Visual FaQtory"
- Updated banner: now shows `ComfyUI + Venice + Veo` (removed retired backends)
- Updated startup log: 3 lines merged to 1 compact line
- Version bump: v0.8.10-beta → v0.9.0-beta

**Backend removal (Qwen, AnimateDiff, LTX-Video)**
- Removed `qwen_image_python_backend.py` and `ltx_video_backend.py` entirely
- Removed `QwenImageComfyUIBackend`, `QwenImageBackend`, `AnimateDiffBackend` from `backends.py`
- Removed all qwen/animatediff/ltx routing from `sliding_story_engine.py` and `visual_faqtory.py`
- Removed dead workflow files: `animatediff_i2v.json`, `animatediff_morph_i2v.json`, `qwen_image_*.json`
- Removed config examples: `config-animatediff.example.yaml`, `config-qwen-svd.example.yaml`, `config-ltx.yaml`
- Removed backend test files for deleted backends
- Cleaned `requirements.txt` — removed Qwen/LTX deps; kept: pyyaml, requests, websocket-client, google-genai, pillow, numpy, fastapi, uvicorn, qrcode, obsws_python
- Removed morph from `describe_backend_config()` output (cleaner log line)

**Log consolidation (at source, not filtered)**
- `[FaQtory]` startup: 3 lines → 1 (`Starting run — mode: X | reinject: Y | id: Z`)
- `[SlidingStory]` startup: 8 lines → 2 (story name + Venice duration)
- Removed: "Story copied to", "Backend temp dir", "Backend routing" (duplicate), "Venice video backend detected"
- `[Timing]` lines: merged multi-line input log to 1; added `[Timing]` prefix to resolved line
- Removed duplicate "Resolved Timing" from engine (timing.py already logs it)
- Frame extraction logs: demoted to DEBUG (noise after job-complete log)
- `[Venice] Retrying` messages: demoted from WARNING to DEBUG

**Cycle line ANSI colour**
- Cycle number in `Cycle N/M — window paragraphs [...]` is now bright green (`\033[92m`) for easy visual scanning in terminal output

**Documentation**
- Moved all docs except README.md into `doc/` folder
- README.md rewritten: logo, short description, quickstart, links to doc/
- All docs updated to v0.9.0-beta

---

## v0.8.10-beta (patch 12)

### Spinner + polling improvements

- ETA estimate from rolling job history (`run/.venice_job_timings.json`, last 20 jobs per model)
- Spinner displays `~Ns left` instead of empty when no progress field returned by API
- Poll interval: 5s → 3s (safe, no rate limit risk on Venice)
- `_save_job_timing` called on every successful video job for self-calibration

## v0.8.10-beta (patch 11)

- Spinner redraw rate: 0.5s → 0.1s (10 fps)

## v0.8.10-beta (patch 10)

### Live spinner for all Venice ops

- `_LiveSpinner` background-thread class: redraws stderr at configurable fps, independent of poll/HTTP timing
- Spinner covers: `text2img`, `img2img`, `text2vid`, `img2vid`
- Progress bar when Venice returns numeric progress field (all known aliases checked)
- Clean `\r`-based line reuse — logger output never interleaved with spinner
- `threading.Event.wait(interval)` based clock (daemon thread, zero cost when idle)

## v0.8.10-beta (patch 9)

- fps default: 30 → 24

## v0.8.10-beta (patch 8)

### Duration snap-to-nearest retry

- When Venice returns `invalid_enum_value` on `duration` with `options[]`, backend snaps to nearest valid value and retries automatically
- Logs: `[Venice] Duration '6s' not accepted (valid: ['5s','10s','15s']). Snapping to nearest: '5s'`

## v0.8.10-beta (patch 7)

### text_to_video_first_cycle=false fully wired + per-op durations

- `text_to_video_first_cycle: false` now generates anchor keyframe via `text2img → img2vid` on cycle 1
- Per-op duration engine fix: engine reads `venice.video.text2vid/img2vid.duration_seconds` directly, passes correct value per cycle as `request.duration_seconds`
- All image/video/morph fallback dims updated to 1280×720
- `hide_watermark: true` by default
- img2vid model: `wan-2.1-pro-image-to-video`
- fps: 8 → 30 (later revised to 24 in patch 9)

## v0.8.10-beta (patch 6)

### Per-op duration/aspect_ratio/resolution overrides

- `VeniceConfig` gains 6 new Optional fields: `text2vid_*` / `img2vid_*` for duration, aspect_ratio, resolution
- Config loader parses `venice.video.text2vid.*` and `venice.video.img2vid.*` sub-blocks
- `_select_video_duration/aspect_ratio/resolution` accept `op` param with inheritance chain: request → per-op config → global config
- `_build_video_payload` derives `op` from `source_image` and passes to all selectors

## v0.8.10-beta (patch 5)

### Duration pass-through, allowed_durations removed

- `_select_video_duration` rewritten: no snapping, passes configured duration directly to Venice
- `video_allowed_durations` field, `_VENICE_DEFAULT_ALLOWED_VIDEO_DURATIONS` constant, loader line, from_dict fallback all removed
- `allowed_durations` removed from both config files

## v0.8.10-beta (patch 4)

### aspect_ratio dual-model contradiction resolved

- `aspect_ratio` added back to `_optional_queue_fields()` (ovi-image-to-video rejects it)
- `_select_aspect_ratio` always `str()`-coerces value (YAML parses unquoted `16:9` as int 969)
- Both configs: `aspect_ratio: "16:9"` (quoted)

## v0.8.10-beta (patch 3)

### Restore-Required-Stripped-Fields mechanism

- `_queue_video_request` tracks `restored_fields` set
- On 400 where previously-stripped field is now Required: field restored, added to `restored_fields` (never stripped again)
- `aspect_ratio` removed from `_optional_queue_fields()` (reverted in patch 4)

## v0.8.10-beta (patch 2)

### resolution blind spot + aspect_ratio wrong strip

- `resolution` added to `_optional_queue_fields()` and moved from hardcoded base payload to conditional block
- `_queue_retry_candidates` rewritten to parse `issues[]` array: classifies fields as unsupported vs required, never strips required fields; blob search as fallback
- Model IDs updated: `wan-2-7-text-to-video`, `wan-2-7-image-to-video`, `flux-2-max`, `qwen-edit`

## v0.8.10-beta (patch 1)

### Stale Venice model IDs

- Fixed 4 dead model IDs in `worqspace/config.yaml`
- `grok-imagine-text-to-video` → `wan-2.5-preview-text-to-video`
- `ovi-image-to-video` → `wan-2.5-preview-image-to-video`
- `flux-2-pro` → `z-image-turbo`
- `qwen-image-2` → `qwen-edit`
- `resolution: 576p` → `720p`

## v0.8.5-beta

Initial Venice native backend release.
