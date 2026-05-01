# Release Notes

## v0.9.1-beta

### Fix: per-cycle pingpong silently did nothing without interpolation

`Finalizer._process_cycle_video()` early-returned the unchanged input when
`per_cycle_interpolation: false`, even if `per_cycle_pingpong: true`. Pingpong
is now fully independent of interpolation — any of the four combinations
(neither / pingpong-only / interpolation-only / both) work as expected.

When pingpong runs, the produced clip is forward + reverse-without-seam-frame
(perfect-loop shape), preserving fps. End-to-end ffmpeg verification: a 2s
input becomes a 4s output with the correct frame count, all combinations
verified.

### Fix: lastframe extraction looped on itself when pingpong was on

When pingpong is applied to a cycle clip, the clip's actual final frame
equals its first frame (it ran forward then reversed). The previous engine
extracted the lastframe from the pingpong'd file, so the next cycle re-used
the *start* of the previous cycle's content — visually looping on itself
instead of progressing.

The engine now extracts the lastframe from the **original pre-pingpong
video** when `per_cycle_pingpong: true`, so the next cycle picks up from the
true end-of-content frame. Verified: pingpong-last vs source-first mean
pixel diff ≈ 0.36, vs source-last ≈ 21.77. Without the fix the engine was
reading the (essentially identical to source-first) frame.

### Feature: `post_run_pingpong` + `post_run_interpolation` (independent stages)

Post-stitch finalizer stages are now independently controllable via
`finalizer.post_run_pingpong` and `finalizer.post_run_interpolation`. The
master `enabled` flag still gates everything.

  - `enabled: true` + `post_run_interpolation: true` (default when unset) +
    `post_run_pingpong: false` → legacy: minterpolate + upscale →
    `final_60fps_1080p.mp4`
  - `enabled: true` + `post_run_pingpong: true` +
    `post_run_interpolation: false` → mirror final master into perfect loop →
    `final_pingpong.mp4`
  - `enabled: true` + both true → pingpong feeds into interpolation +
    upscale → `final_60fps_1080p.mp4`

Backwards compatible: configs that only set `enabled: true` continue to
behave exactly as before (interpolation + upscale, no pingpong).

### Feature: `crowd_control.inject_source_mode` (audience prompt routing)

New config field controls *how* an audience-injected prompt drives the next
cycle, independent of `inject_mode` (which only controls what TEXT goes in):

  - `inject_source_mode: "as_image_source"` (default) — IMG2VID with
    previous lastframe as the SOURCE image. Hard visual continuity from
    cycle N-1.
  - `inject_source_mode: "as_reference"` — TEXT2VID with previous lastframe
    as a REFERENCE image (when the model supports `reference_image_urls`).
    Audience prompt is the primary visual driver; lastframe acts as a soft
    style/identity tether.

`inject_mode` (append/replace) is now orthogonal — append concatenates
audience text onto the story window, replace uses audience text alone. Any
of the four combinations from the spec work as designed.

`reinject` and `require_morph` are auto-suppressed during crowd-driven
cycles (in either source mode) so the audience prompt isn't diluted by an
img2img remix of the prior frame.

### Backend: Venice now honors per-request `reference_image_paths`

`VeniceBackend._build_video_payload` previously only consumed the static
`venice.video.reference_image_urls` list from config. It now ALSO accepts
per-request `Path` objects passed via `GenerationRequest.reference_image_paths`,
data-URLs them, and merges with any static URLs. Required for the new
`as_reference` crowd mode but generally useful for any caller wanting to
attach a per-call reference image. If the model doesn't support reference
images, both sources are dropped with a log warning.

### Config

- `worqspace/config.yaml`: `post_run_pingpong` + `post_run_interpolation`
  documented as commented-out reference (no behaviour change). New
  `crowd_control.inject_source_mode: "as_image_source"` set explicitly.
- `worqspace/config.example.yaml`: both new flags fully populated with
  inline docs.

---

## v0.9.0-beta (patch)

### Fix: Venice aspect_ratio "107:60" HTTP 400 invalid_enum_value

**Symptom**
```
[Venice] Image generation failed: Venice API error HTTP 400:
  aspect_ratio: Invalid enum value. Expected 'auto' | '1:1' | '3:2' | '16:9'
  | '21:9' | '9:16' | '2:3' | '3:4' | '4:5', received '107:60'
[SlidingStory/Venice] Evolved keyframe failed, falling back to image_to_video
```

**Root cause**
`_aspect_ratio_from_dims()` in `vfaq/venice_backend.py` returned the raw
GCD-reduced ratio of the requested pixel dimensions (e.g. 856×480 →
`107:60`, since `gcd(856, 480) = 8`). Mathematically correct, but Venice's
`/image/edit` and `/video/queue` endpoints accept `aspect_ratio` only from
a fixed enum. As a secondary issue, the `venice.image.aspect_ratio` config
key was silently ignored — `VeniceConfig` had no field for it, so even with
the value set in YAML the code fell through to the broken GCD path.

**Fix**
1. New `_snap_aspect_ratio()` helper snaps any aspect-ratio string to the
   nearest Venice-valid enum value via log-distance (so 856:480 → 16:9, not
   3:2).
2. `_aspect_ratio_from_dims()` now routes through the snapper — it can no
   longer return an invalid value.
3. `VeniceConfig` gains `image_aspect_ratio` + `image_resolution` fields,
   and `from_dict()` parses them from `venice.image.aspect_ratio` /
   `venice.image.resolution`.
4. New `_select_image_aspect_ratio()` mirrors `_select_aspect_ratio()` for
   the image endpoint: per-request → config → snapped pixel-dim fallback.
5. `_edit_image` now uses the new selector instead of calling
   `_aspect_ratio_from_dims` directly.
6. Defence-in-depth: `_queue_video_request` gains an `aspect_ratio_snapped`
   retry branch mirroring the existing `resolution_snapped` / duration
   logic — if a future model rejects an aspect_ratio with
   `invalid_enum_value`, we snap to the nearest of the model-supplied
   options and retry once.

**Config cleanup**
- Removed dead `width / height / aspect_ratio / resolution` keys from the
  top-level `backend / image_backend / video_backend / morph_backend`
  blocks (they are not consumed when `type: venice`; only
  `venice.image.*` / `venice.video.*` are read).
- `worqspace/config.yaml` annotated with commented-out reference for every
  available parameter — no behaviour changes vs prior version.
- Added `worqspace/config.example.yaml` — fully-filled reference config
  with every available parameter set.

---

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
