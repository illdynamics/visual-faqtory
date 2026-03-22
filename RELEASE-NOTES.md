# Release Notes — Visual FaQtory v0.6.0-beta

**"Veo Backend — Cloud Video Generation Goes Brrrr"**

Release Date: 2026-03-22

---

## Summary

v0.6.0-beta introduces a first-class Google Veo video generation backend alongside the existing ComfyUI backend. Veo generates video directly from text or images via the Google Gen AI SDK — no local GPU required. The sliding story engine has been made backend-aware, with Veo-specific cycle routing that maps the existing pipeline semantics to Veo's native text_to_video, image_to_video, and first_last_frame modes. A new loop closure feature enables seamless looping video output. All existing ComfyUI functionality is fully preserved.

---

## What's New in v0.6.0-beta

### Google Veo Backend (`vfaq/veo_backend.py`)

- **Full Veo integration** via `google-genai` SDK with support for:
  - `text_to_video` — generate video purely from text prompt
  - `image_to_video` — generate video from a starting image + prompt
  - `first_last_frame` — interpolation between start and end frames (replaces morph)
  - `extend_video` — extend a previously Veo-generated clip
- **Dual provider support:**
  - `provider: gemini` — Google Gemini Developer API (API key auth)
  - `provider: vertex` — Google Vertex AI (ADC / service account auth)
- **Auth resolution chain** for gemini: `GOOGLE_API_KEY` → `GEMINI_API_KEY` → `GOOGLE_API_TOKEN` → `GEMINI_API_TOKEN`
- **Retry with exponential backoff + jitter** for transient failures, with clear separation of auth/quota/parameter errors
- **Operation polling** with configurable interval and timeout
- **Immediate file download** — video bytes saved to disk as soon as generation completes
- **Per-generation observability logging:** provider, model, mode, duration, auth mode, seed, latency, attempt count

### Backend Abstraction (`vfaq/backends.py`)

- **`BackendType.VEO`** enum added
- **`GenerationRequest` extended** with Veo-specific fields:
  - `veo_mode`, `last_frame_path`, `input_video_path`
  - `reference_image_paths[]`, `reference_image_types[]`
  - `generate_audio`, `aspect_ratio`, `resolution`, `person_generation`
  - `continuity_strength`, `mutation_strength`, `identity_lock_strength`, `loop_closure_strength`
  - `sample_count`, `storage_uri`, `compression_quality`
- All new fields are **backward-compatible** — existing ComfyUI/Mock configs work without changes
- **Factory function** updated: `create_backend()` lazily imports `VeoBackend` to avoid hard google-genai dependency

### Veo-Aware Sliding Story Engine (`vfaq/sliding_story_engine.py`)

- **Backend-aware cycle routing:**
  - Veo cycles skip the separate txt2img→img2img→img2vid pipeline
  - Cycle 1: `text_to_video` or `image_to_video` (if base image exists)
  - Cycle n>1: `image_to_video` from previous last frame
  - If `require_morph=true`: generates target keyframe then uses `first_last_frame`
- **Loop closure** (new `enable_loop_closure` config):
  - Final cycle generates a clip from last frame back to cycle-1 anchor frame
  - Creates seamless loop when video is played on repeat
- **`_extract_first_frame_ffmpeg()`** — new helper to extract first frame for anchor storage
- **`SlidingStoryConfig`** extended with `veo_config` and `enable_loop_closure` fields
- **All ComfyUI/Mock cycle logic untouched** — the Veo path is a clean `if is_veo: ... continue` block

### Config & CLI Updates

- **`config-veo.yaml`** — complete Veo backend config template with inline documentation
- **`config.yaml`** — updated header, backend comments now list all three backends
- **CLI** — `--backend veo` option, `backends` command lists Veo availability
- **`requirements.txt`** — `google-genai>=1.0` added (optional, only for Veo)

### Pipeline Wiring (`vfaq/visual_faqtory.py`)

- `_build_story_config()` now injects `veo:` section into backend config
- Veo duration defaults applied when `img2vid_duration_sec` not set
- `veo_config` and `enable_loop_closure` passed through to story engine

---

## Backward Compatibility

- **No breaking changes.** All existing `config.yaml` files with `backend.type: comfyui` or `backend.type: mock` work exactly as before.
- **No new mandatory dependencies.** `google-genai` is only imported when `backend.type: veo` is selected.
- **No behavioral changes** to the ComfyUI or Mock backends.

---

## Quick Start — Veo Backend

```bash
# 1. Install google-genai
pip install google-genai

# 2. Set API key
export GEMINI_API_TOKEN=<your-gemini-api-key>

# 3. Copy Veo config
cp worqspace/config-veo.yaml worqspace/config.yaml

# 4. Run
python vfaq_cli.py -n my-veo-project
```

---

## Known Limitations (v0.6.0-beta)

- `extend_video` mode requires prior Veo-generated clips (not arbitrary MP4s)
- Veo `generate_image` for text mode produces a short clip + extracts frame (no native txt2img)
- `negative_prompt` support varies by Veo model — unsupported models log a warning and continue
- Reference images (`STYLE`/`ASSET`) availability depends on model version
- Loop closure quality depends on visual similarity between last and first frames

---

# Release Notes — Visual FaQtory v0.5.8-beta

**"Crowd Replace Visual Reset & OBS Prewarm Swap"**

Release Date: 2026-02-27

---

## Summary

v0.5.8-beta delivers two long-awaited fixes: Crowd Control `replace` mode now performs a full visual reset (TEXT2IMG, no last-frame bleed) for every cycle it fires — including cycle 1 — and OBS A/B switching has been upgraded to a prewarm swap that eliminates the ~1s black flash by warming up the incoming source behind the current one before cutting over.

---

## What's New in v0.5.8-beta

### Crowd Control — Force TEXT2IMG on Replace Mode (`vfaq/sliding_story_engine.py`)

- **`crowd_replace_active` flag** — computed per-cycle: `True` when a crowd entry was consumed _and_ `inject_mode == "replace"`.
- **`effective_reinject`** — `config.reinject AND NOT crowd_replace_active`. Crowd replace overrides the global reinject setting for that cycle only; subsequent cycles resume normal chaining from the newly generated last frame.
- **`effective_require_morph`** — `config.require_morph AND effective_reinject`. Morph requires both chain endpoints; a replace-reset cycle has no valid start image so morph is suppressed for that cycle.
- **Cycle 1 override** — even if `base_image_path` is set, a crowd replace active in cycle 1 bypasses the base image and runs TEXT2IMG, preventing visual bleed from the base image into a wholly different crowd prompt.
- **Cycle > 1 override** — a new `crowd_replace_active` branch runs `GenerationRequest(mode=InputMode.TEXT, init_image_path=None)` and omits `start_image` from `briq_data["paths"]`, proving no last-frame was used.
- **Loud logging** — every visual reset emits `[SlidingStory] Crowd REPLACE visual reset: TEXT2IMG keyframe (reinject overridden)`.
- **Briq telemetry** — `briq_data["crowd_control"]` now includes `visual_reset: bool` and `effective_reinject: bool` for post-run inspection.

### OBS A/B — Prewarm Swap (`obs-swap.py`)

- **Full rewrite** of `obs-swap.py` implementing the prewarm sequence:
  1. Z-order: current source on top (`index 0`), incoming target below (`index 1`).
  2. Enable target while hidden behind current.
  3. Force media reload via `trigger_media_input_action(RESTART)` — eliminates the need for "Close file when inactive".
  4. Fallback: if RESTART fails, toggles `local_file` setting to force re-open.
  5. Poll `get_media_input_status` until `media_state == OBS_MEDIA_STATE_PLAYING` or `cursor > 0`.
  6. Disable current — cut is instantaneous, frame already buffered.
  7. Reorder target to `index 0` so the next swap works in reverse.
- **Fail-open**: timeout or WebSocket errors log a warning and proceed — the swap still completes.
- **Configurable prewarm** via `--prewarm SECONDS` CLI arg or `OBS_PREWARM_SEC` env var (default `0.8`).
- **OBS settings overridable** via env: `OBS_HOST`, `OBS_PORT`, `OBS_PASSWORD`, `OBS_SCENE`.

### OBS Watcher — Prewarm Config (`vf-obs-watcher-same-machine.sh`)

- Passes `--prewarm "$OBS_PREWARM_SEC"` to `obs-swap.py` on every swap call.
- Exports `OBS_PREWARM_SEC` with default `0.8` — override before launching the watcher or set in your environment.
- Watcher startup log now prints the active prewarm value.
- **User note**: "Close file when inactive" can be disabled on both OBS sources — media reload is now WebSocket-driven.

---

## What's New in v0.5.7-beta (Previous)

**"Strict Timing Normalization & Per-Cycle Interpolation"**

Release Date: 2026-02-21

### Strict Timing Normalization

### Strict Timing Normalization
- Implemented a `TimingResolver` module (`vfaq/timing.py`) to normalize video frames, FPS, and duration.
- Configurable `timing_authority` (`frames`, `duration`, or `fps`) dictates which variable is treated as truth, adjusting others accordingly.
- All timing calculations now occur *before* generation, ensuring deterministic video length.

### Per-Cycle Interpolation
- Optional feature to interpolate each raw SVD video to a higher FPS immediately after generation.
- Interpolated videos are stored in a new `run/videos_interpolated/` directory.
- The final stitch process uses these interpolated videos (with fallback to raw if interpolation fails for a cycle).
- Configurable `finalizer.per_cycle_interpolation` and `finalizer.interpolate_fps` settings in `worqspace/config.yaml`.

---

## What's New in v0.5.6-beta

### Reinject Mode Default ON
- Reinject is now the default behavior — every cycle produces an img2img keyframe from the previous cycle's last frame
- CLI: `--reinject` / `-r` (default ON), `--no-reinject` / `-R` to disable
- Denoise is sampled from configurable range per cycle

### Directory Refactor
- **`qodeyard/`** → **`run/`** — current run artifacts live here
- **`worqspace/qonstructions/`** → **`worqspace/saved-runs/`** — archived runs
- Clean subdirectory structure: `run/videos/`, `run/frames/`, `run/briqs/`, `run/meta/`
- Run inputs (config, story, prompts, base files) are copied into `run/meta/` at start

### Project Saving
- After completion, prompts for project name (or accepts `-n` flag)
- Moves `run/` → `worqspace/saved-runs/<project-name>/`
- Auto-suffix (`-001`, `-002`) if name already exists
- Deliverable renamed to `<project-name>.mp4` in saved run root

### Finalizer Working Again
- Complete pipeline: stitch → interpolate 60fps → upscale 1080p → audio mux
- Output naming: `final_video.mp4` → `final_video_60fps.mp4` → `final_video_60fps_1080p.mp4` → `final_video_60fps_1080p_audio.mp4`
- GPU encoding (h264_nvenc) with CPU fallback (libx264)

### Audio Mux + Sync
- Base audio auto-detected from `worqspace/base_audio/`
- `audio.sync_video_audio: true` auto-computes cycle count from audio duration
- Audio duration via ffprobe (no librosa dependency)
- Video trimmed to audio duration on mux

### ComfyUI-Only Backend
- Single production backend: ComfyUI API
- Mock backend retained for testing only
- LoRA injection support with automatic workflow wiring

### Working Input Modes
- **text**: txt2img → img2vid
- **image**: base image from `worqspace/base_images/` → img2img → img2vid
- **video**: frame extraction from `worqspace/base_video/` → img2img → img2vid
- **auto**: auto-detect mode (video > image > text)

### Dry Run Mode
- `--dry-run` validates config, resolves inputs, writes state JSON, exits before generation
- Useful for testing config changes without burning GPU time

### Briq State Tracking
- Every cycle writes JSON to `run/briqs/cycle_NNN.json`
- Includes: paragraph window, prompt, seed, denoise, paths, backend info
- Run-level state in `run/faqtory_state.json`

---

## What Was Removed

The following experimental features from v0.5.5 and earlier have been fully removed (code, config, docs, CLI flags):

- **Deforum stack** (agents, camera presets, init_pool, deformer)
- **Audio reactivity** (librosa, sounddevice, beat detection)
- **Turbo mode** (live frame generation engine)
- **Crowd mode** (prompt server, queue, rate limiting)
- **Stream / Longcat mode** (autoregressive video continuation)
- **Video2Video** (true v2v latent pipeline)
- **Psycho Mode** (entropy injection system)
- **OSC output** (TouchDesigner integration)
- **MIDI sidecar**
- **Replicate backend** (API-based generation)
- **Diffusers backend** (local HuggingFace pipeline)
- **Story Engine** (beat-based, non-paragraph story engine)
- **Init Pool** (motion-aware init image pools)
- **Color Stability Controller** (per-frame color correction)
- **Split backend** (separate image/video backends)
- **Prompt drift tags**
- **LLM/OpenAI/Gemini integration**

---

## Files Changed (v0.5.8-beta)

### New Files
- `vfaq/timing.py` — New module for timing normalization.

### Modified Files
- `vfaq/sliding_story_engine.py` — Integrated TimingResolver, Finalizer instantiation, and per-cycle interpolation logic. Removed `_concat_videos_ffmpeg`.
- `vfaq/backends.py` — `GenerationRequest` updated to include `Optional[video_frames]` and `float` for `video_fps`. `MockBackend._create_placeholder_video` updated to use `effective_frames`.
- `vfaq/finalizer.py` — Added `per_cycle_interpolation` flag, `videos_interpolated_dir`, and `_interpolate_cycle` method.
- `worqspace/config.yaml` — Added `finalizer.per_cycle_interpolation` setting.
- `README.md` — Updated Features, Directory Structure, and Finalizer Output Naming sections.
- `RELEASE-NOTES.md` — Updated with v0.5.8-beta release notes.

### Preserved Files (already clean from v0.5.5)
- `vfaq/visual_faqtory.py` — Main orchestrator (was already v0.5.6-ready)
- `vfaq/prompt_synth.py` — Deterministic prompt synthesis
- `vfaq/prompt_bundle.py` — Prompt file loading
- `vfaq/instruqtor.py` — Instruction preparation agent
- `vfaq/inspeqtor.py` — Quality inspection agent
- `vfaq/base_folders.py` — Input file detection
- `vfaq/image_metrics.py` — Image quality metrics

---

## Acceptance Tests

1. ✅ `python vfaq_cli.py --help` works; no removed features mentioned
2. ✅ `python -c "import vfaq"` does not crash; no broken imports
3. ✅ ConstruQtor instantiation works without `get_stream_config` error
4. ✅ `python vfaq_cli.py --dry-run -b mock` creates `./run` structure
5. ✅ No dead imports or missing modules in any Python file
6. ✅ `requirements.txt` matches actual imports
7. ✅ No removed features referenced in code, config, or docs

---

## CLI Examples

```bash
python vfaq_cli.py                              # Default run (reinject ON)
python vfaq_cli.py -n my-project                # Named project
python vfaq_cli.py --no-reinject                # Disable reinject
python vfaq_cli.py -R                           # Disable reinject (short)
python vfaq_cli.py --mode image                 # Image mode
python vfaq_cli.py --mode video                 # Video mode
python vfaq_cli.py -n test -b mock --dry-run    # Mock dry run
python vfaq_cli.py status                       # Check status
python vfaq_cli.py backends                     # List backends
```

---

*Visual FaQtory v0.5.8-beta — Built by Ill Dynamics / WoNQ*
