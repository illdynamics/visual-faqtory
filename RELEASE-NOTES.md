# Release Notes — Visual FaQtory v0.5.6-beta

**"Clean Base + Reinject Default + Run/Saved-Runs Refactor"**

Release Date: 2025-02-12

---

## Summary

v0.5.6-beta is a clean, stable foundation for the Visual FaQtory pipeline. Previous versions (v0.5.5 and earlier) accumulated many experimental systems — turbo mode, crowd prompts, audio reactivity, deforum stacks, longcat streaming, multiple backends — that created complexity without production stability. This release strips everything back to the core working pipeline and delivers a reliable, reproducible visual generation system.

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

## Files Changed

### New Files
- `RELEASE-NOTES.md` — This file

### Rewritten Files
- `vfaq/construqtor.py` — Removed v2v, stream, stability; fixed broken `get_stream_config` import
- `vfaq/backends.py` — Removed v2v, stream, longcat from all backends (~600 lines removed)
- `README.md` — Complete rewrite documenting only v0.5.6-beta features
- `DOCUMENTATION.md` — Complete rewrite with accurate docs

### Cleaned Files
- `vfaq/__init__.py` — Clean exports, no dead references
- `vfaq/visual_briq.py` — Removed stream mode fields from GenerationSpec
- `vfaq/sliding_story_engine.py` — Cleaned docstring (removed deforum reference)
- `worqspace/config.yaml` — Minimal canonical config, no zombie keys

### Removed Files
- `vfaq/utils_sanitize.py` — Crowd prompt sanitizer (no longer imported)
- `vfaq/video_preprocess.py` — V2V preprocessing (no longer imported)

### Preserved Files (already clean from v0.5.5)
- `vfaq/visual_faqtory.py` — Main orchestrator (was already v0.5.6-ready)
- `vfaq/finalizer.py` — Stitch + interpolate + upscale pipeline
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

*Visual FaQtory v0.5.6-beta — Built by Ill Dynamics / WoNQ*
