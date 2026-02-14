# Visual FaQtory v0.5.6-beta — Documentation

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Configuration Reference](#3-configuration-reference)
4. [Story Engine](#4-story-engine)
5. [Input Modes](#5-input-modes)
6. [Reinject Mode](#6-reinject-mode)
7. [Backend Configuration](#7-backend-configuration)
8. [LoRA Support](#8-lora-support)
9. [Audio Support](#9-audio-support)
10. [Finalizer Pipeline](#10-finalizer-pipeline)
11. [Project Saving](#11-project-saving)
12. [Prompt Files](#12-prompt-files)
13. [Briq State Tracking](#13-briq-state-tracking)
14. [CLI Reference](#14-cli-reference)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. Overview

Visual FaQtory is an automated long-form AI visual generation pipeline. It reads a story, splits it into paragraphs, and generates a continuous visual narrative using a sliding window engine. Each cycle produces a keyframe image and a transition video. Cycles are chained by feeding the last frame of each video into the next cycle's generation step (reinject mode).

The pipeline:
```
story.txt → paragraph sliding window → per-cycle generation
  → keyframe (txt2img or img2img) → video (img2vid or morph)
  → finalizer (stitch → 60fps → 1080p → audio mux)
  → save to worqspace/saved-runs/<project-name>
```

---

## 2. Architecture

### Pipeline Agents

The core pipeline uses three agents in sequence:

- **InstruQtor** — Reads config and prompt files, prepares VisualBriq instruction packets per cycle.
- **ConstruQtor** — Validates inputs, calls the backend for image and video generation.
- **InspeQtor** — Runs quality checks on generated content.

### Orchestration

`VisualFaQtory` (visual_faqtory.py) is the thin orchestrator that wires everything together:

1. Load config, detect inputs (base image/video/audio)
2. Run `sliding_story_engine` (paragraph window → cycle generation)
3. Run `Finalizer` (stitch → interpolate → upscale)
4. Mux audio (if present)
5. Save run to `worqspace/saved-runs/<project-name>`

### Directory Layout

**During a run:**
```
run/
├── videos/          # Per-cycle MP4s: video_000.mp4, video_001.mp4, ...
├── frames/          # Keyframes and last-frames per cycle
├── briqs/           # Per-cycle JSON: cycle_000.json, cycle_001.json, ...
├── meta/            # Config snapshot, story snapshot, base inputs
├── final_video.mp4
├── final_video_60fps.mp4
├── final_video_60fps_1080p.mp4
├── final_video_60fps_1080p_audio.mp4  (if audio present)
└── faqtory_state.json
```

**After saving:**
```
worqspace/saved-runs/<project-name>/
├── <project-name>.mp4    # Final deliverable
├── videos/
├── frames/
├── briqs/
├── meta/
└── faqtory_state.json
```

---

## 3. Configuration Reference

All configuration lives in `worqspace/config.yaml`. The minimal canonical config:

```yaml
# Paths
paths:
  output_dir: ./run
  briqs_dir: ./run/briqs

# Input mode
input:
  mode: text              # auto | text | image | video

# Backend
backend:
  type: comfyui           # comfyui | mock
  api_url: http://localhost:8188
  timeout: 600
  width: 1024
  height: 576

# ComfyUI-specific checkpoint names
comfyui:
  sdxl_ckpt: juggernautXL_ragnarokBy.safetensors
  svd_ckpt: svd_xt_1_1.safetensors

# LoRA (optional)
lora:
  enabled: false
  path: /path/to/your/lora.safetensors
  strength: 0.6
  backend: comfyui

# Story engine
paragraph_story:
  max_paragraphs: 2
  img2vid_duration_sec: 3
  img2img_denoise_min: 0.38
  img2img_denoise_max: 0.58
  video_fps: 8
  seed_base: 2222
  rolling_window: true
  require_morph: false

# Audio
audio:
  enabled: true
  sync_video_audio: false
  cycle_seconds: 3
  allow_ext: [wav, mp3, flac]

# Finalizer
finalizer:
  enabled: true
  interpolate_fps: 60
  upscale_resolution: 1920x1080
  scale_algo: bicubic
  encoder_preference: [h264_nvenc, libx264]
  quality:
    crf: 16
```

---

## 4. Story Engine

The sliding story engine (`sliding_story_engine.py`) reads `worqspace/story.txt` and splits it into paragraphs separated by blank lines.

### Windowing Schedule

For P paragraphs and max window size M:

1. **Ramp-up** (cycles 1..M): Window grows from [1] to [1..M]
2. **Sliding** (cycles M+1..P): Fixed window slides forward by one
3. **Ramp-down** (after last paragraph): Window shrinks to single paragraph

Example with P=6, M=3:
```
Cycle 0: [1]
Cycle 1: [1,2]
Cycle 2: [1,2,3]    ← full window
Cycle 3: [2,3,4]    ← sliding
Cycle 4: [3,4,5]
Cycle 5: [4,5,6]
Cycle 6: [5,6]      ← ramp-down
Cycle 7: [6]
```

### Per-Cycle Generation

Each cycle:
1. Build prompt from paragraph window text
2. Append motion_prompt, style_hints, evolution_lines
3. Generate keyframe (txt2img or img2img with reinject)
4. Generate video from keyframe (img2vid)
5. Extract last frame for next cycle
6. Write briq JSON for reproducibility

---

## 5. Input Modes

### Text Mode (default)
- Cycle 0 uses txt2img to generate the first keyframe from story prompt
- Subsequent cycles use img2img (reinject) from the last frame

### Image Mode
- Auto-detects the newest image in `worqspace/base_images/` (png/jpg/jpeg/webp)
- Cycle 0 uses that image as the starting point for img2img
- Subsequent cycles chain normally via reinject

### Video Mode
- Auto-detects the newest video in `worqspace/base_video/` (mp4/mov/mkv/webm)
- Extracts first frame using ffmpeg, resized/padded to configured dimensions
- Uses extracted frame as if it were a base image (image mode from cycle 0)
- Subsequent cycles chain normally

### Auto Mode
Set `input.mode: auto` in config to auto-detect: video > image > text.

---

## 6. Reinject Mode

Reinject is the default behavior (ON). It means:

**With reinject (default):**
- Every cycle runs img2img on the last frame from the previous cycle
- Denoise is sampled uniformly from `[img2img_denoise_min, img2img_denoise_max]`
- This produces a new keyframe that evolves the visual narrative
- Video is then generated from the new keyframe

**Without reinject (`--no-reinject` / `-R`):**
- The last frame is used directly as conditioning for img2vid
- No img2img keyframe step is run
- Results in more stable but less evolving visuals

---

## 7. Backend Configuration

### ComfyUI (Production)

```yaml
backend:
  type: comfyui
  api_url: http://localhost:8188
  timeout: 600
  width: 1024
  height: 576
```

The ComfyUI backend supports:
- `generate_image()` — SDXL txt2img and img2img workflows
- `generate_video()` — SVD img2vid workflows
- `generate_morph_video()` — Morph between two images (requires workflow)

Custom workflows can be specified:
```yaml
backend:
  workflow_image: path/to/custom_image.json
  workflow_video: path/to/custom_video.json
  workflow_morph: path/to/morph_i2v.json
```

### Mock (Testing)

```yaml
backend:
  type: mock
```

Generates placeholder images and videos using PIL/ffmpeg. No GPU required.

---

## 8. LoRA Support

Optional LoRA injection for ComfyUI workflows:

```yaml
lora:
  enabled: true
  path: /absolute/path/to/your/lora.safetensors
  strength: 0.6
  backend: comfyui
```

The LoRA loader is automatically injected into image generation workflows. Video generation workflows are not affected by LoRA.

Requirements:
- The `.safetensors` file must exist at the specified path
- The path must be within a `models/loras` directory structure
- Backend must be `comfyui`

CLI overrides: `--lora-enabled`, `--no-lora`, `--lora-path`, `--lora-strength`

---

## 9. Audio Support

Audio is optional but fully supported as a final mux step.

### Setup

Place audio files in `worqspace/base_audio/` (supported: wav, mp3, flac). The pipeline auto-detects the newest file.

### Audio Sync

When `audio.sync_video_audio: true` and an audio file is present:
- Audio duration is determined via ffprobe
- Cycle count is computed: `ceil(duration / cycle_seconds)`
- Maximum overrun: 2.0 seconds (from ceil rounding)

### Audio Muxing

After the finalizer completes (stitch → interpolate → upscale), the audio is muxed:
- Video is trimmed to audio duration (no trailing silence)
- Output: `final_video_60fps_1080p_audio.mp4`

---

## 10. Finalizer Pipeline

Runs automatically after all cycles complete:

1. **Stitch** — Concatenate all cycle videos → `final_video.mp4`
2. **Interpolate** — Interpolate to 60fps via minterpolate → `final_video_60fps.mp4`
3. **Upscale** — Scale to 1920×1080 → `final_video_60fps_1080p.mp4`
4. **Audio Mux** — If audio present → `final_video_60fps_1080p_audio.mp4`

Encoder preference: `h264_nvenc` (GPU) → `libx264` (CPU fallback).

---

## 11. Project Saving

After completion:
- If `--name` was passed, uses that name
- Otherwise, prompts: "Project name to save as: "
- Name is sanitized for filesystem safety
- If folder exists, auto-suffixed: `name-001`, `name-002`, etc.
- `run/` is moved to `worqspace/saved-runs/<name>/`
- Best deliverable is renamed to `<name>.mp4`

---

## 12. Prompt Files

All prompt files live in `worqspace/`:

| File | Purpose | Required |
|------|---------|----------|
| `story.txt` | Main narrative paragraphs | Yes |
| `motion_prompt.md` | Camera/motion hints | No |
| `style_hints.md` | Style modifiers | No |
| `evolution_lines.md` | Per-cycle evolution guidance | No |
| `negative_prompt.md` | Negative prompt | No |
| `transient_tasq.md` | Per-run overrides | No |

Prompt synthesis is deterministic (no LLM). The prompt for each cycle combines:
- Paragraph window text (base prompt)
- Style hints
- Motion prompt
- Evolution line for current cycle
- Transient task (if present)

---

## 13. Briq State Tracking

Every cycle writes JSON into `run/briqs/`:
```json
{
  "cycle_index": 0,
  "paragraph_window": [1, 2],
  "paragraph_text": "...",
  "prompt_final": "...",
  "negative_prompt": "...",
  "seed": 2222,
  "denoise": 0.42,
  "input_mode": "text",
  "paths": {
    "keyframe": "run/frames/keyframe_000.png",
    "video": "run/videos/video_000.mp4",
    "last_frame": "run/frames/lastframe_000.png"
  },
  "backend": "comfyui"
}
```

The run-level state is tracked in `run/faqtory_state.json`:
```json
{
  "run_id": "run_20250212_143000_abc123",
  "version": "v0.5.6-beta",
  "start_time": "...",
  "end_time": "...",
  "mode": "text",
  "reinject": true,
  "cycles_planned": 8,
  "cycles_completed": 8,
  "finalizer_outputs": {...},
  "saved_to": "worqspace/saved-runs/my-project"
}
```

---

## 14. CLI Reference

```bash
# Default run (reinject ON)
python vfaq_cli.py

# Named project
python vfaq_cli.py -n cyberpunk-set

# Disable reinject
python vfaq_cli.py --no-reinject
python vfaq_cli.py -R

# Override mode
python vfaq_cli.py --mode image
python vfaq_cli.py --mode video

# Mock backend test
python vfaq_cli.py -n test -b mock

# Dry run (validate only)
python vfaq_cli.py --dry-run

# With LoRA
python vfaq_cli.py --lora-enabled --lora-path /path/to/lora.safetensors --lora-strength 0.7

# Check status
python vfaq_cli.py status

# List backends
python vfaq_cli.py backends
```

---

## 15. Troubleshooting

**"Backend not fully available"**
- Check ComfyUI is running at the configured `api_url`
- Verify with: `curl http://localhost:8188/system_stats`

**"Configured SDXL/SVD checkpoint not found"**
- Place checkpoint files in ComfyUI's models directory
- Restart ComfyUI or click "Reload models"
- Update `comfyui.sdxl_ckpt` / `comfyui.svd_ckpt` in config

**"LoRA file not found"**
- Ensure `lora.path` points to an existing `.safetensors` file
- Path must be within a `models/loras` directory structure

**No cycle videos generated**
- Check story.txt has at least 2 paragraphs separated by blank lines
- Run with `--dry-run` first to validate config

**Audio mux fails**
- Ensure ffmpeg and ffprobe are installed
- Check audio file format (wav/mp3/flac)
- Verify audio file is not corrupted

**Finalizer encoder fails**
- h264_nvenc requires NVIDIA GPU with NVENC support
- Falls back to libx264 automatically
- Ensure ffmpeg is compiled with the appropriate encoders

---

*Visual FaQtory v0.5.6-beta — Built by Ill Dynamics / WoNQ*
