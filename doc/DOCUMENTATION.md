# Visual FaQtory v0.9.3-beta — Documentation

## Venice native backend (Primary)

Venice support is implemented as a native HTTP backend rather than a ComfyUI workflow wrapper. Visual FaQtory maps its modes like this:

- text2img → `POST /image/generate`
- img2img → `POST /image/edit`
- text2vid → `POST /video/queue` with the configured text-to-video model
- img2vid → `POST /video/queue` with `image_url`
- morph / loop-closure → `POST /video/queue` with `image_url` + `end_image_url`

Use `worqspace/config-venice.yaml` as the base template. Required secret: `VENICE_API_KEY`.

```yaml
backend:
  type: hybrid
image_backend:
  type: venice
video_backend:
  type: venice

venice:
  api_key: ${VENICE_API_KEY}
  models:
    text2img: z-image-turbo
    img2img: qwen-edit
    text2vid: wan-2.5-preview-text-to-video
    img2vid: wan-2.5-preview-image-to-video
  image:
    width: 1024
    height: 576
    cfg_scale: 7.5
    steps: 8
  video:
    duration_seconds: 5
    aspect_ratio: "16:9"
    resolution: "720p"
  text_to_video_first_cycle: true
  enable_end_frame_morph: true
```

The backend validates configured model IDs against `GET /models` when credentials are present, and normalizes still images to PNG on disk so the rest of the pipeline keeps its existing naming assumptions.
For Venice video, `resolution`, `aspect_ratio`, and `audio` are model-dependent optional fields: unsupported fields are auto-omitted safely, and supported `resolution` values still use snap-to-nearest when the configured enum is invalid.

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
15. [Crowd Control](#15-crowd-control)
16. [Troubleshooting](#16-troubleshooting)
17. [Veo Backend](#17-veo-backend)

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
  type: venice            # comfyui | animatediff | venice | veo | mock
  api_url: http://localhost:8188
  timeout: 600
  width: 1024
  height: 576

# Venice-specific configuration (primary)
venice:
  api_key: ${VENICE_API_KEY}
  models:
    text2img: z-image-turbo
    img2img: qwen-edit
    text2vid: wan-2.5-preview-text-to-video
    img2vid: wan-2.5-preview-image-to-video

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

# Finalizer
finalizer:
  enabled: true
  interpolate_fps: 60
  upscale_resolution: 1920x1080
  scale_algo: bicubic
  encoder_preference: [h264_nvenc, libx264]
```

---

## 4. Story Engine

The sliding story engine (`sliding_story_engine.py`) reads `worqspace/story.txt` and splits it into paragraphs separated by blank lines.

### Windowing Schedule

For P paragraphs and max window size M:

1. **Ramp-up** (cycles 1..M): Window grows from [1] to [1..M]
2. **Sliding** (cycles M+1..P): Fixed window slides forward by one
3. **Ramp-down** (after last paragraph): Window shrinks to single paragraph

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
- Uses extracted frame as if it were a base image

---

## 6. Reinject Mode

Reinject is the default behavior (ON). It means:

**With reinject (default):**
- Every cycle runs img2img on the last frame from the previous cycle
- This produces a new keyframe that evolves the visual narrative
- Video is then generated from the new keyframe

---

## 7. Backend Configuration

### Venice (Primary Native Backend)

```yaml
backend:
  type: venice
```
See the [Venice section at the top](#venice-native-backend-primary) for full details.

### ComfyUI (Production Workflow Wrapper)

```yaml
backend:
  type: comfyui
  api_url: http://localhost:8188
  timeout: 600
```

The ComfyUI backend supports:
- `generate_image()` — SDXL txt2img and img2img workflows
- `generate_video()` — ComfyUI img2vid workflows (SVD by default)
- `generate_morph_video()` — Morph between two images (requires workflow)

### Qwen-Image (Split image/video mode)

Visual FaQtory supports two image-only Qwen backends for hybrid setups:

- `qwen_image_comfyui` for ComfyUI-driven Qwen text2img/img2img
- `qwen_image_python` / `qwen_python` for native local Python Qwen text2img/img2img via diffusers

Supported hybrid config:
```yaml
backend:
  type: hybrid
image_backend:
  type: qwen_image_comfyui
  workflow_image: ./worqspace/workflows/qwen_image_t2i.json
video_backend:
  type: venice
```

### AnimateDiff Backend

```yaml
video_backend:
  type: animatediff
  api_url: http://localhost:8188
```
Used primarily for local SVD-style video generation via ComfyUI with AnimateDiff nodes.

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

---

## 9. Audio Support

Audio is optional but fully supported as a final mux step. Place files in `worqspace/base_audio/`.

---

## 10. Finalizer Pipeline

Runs automatically after all cycles complete:

1. **Stitch** — Concatenate all cycle videos
2. **Interpolate** — Interpolate to 60fps
3. **Upscale** — Scale to 1080p
4. **Audio Mux** — If audio present

---

## 11. Project Saving

Moves `run/` to `worqspace/saved-runs/<name>/` and renames the best deliverable to `<name>.mp4`.

---

## 12. Prompt Files

All prompt files live in `worqspace/`: `story.txt`, `motion_prompt.md`, `style_hints.md`, `evolution_lines.md`, `negative_prompt.md`.

---

## 13. Briq State Tracking

Every cycle writes JSON into `run/briqs/` for full reproducibility and telemetry.

---

## 14. CLI Reference

```bash
python vfaq_cli.py        # Run pipeline
python vfaq_cli.py crowd  # Start Crowd Control server
python vfaq_cli.py status # Check status
```

---

## 15. Crowd Control

Crowd Control enables live audience prompt injection during streams via a QR code submission page.

### Lifecycle Management
- **Claim**: The generator claims the next prompt from the queue at the start of a cycle.
- **Bake**: The prompt is baked into a crowd keyframe (reinject_keyframe mode).
- **Ack**: Prompt is acked only after successful generation.
- **Requeue**: Failed generations return the prompt to the queue.

### Configuration Reference

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `false` | Master switch |
| `base_url` | `http://127.0.0.1:8808/visuals` | Server endpoint |
| `ack_after_success` | `true` | Only ack on success |
| `requeue_on_failure` | `true` | Return to queue on fail |
| `claim_timeout_seconds` | `900` | Stale claim recovery |

---

## 16. Troubleshooting

- **Backend unreachable**: Check `api_url` or API keys.
- **No videos**: Check `story.txt` paragraph spacing.
- **Finalizer fails**: Ensure `ffmpeg` and `ffprobe` are in PATH.

---

## 17. Veo Backend

Native Google Veo support via Gemini Developer API or Vertex AI.

```yaml
backend:
  type: veo
veo:
  provider: gemini
  model: veo-3.1-generate-preview
```

---

*Visual FaQtory v0.9.3-beta — Built by Ill Dynamics / WoNQ*
