# Visual FaQtory v0.9.0-beta — Documentation

## Qwen-Image via ComfyUI (split image/video mode)

Visual FaQtory now supports two image-only Qwen backends while keeping a separate `video_backend` and `morph_backend` for SVD or other video generators:

- `qwen_image_comfyui` for ComfyUI-driven Qwen text2img/img2img
- `qwen_image_python` / `qwen_python` for native local Python Qwen text2img/img2img via diffusers

Neither path should be presented as a full standalone image+video pipeline backend.

Supported config shapes:

```yaml
backend:
  type: hybrid
  width: 1024
  height: 576

image_backend:
  type: qwen_image_comfyui
  api_url: http://localhost:8188
  workflow_image: ./worqspace/workflows/qwen_image_t2i.json
  workflow_img2img: ./worqspace/workflows/qwen_image_i2i.json

video_backend:
  type: comfyui
  api_url: http://localhost:8188
  workflow_video: ./worqspace/workflows/svd_img2vid.json

morph_backend:
  type: comfyui
  api_url: http://localhost:8188
  workflow_morph: ./worqspace/workflows/morph_i2v.json
```

Workflow requirements:
- `qwen_image_comfyui` is image-only. Use a separate `video_backend` for img2vid and a separate `morph_backend` for loop closure / morph.
- `workflow_image` must be a valid ComfyUI Qwen text-to-image workflow JSON.
- `workflow_img2img` should contain at least one `LoadImage` node so reinject / resume images can be uploaded and substituted safely.
- `workflow_morph` must be configured explicitly for two-image morph / loop-closure generation; no default morph path is assumed.
- Runtime prompt injection is graph-aware: the backend patches whichever conditioning nodes are actually wired into `KSampler`, including nodes that expose `prompt` instead of `text`.

## Venice native backend

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
18. [LTX-Video Backend (NEW)](#18-ltx-video-backend)

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
  type: comfyui           # comfyui | animatediff | venice | veo | ltx_video | mock
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
- `generate_video()` — ComfyUI img2vid workflows (SVD by default)
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

### Qwen-Image via ComfyUI (image stage only)

```yaml
backend:
  type: hybrid

image_backend:
  type: qwen_image_comfyui
  api_url: http://localhost:8188
  workflow_image: ./worqspace/workflows/qwen_image_t2i.json
  workflow_img2img: ./worqspace/workflows/qwen_image_i2i.json

video_backend:
  type: comfyui   # or veo / ltx_video

morph_backend:
  type: comfyui   # required for loop-closure / morph
```

`qwen_image_comfyui` only handles the image stage. It should normally be used in split-capability configs like the example above. Using it as the only backend leaves img2vid and loop-closure/morph generation unconfigured by design.

### Qwen-Image via native Python (image stage only)

```yaml
backend:
  type: hybrid

image_backend:
  type: qwen_image_python

video_backend:
  type: animatediff
  api_url: http://localhost:8188

morph_backend:
  type: animatediff
  api_url: http://localhost:8188
  workflow_morph: ./worqspace/workflows/animatediff_morph_i2v.json

qwen_python:
  model_id: Qwen/Qwen-Image
  img2img_model_id: Qwen/Qwen-Image
  edit_model_id: null
  local_model_path: null
  local_img2img_model_path: null
  local_edit_model_path: null
  cache_dir: null
  device: cuda
  torch_dtype: bfloat16
  enable_model_cpu_offload: true
  enable_sequential_cpu_offload: false
  enable_attention_slicing: true
  enable_vae_slicing: true
  use_edit_pipeline_for_img2img: false
  local_files_only: false
  max_sequence_length: null
  num_images_per_prompt: 1
```

`qwen_image_python` is image-only and uses explicit Diffusers Qwen pipeline classes now:
- `QwenImagePipeline` for text2img
- `QwenImageImg2ImgPipeline` for reinject/img2img
- `QwenImageEditPipeline` only when `use_edit_pipeline_for_img2img: true`

The backend writes to the normal artifact path (`{output_dir}/{atom_id}_image.png`), keeps a process-level pipeline cache for repeated cycles, supports both Hub IDs and fully local model folders, and fails clearly if reinject/img2img is requested without a working img2img/edit pipeline.

Optional install:

```bash
pip install torch pillow transformers>=4.51.3 accelerate huggingface_hub
pip install git+https://github.com/huggingface/diffusers
```

Backward-compatible config support remains for the older `qwen_image_python:` section name, but `qwen_python:` is the preferred native backend section going forward.

### AnimateDiff Backend

```yaml
backend:
  type: hybrid

image_backend:
  type: comfyui
  api_url: http://localhost:8188

video_backend:
  type: animatediff
  api_url: http://localhost:8188
  # workflow_video: ./worqspace/workflows/animatediff_i2v.json

morph_backend:
  type: animatediff
  api_url: http://localhost:8188
  workflow_morph: ./worqspace/workflows/animatediff_morph_i2v.json

animatediff:
  checkpoint: juggernautXL_ragnarokBy.safetensors
  motion_model: animatediff_lightning_4step_comfyui.safetensors
  frame_rate: 8
  frame_count: 24
  steps: 20
  cfg_scale: 4.0
  sampler_name: euler
  scheduler: normal
  denoise_strength: 0.75
  context_length: 16
  context_stride: 1
  context_overlap: 4
  closed_loop: false
  pingpong: false
  negative_prompt: low quality, blurry, distorted
```

Notes:
- `workflow_video` is optional for AnimateDiff. When omitted, Visual FaQtory builds a default ComfyUI API workflow.
- Custom AnimateDiff `workflow_video` files must expose at least one `LoadImage` node and a real video/animation output node.
- `workflow_morph` is still explicit-only. A starter two-image AnimateDiff morph graph is bundled, but Visual FaQtory does not auto-generate one dynamically.
- Supported aliases in the `animatediff:` section include `fps`/`frame_rate`, `cfg`/`cfg_scale`, `sampler`/`sampler_name`, and `denoise`/`denoise_strength`.

### Google Veo (Cloud)

```yaml
backend:
  type: veo
  width: 1024
  height: 576

veo:
  provider: gemini                    # gemini | vertex
  model: veo-3.1-generate-preview
  duration_seconds: 8
  default_mode: image_to_video
  enable_loop_closure: false
```

The Veo backend generates video directly via Google's Gen AI SDK. No local GPU required. Supports text_to_video, image_to_video, first_last_frame, and extend_video modes.

Requires: `pip install google-genai` and `GEMINI_API_TOKEN` environment variable (or Vertex ADC credentials). See `worqspace/config-veo.yaml` for full configuration reference.

### LTX-Video (Self-Hosted)

```yaml
backend:
  type: ltx_video
  width: 768
  height: 512

ltx_video:
  enabled: true
  runner: python_api                  # python_api | cli | comfyui
  mode: auto                          # auto | t2v | i2v
  repo_path: ~/LTX-Video
  python_bin: python3
  pipeline_config: configs/ltxv-2b-0.9.8-distilled.yaml
  model_variant: 2b_distilled         # 2b_distilled | 2b_full | 13b_distilled | 13b_full
  frame_rate: 30
  offload_to_cpu: false
  image_cond_noise_scale: 0.15
  conditioning_strength_start: 1.0
  conditioning_strength_end: 0.85
  enable_end_keyframe: true
  end_keyframe_strategy: self_loop_evolved
  use_prompt_enhancement: true
  max_frames_per_clip: 241
  enable_loop_closure: false
  loop_closure_strength: 0.90
  loop_seam_threshold: 0.10
```

The LTX-Video backend generates video on your local GPU. No cloud API keys or quotas.

**Capabilities:**
- `generate_video()` — t2v (text-to-video) and i2v (image-to-video) via conditioning
- `generate_image()` — Generates a minimal 9-frame clip and extracts the first frame. This is NOT native still-image generation — LTX is a video model. For production keyframe quality, consider ComfyUI for images.
- `generate_morph_video()` — Two-keyframe conditioned generation. NOT true latent morphing — it is a conditioned transition with start/end frame guidance.

**Mode selection (auto):**
- If a source image exists (base image or previous last frame) → image-to-video (i2v)
- Otherwise → text-to-video (t2v)
- Config `mode: t2v` or `mode: i2v` can force a specific mode

**Frame count constraint:**
LTX-Video's VAE requires frame counts of the form 8k+1 (9, 17, 25, 33, ..., 241). VFaQ automatically snaps the requested frame count up to the nearest valid value.

**Model variants:**
| Variant | VRAM | Speed | Quality |
|---------|------|-------|---------|
| `2b_distilled` | ~8 GB | Fast | Good (recommended) |
| `2b_full` | ~10 GB | Medium | Better |
| `13b_distilled` | ~24 GB | Medium | Very good |
| `13b_full` | ~28 GB | Slow | Best |

**Reinject mapping:**
ComfyUI uses `denoise_strength` for img2img. LTX uses `conditioning_strengths` and `image_cond_noise_scale`. The mapping is approximate: higher conditioning strength = more influence from the source image (lower effective denoise), higher noise scale = more creative freedom (higher effective denoise). There is no exact 1:1 mapping — this is documented by design.

**Setup:**
```bash
git clone https://github.com/Lightricks/LTX-Video.git ~/LTX-Video
cd ~/LTX-Video && pip install -e .
# Download model weights and pipeline config YAML
```

See `worqspace/config-ltx.yaml` for a complete configuration template with inline documentation.

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

# AnimateDiff single-backend override
python vfaq_cli.py -b animatediff -n ad-test

# Venice single-backend override
python vfaq_cli.py -b venice -n venice-test

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

`-b/--backend` only overrides `backend.type` for single-backend runs. Split capability routing is first-class; use `backend.type: hybrid` plus `image_backend`, `video_backend`, and `morph_backend` in YAML for Qwen/ComfyUI/AnimateDiff/Venice mixed setups.

## 15. Crowd Control

Crowd Control enables live audience prompt injection during streams. Viewers scan a QR code, land on a minimal web page, and submit prompts that get injected into the Visual FaQtory's next generation cycle.

### Architecture

The system has two sides:

**Server** (FastAPI, runs on the visuals machine):
- Serves the HTML prompt page at `{prefix}/`
- Serves a QR code image at `{prefix}/qr.png`
- Accepts prompt submissions at `{prefix}/api/submit`
- Provides a token-protected pop endpoint at `{prefix}/api/next`
- Health check at `{prefix}/api/health`
- SQLite database for queue, rate limiting, and audit trail

**Client** (built into the generator):
- At the start of each cycle, the sliding story engine calls `/api/next`
- If a crowd prompt exists, it's injected into the stacked prompt
- If the server is down or unreachable, the engine continues in story mode (fail-open)
- All errors are caught and logged — generation never hard-fails

### Quick Start

**1. Start the Crowd Control server on the visuals machine:**

```bash
python vfaq_cli.py crowd --token MY_SECRET_TOKEN --public-url http://192.168.1.50:8808/visuals
```

**2. In OBS, add a Browser Source or Image Source** pointing at:
```
http://<visuals-host>:8808/visuals/qr.png
```

**3. Enable crowd control in `worqspace/config.yaml`:**

```yaml
crowd_control:
  enabled: true
  base_url: "http://127.0.0.1:8808/visuals"
  pop_token: "MY_SECRET_TOKEN"
```

**4. Run Visual FaQtory** — it will check the queue each cycle:

```bash
python vfaq_cli.py
```

### Configuration Reference

All settings in `worqspace/config.yaml` under `crowd_control:`:

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `false` | Master switch for crowd prompt injection |
| `base_url` | `http://127.0.0.1:8808/visuals` | URL the generator uses to reach the server |
| `pop_token` | `CHANGE_ME_LONG_RANDOM` | Bearer token for the `/api/next` endpoint |
| `timeout_seconds` | `1.0` | HTTP timeout for generator→server calls |
| `inject_label` | `Audience mutation request` | Label injected into the prompt |
| `inject_mode` | `append` | `append` adds to story, `replace` overrides story |
| `max_chars` | `300` | Maximum prompt length from viewers |
| `rate_limit_seconds` | `600` | Seconds between submissions per IP |
| `max_queue` | `100` | Maximum pending prompts in queue |
| `badwords_path` | `worqspace/badwords.txt` | Path to bad word filter file |
| `public_url` | `https://wonq.tv/visuals` | URL the QR code points to |
| `prefix` | `/visuals` | URL path prefix for all routes |
| `db_path` | `worqspace/crowdcontrol.sqlite3` | SQLite database file path |

### Security & Abuse Controls

- **Rate limiting**: 1 submission per IP per `rate_limit_seconds` (default 10 minutes)
- **Bad word filtering**: Regex word-boundary matching from `worqspace/badwords.txt`
- **Queue cap**: Rejects submissions when queue reaches `max_queue`
- **Input sanitization**: Strips whitespace, collapses spaces, removes newlines, enforces max length
- **Token protection**: `/api/next` requires Bearer token or query `?token=...`
- **Proxy IP detection**: Reads `CF-Connecting-IP` → `X-Forwarded-For` → `request.client.host`
- **Audit trail**: All submissions (accepted and rejected) are recorded in SQLite

### Bad Words File

Edit `worqspace/badwords.txt` to add or remove words/phrases. Format:
- One word or phrase per line
- Lines starting with `#` are comments
- Blank lines are ignored
- Matching is case-insensitive with word boundaries (e.g., "ass" won't match "class")

### Fail-Open Behavior

The generator is designed to never hard-fail due to crowd control issues:
- If the server is unreachable → continues story mode
- If the pop request times out → continues story mode
- If the token is wrong → logs warning, continues story mode
- If the crowd control module fails to import → logs warning, continues story mode
- If the queue is empty → continues story mode normally

### Reverse Proxy Setup (Future)

The Crowd Control server is designed to sit behind a reverse proxy. The planned setup:

```
https://wonq.tv/visuals  →  http://<visuals-zt-ip>:8808/visuals
```

This will be handled by WoNQ.TV infra (Traefik/NGINX/Cloudflare) over ZeroTier or WireGuard. The server already supports:
- Configurable `prefix` for path-based routing
- Configurable `public_url` for QR code generation
- Proxy IP extraction from `CF-Connecting-IP` and `X-Forwarded-For`

No changes to the Crowd Control code are needed — just set `public_url` to the final public URL and configure the reverse proxy.

### Prompt Injection Flow

When a crowd prompt is served, it appears in the stacked prompt like this (append mode):

```
[paragraph 1 text]

[paragraph 2 text]

[AUDIENCE MUTATION REQUEST]
cyberpunk jellyfish swimming through neon rain
```

In replace mode, the crowd prompt completely overrides the story prompt for that cycle.

The briq JSON for each cycle records crowd control state:

```json
{
  "crowd_control": {
    "used": true,
    "prompt_preview": "cyberpunk jellyfish swimming through neon rain",
    "inject_mode": "append"
  }
}
```

### CLI Reference

```bash
python vfaq_cli.py crowd [OPTIONS]

Options:
  --host TEXT          Bind host (default: 0.0.0.0)
  --port INT           Bind port (default: 8808)
  --prefix TEXT        URL prefix (default: /visuals)
  --public-url TEXT    Public URL for QR code
  --db-path TEXT       SQLite database path
  --token TEXT         Bearer token for /api/next
  --badwords TEXT      Bad words filter file
  --max-chars INT      Max prompt length (default: 300)
  --rate-limit INT     Rate limit seconds per IP (default: 600)
  --max-queue INT      Max queue length (default: 100)
```

Environment variables:
- `VF_CROWD_TOKEN`: Alternative to `--token`
- `VF_CROWD_ALLOW_NO_TOKEN=true`: Skip token requirement (dev only)

---

## 15A. ComfyUI workflow reality

Visual FaQtory does **not** bundle heavyweight ComfyUI workflow JSONs. The config examples and filenames under `worqspace/workflows/` are operator-facing contracts only. You are expected to export your own **API format JSON** workflows from the ComfyUI instance that actually has the right custom nodes, checkpoints, and motion models installed.

### Known-good workflow contracts

- **Qwen text2img (ComfyUI path)** — API-format ComfyUI JSON for Qwen still-image generation. `workflow_image` is required for `qwen_image_comfyui`.
- **Qwen text2img/img2img (Python path)** — native diffusers loading via `qwen_image_python`, explicitly using `QwenImagePipeline` for text2img and `QwenImageImg2ImgPipeline` for reinject/img2img, with optional `QwenImageEditPipeline` only when explicitly configured.
- **Qwen img2img** — API-format JSON with at least one `LoadImage` node so reinject and resume frames can be uploaded.
- **SVD img2vid** — API-format JSON with a `LoadImage` start frame and a real video output path.
- **AnimateDiff img2vid** — API-format JSON with `LoadImage`, an AnimateDiff loader node, and a real video output node such as `VHS_VideoCombine`; or omit `workflow_video` and let Visual FaQtory build the default AnimateDiff graph.
- **AnimateDiff morph** — explicit two-image workflow with two image inputs and an AnimateDiff-compatible video output path. The repo now bundles a starter graph at `worqspace/workflows/animatediff_morph_i2v.json`, but Visual FaQtory still does not auto-build one dynamically at runtime.
- **ComfyUI morph** — explicit two-image morph / loop-closure workflow with two `LoadImage` nodes and a real video output path.

### Bad or missing node graph troubleshooting

- Export **API format JSON**, not the regular UI workflow export.
- Keep graph node classes aligned with the custom nodes installed on the ComfyUI host.
- Keep checkpoint names, motion model names, and optional motion LoRA names aligned with what ComfyUI actually advertises.
- Missing `LoadImage`, AnimateDiff loader, or video-output nodes are now treated as configuration errors, not silently guessed around.
- `workflow_morph` is always explicit. Visual FaQtory will not invent a generic two-image morph graph for you.
- See `worqspace/workflows/README.md` for the shorter operator checklist that lives next to the example filenames.

## 16. Troubleshooting

**"Backend not fully available"**
- Check ComfyUI is running at the configured `api_url`
- Verify with: `curl http://localhost:8188/system_stats`

**"Configured SDXL/SVD checkpoint not found"**
- Place checkpoint files in ComfyUI's models directory
- Restart ComfyUI or click "Reload models"
- Update `comfyui.sdxl_ckpt` / `comfyui.svd_ckpt` in config

**"Workflow JSON fails or patches the wrong nodes"**
- Re-export the workflow in **API format JSON**
- Confirm the graph contains the expected `LoadImage`, conditioning, sampler, AnimateDiff loader, and/or video output nodes for that mode
- Check `worqspace/workflows/README.md` against the workflow you exported

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

*Visual FaQtory v0.9.0-beta — Built by Ill Dynamics / WoNQ*

---

## 17. Veo Backend

### 17.1 Overview

The Veo backend generates video directly from text or images via Google's Veo API. No local GPU is required — video generation runs on Google's infrastructure.

**Key differences from ComfyUI:**
- No separate txt2img → img2img → img2vid pipeline. Veo generates video directly.
- Duration is controlled natively (5-8 seconds per clip).
- The `first_last_frame` mode replaces ComfyUI's morph workflow.
- Output is downloaded immediately after generation completes.

### 17.2 Authentication

Veo supports two providers:

**Gemini Developer API** (`provider: gemini`):
Set one of these environment variables (checked in order):
1. `GOOGLE_API_KEY`
2. `GEMINI_API_KEY`
3. `GOOGLE_API_TOKEN` (legacy alias)
4. `GEMINI_API_TOKEN` (legacy alias)

**Vertex AI** (`provider: vertex`):
Uses Application Default Credentials (ADC). Set:
- `GOOGLE_CLOUD_PROJECT` — your GCP project ID
- `GOOGLE_CLOUD_LOCATION` — region (default: `us-central1`)

### 17.3 Configuration

Copy `worqspace/config-veo.yaml` to `worqspace/config.yaml` for a complete Veo configuration template.

Key settings in the `veo:` section:

| Key | Default | Description |
|-----|---------|-------------|
| `provider` | `gemini` | `gemini` or `vertex` |
| `model` | `veo-3.1-generate-preview` | Primary Veo model ID |
| `fast_model` | `veo-3.1-fast-generate-preview` | Fast model (lower quality) |
| `prefer_fast_model` | `false` | Use fast model by default |
| `duration_seconds` | `8` | Clip duration (5-8, clamped) |
| `aspect_ratio` | `16:9` | Output aspect ratio |
| `resolution` | `720p` | Output resolution |
| `generate_audio` | `false` | Generate audio track |
| `enable_last_frame` | `true` | Enable first_last_frame mode |
| `enable_extension` | `false` | Enable video extension |
| `enable_loop_closure` | `false` | Generate loop-closure clip |
| `poll_interval` | `10.0` | Seconds between operation polls |
| `poll_timeout` | `600.0` | Max wait for generation |
| `max_retries` | `3` | Retry attempts for transient errors |

### 17.4 Generation Modes

**text_to_video** — Generate video purely from a text prompt. Used for cycle 1 in text input mode.

**image_to_video** — Generate video from a starting image + optional prompt. Used for subsequent cycles (previous last frame → next clip).

**first_last_frame** — Interpolation between a start frame and an end frame. Replaces the ComfyUI morph workflow. Enabled when `require_morph: true` and `enable_last_frame: true`.

**extend_video** — Extend a previously Veo-generated clip. Requires `enable_extension: true` and a valid prior Veo clip.

### 17.5 Story Engine Routing

When `backend.type: veo`, the sliding story engine routes cycles differently:

```
Cycle 1 (no base image): text_to_video
Cycle 1 (base image):    image_to_video (base image as anchor)
Cycle n>1:               image_to_video (previous last frame)
Cycle n>1 (morph=true):  first_last_frame (last frame → new keyframe)
Loop closure cycle:      first_last_frame (final last frame → cycle-1 anchor)
```

All existing ComfyUI pipeline logic is untouched — the Veo routing is a clean separate code path.

### 17.6 Loop Closure

When `enable_loop_closure: true` in the `veo:` config, the engine generates one additional clip after all story cycles complete. This clip uses `first_last_frame` mode to transition from the final video's last frame back to the first cycle's anchor frame, creating a seamless loop.

This is ideal for DJ sets and live visuals where the video needs to loop continuously.

### 17.7 Retry Behavior

The Veo backend classifies failures into three categories:
- **Auth errors** (invalid key, permission denied) — immediate fail, no retry
- **Parameter errors** (unsupported combination) — immediate fail, no retry
- **Transient errors** (quota, timeout, server error) — exponential backoff with jitter, up to `max_retries` attempts

### 17.8 Observability

Every Veo generation logs:
- Backend, provider, model
- Auth mode (api_key or adc)
- Request mode (text_to_video, image_to_video, etc.)
- Duration, aspect ratio, resolution
- Seed
- Output path
- Operation polling status
- Generation latency
- Retry/backoff events

Per-cycle briq JSON includes `veo_mode` and `veo_metadata` fields for full reproducibility.

### 17.9 Quick Start

```bash
# Install Veo dependency
pip install google-genai

# Set API key
export GEMINI_API_TOKEN=your-api-key-here

# Use Veo config
cp worqspace/config-veo.yaml worqspace/config.yaml

# Write story
echo "A cyberpunk cityscape at night, neon lights reflecting off wet streets" > worqspace/story.txt

# Run
python vfaq_cli.py -n veo-test
```

---

## 18. LTX-Video Backend

### Overview

LTX-Video is a self-hosted video generation model that runs entirely on your local GPU. No cloud API keys, no quotas, no external dependencies beyond the model weights. Visual FaQtory supports LTX-Video as a first-class backend alongside ComfyUI and Veo.

LTX-Video is best suited for:
- Offline or air-gapped environments
- Unlimited generation without API costs
- Full control over model weights and inference parameters
- Privacy-sensitive content generation

### Setup

```bash
# 1. Clone the LTX-Video repository
git clone https://github.com/Lightricks/LTX-Video.git ~/LTX-Video
cd ~/LTX-Video

# 2. Install in a dedicated environment (recommended)
python -m venv ~/ltx-env
source ~/ltx-env/bin/activate
pip install -e .

# 3. Download model weights
# Place the pipeline YAML config in ~/LTX-Video/configs/
# Example: configs/ltxv-2b-0.9.8-distilled.yaml
# Weights are available from Hugging Face: https://huggingface.co/Lightricks/LTX-Video

# 4. Configure Visual FaQtory
cp worqspace/config-ltx.yaml worqspace/config.yaml
# Edit: set ltx_video.repo_path and ltx_video.python_bin
```

### Model Variants

| Variant | Config key | VRAM | Notes |
|---------|-----------|------|-------|
| 2B Distilled | `2b_distilled` | ~8 GB | Fastest, good quality. Recommended for most use cases. |
| 2B Full | `2b_full` | ~10 GB | Slightly better quality, slower. |
| 13B Distilled | `13b_distilled` | ~24 GB | High quality, needs large GPU. |
| 13B Full | `13b_full` | ~28 GB | Highest quality, slowest. |

Select via `ltx_video.model_variant` in config and ensure the matching `pipeline_config` YAML is present.

### How text2vid vs img2vid Is Chosen

When `ltx_video.mode: auto` (the default):

1. If a source image exists (base image for cycle 1, or last frame for subsequent cycles) → **i2v** (image-to-video)
2. If no source image exists → **t2v** (text-to-video)

You can force a specific mode:
- `mode: t2v` — always text-to-video, ignoring source images
- `mode: i2v` — always image-to-video (falls back to t2v if no source available)

### Runner Modes

**`python_api`** (recommended): Runs LTX-Video inference via a subprocess using the configured `python_bin`. Calls the official `infer(InferenceConfig(...))` API. The subprocess approach keeps LTX's heavy dependencies isolated from VFaQ's environment.

**`cli`**: Calls `inference.py` in the LTX-Video repo root as a subprocess. Passes all parameters via command-line arguments with underscore names (matching the official LTX-Video CLI). More isolated but slightly slower due to process startup overhead.

**`comfyui`** (v0.6.3+): Sends LTX-Video workflows to a ComfyUI server with ComfyUI-LTXVideo custom nodes installed. Uses the same HTTP API as the standard ComfyUI backend (POST /prompt, GET /history, GET /view). No `repo_path` or `python_bin` needed — the ComfyUI server handles all model loading and inference.

ComfyUI runner config:
```yaml
ltx_video:
  runner: comfyui
  comfyui_api_url: http://localhost:8188   # Your ComfyUI server
  comfyui_timeout: 600                      # Polling timeout (seconds)
  comfyui_ckpt: ltxv_2b_0.9.8_distilled_fp8.safetensors  # Checkpoint in ComfyUI models dir
  # comfyui_workflow_t2v: path/to/custom_t2v.json   # Optional custom workflow
  # comfyui_workflow_i2v: path/to/custom_i2v.json   # Optional custom workflow
```

Custom workflows support placeholder substitution: `%PROMPT%`, `%NEGATIVE%`, `%WIDTH%`, `%HEIGHT%`, `%NUM_FRAMES%`, `%SEED%`, `%FPS%`, `%CKPT%`, `%COND_STRENGTH_START%`, `%COND_STRENGTH_END%`.

### Frame Count Constraint

LTX-Video's VAE requires frame counts of the form **8k+1** (9, 17, 25, 33, 41, ..., 241). Visual FaQtory automatically snaps the requested frame count up to the nearest valid value. For example:

- Requested 30 frames → snapped to 33 (8×4 + 1)
- Requested 100 frames → snapped to 105 (8×13 + 1)
- Requested 240 frames → snapped to 241 (8×30 + 1)

The snap direction is always **up** to avoid losing requested duration. The snapped value is logged.

### Reinject Mapping

ComfyUI uses `denoise_strength` for img2img evolution. LTX-Video has no direct equivalent. Instead:

| ComfyUI concept | LTX-Video equivalent | Notes |
|----------------|---------------------|-------|
| `denoise_strength` | `image_cond_noise_scale` + `conditioning_strengths` | Approximate, not exact |
| Low denoise (0.2) | High conditioning strength (0.95) + low noise (0.05) | Strong source preservation |
| High denoise (0.7) | Lower conditioning strength (0.7) + higher noise (0.25) | More creative freedom |

This mapping is approximate by design. LTX conditioning is fundamentally different from ComfyUI's img2img denoising.

### Loop Seam Quality

LTX-Video does not have a native loop mode. Loop continuity is achieved by:

1. **Cycle continuity**: Previous cycle's last frame becomes next cycle's conditioning image (i2v)
2. **Loop closure**: Optional final clip transitions from last frame back to cycle-1 anchor frame using two-keyframe conditioning

The `loop_seam_threshold` config value controls a warning threshold. After generating a transition video, VFaQ computes a pixel MSE between the generated last frame and the target end frame. If the score exceeds the threshold, a warning is logged and `seam_warning: true` is added to briq metadata.

Typical seam scores:
- < 0.05: Excellent continuity
- 0.05–0.10: Acceptable for most content
- 0.10–0.20: Visible seam, consider adjusting conditioning strengths
- > 0.20: Poor match, may need manual intervention

### Limitations

- **No native still-image generation.** `generate_image()` produces a 9-frame clip and extracts the first frame. For production keyframes, consider ComfyUI.
- **No true latent morphing.** `generate_morph_video()` uses two-keyframe conditioned generation. The result is a conditioned transition, not mathematically smooth latent interpolation.
- **`long_clip_strategy: chain`** is defined in config but not yet implemented. Clips are capped at `max_frames_per_clip` (default 241 = ~8s at 30fps).
- **LTX API surface may change** between model versions. The `python_api` runner script may need adjustment for newer LTX releases.
- **No negative prompt guarantee.** Some LTX model variants may not fully respect negative prompts.

### Configuration Reference

See `worqspace/config-ltx.yaml` for a complete template with inline documentation of every parameter. Key parameters:

```yaml
ltx_video:
  runner: python_api                  # python_api | cli | comfyui
  mode: auto                          # auto | t2v | i2v
  repo_path: ~/LTX-Video              # Path to cloned LTX-Video repo
  python_bin: python3                  # Python with ltx_video installed
  pipeline_config: configs/ltxv-2b-0.9.8-distilled.yaml
  model_variant: 2b_distilled
  frame_rate: 30
  image_cond_noise_scale: 0.15
  conditioning_strength_start: 1.0
  conditioning_strength_end: 0.85
  enable_end_keyframe: true
  max_frames_per_clip: 241
  enable_loop_closure: false
  loop_seam_threshold: 0.10
```

### Quick Test

```bash
# Verify backend availability
python vfaq_cli.py backends

# Dry run to validate config
python vfaq_cli.py --dry-run -b ltx_video

# Full run with mock story
echo "A cyberpunk cityscape at night, neon lights reflecting off wet streets" > worqspace/story.txt
python vfaq_cli.py -n ltx-test -b ltx_video
```


## Live integration harness

See `LIVE-INTEGRATION-GUIDE.md` for the opt-in live ComfyUI/Qwen/AnimateDiff/Venice validation path. Normal `pytest` runs remain offline by default.
