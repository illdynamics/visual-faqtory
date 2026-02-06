# QonQrete Visual FaQtory - Technical Documentation v0.0.5-alpha

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Core Concepts](#2-core-concepts)
3. [Agent Roles](#3-agent-roles)
4. [Pipeline Flow](#4-pipeline-flow)
5. [Backend Architecture](#5-backend-architecture)
6. [ComfyUI Integration Details](#6-comfyui-integration-details)
7. [Configuration Reference](#7-configuration-reference)
8. [CLI Reference](#8-cli-reference)
9. [Finalizer Logic](#9-finalizer-logic)
10. [Failure Modes & Debugging](#10-failure-modes--debugging)
11. [Performance Notes](#11-performance-notes)

---

## 1. System Overview

Visual FaQtory is a 3-agent + finalizer pipeline that generates long-form evolving visual content from text prompts or base images. Each cycle produces a loopable video segment that chains into the next, with LLM-guided or algorithmic prompt evolution.

### Architecture Diagram

```
User
 ↓
vfaq_cli.py  (-n <project-name>)
 ↓
VisualFaQtory
 ├─ InstruQtor          (parse tasq.md → create VisualBriq)
 ├─ ConstruQtor
 │   ├─ SDXL (image)    (txt2img or img2img via backend)
 │   └─ SVD (video)     (img2vid via backend)
 ├─ InspeQtor           (FFmpeg loop + evolution suggestion)
 └─ Finalizer           (ffmpeg concat → final_output.mp4)
 ↓
worqspace/qonstructions/<project-name>/
 ├─ briqs/
 ├─ images/
 ├─ videos/
 ├─ factory_state.json
 ├─ config_snapshot.yaml
 └─ final_output.mp4
```

---

## 2. Core Concepts

### Briq (VisualBriq)

The fundamental instruction unit passed between agents. Contains prompt, mode, specs, seed, base paths, status, and outputs. Serializable to JSON for state persistence.

### Cycle

One complete pass through the 3-agent pipeline producing a single looped video segment. Cycle 0 starts from tasq.md; cycle N>0 evolves from the previous cycle's output.

### Worqspace

The input directory containing `config.yaml` (mechanical parameters) and `tasq.md` (creative intent). Also hosts the `qonstructions/` directory for named project storage.

### Qonstructions

Project-based archival storage under `worqspace/qonstructions/<project-name>/`. Each project contains all artifacts: briqs, images, videos, state, config snapshot, and final output.

---

## 3. Agent Roles

### InstruQtor

First agent. Parses `tasq.md` for creative intent, applies config.yaml mechanical parameters, creates a VisualBriq. For cycle 0: uses tasq prompt. For cycle N>0: evolves from previous prompt using InspeQtor's suggestion. Enforces strict config/tasq separation — mechanical parameters in tasq.md are ignored with warnings.

### ConstruQtor

Second agent. Validates required inputs per mode (fail-fast), calls the configured backend to generate images and videos. Mode handling:
- TEXT: txt2img → img2vid
- IMAGE: skip image gen, feed base image to img2vid
- VIDEO: extract middle frame → img2img → img2vid

### InspeQtor

Third agent. Creates loopable video using FFmpeg (pingpong or crossfade), generates evolution suggestions for the next cycle using LLM or algorithmic fallback.

---

## 4. Pipeline Flow (Step-by-Step)

```
CYCLE 0 (text mode):
  1. InstruQtor parses tasq.md → VisualBriq (mode=TEXT)
  2. ConstruQtor generates image from prompt (txt2img)
  3. ConstruQtor generates video from image (img2vid)
  4. ConstruQtor saves as cycle0000_raw.mp4
  5. InspeQtor creates pingpong loop → cycle0000_video.mp4
  6. InspeQtor generates evolution suggestion

CYCLE 0 (image mode):
  1. InstruQtor parses tasq.md → VisualBriq (mode=IMAGE, base_image set)
  2. ConstruQtor skips image gen, uses base_image directly
  3. ConstruQtor generates video from base_image (img2vid)
  4-6. Same as above

CYCLE N>0 (always video mode):
  1. InstruQtor receives previous briq + evolution suggestion
  2. InstruQtor evolves prompt (80% same, 20% variation)
  3. InstruQtor sets mode=VIDEO, base_video=previous looped video
  4. ConstruQtor extracts middle frame from base_video
  5. ConstruQtor runs img2img on frame with evolved prompt
  6. ConstruQtor generates video from evolved image (img2vid)
  7-9. Same loop + evolution flow

FINALIZATION (after all cycles):
  1. Finalizer collects all cycleNNNN_video.mp4 in order
  2. Concatenates via ffmpeg (stream-copy preferred, re-encode fallback)
  3. Saves as final_output.mp4
  4. Logs duration, fps, resolution, file size
```

---

## 5. Backend Architecture

All backends implement the `GeneratorBackend` interface:

```python
class GeneratorBackend(ABC):
    def generate_image(request: GenerationRequest) -> GenerationResult
    def generate_video(request: GenerationRequest, source_image: Path) -> GenerationResult
    def check_availability() -> tuple[bool, str]
```

Available backends:

| Backend | Image Gen | Video Gen | Cost | NVENC |
|---------|-----------|-----------|------|-------|
| mock | placeholder | placeholder | Free | Fallback |
| comfyui | SDXL | SVD | Free* | Yes |
| diffusers | SDXL | SVD | Free* | Fallback |
| replicate | SDXL | SVD | $$$ | N/A |

All backends try h264_nvenc first, fallback to libx264 automatically.

---

## 6. ComfyUI Integration Details

### Validation

Before generation, ComfyUI backend:
1. Fetches `/object_info` to get available nodes and models
2. Validates SDXL checkpoint exists in `CheckpointLoaderSimple` options
3. Validates SVD checkpoint exists in `ImageOnlyCheckpointLoader` options
4. Fails early with actionable error messages if models are missing

### Required ComfyUI Nodes

- `CheckpointLoaderSimple` (SDXL)
- `ImageOnlyCheckpointLoader` (SVD)
- `KSampler`
- `CLIPTextEncode`
- `EmptyLatentImage`
- `VAEDecode`
- `SVD_img2vid_Conditioning`
- `VHS_VideoCombine` (from VideoHelperSuite)
- `SaveImage`
- `LoadImage`

### Config

```yaml
backend:
  type: comfyui
  api_url: http://localhost:8188
  timeout: 600

comfyui:
  sdxl_ckpt: sd_xl_base_1.0.safetensors
  svd_ckpt: svd_xt.safetensors
```

---

## 7. Configuration Reference

### config.yaml (complete)

```yaml
input:
  tasq_file: tasq.md                    # Creative intent file

backend:
  type: comfyui                          # mock, comfyui, diffusers, replicate
  api_url: http://localhost:8188
  timeout: 600

comfyui:
  sdxl_ckpt: sd_xl_base_1.0.safetensors # Must match ComfyUI's available models
  svd_ckpt: svd_xt.safetensors

generation:
  clip_seconds: 8                        # Raw video duration
  width: 1280
  height: 720
  cfg_scale: 6.0
  steps: 30
  sampler: euler_ancestral
  video_frames: 240
  video_fps: 30
  motion_bucket_id: 127
  noise_aug_strength: 0.02

chaining:
  denoise_strength: 0.4                  # 0.3-0.5 recommended

looping:
  method: pingpong                       # pingpong or crossfade
  output_fps: 30
  output_codec: h264_nvenc               # Falls back to libx264
  output_quality: 16

prompt_drift:
  quality_tags: [masterpiece, best quality, highly detailed]
  negative_prompt: "blurry, low quality, watermark, deformed"

cycle:
  max_cycles: 0                          # 0 = unlimited
  target_duration_hours: 2.0
  delay_seconds: 2.0
  max_retries: 3
  continue_on_error: true

llm:
  provider: mock                         # mock, openai, google
  model: gpt-4o-mini
  api_key_env: OPENAI_API_KEY
```

### tasq.md (allowed fields)

```yaml
---
title: My Visual Project          # Descriptive title
mode: text                         # text or image
backend: comfyui                   # Backend hint (optional)
input_image: inputs/my_image.png   # For image mode
seed: 42                           # Initial seed
---

Your creative prompt text here...

## Negative
Things to avoid in generation...
```

**Forbidden in tasq.md:** fps, duration, resolution, width, height, video_frames, clip_seconds, steps, cfg_scale, sampler, motion_bucket_id, noise_aug_strength, denoise_strength, output_fps, output_codec, output_quality, crossfade_frames.

---

## 8. CLI Reference

```
vfaq_cli.py [-w WORQSPACE] [-o OUTPUT] COMMAND [OPTIONS]

Global options:
  -w, --worqspace DIR    Worqspace directory (default: ./worqspace)
  -o, --output DIR       Default output dir (default: ./qodeyard)

Commands:
  run       Run the visual generation pipeline
  single    Run a single test cycle
  status    Show pipeline status
  backends  List available backends
  assemble  Assemble per-cycle videos into final_output.mp4
  clean     Clean output / state files

run options:
  -n, --name NAME        Project name
  -c, --cycles N         Number of cycles
  --hours H              Target hours of content
  -b, --backend TYPE     Override backend
  --delay SECONDS        Inter-cycle delay
  --fresh                Ignore saved state

single options:
  -n, --name NAME        Project name
  --cycle N              Cycle index (default: 0)
  -b, --backend TYPE     Override backend

assemble options:
  -n, --name NAME        Project name
  --preview              Preview mode (first N videos)
  --preview-count N      Videos in preview (default: 10)

clean options:
  -n, --name NAME        Project name
  --all                  Remove all artifacts (not just state)
```

---

## 9. Finalizer Logic

The Finalizer runs after all cycles complete:

1. **Validation**: Checks for failed cycles — if any exist, finalization is aborted with a report identifying which cycles failed.

2. **Collection**: Gathers all `cycleNNNN_video.mp4` files in chronological order from the project's `videos/` directory.

3. **Stream-copy attempt**: First tries `ffmpeg -c copy` concatenation (fastest, no re-encoding). This works when all videos share identical codec, resolution, and fps.

4. **Re-encode fallback**: If stream-copy fails, re-encodes using preferred codec (h264_nvenc) with libx264 fallback.

5. **Output**: Produces `final_output.mp4` in the project root.

6. **Logging**: Reports final duration (seconds/minutes), fps, resolution, and file size.

---

## 10. Failure Modes & Debugging

### No videos generated
1. Check backend: `python vfaq_cli.py backends`
2. Verify FFmpeg: `ffmpeg -version`
3. Check project videos/ directory for raw files

### ComfyUI not reachable
1. Start ComfyUI first: `python main.py --listen`
2. Verify api_url in config.yaml
3. Test: `curl http://localhost:8188/system_stats`

### SDXL/SVD checkpoint not found
1. Check ComfyUI models: `curl http://localhost:8188/object_info | jq '.CheckpointLoaderSimple.input.required.ckpt_name[0]'`
2. Ensure checkpoint filenames in config.yaml match exactly
3. Restart ComfyUI after adding models

### Finalization failed
1. Check for failed cycles in factory_state.json
2. Verify all cycle videos exist in videos/ directory
3. Run manually: `python vfaq_cli.py assemble -n <project>`

### State file issues
```bash
python vfaq_cli.py clean -n my-project         # Reset state only
python vfaq_cli.py clean -n my-project --all    # Full reset
```

### Mechanical params in tasq.md
If you see warnings about "forbidden keys", move those parameters from tasq.md to config.yaml. tasq.md is for creative intent only.

---

## 11. Performance Notes

- Each cycle produces ~16s of looped video (8s raw → 16s pingpong)
- Mock backend is instant (CPU only, for testing)
- ComfyUI with SDXL+SVD: ~30-60s per cycle on modern GPU
- NVENC encoding is 5-10x faster than libx264
- Stream-copy finalization is near-instant regardless of video count
- State is saved after each cycle — resume anytime with `--resume`
- LLM calls add ~1-2s per cycle (optional, basic fallback is instant)

### Duration Estimation

| Target | Cycles | Est. Time (GPU) |
|--------|--------|-----------------|
| 30 min | ~113 | ~1 hour |
| 1 hour | ~225 | ~2 hours |
| 2 hours | ~450 | ~4 hours |

---

*Documentation for QonQrete Visual FaQtory v0.0.5-alpha*
*Part of the WoNQ Cinematic Universe*
