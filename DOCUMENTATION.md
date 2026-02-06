# QonQrete Visual FaQtory - Technical Documentation v0.0.7-alpha

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
10. [Post-Stitch Finalizer (Interpolation + Upscale)](#10-post-stitch-finalizer-interpolation--upscale)
11. [Failure Modes & Debugging](#11-failure-modes--debugging)
12. [Performance Notes](#12-performance-notes)

---

## 1. System Overview

Visual FaQtory is a 3-agent + finalizer pipeline that generates long-form evolving visual content from text prompts or base images. Each cycle produces a video segment that chains into the next, with LLM-guided or algorithmic prompt evolution. After all cycles complete, the post-stitch finalizer interpolates to 60fps and upscales to 1080p.

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
 ├─ InspeQtor           (passthrough or loop + evolution suggestion)
 ├─ Finalizer           (ffmpeg concat → final_output.mp4)
 └─ Post-Stitch         (interpolate 60fps → upscale 1080p → final_60fps_1080p.mp4)
 ↓
worqspace/qonstructions/<project-name>/
 ├─ briqs/
 ├─ images/
 ├─ videos/
 ├─ factory_state.json
 ├─ config_snapshot.yaml
 ├─ final_output.mp4          (base master — 8fps, 1024×576)
 └─ final_60fps_1080p.mp4     (final deliverable — 60fps, 1920×1080)
```

---

## 2. Core Concepts

### Briq (VisualBriq)

The fundamental instruction unit passed between agents. Contains prompt, mode, specs, seed, base paths, status, and outputs. Serializable to JSON for state persistence.

### Cycle

One complete pass through the 3-agent pipeline producing a single video segment. Cycle 0 starts from tasq.md; cycle N>0 evolves from the previous cycle's output.

### Worqspace

The input directory containing `config.yaml` (mechanical parameters) and `tasq.md` (creative intent). Also hosts the `qonstructions/` directory for named project storage.

### Qonstructions

Project-based archival storage under `worqspace/qonstructions/<project-name>/`. Each project contains all artifacts: briqs, images, videos, state, config snapshot, and final outputs.

### Base Master vs Deliverable

- **Base Master** (`final_output.mp4`): Raw stitched video at original fps/resolution. Never modified after creation.
- **Final Deliverable** (`final_60fps_1080p.mp4`): Post-processed output with 60fps interpolation and 1080p upscale. The file you ship.

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

Third agent. Processes cycle video and generates evolution suggestions. Two operating modes:

- **Looping enabled** (`looping.enabled: true`): Creates loopable video using FFmpeg (pingpong forward+reverse or crossfade). Each cycle produces a seamless loop.
- **Passthrough mode** (`looping.enabled: false`): Re-encodes raw video with consistent codec/fps but NO reverse loop. Produces forward-evolving visuals that flow continuously when stitched. This is the recommended mode when using the post-stitch finalizer.

---

## 4. Pipeline Flow (Step-by-Step)

```
CYCLE 0 (text mode):
  1. InstruQtor parses tasq.md → VisualBriq (mode=TEXT)
  2. ConstruQtor generates image from prompt (txt2img)
  3. ConstruQtor generates video from image (img2vid)
  4. ConstruQtor saves as cycle0000_raw.mp4
  5. InspeQtor processes video:
     - looping.enabled=true:  pingpong/crossfade loop → cycle0000_video.mp4
     - looping.enabled=false: passthrough re-encode → cycle0000_video.mp4
  6. InspeQtor generates evolution suggestion

CYCLE 0 (image mode):
  1. InstruQtor parses tasq.md → VisualBriq (mode=IMAGE, base_image set)
  2. ConstruQtor skips image gen, uses base_image directly
  3. ConstruQtor generates video from base_image (img2vid)
  4-6. Same as above

CYCLE N>0 (always video mode):
  1. InstruQtor receives previous briq + evolution suggestion
  2. InstruQtor evolves prompt (80% same, 20% variation)
  3. InstruQtor sets mode=VIDEO, base_video=previous cycle video
  4. ConstruQtor extracts middle frame from base_video
  5. ConstruQtor runs img2img on frame with evolved prompt
  6. ConstruQtor generates video from evolved image (img2vid)
  7-9. Same processing + evolution flow

FINALIZATION (after all cycles):
  1. Finalizer collects all cycleNNNN_video.mp4 in order
  2. Concatenates via ffmpeg (stream-copy preferred, re-encode fallback)
  3. Saves as final_output.mp4 (BASE MASTER)
  4. Logs duration, fps, resolution, file size

POST-STITCH FINALIZER (if enabled, runs ONCE):
  1. Checks if final_60fps_1080p.mp4 already exists → skip if so
  2. Detects encoder: h264_nvenc (GPU) or libx264 (CPU) fallback
  3. STEP 1: Interpolate final_output.mp4 → 60fps (minterpolate MCI)
  4. STEP 2: Upscale interpolated output → 1920×1080 (bicubic)
  5. Saves as final_60fps_1080p.mp4 (FINAL DELIVERABLE)
  6. Cleans up intermediate temp file
  7. Logs deliverable metadata
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
  clip_seconds: 4.0                      # Raw video duration
  width: 1024
  height: 576
  cfg_scale: 6.0
  steps: 30
  sampler: euler_ancestral
  video_frames: 24
  video_fps: 8
  motion_bucket_id: 127
  noise_aug_strength: 0.02

chaining:
  denoise_strength: 0.4                  # 0.3-0.5 recommended

looping:
  enabled: false                         # false = passthrough (forward-evolving)
  method: pingpong                       # pingpong or crossfade (when enabled)
  output_fps: 8
  output_codec: h264_nvenc               # Falls back to libx264
  output_quality: 16

finalizer:
  enabled: true                          # Post-stitch interpolation + upscale
  interpolate_fps: 60                    # Target frame rate
  upscale_resolution: 1920x1080          # Target resolution
  scale_algo: bicubic                    # Scaling algorithm
  encoder_preference:                    # GPU-first with fallback
    - h264_nvenc
    - libx264
  quality:
    crf: 16                              # CRF / NVENC CQ (lower = better)

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
  model: gpt-4.1-mini
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

## 9. Finalizer Logic (Stitching)

The Finalizer stitching runs after all cycles complete:

1. **Validation**: Checks for failed cycles — if any exist, finalization is aborted with a report identifying which cycles failed.

2. **Collection**: Gathers all `cycleNNNN_video.mp4` files in chronological order from the project's `videos/` directory.

3. **Stream-copy attempt**: First tries `ffmpeg -c copy` concatenation (fastest, no re-encoding). This works when all videos share identical codec, resolution, and fps.

4. **Re-encode fallback**: If stream-copy fails, re-encodes using preferred codec (h264_nvenc) with libx264 fallback.

5. **Output**: Produces `final_output.mp4` (base master) in the project root.

6. **Logging**: Reports duration (seconds/minutes), fps, resolution, and file size.

---

## 9.5. Prompt Bundle System (v0.0.7-alpha)

### Overview

The Prompt Bundle system allows users to split creative intent across multiple worqspace files, giving the LLM richer context and the user finer-grained control.

### Files

| File | Required | Purpose |
|------|----------|---------|
| `tasq.md` | YES | Base creative prompt + frontmatter |
| `negative_prompt.md` | no | Negative prompt source of truth |
| `style_hints.md` | no | Style constraints, evolution rules |
| `motion_prompt.md` | no | Video motion intent, camera direction |

### Loader: `vfaq/prompt_bundle.py`

The `load_prompt_bundle()` function reads all files from worqspace and returns a `PromptBundle` dataclass. The InstruQtor caches this bundle and passes it to the LLM on every cycle.

### Negative Prompt Precedence

1. tasq.md frontmatter `negative_prompt:` (power-user override)
2. `negative_prompt.md` file content
3. `## Negative` section inside tasq.md body
4. config.yaml `prompt_drift.negative_prompt`
5. LLM-returned negative (only if source was config default)

### Mechanical Separation

The PromptBundle loader enforces that creative files do NOT contain mechanical parameters. If `tasq.md` frontmatter contains keys like `fps`, `resolution`, `steps`, etc., they are logged as warnings and IGNORED. These belong exclusively in `config.yaml`.

### VisualBriq Fields

Every briq JSON now contains these additional fields for auditability:

- `style_hints` — contents of style_hints.md
- `motion_prompt` — contents of motion_prompt.md
- `video_prompt` — dedicated video prompt (LLM-generated or derived)
- `motion_hint` — short LLM-generated motion guidance

### Split Backend Config (v0.0.7-alpha)

Use different backends for image and video generation:

```yaml
backends:
  image:
    type: comfyui
    api_url: http://image-gpu:8188
  video:
    type: comfyui
    api_url: http://video-gpu:8188
```

If `backends:` is present, it takes priority over legacy `backend:`. If `backends.video` is omitted, it defaults to the image config. The global `comfyui:` section is auto-merged for ckpt visibility.

---

## 10. Post-Stitch Finalizer (Interpolation + Upscale)

### Overview

The post-stitch finalizer is a **permanent pipeline capability** that runs exactly once after stitching. It transforms the base master into a cinema-smooth deliverable.

### Pipeline Position

```
final_output.mp4 (8fps, 1024×576)
  → STEP 1: minterpolate to 60fps
  → STEP 2: bicubic upscale to 1920×1080
  → STEP 3: encode with h264_nvenc or libx264
final_60fps_1080p.mp4 (60fps, 1920×1080)
```

### Non-Negotiable Rules

- ❌ MUST NOT run per cycle
- ❌ MUST NOT run before final stitching
- ❌ MUST NOT modify the raw stitched master
- ❌ MUST NOT double-run if pipeline is resumed or re-entered
- ✅ MUST run once, after final stitching, before pipeline exits

### Encoder Detection

The finalizer programmatically detects encoder availability:

1. Attempts h264_nvenc with a minimal test encode (`nullsrc` → `/dev/null`)
2. If NVENC fails (no GPU, driver issue, etc.), falls back to libx264
3. If ALL encoders fail, the finalizer aborts with a clear error

**NVENC args**: `-c:v h264_nvenc -cq 16 -preset p5 -pix_fmt yuv420p`
**libx264 args**: `-c:v libx264 -crf 16 -preset slow -pix_fmt yuv420p`

### Interpolation Details

Uses FFmpeg's `minterpolate` filter with motion-compensated interpolation:

```
minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1
```

- `mi_mode=mci`: Motion Compensated Interpolation (best quality)
- `mc_mode=aobmc`: Adaptive Overlapped Block Motion Compensation
- `me_mode=bidir`: Bidirectional motion estimation
- `vsbmc=1`: Variable-size block motion compensation

This preserves glitch artifacts and AI-generated visual characteristics while creating smooth temporal transitions.

### Idempotency

If `final_60fps_1080p.mp4` already exists when the finalizer is invoked, it skips entirely. This prevents double-processing on pipeline resume or re-entry.

### Quality Goals

- Preserve glitch artifacts and AI generation aesthetics
- Enhance temporal continuity (smooth frame transitions)
- Avoid reverse-loop dependence (forward-evolving visuals)
- Produce cinema-smooth, high-resolution deliverables
- All post-processing is FFmpeg-based (no VRAM needed)

---

## 11. Failure Modes & Debugging

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

### Post-stitch finalizer failed
1. Check ffmpeg has minterpolate support: `ffmpeg -filters | grep minterpolate`
2. Check encoder: `ffmpeg -encoders | grep h264` (need nvenc or x264)
3. Check disk space — interpolated video can be large
4. Check logs for specific step failure (interpolation vs upscale)

### State file issues
```bash
python vfaq_cli.py clean -n my-project         # Reset state only
python vfaq_cli.py clean -n my-project --all    # Full reset
```

### Mechanical params in tasq.md
If you see warnings about "forbidden keys", move those parameters from tasq.md to config.yaml. tasq.md is for creative intent only.

---

## 12. Performance Notes

- Each cycle produces ~4-8s of video (configurable via `clip_seconds`)
- Mock backend is instant (CPU only, for testing)
- ComfyUI with SDXL+SVD: ~30-60s per cycle on modern GPU
- NVENC encoding is 5-10x faster than libx264
- Stream-copy finalization is near-instant regardless of video count
- State is saved after each cycle — resume anytime
- LLM calls add ~1-2s per cycle (optional, basic fallback is instant)

### Post-Stitch Finalizer Performance

- Interpolation (minterpolate) is CPU-intensive: ~5-15 minutes per minute of source video
- Upscale is relatively fast: ~1-3 minutes per minute of interpolated video
- NVENC encoding is 5-10x faster than libx264 for the upscale step
- Total post-stitch time for a 2-hour source: ~30-60 minutes (varies by CPU)

### Duration Estimation

| Target | Cycles (~4s each) | Est. Gen Time (GPU) | Post-Stitch |
|--------|-------------------|---------------------|-------------|
| 30 min | ~450 | ~4 hours | ~15-30 min |
| 1 hour | ~900 | ~8 hours | ~30-60 min |
| 2 hours | ~1800 | ~16 hours | ~60-120 min |

---

*Documentation for QonQrete Visual FaQtory v0.0.7-alpha*
*Part of the WoNQ Cinematic Universe*
