# QonQrete Visual FaQtory - Technical Documentation v0.3.5-beta

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
13. [Turbo Audio-Reactive Mode](#13-turbo-audio-reactive-mode)
14. [MIDI Sidecar (v0.3.5-beta)](#14-midi-sidecar-v035-beta)
15. [TouchDesigner Integration (v0.3.5-beta)](#15-touchdesigner-integration-v035-beta)
16. [Stream / Longcat Mode (v0.3.5-beta)](#16-stream--longcat-mode-v035-beta)
17. [Color Stability Controller (v0.3.5-beta)](#17-color-stability-controller-v035-beta)
18. [Macro Control Semantics (v0.3.5-beta)](#18-macro-control-semantics-v035-beta)

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

## 9.5. Prompt Bundle System (v0.3.0-beta)

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

### Split Backend Config (v0.3.0-beta)

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

## 13. Turbo Audio-Reactive Mode

The TURBO engine can now respond to live audio input to modulate frame
parameters in real time. This feature uses a lightweight controller
(`vfaq/audio_reactive.py`) that captures audio from a selected device
via the `sounddevice` library. It computes a smoothed RMS energy and a
simple beat (peak) detection and exposes this state to the TURBO loop
without blocking frame generation.

### Configuring Audio Input

In your `config.yaml` under the `turbo` section you can enable and
customise the audio controller:

```yaml
turbo:
  audio_reactive:
    enabled: false          # Set true to enable by default (or use live toggle)
    device: null            # Name/index of input device; null uses system default
    sample_rate: 44100      # Sample rate in Hz
    block_size: 1024        # Buffer size for processing (smaller = lower latency)
    rms_smooth: 0.25        # EMA smoothing constant (0–1)
    beat_threshold: 0.6     # Normalised RMS threshold for beat detection
    mappings:
      rms_to_cfg: [1.2, 2.0]      # Map 0→1 RMS to this CFG range
      rms_to_seed_drift: [0, 6]   # Map 0→1 RMS to integer seed drift
      beat_macro: "DROP"         # Macro triggered on beats (DROP/BUILD/CHILL)
  live_toggle:
    enabled: true
    toggle_file: "macro_AUDIO"
  priority:
    crowd_overrides_audio: true
```

When `live_toggle.enabled` is true (recommended), the presence of
`live_output/macro_AUDIO` determines whether audio reactivity is active.
Touching this file (e.g. `touch live_output/macro_AUDIO`) turns on the
audio controller; removing it turns the controller off.

**ARMED vs ACTIVE (v0.3.5-beta):** To use `macro_AUDIO` toggling, you must
first set `turbo.audio_reactive.enabled: true` in the config. This *arms*
the audio-reactive feature. The toggle file then controls whether it is
*active*:

- `enabled: false` → audio OFF, `macro_AUDIO` file is **ignored** (logged)
- `enabled: true` + `live_toggle.enabled: true` + file exists → audio ON
- `enabled: true` + `live_toggle.enabled: true` + file missing → audio OFF
- `enabled: true` + `live_toggle.enabled: false` → audio always ON

The default config ships with `enabled: false` (safe for stage — no
surprise audio capture) and `live_toggle.enabled: true` (ready for MIDI
control once you arm it).

### How It Works

On each frame, the TURBO loop polls the audio controller for its latest
state. The smoothed RMS value is clamped to [0, 1] and linearly mapped
to the configured CFG and seed drift ranges. When a beat is detected
(RMS crosses `beat_threshold` from below), the specified macro (by
default `DROP`) is triggered—unless a manual macro override is in
effect.

### Crowd Override

When a crowd prompt is active and `crowd_overrides_audio` is true, the
engine freezes the CFG and seed drift values to their base settings and
temporarily disables audio-driven macros. Once the crowd prompt expires,
audio modulation resumes automatically.

### Failure Modes

If `numpy` or `sounddevice` are not installed, or if the specified
audio device cannot be opened, the controller will disable itself and
log a warning. Frame generation continues uninterrupted, and CFG/seed
values remain at their base configuration. This makes the system safe
for on-stage use even when audio hardware is unavailable.

## 14. MIDI Sidecar (v0.3.5-beta)

The MIDI sidecar is an optional companion process that allows physical
controllers—such as USB MIDI devices, Arduino boards, or DJ gear—to
control TURBO macros and continuous parameters without touching the
core engine. The sidecar runs as a **separate process** and never
imports the Turbo engine, ComfyUI, or any GPU libraries. It simply
creates or removes files in `live_output/` that the TURBO loop already
polls. This design keeps the system gig‑safe: even if the sidecar
crashes, frame generation continues unaffected.

### Configuration (`config.yaml`)

Add or update the following section in `config.yaml` to enable and configure the
MIDI sidecar. The new schema supports more advanced mappings, including note
modes (momentary vs toggle) and smoothing for CC values:

```yaml
midi:
  enabled: false            # Set true to enable MIDI sidecar
  in_name: null             # Case‑insensitive substring to match input port
  in_port: null             # Exact port name (overrides in_name)
  live_output_dir: "live_output" # Directory where macro files are written
  poll_sleep_ms: 2          # Sleep duration (ms) when no MIDI messages are pending
  log_level: "INFO"        # Logging level: DEBUG, INFO, WARN, ERROR
  note_off_cleanup: true    # Remove momentary macro files on note off
  mapping:
    notes:
      36: { action: "DROP",  mode: "momentary" }   # C1 holds DROP while pressed
      37: { action: "BUILD", mode: "momentary" }   # D1 holds BUILD while pressed
      38: { action: "CHILL", mode: "momentary" }   # D#1 holds CHILL while pressed
      39: { action: "AUDIO", mode: "toggle" }      # F1 toggles audio reactivity on each press
    cc:
      10: { action: "INTENSITY", min: 0.0, max: 1.0, smoothing: 0.0 }  # CC10 controls intensity
      7:  { action: "ENERGY",    min: 0.0, max: 1.0, smoothing: 0.1 }  # CC7 controls energy with smoothing
```

### How It Works (v0.3.5-beta)

When the sidecar receives a MIDI **note** message, it performs one of two behaviours
based on the configured *mode*:

* **Momentary** (default): On `note_on` (velocity > 0), the sidecar creates or
  touches a file named `macro_<ACTION>` (e.g. `macro_DROP`). On `note_off` or
  `note_on` with velocity 0, it removes the file to clear the override. This
  behaves like holding down a button: the macro stays active only while you
  keep the note pressed.
* **Toggle**: On `note_on` (velocity > 0), the sidecar checks for the
  existence of the file. If present, it removes it; if absent, it creates it. This
  behaves like an on/off switch: each press toggles the macro state. Note off
  messages are ignored for toggle mappings.

Actions map to macro file names as follows: `DROP` → `macro_DROP`, `BUILD` →
`macro_BUILD`, `CHILL` → `macro_CHILL`, `AUDIO` → `macro_AUDIO`. If
`note_off_cleanup` is set to `false` in the config or via CLI, note off
messages are ignored, so momentary macros stay active until explicitly
cleared.

For **control change (CC)** messages, the sidecar normalises the controller
value (0–127) to the configured `min`–`max` range. It then optionally
applies exponential moving average smoothing: each update is combined with
the previous output using the `smoothing` factor (0.0 = no smoothing,
1.0 = very slow). The resulting floating point number is written to
`macro_INTENSITY` or `macro_ENERGY` depending on the `action`. These
continuous parameters are read by Turbo on each frame and applied to
`cfg_scale` and `seed_drift` multiplicatively. If the files are absent, Turbo
reverts to its base values.

### Running the Sidecar

To start the MIDI sidecar, run:

```bash
python sidecar/midi_sidecar.py --name "Arduino" --out-dir live_output
```

The sidecar uses the default configuration from `config.yaml` but allows
extensive CLI overrides:

- `--config <file>` — Path to a custom YAML config (defaults to
  `worqspace/config.yaml` if it exists).
- `--port <exact>` — Use an exact MIDI input port name (overrides config).
- `--name <substring>` — Pick the first input whose name contains this
  substring (case‑insensitive). Use this to auto‑select controllers like
  “Arduino” or “DJ”.
- `--list` — Print available MIDI input ports and exit.
- `--out-dir <path>` — Directory where macro files are written. Defaults to
  the `live_output_dir` in the config.
- `--log-level` — Set logging verbosity (DEBUG, INFO, WARN, ERROR).
- `--dry-run` — Do not touch or write any files; log actions instead.
- `--note-off-cleanup` — Enable or disable removal of momentary macros on
  note off (overrides config).
- `--default-note-mode` — Set a fallback note mode (`momentary` or
  `toggle`) for notes lacking an explicit `mode` in the config.
- `--verbose-cc` — Log CC values at INFO instead of DEBUG.
- `--self-test` — Run a test of file I/O helpers (touch/remove/atomic
  write) in a temporary directory and exit.

Run `python sidecar/midi_sidecar.py --help` for a complete list of options.

### Why File‑Based Control Is Safe

Because the sidecar only touches files, there is **zero coupling**
between hardware control and the core generation loop. Turbo polls
`live_output/` for macro files via a background thread (v0.3.5-beta,
≤200ms response) and applies macros and continuous parameters. If the
sidecar is not running or no MIDI device is connected, the system
behaves normally. Likewise, if the sidecar crashes, the existing macros
persist until cleared, and frame generation continues uninterrupted.

**Turbo never auto-deletes macro files.** The sidecar is solely
responsible for file lifecycle. File removal always reverts state.

### Explicit Non-Goals

- The sidecar does **not** manage the Turbo lifecycle (start/stop/restart).
- The sidecar does **not** guarantee MIDI device hotplug on all OSes.
  Reconnection is attempted on device loss, but hotplug behavior varies.
- MIDI is **optional, never required**. Turbo runs identically without it.

<!-- End of MIDI Sidecar section -->

## 15. TouchDesigner Integration (v0.3.5-beta)

The **TouchDesigner integration** enables you to layer real‑time GPU effects on top
of AI‑generated frames without blocking or slowing down the generation loop. TouchDesigner handles
feedback loops, kaleidoscope, glitch, colour correction and other shader FX,
while Visual FaQtory continuously writes the latest frame and prompt overlays
to disk. The two systems communicate exclusively via file watching (for the
images and overlay text files) and optionally via OSC messages for state
synchronisation.

### Why Use TouchDesigner?

Visual FaQtory specialises in generating evocative AI imagery, but it does
not perform heavy post‑processing. TouchDesigner, on the other hand, excels at
real‑time video manipulation on the GPU. By combining the two you get the
best of both worlds: AI‑generated content evolving over minutes or hours and
shader‑based FX reacting instantaneously to audio and MIDI. The AI never
blocks the FX pipeline, and the FX never waits for a new AI frame.

### Contract and File Exchange

- **Frames**: Turbo writes `live_output/current_frame.jpg` on every frame. A
  **Movie File In TOP** in TouchDesigner watches this path.
- **Overlays**: Turbo (via `overlay_writer.py`) writes `now.txt` and
  `next.txt` containing the current and upcoming prompts. A **Text DAT** in
  TouchDesigner reads these files and feeds the text into a **Text TOP** for
  overlay.
- **Macros**: TouchDesigner can respond to the same macro files used by
  Turbo (`macro_DROP`, `macro_BUILD`, `macro_CHILL`, `macro_AUDIO`,
  `macro_INTENSITY`, `macro_ENERGY`) if you wish to synchronise FX
  behaviours. These files live in `live_output/` and can be read or
  ignored depending on your patch design.
- **OSC (Optional)**: If `osc.enabled` is set to `true` in `config.yaml`,
  Turbo broadcasts its current macro, crowd activity flag, and combined
  energy level (max of intensity and energy) on every frame (rate limited
  by `osc.send_ms`). TouchDesigner can receive these values via an **OSC In
  CHOP** to drive additional parameters.

### TouchDesigner Network Contract

> **No `.toe` file is shipped.** TouchDesigner projects are binary files that
> cannot be meaningfully version-controlled. This repository provides the
> **integration contract** — file paths, OSC schema, macro semantics, and a
> network blueprint — not the binary project. You must create the `.toe`
> yourself in TouchDesigner using the blueprint below.

The repository includes a network blueprint at
`touchdesigner/NETWORK_CONTRACT.txt`. This is a **plain-text description** of
the network to be built in TouchDesigner. It outlines the following components:

1. **Movie File In TOP** watching `live_output/current_frame.jpg`.
2. **Cache TOP** to buffer the incoming frame and prevent flicker.
3. **Level / HSV / Colour Correct TOPs** to adjust brightness, contrast and hue.
4. **Feedback TOP Loop** using a **Feedback TOP**, **Transform TOP** and
   **Composite TOP** to create trails and zoom effects.
5. **Displace TOP** driven by an **Analyze CHOP** reading audio RMS and peaks.
6. **Kaleidoscope and Glitch FX** via **Kaleido TOP** and **Glitch TOP**.
7. **Composite TOP** layering processed video with text overlays fed from
   **Text DATs** reading `now.txt` and `next.txt`.
8. **Audio Device In CHOP** and **Analyze CHOP** computing RMS and peaks to
   drive feedback amount, strobe, zoom and other FX parameters.
9. **MIDI In CHOP** connected to your controller. Map knobs and faders to
   FX depths and toggles. Use the same controller for the MIDI sidecar to
   drive AI macros by connecting it to both the sidecar and TouchDesigner.
10. **NDI Out TOP**, **Spout/Syphon Out TOP** or a fullscreen window to
    deliver the final output to OBS or a projector.

You should recreate this network in TouchDesigner, save it as a `.toe` file,
and adjust parameters to taste. The blueprint ensures a solid starting
point that synchronises perfectly with Visual FaQtory's file-based control
layer.

### Wiring Audio and MIDI

TouchDesigner runs its own audio analysis in parallel to the AI audio
reactivity. Use an **Audio Device In CHOP** to bring in your DJ audio or
microphone, followed by **Analyze CHOPs** (e.g. RMS, Beat, Band RMS) to
generate control signals. Use a **MIDI In CHOP** to capture hardware
controller input. These control signals can drive the feedback amount, glitch
intensity, displacement strength, kaleidoscope angle and other FX. Because
TouchDesigner’s CHOPs are highly optimized, the effects react with
millisecond latency, independent of AI generation time.

### Performance and Safety Notes

- **Separation of Concerns**: Turbo runs diffusion on the GPU and is
  inherently slower than real-time. TouchDesigner runs shader effects on the
  GPU and is extremely fast. By decoupling them via file watching and
  optional OSC, each system operates at its own pace without causing frame
  drops in the other.
- **MIDI and Audio Sharing**: You can route the same MIDI device to both
  the MIDI sidecar (for AI control) and TouchDesigner (for FX control). Each
  process polls independently so there is no contention or latency.
- **Tuning**: For best results, match the output resolution in TouchDesigner
  to `turbo.width × turbo.height` and set the output frame rate to your
  streaming or projection target (e.g. 60 fps). Use the **Cache TOP** to
  smooth frame updates and avoid tearing when the AI frame rate is lower
  than the display frame rate.

- **OBS/Projection Pipelines**: You can ingest the TouchDesigner output via
  NDI into OBS for streaming (Option A), display it directly on a projector
  (Option B), or mix it with camera feeds and overlays in OBS (Option C).

*Documentation for QonQrete Visual FaQtory v0.3.5-beta*
*Part of the WoNQ Cinematic Universe*

---

## Auto-Duration + Audio Match (v0.1.2)

### Overview
When `base_audio` is provided and `match_audio` is enabled, the system automatically
computes how many cycles are needed to match the audio length, trims the final video
to the exact audio duration, and optionally muxes the audio into the final MP4.

### Configuration
```yaml
duration:
  mode: auto            # auto | fixed | unlimited
  seconds: null         # Used when mode=fixed
  match_audio: false    # Enable audio matching
  trim_strategy: cut    # cut (required)
  mux_audio: true       # Mux audio into final video
```

### CLI Flags
```bash
--match-audio          # Enable audio matching
--duration 180         # Fixed duration (seconds)
```

### Behavior
- `mode: auto` + `match_audio: true` + base_audio → compute required cycles
- `mode: fixed` + `seconds: 60` → render until 60s reached
- `mode: unlimited` → current behavior (cycle count or Ctrl+C)
- Only full cycles are rendered (no partial cycles)
- Trim is deterministic (stream copy → re-encode fallback)

---

## Stream Mode: Autoregressive Continuation (“Longcat”) (v0.3.0)

### Overview
Longcat mode produces a truly continuous “infinite” video by loading a tail clip of up to `context_frames` frames from the end of the previously generated segment, **extracting the last frame of that clip** to use as the conditioning image, synthesising new frames beyond that clip, and appending the new frames to the timeline. This process repeats autoregressively until the requested duration is reached. Unlike the old sliding‑window restyle, longcat **extends** the video; it doesn’t just restyle the same window, and only the last frame of the tail clip participates in the temporal diffusion.

Each iteration loads only a small tail clip into the GPU, extracts its last frame, and then executes a short SVD continuation workflow that uses this frame as the init image to generate `generate_frames` genuinely new frames. Only the new frames are appended; the tail clip itself is not reused. The procedure repeats until `video_frames` (or cycle duration) is satisfied. VRAM usage is predictable because context and generation lengths are capped.

### Configuration
```yaml
stream:
  enabled: false            # Enable longcat autoregressive continuation
  mode: "longcat"          # Longcat mode (default when stream is present)
  context_frames: 16       # Number of tail frames to condition on
  generate_frames: 16      # New frames per iteration
  max_iterations: 999      # Safety cap on iterations
  checkpoint: "svd_xt.safetensors"  # SVD model to use
  overlap_strategy: "tail" # How to handle overlaps (reserved for future)
```
If `stream.enabled` is true the pipeline switches into longcat mode on cycles >0. The CLI flag `--stream` sets this property on the fly. Legacy `stream_mode` configuration is still recognised for backward compatibility but will use the old sliding‑window behaviour; migrating to the new `stream` section is recommended.

### How It Works
1. **Extract Tail** — At each iteration the last `context_frames` frames are pulled from the current video (via ffmpeg) and re‑encoded as a tail clip. The last frame of this clip is extracted and becomes the conditioning image for SVD temporal diffusion; the rest of the clip is discarded.
2. **Generate** — A ComfyUI workflow (derived from `worqspace/workflows/stream_continuation.json`) is customised to load the tail clip, set the `ckpt_name` to the configured checkpoint, and feed the extracted last frame into `SVD_img2vid_Conditioning`. It generates `generate_frames` new frames beyond the tail; seeds, steps, CFG and denoise values propagate from your `config.yaml`.
3. **Append** — Only the newly generated frames are appended onto the evolving video using ffmpeg. The next iteration uses this updated video as its starting point; the tail clip itself is not re‑inserted.
4. **Repeat** — Steps 1–3 repeat until the desired number of frames (or seconds) is produced or `max_iterations` is hit. This yields a seamless long‑form output.

### VRAM & Performance
Longcat is **VRAM‑heavy** and **slow** compared to TURBO or single‑cycle generation. Each iteration must fit both the context and new frames into GPU memory. For safety the implementation hard‑caps `context_frames` and `generate_frames` at 64 and halves them on OOM errors. Expect 4×–8× slower rendering than normal cycles. Use a dedicated GPU with ample memory and run offline; longcat is not designed for live performance.

### Fallbacks
If longcat continuation fails (e.g., ComfyUI unavailable), the engine falls back to:
1. **Video2Video** using the context clip as input (quality reduced).
2. **img2vid** from the last frame when all else fails.

The old sliding‑window stream (v0.2.0) still exists under the `stream_mode` section for legacy projects.

---

## TURBO Live Mode (v0.2.5)

### Overview
Real-time single-frame generation for OBS / TouchDesigner integration.
Generates frames continuously and writes to `live_output/current_frame.jpg`.

### Quick Start
```bash
# 1. Start ComfyUI with SDXL Turbo model
# 2. Configure turbo section in config.yaml
# 3. Run:
python vfaq_cli.py live --turbo

# With crowd prompts:
python vfaq_cli.py live --turbo --crowd
```

### OBS Setup
1. Add Image Source → `live_output/current_frame.jpg`
2. Add Text (GDI+) from file → `live_output/now.txt`
3. Add Text (GDI+) from file → `live_output/next.txt`

### DJ Macro Triggers
Touch these files to trigger macros:
- `live_output/macro_DROP` → Intense energy boost
- `live_output/macro_BUILD` → Rising tension
- `live_output/macro_CHILL` → Return to base state

### Seed Modes
- `fixed` — Same seed every frame (static evolution)
- `drift` — Seed increments by `seed_drift` each frame
- `beatjump` — Large seed jumps for visual variety

---

## Crowd Queue System (v0.2.5)

### Overview
Live audience can submit prompts via phone during performances.
Server runs on LAN, no internet required.

### Endpoints
- `GET /` — HTML submit page
- `POST /submit` — Submit prompt
- `GET /status` — Queue status
- `GET /queue` — Top N queued items

### Rate Limiting
Per-IP and per-name sliding window rate limits prevent spam.

### Moderation
- Length enforcement (min 3, max 120 chars)
- Banned words list
- Optional regex patterns
- URL stripping
- ASCII-only mode (optional)

### Integration Modes
- **blend** — Crowd prompt appended to base prompt for `duration_seconds`
- **takeover** — Crowd prompt completely replaces base prompt temporarily
- **timed_slot** — Every N seconds, next crowd prompt gets a time slot

### Network Safety
- Crowd server is LAN-only by default
- For public exposure: use Cloudflare Tunnel + auth token
- `crowd.server.auth.enabled: true` + `token: "secret"` for simple auth

---

## 16. Stream / Longcat Mode (v0.3.5-beta)

### Overview

Stream (Longcat) mode generates long-form videos through true autoregressive
continuation using SVD temporal diffusion. **This is an offline cinematic
mode — it is NOT real-time.** For live performance, use TURBO mode.

### How It Works

Each iteration of the longcat loop:

1. **Extracts context tail** — The last `context_frames` frames from the
   current video are extracted as a short clip using ffmpeg.
2. **Extracts last frame** — The workflow uses `GetImageFromBatch` to grab
   the very last frame from the context clip.
3. **SVD temporal diffusion** — The last frame is fed into
   `SVD_img2vid_Conditioning` as the init_image, which generates
   `generate_frames` genuinely NEW frames using temporal diffusion.
4. **Appends to timeline** — Only the new frames are appended. The context
   is NOT included in the output.
5. **Repeats** — The new video becomes the source for the next context
   extraction. This continues until target duration or `max_iterations`.

### Workflow

The SVD workflow (`worqspace/workflows/stream_continuation.json`) contains:

```
VHS_LoadVideo → GetImageFromBatch (last frame)
  → ImageOnlyCheckpointLoader (SVD checkpoint)
  → SVD_img2vid_Conditioning
  → KSampler (temporal diffusion)
  → VAEDecode
  → VHS_VideoCombine
```

### VRAM Safety

Stream mode is VRAM-heavy. A 12GB GPU can handle approximately 16 context +
16 generation frames. The config provides hard caps:

```yaml
stream:
  vram_safety:
    max_context_frames: 24   # Hard cap
    max_generate_frames: 24  # Hard cap
    oom_retry: true           # Halve and retry on OOM
```

On OOM: `generate_frames` is halved and the iteration retried. If still OOM,
the stream aborts cleanly, keeping all segments generated so far.

### Configuration

```yaml
stream:
  enabled: true
  mode: "longcat"
  backend: "comfyui"
  temporal_model: "svd"
  context_frames: 16
  generate_frames: 16
  max_iterations: 999
  fps: 8
  checkpoint: "svd_xt.safetensors"
  overlap_strategy: "tail"
  vram_safety:
    max_context_frames: 24
    max_generate_frames: 24
    oom_retry: true
```

### Honest Limitations

- **Slow**: Each iteration involves a full SVD inference pass (20+ steps).
  A 16-frame chunk takes 10-60 seconds depending on GPU.
- **VRAM-heavy**: SVD models require significant GPU memory. 8GB GPUs
  should reduce `generate_frames` to 8.
- **Not real-time**: Use TURBO for live visuals.

---

## 17. Color Stability Controller (v0.3.5-beta)

### The Problem

Iterative diffusion accumulates color drift. After 100+ frames:
- Saturation collapses
- Single hue dominates
- Detail/edges disappear
- Output converges to a uniform "green blob"

This is diffusion feedback collapse, not user error. It happens because each
frame is conditioned on the previous one, and small biases compound.

### The Solution

The `StabilityController` in `vfaq/color_stability.py` provides three
mechanisms, all CPU-side and frame-safe:

#### 1. Palette Anchoring (LAB Color Space)

- The first output frame is captured as the "anchor".
- On each subsequent frame, the mean and standard deviation of the LAB A/B
  channels are matched to the anchor's statistics.
- The correction is blended with the original by `strength` (default 0.6).
- Uses CIELAB for perceptually uniform corrections.

#### 2. Collapse Detection

Three metrics are monitored every frame:
- **Mean saturation** (HSV S-channel) — below `sat_floor` = collapsed
- **Hue dominance** — single 30° sector above `hue_dom_ratio` = collapsed
- **Edge energy** (Sobel magnitude mean) — below `edge_floor` = collapsed

If ALL three metrics cross thresholds for `consecutive_frames` frames
(default 12), a collapse event is triggered.

#### 3. Collapse Mitigation

On collapse detection:
- CFG is reduced by `reduce_cfg` (default 0.15)
- Seed drift is reduced by `reduce_seed_drift` factor (default 0.5)
- Micro-noise (Gaussian, sigma=0.005) is injected to break the attractor
- Palette is re-anchored from the mitigated frame

### Performance

- CPU-only, pure numpy (no OpenCV dependency)
- <2ms per frame at 1024×576
- In live TURBO mode, skips correction if 10ms time budget is exceeded
- Never blocks frame output

### Configuration

```yaml
stability:
  enabled: true
  anchor: "first_frame"
  method: "lab_palette"
  strength: 0.6
  every_n_frames: 1
  collapse_detection:
    enabled: true
    sat_floor: 0.08
    hue_dom_ratio: 0.72
    edge_floor: 3.0
    consecutive_frames: 12
  mitigation:
    reduce_cfg: 0.15
    reduce_seed_drift: 0.5
    micro_noise_sigma: 0.005
```

### Where It Runs

- **TURBO mode**: Applied to every output frame via `process_frame_bytes()`
- **Normal cycles**: Applied post-generation to all output frames (ConstruQtor)
- **STREAM/LONGCAT (v0.3.5-beta)**: **In‑loop** per‑iteration stability. After
  each longcat iteration the last frame of the generated segment is extracted
  and processed by the `StabilityController`. If collapse is detected, CFG and seed drift
  are reduced for the next iteration. Micro‑noise is applied to the probe frame for anchoring
  purposes, but the corrected frame is **not** used as the next init image. This mechanism
  prevents the autoregressive feedback loop from converging into monochrome by adjusting generation
  parameters rather than feeding corrected frames back into the diffusion model. The stability
  check runs on a single frame per iteration (CPU‑side, <5 ms) and never blocks GPU rendering.

---

## 18. Macro Control Semantics (v0.3.5-beta)

### The Contract

Turbo **never** auto-deletes macro files. File presence equals state.

| Macro     | Type      | Active When                | Revert When           | Content      |
|-----------|-----------|----------------------------|-----------------------|--------------|
| DROP      | momentary | `macro_DROP` file exists   | File removed          | (ignored)    |
| BUILD     | momentary | `macro_BUILD` file exists  | File removed          | (ignored)    |
| CHILL     | momentary | `macro_CHILL` file exists  | File removed          | (ignored)    |
| AUDIO     | toggle    | `macro_AUDIO` file exists  | File removed          | (ignored)    |
| INTENSITY | value     | Float in `macro_INTENSITY` | File removed → 0     | Float 0.0–1.0 |
| ENERGY    | value     | Float in `macro_ENERGY`    | File removed → 0     | Float 0.0–1.0 |

Priority for momentary macros: DROP > BUILD > CHILL (first found wins).
If no momentary macro file is present, state reverts to CHILL.

### MIDI Behavior

- `NOTE_ON` → sidecar creates file → Turbo sees it → macro active
- `NOTE_OFF` → sidecar removes file → Turbo reverts to CHILL
- File removal always reverts state (no stale macros)

### Alignment with MIDI Sidecar

The sidecar is responsible for file lifecycle:
- `NOTE_ON` → sidecar creates `macro_DROP` (momentary) or flips
  `macro_AUDIO` (toggle)
- `NOTE_OFF` → sidecar removes `macro_DROP` (momentary only)
- `CC` → sidecar writes float to `macro_INTENSITY` / `macro_ENERGY`

Turbo only reads. This means you can also control macros from:
- Shell scripts (`touch macro_DROP` / `rm macro_DROP`)
- OSC bridges
- Web interfaces
- Any tool that creates/removes files

### Background Polling (v0.3.5-beta)

Macro files and the audio toggle are polled by a background daemon thread
every 100-200ms, independent of the render loop. The render loop reads
only cached values. This guarantees ≤200ms response to `macro_AUDIO`
toggle even when individual frame generation takes multiple seconds.

The control layer is implemented as logical components inside TurboEngine,
not as separate classes:
1. **Audio influence** — RMS/beat detection → CFG/seed drift/macro
2. **Macro influence** — file-based momentary/toggle/value macros
3. **Crowd override** — when crowd is active, audio influence is zeroed

### Sidecar Crash Isolation

The MIDI sidecar runs as a separate process. It never imports Turbo, GPU,
or ComfyUI modules. If the sidecar crashes:
- Turbo continues running
- Macro files remain in their last state
- Audio and generation are unaffected
