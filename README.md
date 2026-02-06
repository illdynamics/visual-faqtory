# QonQrete Visual FaQtory v0.0.7-alpha
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
![Repo Views](https://komarev.com/ghpvc/?username=illdynamics-visual-faqtory&label=Repo+Views&color=blue)

![Splash](visual-faqtory.png)

```
 ██╗   ██╗██╗███████╗██╗   ██╗ █████╗ ██╗         ███████╗ █████╗  ██████╗ ████████╗ ██████╗ ██████╗ ██╗   ██╗
 ██║   ██║██║██╔════╝██║   ██║██╔══██╗██║         ██╔════╝██╔══██╗██╔═══██╗╚══██╔══╝██╔═══██╗██╔══██╗╚██╗ ██╔╝
 ██║   ██║██║███████╗██║   ██║███████║██║         █████╗  ███████║██║   ██║   ██║   ██║   ██║██████╔╝ ╚████╔╝
 ╚██╗ ██╔╝██║╚════██║██║   ██║██╔══██║██║         ██╔══╝  ██╔══██║██║▄▄ ██║   ██║   ██║   ██║██╔══██╗  ╚██╔╝
  ╚████╔╝ ██║███████║╚██████╔╝██║  ██║███████╗    ██║     ██║  ██║╚██████╔╝   ██║   ╚██████╔╝██║  ██║   ██║
   ╚═══╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝    ╚═╝     ╚═╝  ╚═╝ ╚══▀▀═╝    ╚═╝    ╚═════╝ ╚═╝  ╚═╝   ╚═╝
```

> **Automated Long-form AI Visual Generation for Music, DJ Sets & Experimental AV**
>
> **v0.0.7-alpha** — Prompt Bundle system, split backends, video_prompt support, LLM-aware creative context

---

## What This Does

Visual FaQtory takes a text prompt or base image and generates hours of evolving, forward-evolving visual content. Each cycle produces a video segment that morphs into the next, creating an infinite visual journey. After all cycles complete, the pipeline automatically interpolates to 60fps and upscales to 1920×1080 for cinema-smooth deliverables. Perfect for DJ sets, video installations, streams, and experimental AV.

---

## What's New in v0.0.7-alpha

**Prompt Bundle System** — Drop markdown files into `worqspace/` for full creative control:
- `tasq.md` — Base prompt (existing, required)
- `negative_prompt.md` — What to avoid (optional, overrides config defaults)
- `style_hints.md` — Style constraints and evolution direction (optional)
- `motion_prompt.md` — Video motion intent and camera guidance (optional)

All files are loaded, passed to the LLM, and stored in every briq JSON for full auditability. Missing files fall back to existing behavior — fully backward-compatible.

**Split Backend Config** — Use different engines for image vs video generation:
```yaml
backends:
  image:
    type: comfyui
    api_url: http://image-gpu:8188
  video:
    type: comfyui
    api_url: http://video-gpu:8188
```

**video_prompt Support** — Dedicated video prompt (separate from image prompt) for text-conditioned video workflows. SVD default ignores it gracefully.

**LLM-Aware Context** — InstruQtor and InspeQtor now receive full bundle context (style, motion, negative) for informed creative decisions.

---

## Supported Modes

| Mode | Cycle 0 | Cycle N>0 |
|------|---------|-----------|
| `text` | text → image → video | video → video (evolution) |
| `image` | image → video (skip img gen) | video → video (evolution) |
| `video` | ❌ Not valid for cycle 0 | video → video (evolution) |

After cycle 0, the pipeline always chains: previous video → extract frame → img2img → img2vid. Visual identity is **never hard-reset** between cycles.

---

## Requirements

- **Python** 3.10+
- **FFmpeg** (with h264_nvenc or libx264)
- **GPU** (for real backends; mock requires none)
- **pyyaml**, **pillow** (pip install)

---

## Quick Start

```bash
# 1. Enter directory
cd visual-faqtory-v0.0.7-alpha

# 2. Install dependencies
pip install pyyaml pillow

# 3. Quick smoke test (mock backend, no GPU needed)
python quick_test.py

# 4. Run 3 cycles as a named project
python vfaq_cli.py run -n test-run -c 3 -b mock --delay 1

# 5. Check the project output
ls worqspace/qonstructions/test-run/videos/

# 6. Assemble final video
python vfaq_cli.py assemble -n test-run
```

---

## CLI Reference

```bash
# Run generation
python vfaq_cli.py run [OPTIONS]
  -n, --name NAME      Project name (stored in worqspace/qonstructions/<n>/)
  -c, --cycles N       Run N cycles (default: unlimited)
  --hours H            Target H hours of content
  -b, --backend TYPE   Override backend (mock/comfyui/diffusers/replicate)
  --delay SECONDS      Delay between cycles (default: 2)
  --fresh              Start fresh (ignore saved state)

# Single test cycle
python vfaq_cli.py single [-n NAME] [--cycle N] [-b BACKEND]

# Check status
python vfaq_cli.py status [-n NAME]

# List available backends
python vfaq_cli.py backends

# Assemble all videos into final_output.mp4
python vfaq_cli.py assemble [-n NAME] [--preview]

# Clean up
python vfaq_cli.py clean [-n NAME] [--all]
```

---

## Worqspace Layout (Prompt Bundle)

```
worqspace/
├── tasq.md                   # Base creative prompt (REQUIRED)
├── negative_prompt.md        # What to avoid (optional)
├── style_hints.md            # Style + evolution constraints (optional)
├── motion_prompt.md          # Video motion intent (optional)
├── config.yaml               # Mechanical parameters (REQUIRED)
├── inputs/                   # Base images for image mode
├── examples/                 # Example configs and templates
└── qonstructions/            # Project output directories
```

---

## Project-Based Runs

When you use `-n <project-name>`, all outputs go into a structured project directory:

```
worqspace/qonstructions/<project-name>/
├── briqs/                    # VisualBriq JSON state files
├── images/                   # Generated source images
├── videos/                   # Per-cycle MP4s + raw videos
│   ├── cycle0000_raw.mp4
│   ├── cycle0000_video.mp4
│   ├── cycle0001_raw.mp4
│   └── cycle0001_video.mp4
├── factory_state.json        # Pipeline state (resumable)
├── config_snapshot.yaml      # Config used for this run
├── final_output.mp4          # Stitched base master (8fps, 1024×576)
└── final_60fps_1080p.mp4     # Final deliverable (60fps, 1920×1080)
```

If you omit `-n`, the run uses a temporary directory (`qodeyard/`). After completion, you're prompted to save it as a named project.

---

## Pipeline Flow

```
cycle generation (InstruQtor → ConstruQtor → InspeQtor)
  → per-cycle video (passthrough or loop)
  → cycle stitching (stream-copy / re-encode)
  → final_output.mp4 (BASE MASTER — 8fps, 1024×576)
  → POST-STITCH FINALIZER:
       → interpolate to 60fps (minterpolate MCI)
       → upscale to 1920×1080 (bicubic)
       → encode (h264_nvenc / libx264)
  → final_60fps_1080p.mp4 (FINAL DELIVERABLE)
  → pipeline exit
```

---

## Config vs tasq.md (Strict Separation)

**tasq.md** = Creative intent ONLY:
- `title`, `mode`, `backend`, `input_image`/`base_image`
- Descriptive prompt text
- Negative prompt text

**config.yaml** = Mechanical truth ONLY:
- `width`, `height`, `fps`, `duration`, `steps`
- `video_frames`, `clip_seconds`, `cfg_scale`
- All diffusion parameters, codec settings, finalizer settings, etc.

Mechanical parameters in tasq.md are **ignored with a warning**.

---

## Post-Stitch Finalizer Config

```yaml
finalizer:
  enabled: true                    # Set to false to skip post-stitch processing
  interpolate_fps: 60              # Target frame rate
  upscale_resolution: 1920x1080   # Target resolution
  scale_algo: bicubic              # Scaling algorithm
  encoder_preference:              # GPU-first with CPU fallback
    - h264_nvenc
    - libx264
  quality:
    crf: 16                        # CRF / NVENC CQ value (lower = better)
```

---

## Backend Options

| Backend | Availability | Setup |
|---------|-------------|-------|
| `mock` | ✅ Always | None needed |
| `comfyui` | ✅ Works | ComfyUI server + SDXL/SVD checkpoints |
| `diffusers` | ⚠️ Needs CUDA | `pip install torch diffusers` |
| `replicate` | ⚠️ Needs token | `REPLICATE_API_TOKEN` env var |

ComfyUI validates SDXL and SVD checkpoint availability via `/object_info` before generating. NVENC encoding is preferred; libx264 is automatic fallback.

---

## Known Limitations (v0.0.7-alpha)

- Video mode does frame extraction (not true video2video via AnimateDiff)
- ComfyUI needs VideoHelperSuite nodes for video output
- Diffusers backend requires CUDA (no CPU fallback)
- LLM evolution is optional (basic fallback always works)
- Post-stitch interpolation (minterpolate) is CPU-intensive and can be slow for long videos
- Default SVD workflow ignores text prompts (motion_prompt.md stored for auditability but not used by SVD directly)
- Split backends share the same project directory (no per-backend output isolation)

---

## License

Visual FaQtory is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
See the [LICENSE](LICENSE) file for full text.

---

Built by **Ill Dynamics / WoNQ** for the drum & bass massive 🎵

```
░▒▓█ ONE LOVE █▓▒░
```

![Scarf](https://static.scarf.sh/a.png?x-pxid=dc67438c-3388-46cd-baa7-7a0374420474)
