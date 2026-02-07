# QonQrete Visual FaQtory v0.3.5-beta
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

> **v0.3.5-beta** — Production ready, stage safe, no hand-waving. Fixed stream/longcat (true autoregressive continuation), unified macro semantics, color stability controller, TouchDesigner integration contract (no `.toe` shipped).

---

## What This Does

Visual FaQtory takes a text prompt or base image and generates hours of evolving, forward-evolving visual content. Each cycle produces a video segment that morphs into the next, creating an infinite visual journey. After all cycles complete, the pipeline automatically interpolates to 60fps and upscales to 1920×1080 for cinema-smooth deliverables. Perfect for DJ sets, video installations, streams, and experimental AV.

---

## What's New in v0.3.5-beta

**⚡ Audio Toggle ≤200ms** — Background polling thread ensures macro response within 200ms regardless of frame generation time.

**🎬 Longcat Actually Works** — Default runs now produce real extended output. Target duration computed from `target_seconds` → `target_frames` → `generate_frames × max_iterations`.

**🎨 Stability Everywhere** — Color collapse prevention applied to all generation paths (offline, stream, Turbo).

**📡 VRAM Logging** — INFO-level estimates before each iteration. Clear warnings when safety caps hit.

**📖 No More Lies** — Every doc claim verified against code. TouchDesigner section states no `.toe` shipped. Longcat conditioning honesty documented.

---

## What Was New in v0.3.4-beta

**🔧 Fixed Stream/Longcat: True Autoregressive Continuation** — v0.3.3 Stream mode only restyled context frames. v0.3.4 uses SVD temporal diffusion to generate genuinely new frames beyond the context window. Each iteration takes the last frame, runs it through `SVD_img2vid_Conditioning`, and appends the new frames to the timeline. Stream output now actually grows every iteration and `generate_frames` controls how much.

**STREAM is offline cinematic. TURBO is live. Don't confuse them.**

**🎛️ Unified Macro Semantics** — Turbo no longer auto-deletes macro files. The contract is now deterministic: file exists = macro active. MIDI NOTE_ON creates the file, NOTE_OFF removes it. Turbo just reads. Works with any file-based trigger (scripts, OSC bridges, etc.).

**🎨 Long-Run Stability Controller** — New `vfaq/color_stability.py` prevents diffusion feedback collapse (the "green blob" problem). Uses CIELAB palette anchoring to the first frame, detects collapse via saturation/dominance/edge metrics, and mitigates by adjusting CFG, seed drift, and injecting micro-noise. CPU-side, <2ms per frame.

**🎥 TouchDesigner Integration Contract** — The `touchdesigner/` directory includes a network blueprint (`NETWORK_CONTRACT.txt`) and a Python builder (`td_setup.py`) describing the complete FX chain. Audio Device In, Analyze CHOP, Feedback loop, Displace, HUD overlay, MIDI In, OSC In. No binary `.toe` is shipped — you create the project in TD using the contract. TD keeps running even when AI stalls.

**🎧 Audio-Reactive Finalization** — Explicit audio-paused state when crowd override is active. Audio failure disables the controller, never crashes the frame loop. No GPU rebuilds from audio events.
```
- Trims final video to exact audio duration
- Muxes audio into final MP4

**🔄 Stream Mode (Longcat)** — True autoregressive continuation:
```bash
python vfaq_cli.py run -c 20 --stream      # Enable longcat mode
```
- Cycle N loads a short tail clip of up to `context_frames` frames from Cycle N‑1, **extracts the last frame** of that clip and uses it as the conditioning image for temporal diffusion. The entire tail window is not fed into the model.
- Generates `generate_frames` genuinely new frames beyond the tail clip and appends them to the timeline (the tail frames themselves are not duplicated).
- Repeats until the cycle's target length is met (full autoregression).
- Configurable via the `stream` section (`context_frames`, `generate_frames`, `max_iterations`, `checkpoint`).
- **Slower and VRAM‑heavy** — designed for offline cinematic runs, not live performances.

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
cd visual-faqtory-v0.3.5-beta

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
  --match-audio        Align visual duration to audio length (v0.1.2)
  --duration SECONDS   Fixed duration mode (v0.1.2)
  --stream             Enable stream continuation mode (v0.2.0)

# TURBO Live Mode (v0.3.5-beta)
python vfaq_cli.py live [OPTIONS]
  --turbo              Enable TURBO frame generation (default)
  --fps N              Target FPS (default: from config)
  --size WxH           Resolution (e.g., 768x432)
  --crowd              Enable crowd prompt server
  --crowd-port PORT    Crowd server port (default: 7777)
  --crowd-token TOKEN  Auth token for crowd submissions

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

## Known Limitations (v0.3.5-beta)

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
