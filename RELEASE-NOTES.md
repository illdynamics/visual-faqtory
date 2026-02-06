# QonQrete Visual FaQtory - Release Notes

## v0.0.7-alpha

### New Features

- **Prompt Bundle System** — Full creative prompt loading from multiple worqspace files:
  - `tasq.md` — Base creative prompt (existing, still required)
  - `negative_prompt.md` — Dedicated negative prompt source of truth (optional, NEW)
  - `style_hints.md` — Style constraints and evolution direction (optional, NEW)
  - `motion_prompt.md` — Video motion intent and camera direction (optional, NEW)
  - All files are loaded via the new `PromptBundle` loader (`vfaq/prompt_bundle.py`)
  - Backward-compatible: missing files fall back to existing behavior

- **LLM-Aware Prompt Refinement** — InstruQtor and InspeQtor LLM templates now receive full bundle context (style hints, motion prompt, negative prompt) so the model can make informed creative decisions. LLM output fields extended to include `motion_hint`, `video_prompt`, and refined `negative_prompt`.

- **video_prompt Support** — Dedicated prompt for the video generation stage, separate from the image prompt. Useful for text-conditioned video workflows (e.g. custom ComfyUI workflows that accept text prompts for video). SVD default workflow ignores it gracefully — no regression.

- **Split Backend Configuration** — Use different backends for image and video generation:
  ```yaml
  backends:
    image:
      type: comfyui
      api_url: http://image-gpu:8188
    video:
      type: comfyui
      api_url: http://video-gpu:8188
  ```
  - `SplitBackend` wrapper delegates image/video to separate backend instances
  - Legacy single `backend:` config still works unchanged
  - ComfyUI global `comfyui:` section (ckpt paths) auto-merged into each split backend

- **VisualBriq Extended** — New fields for full creative auditability:
  - `style_hints`, `motion_prompt`, `video_prompt`, `motion_hint`
  - Backward-compatible: old briq JSONs load with empty defaults

### Improvements

- **Negative Prompt Precedence** — Clear, documented priority chain:
  1. tasq.md frontmatter `negative_prompt:` (power-user override)
  2. `negative_prompt.md` file
  3. `## Negative` section inside tasq.md body
  4. config.yaml `prompt_drift.negative_prompt`
  5. LLM-refined negative (only if source was config default)

- **Mechanical Separation Enforced** — FORBIDDEN_CREATIVE_KEYS validation in PromptBundle loader prevents mechanical parameters (fps, resolution, steps, etc.) from leaking into creative files.

- **ComfyUI video_prompt Injection** — Custom video workflows now receive `video_prompt` via CLIPTextEncode injection when available.

- **InspeQtor Style/Motion Context** — Evolution suggestions now respect style_hints boundaries and consider motion_prompt for continuity.

- **Worqspace Template Files** — Drop-in templates for all prompt bundle files with documentation headers.

- **Example Files** — `prompt_bundle_guide.md` and `split_backend_config.yaml` added to `worqspace/examples/`.

### Quality Goals
- Backward compatible with v0.0.6-alpha configs and briq JSONs
- Deterministic fallback when LLM is disabled
- No silent failure: every file load logged
- All new fields persisted in briq JSON for full auditability

### Output Contract (unchanged)
At pipeline completion:
- `final_output.mp4` — raw stitched base master (8fps, 1024×576) — NEVER MODIFIED
- `final_60fps_1080p.mp4` — final deliverable (60fps, 1920×1080, CRF 16)

---

## v0.0.6-alpha

### New Features

- **Post-Stitch Finalizer: Interpolation → Upscale** — The BANGER upgrade. After all cycles are stitched into `final_output.mp4` (base master), a new finalizer stage runs ONCE to produce `final_60fps_1080p.mp4`:
  - **Step 1 — Interpolation to 60fps** using FFmpeg `minterpolate` (MCI mode with bidirectional motion estimation). Converts the base 8fps video into cinema-smooth 60fps with temporal continuity.
  - **Step 2 — Upscale to 1920×1080** using bicubic scaling. Takes the interpolated output from 1024×576 to full HD.
  - **Encoding** — h264_nvenc (GPU) first with programmatic detection, automatic libx264 (CPU) fallback. CRF 16 / NVENC CQ 16 for high quality. `yuv420p` pixel format enforced.
  - Runs **exactly once**, never per-cycle, never before stitching, never on re-entry if deliverable already exists (idempotent).
  - Preserves the raw stitched master — `final_output.mp4` is never modified.
  - Configurable via `finalizer:` section in config.yaml with enable/disable flag.

- **Looping Disable (Passthrough Mode)** — InspeQtor now respects `looping.enabled: false` in config.yaml. When disabled, raw cycle videos are re-encoded with consistent codec/fps settings but WITHOUT creating a reverse (pingpong) loop. This produces forward-evolving visuals that look like continuous progression rather than small looping segments. Combined with the post-stitch interpolation, this creates smooth, cinematic visual flows.

- **Finalizer Config Section** — New `finalizer:` block in config.yaml:
  ```yaml
  finalizer:
    enabled: true
    interpolate_fps: 60
    upscale_resolution: 1920x1080
    scale_algo: bicubic
    encoder_preference:
      - h264_nvenc
      - libx264
    quality:
      crf: 16
  ```

### Improvements

- **GPU Encoder Detection** — Post-stitch finalizer programmatically tests h264_nvenc availability before use, with automatic libx264 fallback. Detection is based on actual ffmpeg return codes, not assumptions.
- **Verbose Finalizer Logging** — Full logging of encoder detection, interpolation progress, upscale progress, file sizes, and video metadata for both base master and deliverable.
- **Idempotent Finalizer** — If `final_60fps_1080p.mp4` already exists, the post-stitch finalizer skips entirely. Safe for pipeline resume/re-entry.
- **Pipeline Summary Enhanced** — Final pipeline output now shows both stitched master path and final deliverable path.
- **All version strings bumped** — Every file, module docstring, banner, and config reference updated to v0.0.6-alpha.

### Quality Goals

The post-stitch finalizer is designed to:
- Preserve glitch artifacts and AI generation characteristics
- Enhance temporal continuity (smooth frame transitions)
- Avoid reverse-loop dependence (forward-evolving visuals)
- Produce cinema-smooth, high-resolution deliverables
- Avoid VRAM-heavy generation paths (all post-processing is FFmpeg-based)

### Output Contract

At pipeline completion, the following artifacts exist:
- `final_output.mp4` — raw stitched base master (8fps, 1024×576)
- `final_60fps_1080p.mp4` — final deliverable (60fps, 1920×1080, CRF 16)

No other steps may modify these files.

---

## v0.0.5-alpha

### Bug Fixes
- **Fixed IMAGE mode cascade failure** — When cycle 0 failed or when cycle N>0 had no previous briq, IMAGE mode would crash because `base_image_path` was never set from `input_image` in the fallback path. The InstruQtor now correctly propagates `input_image` from tasq.md in all IMAGE mode scenarios, including cycle N>0 fallback.
- **Fixed ComfyUI execution error detection** — `_queue_and_wait` now checks the ComfyUI history `status.status_str` field for execution errors before attempting output download. Previously, a failed execution would appear as "Could not download outputs" with no actionable detail. Now surfaces the actual ComfyUI error.
- **Fixed VHS_VideoCombine `save_output` detection** — The `save_output` and `pingpong` parameters were only checked in VHS `required` inputs, but VHS places these in `optional`/`hidden`. New `_vhs_has_input()` helper checks all input categories. Without `save_output=True`, VHS could write to temp storage causing download failures.
- **Fixed VHS output format handling** — `_queue_and_wait` now checks for `'video'` (singular) key in addition to `'videos'` and `'gifs'`, covering all known VHS_VideoCombine output formats. Also handles dict (single-video) vs list normalization.
- **Improved ComfyUI download diagnostics** — Download failures now log the actual HTTP status, response body, and output node structure. URL parameters use proper `urllib.parse.urlencode` instead of raw f-string interpolation. Empty downloads are caught and reported.

### Improvements
- **Verbose output structure logging** — When ComfyUI generates outputs but no media is downloadable, the error now includes the actual node output keys so you can see exactly what ComfyUI produced.
- **All version strings bumped** — Every file, module docstring, banner, and config reference updated to v0.0.5-alpha.

### Verified Working
- **TEXT mode**: text → image → video ✓
- **IMAGE mode**: input_image → video (cycle 0 and fallback) ✓  
- **VIDEO mode**: previous cycle video → frame extract → img2img → video ✓ (requires successful cycle 0)

---

## v0.0.4-alpha

### New Features
- **Project-based runs** — Use `-n <project-name>` to store all artifacts in `worqspace/qonstructions/<project-name>/` with structured layout (briqs/, images/, videos/, factory_state.json, config_snapshot.yaml, final_output.mp4)
- **Qonstructions directory** — Persistent archival storage for named projects under worqspace/
- **Interactive save** — Unnamed runs prompt user to save as named project after completion
- **Final video stitching (Finalizer)** — Automatic concatenation of all per-cycle MP4s into `final_output.mp4` after pipeline completion, with stream-copy preferred and re-encode fallback
- **NVENC encoding support** — h264_nvenc preferred throughout pipeline with automatic libx264 fallback
- **VERSION file** — Canonical version tracking at repo root

### Improvements
- **Strict config/tasq separation** — Mechanical parameters (fps, duration, resolution, steps, etc.) in tasq.md are now ignored with clear warnings. config.yaml is the single source of truth for technical parameters.
- **Formalized mode handling** — TEXT mode: text→image→video. IMAGE mode: requires input_image, skips image gen. VIDEO mode: only valid for cycle N>0, requires previous cycle output. Pipeline fails fast with clear errors on invalid inputs.
- **ComfyUI model validation** — SDXL and SVD checkpoint availability verified via /object_info before generation, with actionable error messages
- **Pipeline never hard-resets visual identity** — Cycle N>0 always chains from previous output unless explicitly instructed otherwise
- **Documentation overhaul** — Complete rewrite of DOCUMENTATION.md with architecture diagram, all 11 sections, and accurate code-matching content
- **README update** — Reflects v0.0.4-alpha features, project-based workflow, and strict separation rules

### Bug Fixes
- Fixed duplicate backend initialization in VisualFaQtory.__init__
- Fixed InspeQtor using wrong LLM client reference
- Added missing `cycle_video_paths` and `failed_cycles` tracking to CycleState
- Ensured briq save creates parent directories

### Breaking Changes
- Output directory structure changed: videos and raw files now go to `videos/` subdirectory within projects
- tasq.md mechanical parameters (fps, duration, resolution, width, height, etc.) are no longer respected — move to config.yaml
- CLI `-n`/`--name` flag now controls project naming (previously used for assemble output filename)

---

## v0.0.3-alpha

### Features
- Fully wired pipeline: config.yaml + tasq.md actually used
- 3-Agent architecture: InstruQtor → ConstruQtor → InspeQtor
- Video chaining: each cycle uses previous output
- Frame extraction: video mode extracts frame for img2img
- FFmpeg looping: pingpong creates seamless 16s loops
- Evolution suggestions: InspeQtor suggests next cycle variations
- State persistence: resume interrupted runs
- Temp file cleanup: auto-removes intermediate files
- CLI delay option: `--delay 0` for fast testing
- ComfyUI backend fully functional with SDXL + SVD workflows
- ComfyUI object_info validation for SVD checkpoints
- LLM integration (OpenAI, Google Gemini, Mock)
- Mock backend with Pillow-generated placeholder images
- Diffusers and Replicate backends (partial)
- Quick test script (quick_test.py)
- Example tasq.md files for all three modes

---

## v0.0.2-alpha

### Features
- Initial 3-agent pipeline architecture
- VisualBriq data model
- Basic backend abstraction
- ComfyUI workflow generation
- FFmpeg video processing

---

## v0.0.1-alpha

### Features
- Initial proof of concept
- Basic pipeline structure
