# QonQrete Visual FaQtory - Release Notes

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
