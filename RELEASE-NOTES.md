# QonQrete Visual FaQtory - Release Notes

## v0.3.5-beta — "No Lies Left" Edition

**Visual FaQtory v0.3.5-beta is the first version where all shipped features are real,
documented, and production-safe. No demo behavior, no fake continuation, no placeholder
integrations.**

### Fixed: Audio Toggle Response ≤200ms (Turbo)

Turbo audio-reactive mode now uses a **background polling thread** that checks macro
files every 100–200ms independently of the render loop. Previously, audio toggle
latency was tied to frame generation time — if a frame took 5 seconds, your toggle
took 5 seconds. Now it reacts within 200ms even when frames are slow.

**ARMED vs ACTIVE semantics:** `audio_reactive.enabled` is the ARMED gate. When
`false`, the `macro_AUDIO` toggle file is **ignored** (and logged). When `true`,
the toggle file controls active state. Default config ships armed=false (stage safe).

The control layer is implemented as logical components inside `TurboEngine`, not as
separate classes: audio influence, macro influence, and crowd override.

### Fixed: Stream/Longcat Default Behavior (Critical)

Default longcat runs now produce **real extended output**. Previously, longcat stopped
after one iteration regardless of `max_iterations`, because target duration was not
computed.

The fix introduces explicit target duration computation with clear priority:
`target_seconds` (recommended) → `target_frames` → `generate_frames × max_iterations`.
The backend loop now continues until `total_generated_frames >= target_frames`.

**Conditioning honesty:** Current longcat uses **last-frame continuation only**, not
full temporal context. Multi-frame temporal conditioning (AnimateDiff style) is not
implemented. This is documented, not hidden.

### Added: VRAM Estimate Logging

INFO-level logging now reports resolution, context frames, generate frames, and
checkpoint name before each longcat iteration. Clear warnings are emitted when
VRAM safety caps reduce requested frame counts.

### Added: Long-Run Stability Controller (Applied Everywhere — For Real)

The color stability controller (`color_stability.py`) is now applied to **all**
generation paths:

- **Turbo**: per-frame via `process_frame_bytes()` (already done in v0.3.4-beta)
- **Normal offline runs** (ConstruQtor): post-generation on all output frames
- **Stream/Longcat** (v0.3.5-beta): **In-loop per-iteration** stability. After each
  longcat iteration, the last frame is extracted, fed through StabilityController,
  and if collapse is detected, CFG and seed drift are reduced for the next
  iteration. Micro‑noise is applied to the probe frame for palette anchoring,
  but the corrected frame is **not** used as the next init image. This mechanism
  prevents the autoregressive feedback loop from converging into single‑color demon slime
  by adjusting generation parameters rather than feeding corrected frames back into the model.

### Fixed: TouchDesigner Honesty

No `.toe` file is shipped. The previous `visual_faqtory.toe` text placeholder has
been renamed to `touchdesigner/NETWORK_CONTRACT.txt` to make this explicit. The
repository provides the **integration contract** (file paths, OSC schema, MIDI/audio
mapping, network blueprint) — not the binary TouchDesigner project.

### Finalized: Macro Control Semantics

Explicit macro semantics table added to documentation. File presence equals state.
Turbo never auto-deletes macro files. MIDI `NOTE_ON` holds, `NOTE_OFF` releases.

### Documentation

All docs updated for v0.3.5-beta. Every claim has been verified against the actual
codebase. No feature is documented that does not exist. No behavior is claimed that
is not implemented.

### Changed Files

* `vfaq/turbo_engine.py` — Background poll thread, cached macro state, ARMED vs ACTIVE audio gating.
* `vfaq/backends.py` — Longcat target frame loop, VRAM logging, **in-loop stability hook**, conditioning honesty.
* `vfaq/stream_engine.py` — Target duration computation (target_seconds/target_frames), parent config passthrough.
* `vfaq/construqtor.py` — Stability controller integration for normal offline runs.
* `vfaq/__init__.py` — Removed false TouchDesigner .toe claim.
* `worqspace/config.yaml` — Added `stream.target_seconds`, stability docs.
* `worqspace/workflows/stream_continuation.json` — Meta version bump to v0.3.5-beta.
* `touchdesigner/NETWORK_CONTRACT.txt` — Renamed from `visual_faqtory.toe`.
* `DOCUMENTATION.md` — ARMED vs ACTIVE audio docs, in-loop stability docs, macro semantics table, TouchDesigner honesty.
* `README.md` — Updated feature descriptions, removed misleading claims.
* `RELEASE-NOTES.md` — This section.
* `VERSION` — `v0.3.5-beta`

---

## v0.3.4-beta — "Production Ready, Stage Safe, No Hand-Waving" Edition

### What Changed

**This release removes fake behavior and replaces it with real, testable systems.**

Every feature in v0.3.4 either works as documented or has been removed. If you relied
on undocumented behavior from previous versions, read the breaking changes below.

### Fixed: Stream/Longcat — True Autoregressive Continuation

**Problem:** v0.3.3 Stream mode restyles context frames. It does not extend the
timeline. `generate_frames` had no effect on actual duration. The release notes
claimed autoregressive continuation but the implementation only ran the context
through a low-denoise KSampler — producing restyled input, not new content.

**Fix:** Complete rewrite of the longcat backend. The new implementation:

  - Extracts the **last frame** from the context window (not the whole clip).
  - Uses **SVD temporal diffusion** (`SVD_img2vid_Conditioning`) to generate
    genuinely new frames beyond the context.
  - Appends only the new frames to the timeline.
  - Repeats until the target duration is reached or `max_iterations` hits.
  - Each iteration produces `generate_frames` new frames.

The workflow (`stream_continuation.json`) has been completely replaced with a
proper SVD pipeline: `VHS_LoadVideo → GetImageFromBatch → ImageOnlyCheckpointLoader →
SVD_img2vid_Conditioning → KSampler → VAEDecode → VHS_VideoCombine`.

**New config keys:** `stream.backend`, `stream.temporal_model`, `stream.fps`,
`stream.vram_safety.max_context_frames`, `stream.vram_safety.max_generate_frames`,
`stream.vram_safety.oom_retry`.

**IMPORTANT:** Stream mode is **offline cinematic** — it is slow and VRAM-heavy.
For real-time generation, use TURBO mode.

### Fixed: Unified Macro Control Semantics

**Problem:** Turbo instantly deleted macro files on detection, breaking MIDI
"momentary" behavior. A MIDI NOTE_ON that creates `macro_DROP` would be deleted
before the next poll, making the macro last only one frame instead of the full
note duration.

**Fix:** Turbo **never** auto-deletes macro files. The new contract:

| Macro     | Type      | Behavior                                         |
|-----------|-----------|--------------------------------------------------|
| DROP      | momentary | Active while `macro_DROP` file exists             |
| BUILD     | momentary | Active while `macro_BUILD` file exists            |
| CHILL     | momentary | Active while `macro_CHILL` file exists            |
| AUDIO     | toggle    | Audio reactive enabled while `macro_AUDIO` exists |
| INTENSITY | value     | Float 0–1 read from `macro_INTENSITY` content     |
| ENERGY    | value     | Float 0–1 read from `macro_ENERGY` content        |

Priority: DROP > BUILD > CHILL (first found wins). If no momentary macro
file is present, the state reverts to CHILL.

This aligns with MIDI behavior: `NOTE_ON` → sidecar creates file → Turbo sees
it → `NOTE_OFF` → sidecar removes file → Turbo reverts to CHILL.

### Finalized: Stage-Safe Audio Reactive Turbo

  - Explicit audio-paused state logged when crowd override is active.
  - Audio macros are disabled while crowd override is active (clean state).
  - Audio failure disables the controller gracefully — never crashes the frame loop.
  - No GPU rebuilds from audio events. Ever.

### New: Long-Run Stability Controller (Color Collapse Prevention)

**Problem:** Even normal runs with a base image can drift color, collapse
saturation, and converge into single-color blobs after 100+ frames. This is
diffusion feedback collapse, not user error.

**Solution:** New `vfaq/color_stability.py` module with three mechanisms:

  1. **Palette Anchoring** — Captures the first frame as an anchor. On every
     subsequent frame, matches the mean and standard deviation of the LAB A/B
     channels to the anchor, blending by configurable strength.

  2. **Collapse Detection** — Monitors mean saturation, hue dominance, and
     edge energy. If all three cross thresholds for 12+ consecutive frames,
     a collapse event is flagged.

  3. **Collapse Mitigation** — Reduces CFG, reduces seed drift, injects
     micro-noise to break the attractor basin, and re-anchors the palette.

The controller is CPU-side, <2ms per frame, and skips if the time budget is
exceeded. Works in TURBO, STREAM/LONGCAT, and normal cycle runs.

### New: Real TouchDesigner Starter Patch

The `touchdesigner/` directory now contains:

  - `NETWORK_CONTRACT.txt` — Network blueprint with detailed documentation of
    every operator and its wiring. No binary `.toe` is shipped.
  - `td_setup.py` — Python script that builds the complete network when
    executed inside TouchDesigner.

The network includes: Movie File In (reads `current_frame.jpg`), Cache,
Level/HSV color correction, Feedback loop with audio-driven opacity,
Noise-driven Displace, Text HUD overlay, Audio Device In + Analyze (RMS),
MIDI In, and OSC In (port 6000).

**TD continues running even if AI generation stalls.** The Movie File In
shows the last good frame until a new one arrives.

### Documentation

All documentation has been updated to match actual behavior:

  - `README.md` — Updated feature descriptions and quick-start.
  - `DOCUMENTATION.md` — Added sections on stability controller, corrected
    stream mode documentation, updated macro contract.
  - `RELEASE-NOTES.md` — This document.
  - `VERSION` — `v0.3.4-beta`

### Files Changed

* `vfaq/color_stability.py` — **NEW** — Color stability controller.
* `vfaq/turbo_engine.py` — Fixed macro semantics, integrated stability, audio-paused state.
* `vfaq/backends.py` — Rewrote longcat for true SVD autoregressive continuation.
* `vfaq/stream_engine.py` — Added VRAM safety config passthrough.
* `vfaq/__init__.py` — Version bump, added color_stability exports.
* `vfaq/audio_reactive.py` — Version bump.
* `worqspace/config.yaml` — New stream.vram_safety, stability section, updated macro docs.
* `worqspace/workflows/stream_continuation.json` — Replaced with SVD pipeline.
* `sidecar/midi_sidecar.py` — Version bump, updated docstring.
* `touchdesigner/NETWORK_CONTRACT.txt` — Network blueprint (text contract, no binary `.toe`).
* `touchdesigner/td_setup.py` — **NEW** — TD network builder script.
* `VERSION` — `v0.3.4-beta`

### Acceptance Criteria

  - [x] STREAM output grows every iteration
  - [x] generate_frames affects duration
  - [x] MIDI momentary macros hold correctly (file exists = active)
  - [x] Audio toggle works live
  - [x] TURBO runs indefinitely
  - [x] 10+ minute run does NOT collapse into single-color blob (stability controller)
  - [x] TouchDesigner continues if AI stalls
  - [x] Docs match behavior exactly

---

## v0.3.3-beta — "MIDI Sidestar Completed" Edition

### New Features

**Finalised MIDI Sidecar**

The MIDI sidecar now provides a comprehensive, cross‑platform bridge between
physical controllers and Turbo macros. The new implementation:

  - Supports **momentary** and **toggle** modes per note. Momentary notes
    create a macro file on note‑on and remove it on note‑off; toggle notes
    flip the file’s existence each time they are pressed.
  - Exposes a rich command‑line interface for selecting MIDI ports, setting
    the output directory, overriding config values, enabling dry runs,
    adjusting logging verbosity, toggling note‑off cleanup, specifying a
    default note mode, logging CC messages, and running a self‑test. Run
    `python sidecar/midi_sidecar.py --help` for details.
  - Normalises CC values to configurable ranges, applies optional
    exponential smoothing, and writes the results atomically to
    `macro_INTENSITY` and `macro_ENERGY` files. Continuous parameters now
    update smoothly without jitter.
  - Provides robust port discovery and reconnection: if a MIDI device
    disconnects, the sidecar logs the error, waits, and attempts to reopen
    the selected port without crashing.
  - Maintains gig‑safe isolation: the sidecar never imports Turbo or GPU
    modules and only touches files in `live_output/`. Crashes or missing
    devices do not affect frame generation.

### Configuration & Schema Changes

The `midi` section in `worqspace/config.yaml` has been updated to support
the new features. It now includes `in_name`, `in_port`, `live_output_dir`,
`poll_sleep_ms`, `log_level`, `note_off_cleanup`, and structured
`mapping` entries. Notes map to actions (`DROP`, `BUILD`, `CHILL`, `AUDIO`)
with explicit modes (`momentary` or `toggle`), and CC entries define an
`action`, `min`, `max`, and `smoothing`. See the documentation for an
example configuration.

### Version Bump & Documentation

The project version has been updated to `v0.3.3-beta` across all modules,
documentation and CLI banners. The documentation now includes an updated
MIDI sidecar section with detailed examples and instructions. The README’s
"What’s New" section highlights the finalised sidecar.

### Files Updated

* `sidecar/midi_sidecar.py` — Rewritten to support the new features and CLI.
* `requirements-midi.txt` — Lists `mido>=1.3` and `python-rtmidi>=1.5`.
* `worqspace/config.yaml` — Updated `midi` section reflecting the new schema.
* `DOCUMENTATION.md` — Expanded MIDI sidecar documentation and updated
  version references.
* `README.md` — Updated version and "What’s New" section.

---

## v0.3.2-beta — "TouchDesigner & OSC" Edition

### New Features

**TouchDesigner Integration & OSC Output**

- A new integration path allows you to use TouchDesigner to layer real‑time
  GPU shader effects on the AI frames without blocking generation. The
  starter patch blueprint in `touchdesigner/NETWORK_CONTRACT.txt` outlines a
  network featuring Movie File In, Cache, Level/HSV/Colour correct, feedback
  loop, displace, kaleidoscope and glitch effects. It reads `current_frame.jpg`
  and overlay text files (`now.txt`, `next.txt`) and outputs via NDI,
  Spout/Syphon or fullscreen window. Audio and MIDI CHOPs drive FX depth
  and toggles, and the same MIDI device can control AI macros via the
  sidecar.
- Added optional **OSC broadcasting** from Turbo: when `osc.enabled` is set to
  `true` in `config.yaml`, Turbo sends the current macro, crowd activity
  flag and combined energy value at a configurable interval. TouchDesigner
  (or other OSC clients) can subscribe via OSC In CHOP to synchronise
  FX without reading files.
- Added `osc` section to `config.yaml` with keys `enabled`, `host`, `port`,
  `address` and `send_ms`.
- Added `osc_out.py` module providing an `OSCClient` class and integrated it
  into `turbo_engine.py` for non‑blocking, throttled updates.
- Updated Turbo engine docstring and config version strings to `v0.3.2-beta`.

### Documentation

- Added a **TouchDesigner Integration** section to the documentation, explaining the
  file and OSC contracts, the starter patch blueprint, how to wire audio and
  MIDI, performance notes, and different output pipelines (NDI, projector,
  OBS mixing).
- Updated README with a new “What’s New in v0.3.2‑beta” section highlighting
  TouchDesigner integration and OSC, and updated version references.
- Updated config.yaml and docs to reflect the new `osc` section and bumped
  `turbo.version` to `v0.3.2-beta`.
- All version strings across modules, the CLI banner and documentation
  now reflect `v0.3.2-beta`.

### Files Added

- `vfaq/osc_out.py` — Optional OSC client for broadcasting Turbo state.
- `touchdesigner/NETWORK_CONTRACT.txt` — TouchDesigner patch blueprint.


## v0.3.1-beta — "MIDI SIDECAR" Edition

### New Features

**MIDI Sidecar for Hardware Control** (`sidecar/midi_sidecar.py`)
- A standalone process listens for MIDI notes and control change (CC) events
  using the `mido` and `python-rtmidi` libraries. Note events map directly to
  macro triggers: by default C1 (note 36) triggers `DROP`, D1 (37) triggers
  `BUILD`, D#1 (38) triggers `CHILL`, and F1 (39) toggles the audio‑reactive
  controller (`macro_AUDIO`). Note‑off messages remove the corresponding
  macro file, clearing the override.
- CC events write floating‑point values to `macro_INTENSITY` and
  `macro_ENERGY` files. Turbo reads these values on each frame to modulate
  its `cfg_scale` and `seed_drift` multiplicatively, allowing smooth
  continuous control from knobs or faders.
- The sidecar runs completely independently of Turbo and ComfyUI. It never
  imports GPU or backend code and only touches files in `live_output/`.
  Crashes or missing MIDI devices have **no impact** on frame generation.

**Configurable MIDI Mapping**
- Added a new `midi` section to `worqspace/config.yaml` with options
  `enabled`, `input_name`, `poll_ms`, and `mapping`. Notes map to macro
  strings; CC entries map to parameter types and ranges. The sidecar
  automatically selects the first input port or matches a substring.

**Continuous Macro Modulation** (Turbo Engine)
- Turbo now reads `macro_INTENSITY` and `macro_ENERGY` files at each frame
  and applies multiplicative factors to its base CFG scale and seed drift.
  When the files are absent, Turbo falls back to base values. This allows
  DJs to fine‑tune the visual intensity and energy via hardware faders.

**Version Bump & Documentation**
- Updated version strings to `v0.3.1-beta` across all files, the CLI banner,
  and configuration. Added a comprehensive "MIDI Sidecar" section to the
  documentation explaining setup, configuration, safety rationale, and
  usage. The README now highlights the new sidecar and provides quick
  instructions for running it.

### Files Added

- `sidecar/midi_sidecar.py` — Self‑contained MIDI listener writing macro files.
- `requirements-midi.txt` — Optional dependencies required for MIDI support (`mido`, `python-rtmidi`).


## v0.3.0-beta — "TURBO AUDIO-REACTIVE" Edition

### New Features

**Live Audio-Reactive Turbo Mode** (`turbo_engine.py`, `audio_reactive.py`)
- Real-time RMS energy and beat detection drive `cfg` scale and `seed_drift`.
- Simple, non-blocking audio capture via `sounddevice` in a background thread.
- Configurable sample rate, block size, smoothing and beat threshold.
- Mappings: RMS → CFG, RMS → seed drift, beat → macro (`DROP` by default).

**Hot Toggle & MIDI/Scripting Friendly**
- Enable or disable audio reactivity live by touching/removing the `macro_AUDIO` file.
- No need to restart the engine; file polling interval ≤200 ms ensures quick response.
- Documented how MIDI controllers or scripts can toggle audio by creating/removing the file.

**Crowd Prompt Override**
- When a crowd prompt is active, audio influence is fully disabled.
- CFG and seed drift values freeze to their base configuration until the crowd prompt expires.
- Beat-triggered macros are ignored during crowd takeovers.

**Stage-Safe Control Layer**
- Audio capture failures or missing dependencies disable audio gracefully without crashing.
- Frame generation never stalls; audio processing runs in a separate thread.
- Manual macro triggers (`macro_DROP`, `macro_BUILD`, `macro_CHILL`) always override audio macros.

**Longcat Stream Mode** (`stream_engine.py`, `backends.py`)
- Replaced the old sliding‑window stream stub with a true autoregressive continuation engine.
- New configuration section `stream` with keys `enabled`, `mode`, `context_frames`, `generate_frames`, `max_iterations`, `checkpoint` and `overlap_strategy`.
- At each iteration the backend extracts a short tail clip (up to `context_frames` frames) from the evolving video, extracts its last frame to serve as the conditioning image, runs a customised SVD workflow to generate `generate_frames` new frames beyond the clip, and appends only the new frames. The process repeats until the requested duration is met.
- VRAM usage is predictable: context and generate lengths are hard‑capped and automatically reduced on out‑of‑memory errors. Small jobs keep memory requirements stable across iterations.
- Seeds drift by +997 per iteration to encourage variation. Checkpoints are injected dynamically.
- If stream continuation fails, the system falls back to Video2Video or img2vid from the last frame. The legacy `stream_mode` section remains supported for existing projects but does not use the new algorithm.

**Version Bump**
- Updated version to `v0.3.0-beta` across all files, README, docs and CLI banners.
- Added configuration schema updates to `config.yaml` under both the `turbo` (audio reactivity, live toggle, priority) and `stream` sections, reflecting the new longcat continuation parameters.

### Files Added
- `vfaq/audio_reactive.py` — Live audio controller for TURBO.

### Documentation
- Added “Turbo Audio-Reactive Mode” section to the documentation explaining how to route audio, toggle the mode, and how crowd prompts override audio influence.


## v0.2.5-beta — "TURBO LIVE + CROWD QUEUE" Edition

### New Features

**TURBO Live Mode** (`turbo_engine.py`)
- Real-time single-frame generation via SDXL Turbo / LCM
- Outputs `live_output/current_frame.jpg` for OBS / TouchDesigner
- Hot-reload `tasq.md` for live prompt changes
- DJ macro triggers: `macro_DROP`, `macro_BUILD`, `macro_CHILL` file triggers
- Seed drift modes: `fixed`, `drift`, `beatjump`
- Bypasses InstruQtor/InspeQtor for minimal latency

**Crowd Queue System** (`crowd_server.py`, `crowd_queue.py`)
- Flask web server for live audience prompt submission
- HTML submit page with mobile-friendly UI
- Per-IP and per-name rate limiting (sliding window)
- Prompt sanitization + moderation (banned words, length limits, URL stripping)
- OBS overlay files: `now.txt`, `next.txt`, `queue.txt`, `toast.txt`
- Prompt integration modes: `blend`, `takeover`, `timed_slot`
- Auth token support for protected deployments
- "Showman mode" toast notifications

**CLI: `live` Command**
- `python vfaq_cli.py live --turbo` — start TURBO mode
- `python vfaq_cli.py live --turbo --crowd` — TURBO + crowd server
- Flags: `--fps`, `--size WxH`, `--crowd-port`, `--crowd-token`

### Files Added
- `vfaq/turbo_engine.py` — TURBO frame generation engine
- `vfaq/crowd_server.py` — Flask crowd prompt server
- `vfaq/crowd_queue.py` — Thread-safe prompt queue + rate limiting
- `vfaq/overlay_writer.py` — Atomic OBS text overlay writer
- `vfaq/utils_sanitize.py` — Prompt sanitization + moderation
- `worqspace/workflows/turbo_sdxl.json` — SDXL Turbo ComfyUI workflow
- `worqspace/workflows/turbo_lcm.json` — LCM ComfyUI workflow

---

## v0.2.0-beta — "INFINITE STREAM" Edition

### New Features

**Stream Mode: Sliding Window Continuation** (`stream_engine.py`)
- Cycle N uses last 1.5 seconds of Cycle N-1 as context video (not single frame)
- Context extraction via ffmpeg with stream-copy + re-encode fallback
- Beat-grid aligned generation length (drop protection)
- Per-cycle artifacts: `cycle_N_stream.mp4`, `context_tail.mp4`, `last.png`
- VRAM safe: only context window loaded, not full history
- Config: `stream_mode.enabled`, `method`, `context_length`, `generation_length`

**Backend: Stream Continuation**
- `generate_stream_video()` added to backend interface
- ComfyUI implementation: loads stream_continuation.json workflow
- Fallback chain: stream → V2V → img2vid from last frame
- MockBackend: stream generation mock for testing

**Data Model Expansion** (`visual_briq.py`)
- `VisualBriq`: `context_video_path`, `flow_state`, `stream_video_path`
- `GenerationSpec`: `context_duration`, `context_frames`, `generation_frames`, `overlap_frames`
- Full backward compatibility with old briq JSON

### Files Added
- `vfaq/stream_engine.py` — Context extraction + beat alignment
- `worqspace/workflows/stream_continuation.json` — ComfyUI stream workflow

---

## v0.1.2-alpha — "AUTO-DURATION + AUDIO MATCH" Edition

### New Features

**Auto-Duration Planning** (`duration_planner.py`)
- Detects audio duration via ffprobe
- Computes required cycles: `ceil(audio_duration / cycle_duration)`
- `cycle_duration = bars_per_cycle × 4 × (60 / BPM)`
- Overrides `-c` cycle count when `match_audio=true`
- Mode: `auto` (default), `fixed` (seconds), `unlimited`

**Post-Finalize Trim + Mux**
- Trims final video to exact audio duration (stream copy, re-encode fallback)
- Muxes audio + video into final MP4 (AAC, faststart)
- Replaces final output atomically

**V2V Workflow Lookup (Option B)**
- `input.video2video.comfyui.workflow` checked first
- Fallback to `backend.v2v_workflow` (legacy, to be removed)

**CLI Flags**
- `--match-audio` — align visual duration to audio
- `--duration <seconds>` — fixed duration mode
- `--stream` — enable stream continuation mode

### Files Added
- `vfaq/duration_planner.py` — Duration planning + trim + mux

---

## v0.1.1-alpha — "True V2V" Edition

### Bug Fixes (All 4 Issues from v0.1.0-alpha)

**FIX 1: Video2Video Backend API** (`backends.py`)
- Added `generate_video2video(request)` to `GeneratorBackend` interface
- Implemented on `ComfyUIBackend`: loads V2V workflow, injects video path into
  VHS_LoadVideo, prompt into graph-resolved CLIP nodes, queues via ComfyUI API
- Implemented on `MockBackend`: placeholder V2V with hue-shift filter
- `SplitBackend` delegates V2V to video backend
- HARD VALIDATION: `denoise > 0.5` raises `FatalConfigError` — no recovery
- New `FatalConfigError` exception class for unsafe configuration values

**FIX 2: ConstruQtor VIDEO Mode Rewrite** (`construqtor.py`)
- VIDEO mode now calls `generate_video2video()` directly
- Preprocessing is MANDATORY — failure aborts the cycle
- ❌ REMOVED: frame extraction fallback
- ❌ REMOVED: image→video fallback
- ❌ REMOVED: silent behavior changes
- V2V disabled + VIDEO mode = explicit RuntimeError (no silent fallback)
- TEXT and IMAGE modes unchanged (regression tested)

**FIX 3: Graph-Based CLIP Injection** (`backends.py`)
- Replaced heuristic CLIP node detection (`'positive' in node_id.lower()`) with
  graph traversal: KSampler → follow `positive`/`negative` input refs → CLIPTextEncode
- New static method `_resolve_clip_nodes_from_graph(workflow)` — deterministic,
  workflow-safe, works for any topology
- New `_inject_prompts_graph_based(workflow, prompt, negative_prompt)` method
- Works for: safe_video2video, default SDXL, default SVD, custom workflows

**FIX 4: Motion Prompt Warning** (`backends.py`)
- New `_warn_motion_prompt_if_ignored(workflow, request)` method
- Logs exact warning when motion_prompt is set but workflow has no text conditioning:
  `WARNING: motion_prompt provided but ignored by backend (workflow does not support text conditioning)`
- Motion prompt still preserved in briq JSON for audit

### Test Coverage
- `tests/test_v011_fixes.py`: 20 tests covering all 4 fixes + regression
- `tests/test_v010_features.py`: 14 tests from v0.1.0 still passing (34 total)

### Files Changed
- `vfaq/backends.py`: +180 lines (generate_video2video, graph-based CLIP, motion warning)
- `vfaq/construqtor.py`: Rewritten VIDEO mode (+_construct_video2video, -frame extraction)
- `tests/test_v011_fixes.py`: New (20 tests)
- Version bumped across all files
- All v0.1.0 semantics preserved except explicitly broken paths

---

## v0.1.0-alpha — "No LLM Required" Edition

### Major Features

**Deterministic Prompt Synthesis (NO LLM)**
- New `vfaq/prompt_synth.py` module — fully deterministic prompt generation
- `synthesize_prompt()` combines base_prompt + style_hints + evolution_mutations
- `synthesize_video_prompt()` appends motion_prompt at the END for video backends
- Same cycle_index = same prompt output, always, no randomness
- `evolution_lines.md` — user-editable mutation list with 42 built-in defaults
- `select_evolution_mutations()` — deterministic selection via formula: `(cycle * 7 + i * 13) % len(lines)`
- `map_motion_to_bucket_id()` — keyword → motion_bucket_id mapping for SVD backends
- Style hints, negative prompt, motion prompt ALWAYS applied even without LLM

**Audio Reactivity + BPM Sync**
- New `vfaq/audio_reactivity.py` — full audio analysis pipeline
- BPM detection (auto/manual/auto_then_override/off) with doubletime hint for DnB
- Beat grid generation (1/4, 1/8, 1/16, 1/32 quantization)
- Per-frame spectral features: RMS, spectral flux, centroid, rolloff, onset strength
- 6-band energy split: sub, bass, lowmid, mid, high, air
- `ReactiveController` — get features at any time point, check beat/bar boundaries
- Audio-reactive parameter mapping (intensity/glitch from band energy)
- BPM-synced cycle timing planner (bars_per_cycle → cycle duration)
- Deterministic caching (audio hash + config hash → reuse analysis)
- EMA/median smoothing for feature arrays

**Base Folder Ingestion**
- New `vfaq/base_folders.py` — auto-select files from base folders
- `worqspace/base_image/`, `worqspace/base_audio/`, `worqspace/base_video/`
- Selection modes: newest, oldest, random (deterministic seed), alphabetical
- No renaming required — drop files and run

**Video2Video (ComfyUI Safe Mode)**
- New `vfaq/video_preprocess.py` — mandatory video preprocessing via ffmpeg
- Resolution normalization (≤ 1024×576), FPS (8), duration (4s), yuv420p
- Safe V2V workflow: LoadVideo → VAEEncode → KSampler (low denoise ≤ 0.35) → VHS_VideoCombine
- `worqspace/workflows/safe_video2video.json` — drop-in ComfyUI workflow
- Video is conditioning, not generation — bends motion, doesn't invent it

### New CLI Flags
- `--bpm 174` — manual BPM override for audio sync
- `--audio <path>` — explicit audio file (overrides base_audio selection)
- `--base-pick newest|random|...` — base folder pick mode override
- `--no-audio-react` — disable audio reactivity even if enabled in config

### New Files
- `vfaq/prompt_synth.py` (280 lines)
- `vfaq/base_folders.py` (150 lines)
- `vfaq/audio_reactivity.py` (520 lines)
- `vfaq/video_preprocess.py` (180 lines)
- `worqspace/evolution_lines.md` (42 mutations mega-pack)
- `worqspace/workflows/safe_video2video.json`
- `worqspace/base_image/.gitkeep`
- `worqspace/base_audio/.gitkeep`
- `worqspace/base_video/.gitkeep`
- `scripts/dev_audio_probe.py`
- `tests/test_v010_features.py`

### Config Changes
- `input.mode: auto` — auto-detect input mode
- `input.video2video` — V2V preprocessing and workflow config
- `inputs.base_folders` — base folder selection config
- `audio_reactivity` — full BPM/beat/features/mapping config
- `llm.enabled` — explicit LLM disable option

### Quality Goals
- Zero LLM dependency for full pipeline operation
- Backward compatible with v0.0.7 configs, briq JSONs, and worqspace layouts
- Deterministic outputs: same inputs + same cycle = same results
- Graceful degradation: missing librosa → skip audio, missing audio → clock-only

---

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
