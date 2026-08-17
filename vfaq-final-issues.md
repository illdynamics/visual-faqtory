# Visual FaQtory — Deep Codebase Analysis & Issues Report

Generated 2026-08-17 · Branch `main` @ `4d8eb2a` ("fixes")
Version: `v0.9.3-beta` · ~16k lines Python + Bash shell ops

Results of a full manual deep-analyze of the repository, cross-checked by
verifying suspected defects at runtime and by running the test suite
(`75 passed, 9 skipped`). Issues are grouped by severity. Findings marked
**VERIFIED** were reproduced in this session; others are static-analysis
observations.

---

## 🔴 CRITICAL — Hardcoded credentials committed to source control

These are secrets now permanently embedded in the git history. Even though
some are local-dev defaults, they must be rotated/removed and treated as
compromised.

1. **OBS WebSocket password `"Setyup34!"` hardcoded in multiple files**
   - `obs-ids.py` — `PASSWORD = "Setyup34!"`
   - `obs-test.py` — `PASSWORD = "Setyup34!"`
   - `obs-swap.py:63` — `PASSWORD = os.environ.get("OBS_PASSWORD", "Setyup34!")` (unfortunate default)
   - `vf-obs-watcher-same-machine.sh` lines 150, 178, 335 — `os.environ.get('OBS_PASSWORD', 'Setyup34!')` (three occurrences)
   - `obs-swap.py` docstring also advertises the password as the default.
   - **Impact:** anyone with repo access can connect to the OBS WebSocket and take
     over the live stream controls (scene item swapping, media control).

2. **Crowd-control pop token hardcoded in committed config**
   - `worqspace/config.yaml` — `pop_token: "Setyup34Setyup34Setyup34Setyup34"`
   - This is the Bearer token protecting the `/api/next` claim endpoint. A committed
     static token makes the "protected" claim/replay/ack API effectively public to
     anyone who finds the repo.

**Recommendation:** move all secrets to environment variables / `.env` (already
gitignored), rotate the OBS password and the crowd token, and consider expunging
the committed values from history (`git filter-repo` / BFG) since this is a live
streaming control surface.

---

## 🔴 HIGH — Missing modules referenced by the runtime/API

3. **`vfaq/llm_utils.py` does not exist (VERIFIED)**
   - Imported by:
     - `instruqtor.py:175` `from .llm_utils import create_llm_client`
     - `instruqtor.py:554` / `:587` `from .llm_utils import call_llm`
     - `inspeqtor.py:28` `from .llm_utils import create_llm_client, call_llm` (wrapped in try/except)
   - `inspeqtor.py` guards the import, but `instruqtor.py` does **not** (its imports
     only run when `llm_provider`/`self.llm` is truthy, so the path is dormant unless LLM is configured).
   - When `llm_provider` is passed to `InstruQtor`, the constructor calls the
     missing `create_llm_client` and raises `ModuleNotFoundError`. In `InspeQtor`,
     the `try/except ImportError` sets `create_llm_client = None`, so passing a
     provider raises `TypeError: 'NoneType' object is not callable`.
   - **Impact:** any LLM-enabled configuration crashes. The LLM feature is
     advertised but cannot work as shipped.

4. **`vfaq/ltx_video_backend.py` does not exist (VERIFIED)**
   - Referenced by `backends.py` module docstring ("LTXVideoBackend ... see
     ltx_video_backend.py"), referred to throughout `sliding_story_engine.py`
     (`morph_is_ltx`, LTX branches), and advertised in `vfaq_cli.py` help text and
     `cmd_backends` ("Active LTX Config Check").
   - `_create_single_backend` has no `ltx_video` entry → selecting `type: ltx_video`
     silently falls through to `mock` ("Unknown backend … using mock").
   - **Impact:** LTX-Video is documented as a supported backend but is entirely
     absent; misconfigured users silently get mock output.

5. **`qwen_image_python` / `qwen_python` backends do not exist (VERIFIED)**
   - `vfaq/__init__.py` header claims: "NEW: image-only qwen_image_python /
     qwen_python local inference backend".
   - `vfaq_cli.py:358` help text advertises `qwen_image_python/qwen_python`.
   - `_create_single_backend` only registers `qwen_image_comfyui` and `qwen_image`
     (a legacy alias). There is no `qwen_image_python`/`qwen_python` branch.
   - **Impact:** two advertised backend types silently drop to `mock`.

---

## 🔴 HIGH — Undefined names / latent crashes

6. **`traceback` used but never imported in `sliding_story_engine.py` (VERIFIED)**
   - Four occurrences of `traceback.format_exc()`, lines 1827, 2396, 2622, 2874.
   - `import traceback` is absent from the module's imports.
   - These calls sit in the `except Exception` handlers for last-frame extraction.
   - **Impact:** when last-frame extraction fails, the error-handling path itself
     raises `NameError: name 'traceback' is not defined`, masking the real error and
     turning a non-fatal frame-extraction warning into a hard crash of the cycle.

7. **`VisualBriq.from_dict` references non-existent `GenerationSpec` fields (VERIFIED)**
   - `visual_briq.py` (in `from_dict`) does:
     ```python
     d['spec'].setdefault('context_duration', 1.5)
     d['spec'].setdefault('context_frames', None)
     d['spec'].setdefault('generation_frames', None)
     d['spec'].setdefault('overlap_frames', 0)
     ```
   - `GenerationSpec` has none of these fields. Reproduced:
     `TypeError: GenerationSpec.__init__() got an unexpected keyword argument 'context_duration'`.
   - **Impact:** deserializing a `VisualBriq` whose persisted `spec` contains any of
     these keys (e.g. stream-mode briq JSON) crashes. `VisualBriq.from_dict` /
     `.load()` are part of the exported public API.

---

## 🟠 MEDIUM — Dead / orphaned code that misleads users

8. **The entire "agent trio" + companion modules are dead code (VERIFIED)**
   - `instruqtor.py`, `construqtor.py`, `inspeqtor.py`, `prompt_bundle.py`,
     `prompt_synth.py`, `base_folders.py`, and `visual_briq.py` are exported by
     `__init__.py` but are **never imported by any live code path**:
     - `visual_faqtory.py` (orchestrator) → `run_sliding_story` directly.
     - `sliding_story_engine.py` has its own prompt/story helpers.
     - No test file references these modules.
   - The README/`__init__` describe an "InstruQtor → ConstruQtor → InspeQtor" agent
     pipeline that is disconnected from the real `paragraph_story` engine.
   - **Impact:** large maintenance surface, misleading architecture docs, and the
     place where the missing `llm_utils` and the `GenerationSpec` bug live.

9. **`VisualFaQtory._run_finalizer()` is dead code (VERIFIED)**
   - Never called; the runtime uses `run_sliding_story` (which finalizes internally)
     plus `_collect_story_outputs()`.
   - It implements a second, divergent finalization naming scheme
     (`final_video.mp4`, `final_video_60fps.mp4`, `final_video_60fps_1080p.mp4`)
     that differs from the live scheme (`final_output.mp4`, `final_60fps_1080p.mp4`).
   - **Impact:** confusing duplicate logic; future edits could accidentally "fix"
     the wrong path.

10. **`diagnostics.py` and `video_validator.py` are entirely unused (VERIFIED)**
    - `DiagnosticsWriter` / `CycleGuard` are never instantiated anywhere.
    - `validate_video_file` / `await_file_stable` are never called (including by
      `obs-swap.py` / the watchers, despite `video_validator.py` claiming to prevent
      the "swapped but not playing" bug class).
    - **Impact:** advertised safety/diagnostics features don't actually run.

11. **`calculate_blur` / `calculate_entropy` in `image_metrics.py` are unused (VERIFIED)**
    - Only `calculate_frame_similarity` is referenced (by `sliding_story_engine`).

12. **`vfaq/venice_backend.py.patch` is committed development debris (VERIFIED tracked)**
    - A 33-line unified diff whose `--- a/venice_backend.py` path no longer matches
      the actual `vfaq/venice_backend.py` location.
    - Should be removed from the repo.

13. **Duplicate ≈1 MB splash images committed**
    - `visual-faqtory.jpg` (836 KB) and `vfaq/visual-faqtory.jpg` (917 KB) are two
      *different* images. Only `visual-faqtory.jpg` is referenced (`README.md`).
    - `vfaq/visual-faqtory.jpg` appears orphaned; together they add ~1.75 MB of
      image bloat to every clone.

---

## 🟠 MEDIUM — Config/schema mismatches and correctness issues

14. **`finalizer.quality.encoder_preference` is silently ignored (VERIFIED)**
    - `worqspace/config.yaml` sets:
      ```yaml
      quality:
        crf: 11
        encoder_preference: auto
      ```
    - `Finalizer.__init__` reads `self.encoder_preference = cfg.get('encoder_preference', ...)`
      (top-level), never `cfg['quality']['encoder_preference']`.
    - **Impact:** the `encoder_preference: auto` inside `quality` has no effect; the
      effective preference is the top-level list `["h264_nvenc", "libx264"]`.

15. **`base_folders.py` default folder names contradict `visual_faqtory.py` (VERIFIED)**
    - `base_folders.select_base_files` defaults to `base_image`, `base_audio`,
      `base_video` (singular) directories.
    - `visual_faqtory._detect_inputs` scans `base_images`, `base_audio`, `base_video`
      (plural), matching the README's `worqspace/base_images/`.
    - `base_folders.py` is dead code (issue #8), but if revived it would look in
      directories that don't match the active pipeline's naming.

16. **`VeoBackend` does not inherit from `GeneratorBackend` (VERIFIED)**
    - `class VeoBackend:` (no base class), although its docstring says it
      "Implements the GeneratorBackend interface".
    - It's duck-typed (`generate_image`, `generate_video`, `generate_morph_video`,
      `check_availability`), so the factory still works, but `isinstance(x,
      GeneratorBackend)` checks and any future abstract-interface guarantees fail.

17. **Stale version strings scattered across docstrings/comments**
    - `base_folders.py`, `prompt_bundle.py`, `prompt_synth.py`,
      `instruqtor.py`, `construqtor.py`, `inspeqtor.py` all still say
      "Part of Visual FaQtory v0.5.6-beta" (or v0.0.7-alpha / v0.1.1-alpha /
      v0.2.0-beta).
    - `vf-crowd-control.env.example` header says "Part of QonQrete Visual FaQtory
      v0.8.4-beta".
    - `sliding_story_engine.py` docstring says "Part of Visual FaQtory v0.7.0-beta".
    - The version-consistency test only guards a handful of files, so these are missed.

18. **Invalid lowercase `any`/`callable` type annotations (VERIFIED)**
    - `sliding_story_engine.py` uses `Dict[str, any]` (lines 134–139) and
      `Optional[callable]` (line 937) — lowercase names that do not exist.
    - Masked at import time by `from __future__ import annotations`, but
      `typing.get_type_hints()` (or any runtime type introspector) raises
      `NameError`/is incorrect for these signatures.

19. **`_create_single_backend` factory mapping is incomplete/inconsistent**
    - `'venice': None` is populated in the dict and then explicitly handled +
      popped; `veo` is handled via a separate `if` but absent from the dict — works
      but is confusingly structured.
    - Missing `ltx_video`, `qwen_image_python`, `qwen_python` (see #4/#5).
    - `BackendType` enum omits `LTX`/`QwEN_PYTHON`, while the CLI advertises them.

20. **`_concat_stream_copy` and `_concat_reencode` share `_concat_list.txt`**
    - Both write the same temp concat file. They are always called sequentially
      (not concurrently), so this is not currently a data race, but it's fragile.

21. **NVENC encoder args differ between `inspeqtor.py` and `finalizer.py`**
    - `inspeqtor` `_encode_args` uses `-rc vbr -cq N -b:v 0 -profile:v high -preset p5`.
    - `finalizer` `_get_encoder_args` uses only `-cq N -preset p5 -pix_fmt yuv420p`.
    - Since `inspeqtor.py` is dead code this is harmless today, but the divergence
      is another sign of the legacy agent split.

---

## 🟡 LOW — Cosmetic / minor

22. **Leftover stray `print()` in `cmd_backends` (VERIFIED)**
    - `vfaq_cli.py` `cmd_backends` ends with an orphaned
      `print("    # and set comfyui_workflow_t2v / comfyui_workflow_i2v")` that is
      clearly a truncated comment/instruction fragment with no context.

23. **`_generate_video` in `construqtor.py` has a redundant expression**
    - `motion = getattr(briq, 'motion_prompt', None) or None` — the trailing `or None`
      is a no-op for a string attribute.

24. **`image_metrics.py` uses `print()` instead of `logging` (VERIFIED)**
    - `calculate_frame_similarity`, `calculate_blur`, `calculate_entropy` all
      `print(...)` on error, inconsistent with the rest of the codebase's logging.
    - `calculate_blur` returns `0.0` on error with a comment "0 blur (sharp)", which
      is the opposite of the documented meaning ("higher value = more blur").
    - `calculate_blur`/`calculate_entropy` also divide by a hardcoded `max-variance`
      heuristic and emit an unused `min_variance` path.

25. **CLI `-r` short flag overload documented confusingly**
    - `-r` is `--resume`; `--no-reinject` is `-R` (uppercase). The inline comment
      acknowledges this but it's easy to confuse. Consider dropping the `-R` short
      form or documenting loudly.

26. **`venice_backend.py` has an empty `except` with `pass` inside the spinner/timing code**
    - Line 825 area: `pass` in an exception handler, plus a couple of other bare
      `pass` blocks in `_retry_delay` (line 1756). Not bugs by themselves, but they
      silently swallow errors and are worth logging.

27. **Crowd-control port mismatch between config and defaults**
    - `worqspace/config.yaml` says `base_url: "http://127.0.0.1:8000/visuals"` (port 8000),
      while `cmd_crowd`'s default and `vf-crowd-control.env.example` use port 8808.
      The comment in config.yaml even notes "(currently running on port 8000)".

28. **`visual-faqtory.jpg` / `vfaq/visual-faqtory.jpg` are binary assets in the package dir**
    - `vfaq/visual-faqtory.jpg` (917 KB) is inside the Python package directory and
      is never referenced by code; likely should be removed (also see #13).

---

## ✅ Verified-good signals

- Test suite green: **75 passed, 9 skipped** (9 skips are live-integration tests
  gated behind `VF_RUN_LIVE_*` / env flags).
- `run_state.py` atomic state writes, resume/discovery, and frame-extraction
  fallbacks are coherent and well-documented.
- `crowd_control` (models/db/client/server/filtering) is self-consistent and
  has solid claim/ack/requeue lifecycle semantics.
- `VeniceConfig` / `VeoConfig` dataclass parsing is defensive (filtering unknown
  keys, normalizing booleans/ints/floats, snapping aspect ratio).
- The `sliding_story_engine.py` orchestrator is feature-complete for ComfyUI /
  Veo / Venice routing (subject to the `traceback` bug above).

---

## Suggested remediation priority

1. Rotate and remove all hardcoded credentials (#1, #2).
2. Add the missing `llm_utils.py` or remove/guard the LLM feature (#3).
3. Fix the `traceback` import in `sliding_story_engine.py` (#6).
4. Reconcile documented backends vs. factory: implement or clearly remove
   `ltx_video`, `qwen_image_python`, `qwen_python` (#4, #5).
5. Fix `VisualBriq.from_dict` / `GenerationSpec` field mismatch (#7).
6. Delete or properly wire up dead modules (#8–#13); at minimum fix docs so the
   architecture description matches reality.
7. Align config schema (`quality.encoder_preference`, base folder names, crowd
   port) (#14, #15, #27).
8. Clean up minor items (#22–#28).
