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

4. Please remove ANYTHING related to the LTX Video backend. we do not ever want to use this backend anymore. do not break any existing functionality while removing LTX Video completely please.

5. Please remove EVERYTHING related to qwen-image-python and qwen-image and mock. all three we will never ever use again and we do not want to use these backends anymore. do not break any existing functionality while removing these three.

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

     Please remove the instruqtor, construqtor, inspeqtor and prompt_bundle if they are full dead code, and remove any reference to them from the docs, please. do not break any working existing functionality when doing so.

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

    please remove all these unused scripts completely if they are really not used at all or called, as we did not have any issue anymore with obs not playing. from the scripts we do use like obs-swap.py please remove just the parts that are not being used like the video validator. do not break any other working existing functionality.

11. **`calculate_blur` / `calculate_entropy` in `image_metrics.py` are unused (VERIFIED)**
    - Only `calculate_frame_similarity` is referenced (by `sliding_story_engine`).
    ok fix the entropy and image metrics calculators to be working/used/optionally used if we turn them on in the config, make them configurable then. do not break any other working functionality when doing so.

12. **`vfaq/venice_backend.py.patch` is committed development debris (VERIFIED tracked)**
    - A 33-line unified diff whose `--- a/venice_backend.py` path no longer matches
      the actual `vfaq/venice_backend.py` location.
    - Should be removed from the repo.

    please remove that from the repo, and do not break anything working other functionality.

13. **Duplicate ≈1 MB splash images committed**
    - `visual-faqtory.jpg` (836 KB) and `vfaq/visual-faqtory.jpg` (917 KB) are two
      *different* images. Only `visual-faqtory.jpg` is referenced (`README.md`).
    - `vfaq/visual-faqtory.jpg` appears orphaned; together they add ~1.75 MB of
      image bloat to every clone.

    - keep the best one and remove the orphaned one please and do not break anything when doing so.

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

    fix all these issues properly and do not break any working functionality when doing so.

15. **`base_folders.py` default folder names contradict `visual_faqtory.py` (VERIFIED)**
    - `base_folders.select_base_files` defaults to `base_image`, `base_audio`,
      `base_video` (singular) directories.
    - `visual_faqtory._detect_inputs` scans `base_images`, `base_audio`, `base_video`
      (plural), matching the README's `worqspace/base_images/`.
    - `base_folders.py` is dead code (issue #8), but if revived it would look in
      directories that don't match the active pipeline's naming.

    just remove the base_folders.py please and only keep the detect inputs stuff thats all we need I think. if we hae an image or audio or video in one of those base_ folders it should auto detect and do it properly. do not break any existing functionality when doing so.

16. **`VeoBackend` does not inherit from `GeneratorBackend` (VERIFIED)**
    - `class VeoBackend:` (no base class), although its docstring says it
      "Implements the GeneratorBackend interface".
    - It's duck-typed (`generate_image`, `generate_video`, `generate_morph_video`,
      `check_availability`), so the factory still works, but `isinstance(x,
      GeneratorBackend)` checks and any future abstract-interface guarantees fail.

    please make sure this is all fixed properly. and do not break any existing functionality.

17. **Stale version strings scattered across docstrings/comments**
    - `base_folders.py`, `prompt_bundle.py`, `prompt_synth.py`,
      `instruqtor.py`, `construqtor.py`, `inspeqtor.py` all still say
      "Part of Visual FaQtory v0.5.6-beta" (or v0.0.7-alpha / v0.1.1-alpha /
      v0.2.0-beta).
    - `vf-crowd-control.env.example` header says "Part of QonQrete Visual FaQtory
      v0.8.4-beta".
    - `sliding_story_engine.py` docstring says "Part of Visual FaQtory v0.7.0-beta".
    - The version-consistency test only guards a handful of files, so these are missed.

    clean up all invalid version strings please.

18. **Invalid lowercase `any`/`callable` type annotations (VERIFIED)**
    - `sliding_story_engine.py` uses `Dict[str, any]` (lines 134–139) and
      `Optional[callable]` (line 937) — lowercase names that do not exist.
    - Masked at import time by `from __future__ import annotations`, but
      `typing.get_type_hints()` (or any runtime type introspector) raises
      `NameError`/is incorrect for these signatures.

    please fix all these accordingly, do not break anything else along the way.

19. **`_create_single_backend` factory mapping is incomplete/inconsistent**
    - `'venice': None` is populated in the dict and then explicitly handled +
      popped; `veo` is handled via a separate `if` but absent from the dict — works
      but is confusingly structured.
    - Missing `ltx_video`, `qwen_image_python`, `qwen_python` (see #4/#5).
    - `BackendType` enum omits `LTX`/`QwEN_PYTHON`, while the CLI advertises them.

    make sure ltx video, qwen_image_python and qwen_python or qwen_image are fully removed from the repo and every reference to them. in any capitalization. we dont use them and will not ever use them again. make sure you do not break anything else.

20. **`_concat_stream_copy` and `_concat_reencode` share `_concat_list.txt`**
    - Both write the same temp concat file. They are always called sequentially
      (not concurrently), so this is not currently a data race, but it's fragile.

    make this more robust please and do not break anything

21. **NVENC encoder args differ between `inspeqtor.py` and `finalizer.py`**
    - `inspeqtor` `_encode_args` uses `-rc vbr -cq N -b:v 0 -profile:v high -preset p5`.
    - `finalizer` `_get_encoder_args` uses only `-cq N -preset p5 -pix_fmt yuv420p`.
    - Since `inspeqtor.py` is dead code this is harmless today, but the divergence
      is another sign of the legacy agent split.

    inspeqtor is not being used anymore I thought, so if thats the solution to remove inspeqtor.py then do so. make sure that doesnt break anything.

---

## 🟡 LOW — Cosmetic / minor

22. **Leftover stray `print()` in `cmd_backends` (VERIFIED)**
    - `vfaq_cli.py` `cmd_backends` ends with an orphaned
      `print("    # and set comfyui_workflow_t2v / comfyui_workflow_i2v")` that is
      clearly a truncated comment/instruction fragment with no context.

23. **`_generate_video` in `construqtor.py` has a redundant expression**
    - `motion = getattr(briq, 'motion_prompt', None) or None` — the trailing `or None`
      is a no-op for a string attribute.

    construqtor.py is being removed so clean this up. do not break anything.

24. **`image_metrics.py` uses `print()` instead of `logging` (VERIFIED)**
    - `calculate_frame_similarity`, `calculate_blur`, `calculate_entropy` all
      `print(...)` on error, inconsistent with the rest of the codebase's logging.
    - `calculate_blur` returns `0.0` on error with a comment "0 blur (sharp)", which
      is the opposite of the documented meaning ("higher value = more blur").
    - `calculate_blur`/`calculate_entropy` also divide by a hardcoded `max-variance`
      heuristic and emit an unused `min_variance` path.

    fix these accordingly without breaking anything.

25. **CLI `-r` short flag overload documented confusingly**
    - `-r` is `--resume`; `--no-reinject` is `-R` (uppercase). The inline comment
      acknowledges this but it's easy to confuse. Consider dropping the `-R` short
      form or documenting loudly.

      ok fix this so we wont get confused anymore.


26. **`venice_backend.py` has an empty `except` with `pass` inside the spinner/timing code**
    - Line 825 area: `pass` in an exception handler, plus a couple of other bare
      `pass` blocks in `_retry_delay` (line 1756). Not bugs by themselves, but they
      silently swallow errors and are worth logging.

      fix this properly please without breaking anything.

27. **Crowd-control port mismatch between config and defaults**
    - `worqspace/config.yaml` says `base_url: "http://127.0.0.1:8000/visuals"` (port 8000),
      while `cmd_crowd`'s default and `vf-crowd-control.env.example` use port 8808.
      The comment in config.yaml even notes "(currently running on port 8000)".

      make the default be 8000 please on both sides and do not break anything.

28. **`visual-faqtory.jpg` / `vfaq/visual-faqtory.jpg` are binary assets in the package dir**
    - `vfaq/visual-faqtory.jpg` (917 KB) is inside the Python package directory and
      is never referenced by code; likely should be removed (also see #13).

      this one should be removed now, if not remove it and do not break anything.

      also im not sure if we do, but I thought we had some audio sync and bpm sync options in the config, double-check if these are working and if not, fix them accordingly without breaking anything else. and at the very last please tell me on the terminal output how audiuo reactivity works before the video is actually made. the bpm sync I can understand though hehe. fix that as well and do not break anything else.


when all above is fixed, please do a version bump to v0.9.4-beta everywhere you can and inside the VERSION file if we have any (or else create it). then you will deep-analyze the codebase again to the maximum detail and then please update the full documentation accordingly to the current state. then also create/modify the QUICKSTART.md for less technical people to add a easy to follow short step-by-step guide how to use visual faqtory from scratch for multiple goals: promo video generation for people's own tracks that they made or whatever they wanna use it for, live mode with crowd control, or any other modes if we have any.

then please git add commit and push the whole thing.
