# Visual FaQtory v0.9.0-beta — backend validation report

This report now distinguishes between **offline-tested**, **opt-in live harness available**, and **live-tested in this packaging pass**.

## Validation scale

- **Statically inspected** — code paths, config plumbing, and docs were read end-to-end in this repo.
- **Unit tested** — focused automated tests exist and passed in this pass.
- **Smoke tested** — the offline repo-root pytest suite executed successfully.
- **Opt-in live harness available** — a real external-service pytest path exists but is skipped unless explicitly enabled by env vars.
- **Live-tested in this packaging pass** — real external services were actually exercised while producing this repo drop.

## Status matrix

| Area | Statically inspected | Unit tested | Smoke tested | Opt-in live harness available | Live-tested in this packaging pass |
|---|---|---:|---:|---:|---:|
| Qwen hybrid (Qwen + ComfyUI/SVD split routing) | Yes | Yes | Yes | Yes | No |
| AnimateDiff backend | Yes | Yes | Yes | Yes | No |
| Venice backend | Yes | Yes | Yes | Yes | No |
| SRT watcher | Yes | Yes | Yes | No | No |

## Qwen hybrid

**Offline-tested**
- split-capability config parsing
- image-backend routing to `qwen_image_comfyui`
- separate video/morph backend routing
- resume and reinject expectations
- workflow-aware Qwen prompt/image injection

**Evidence**
- Statically inspected: `vfaq/backends.py`, `vfaq/sliding_story_engine.py`, `vfaq/visual_faqtory.py`, `worqspace/config.yaml`, workflow docs
- Unit tested: `tests/test_qwen_hybrid_validation.py`, `tests/test_backend_routing.py`, `tests/test_resume_and_loop_closure.py`, `tests/test_visual_faqtory_config.py`
- Smoke tested: included in the passing repo-root `pytest -q` run below
- Opt-in live harness available: `tests/test_live_integrations.py::TestLiveComfyIntegrations::test_live_qwen_text2img` and `::test_live_qwen_img2img`
- Live-tested in this packaging pass: **No**

**Known limitation**
- Qwen workflow JSONs are operator-supplied and not bundled. A bad ComfyUI graph can still fail at runtime even when the Python-side routing is correct.

## AnimateDiff

**Offline-tested**
- backend selection and config aliases
- default AnimateDiff graph generation
- custom workflow validation
- timing precedence (fps/frame-count/duration)
- morph contract and error handling
- compatibility with finalizer/resume expectations

**Evidence**
- Statically inspected: `vfaq/backends.py`, `vfaq/sliding_story_engine.py`, `vfaq/visual_faqtory.py`, workflow docs
- Unit tested: `tests/test_animatediff_backend.py`, `tests/test_animatediff_validation.py`, `tests/test_backend_routing.py`, `tests/test_resume_and_loop_closure.py`
- Smoke tested: included in the passing repo-root `pytest -q` run below
- Opt-in live harness available: `tests/test_live_integrations.py::TestLiveComfyIntegrations::test_live_animatediff_img2vid`
- Live-tested in this packaging pass: **No**

**Known limitation**
- The live harness still depends on operator-supplied workflow JSON and a real AnimateDiff node stack being installed in ComfyUI.

## Venice

**Offline-tested**
- config parsing and model selection by mode
- env/config auth handling
- text2img / img2img / text2vid / img2vid mapping
- async polling / timeout / cleanup handling
- hybrid routing compatibility
- output naming and metadata capture

**Evidence**
- Statically inspected: `vfaq/venice_backend.py`, `vfaq/backends.py`, `vfaq/sliding_story_engine.py`, `vfaq/visual_faqtory.py`, config/docs
- Unit tested: `tests/test_venice_backend.py`, `tests/test_backend_routing.py`, `tests/test_visual_faqtory_config.py`
- Smoke tested: included in the passing repo-root `pytest -q` run below
- Opt-in live harness available: `tests/test_live_integrations.py::TestLiveVeniceIntegrations::*`
- Live-tested in this packaging pass: **No**

**Known limitation**
- Venice video remains model-dependent and billable. The harness can validate it, but this packaging pass does not claim those calls were actually run.

## SRT watcher

**Offline-tested**
- watch-dir resolution
- startup preload behavior
- direct-write and atomic-move watch events
- A/B slot warm swap logic
- OBS autoswap on/off control flow
- status/smoke-check/systemd/docs alignment

**Evidence**
- Statically inspected: `vf-obs-watcher-srt-endpoints.sh`, env example, systemd unit, live-visuals docs
- Unit tested: `tests/test_srt_watcher.py`
- Smoke tested: included in the passing repo-root `pytest -q` run below
- Opt-in live harness available: **No**
- Live-tested in this packaging pass: **No**

## Passing commands in this pass

```bash
pytest -q
pytest -q tests/test_live_integrations.py -rs
```

Expected default behavior for the live harness command with no env gates: the live tests are **skipped**, not executed.


## Re-verified in v0.9.0-beta

This pass re-verified version consistency and CLI/version-sourcing cleanup only:

- `pytest -q tests/test_cli.py`
- `pytest -q tests/test_version_consistency.py`
- `python -m py_compile vfaq_cli.py vfaq/version.py vfaq/venice_backend.py vfaq/__init__.py vfaq/visual_faqtory.py`

The exact command `pytest -q tests/test_srt_watcher.py` was attempted in this sandbox, but output capture stalled before a reliable final exit/result could be recorded, so SRT watcher status is **not** newly re-certified by this pass.

No new live ComfyUI, Venice, or OBS/SRT network integration was performed in this pass. Live-tested status therefore remains unchanged.
