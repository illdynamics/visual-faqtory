# Visual FaQtory v0.9.3-beta — backend validation report

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
| Venice backend | Yes | Yes | Yes | Yes | No |
| SRT watcher | Yes | Yes | Yes | No | No |
| Crowd Control | Yes | Yes | Yes | No | No |

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

## Crowd Control

**Offline-tested**
- lifecycle management (claim/ack/requeue)
- config parsing and defaults
- SQLite queue integrity
- rate limiting and badword filtering
- generator integration (fail-open)
- smart-reinject interaction

**Evidence**
- Statically inspected: `vfaq/crowd_control/*.py`, `vfaq/sliding_story_engine.py`
- Unit tested: `tests/test_crowd_control_claim_lifecycle.py`, `tests/test_sliding_story_smart_reinject.py`
- Smoke tested: included in the passing repo-root `pytest -q` run below
- Live-tested in this packaging pass: **No**

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
```

**Results:** 75 passed, 9 skipped in 20.25s.

## Re-verified in v0.9.3-beta

This pass re-verified version consistency, documentation alignment, and crowd control lifecycle:

- `pytest -q tests/test_version_consistency.py`
- `pytest -q tests/test_crowd_control_claim_lifecycle.py`
- `pytest -q tests/test_sliding_story_smart_reinject.py`
- `python -m py_compile vfaq_cli.py vfaq/version.py vfaq/venice_backend.py vfaq/__init__.py vfaq/visual_faqtory.py`

No new live Venice or OBS/SRT network integration was performed in this pass. Live-tested status therefore remains unchanged.

