# Visual FaQtory v0.9.9-beta — Persistent Wan Daemon Implementation

## Context

Visual FaQtory's `local_backend.py` currently spawns `mlxgen-generate-wan` as a
fresh `subprocess.Popen` for every cycle. This means the T5 text encoder (~1-2 GB)
is loaded from disk on every cycle even though `--keep-text-encoder` is already
wired into `WanRunner._common()`. The `vfaq_cli.py` main loop is a **persistent
in-process for-loop** (confirmed in `sliding_story_engine.py`), so the process
itself stays alive — but each `_run_with_live_progress()` call spawns and kills a
new mlxgen subprocess, defeating `--keep-text-encoder` entirely.

The fix: replace the per-cycle `subprocess.Popen` approach for the `wan` runner
with a **persistent mlxgen-generate-wan daemon process** that accepts jobs over
stdin (JSON lines) and streams progress back over stdout. The daemon stays alive
for the lifetime of the `vfaq_cli.py` session. The T5 encoder loads once on the
first cycle and is reused for every subsequent cycle.

---

## Files to modify

- `vfaq/local_backend.py` — main implementation
- `vfaq/version.py` — bump version
- `VERSION` — bump version
- `tests/test_wan_runner_config.py` — add daemon lifecycle tests
- `tests/test_wan_runner_sampling.py` — ensure existing tests still pass
- `RELEASE-NOTES.md` — add v0.9.9-beta section

---

## Implementation spec

### 1. New class: `WanDaemon` in `local_backend.py`

Add a new class `WanDaemon` that manages a single long-lived
`mlxgen-generate-wan` subprocess. Place it directly above `WanRunner`.

```python
class WanDaemon:
    """
    Manages a single persistent mlxgen-generate-wan process for the lifetime
    of the Visual FaQtory session.

    mlxgen-generate-wan does NOT natively support a daemon/server mode, so we
    achieve persistence by keeping the process alive between generations using
    a job-queue approach: instead of spawning a new process per cycle, we
    keep ONE process running and feed it jobs.

    IMPORTANT: mlxgen-generate-wan is a one-shot CLI tool — it does not have
    a built-in stdin job loop. So our daemon wraps it differently:

    Strategy: process POOL of size 1.
      - On the first generate call, spawn mlxgen-generate-wan with
        --keep-text-encoder. Capture its stdout/stderr live.
      - When the process exits (generation complete), check output and return.
      - On the NEXT generate call, re-use the SAME Python process object IF
        it is still alive, otherwise respawn.

    Wait — mlxgen-generate-wan exits after each generation. So true persistence
    requires mlxgen to support a server mode. Since it does not, we instead:

    1. Keep the TEXT ENCODER loaded in a tiny Python sidecar process that
       pre-warms the T5 model and passes cached embeddings to mlxgen.
       [Too complex — requires mlxgen Python API access.]

    OR

    2. Use mlxgen's Python importable API directly (mflux.cli.mlx_gen:main
       is the entry point) — call it in-process via importlib rather than
       subprocess, keeping the loaded model in memory between calls.

    Strategy 2 is the correct approach. mlx-gen is pip-installed and its
    Python API is accessible. We call the generation function directly in a
    background thread, keeping the model object alive between cycles.
    """
```

**Concrete implementation using in-process Python API:**

```python
import threading
import importlib
import queue as _queue

class WanDaemon:
    """
    Calls mlx-gen's Wan generation pipeline in-process via its Python API,
    keeping the pipeline object (including T5 text encoder) alive between
    cycles. This is the correct way to benefit from --keep-text-encoder
    semantics across cycles in a persistent vfaq_cli.py session.
    """

    def __init__(self, model_spec: str, base_cmd_flags: Dict[str, Any]):
        """
        Args:
            model_spec: Path to the mlxgen model directory.
            base_cmd_flags: Dict of flag→value pairs that are constant across
                            all cycles (width, height, fps, guidance, etc).
                            Per-cycle values (prompt, image_path, output) are
                            passed at generate() time.
        """
        self._model_spec = model_spec
        self._base_flags = base_cmd_flags
        self._pipeline = None          # holds the loaded mlx-gen pipeline
        self._lock = threading.Lock()  # one generation at a time
        self._loaded = False
        self._logger = logging.getLogger(__name__ + ".WanDaemon")

    def _load_pipeline(self):
        """Import and initialise the mlx-gen Wan pipeline. Called once."""
        try:
            # mlx-gen exposes its pipeline via mflux package internals.
            # Attempt the most likely import paths in order.
            pipeline_mod = None
            for mod_path in (
                "mflux.models.wan.wan_pipeline",
                "mflux.wan.pipeline",
                "mflux.cli.mlx_gen",
            ):
                try:
                    pipeline_mod = importlib.import_module(mod_path)
                    break
                except ImportError:
                    continue
            if pipeline_mod is None:
                raise ImportError(
                    "Could not import mlx-gen Wan pipeline. "
                    "Ensure mlx-gen is installed: pip install mlx-gen"
                )
            # TODO: instantiate the pipeline with self._model_spec and
            # self._base_flags. The exact API depends on the installed
            # mlx-gen version. Log the available attributes for diagnostics.
            self._logger.info(
                f"[WanDaemon] mlx-gen module loaded: {pipeline_mod.__name__}. "
                f"Available: {[a for a in dir(pipeline_mod) if not a.startswith('_')]}"
            )
            self._loaded = True
        except Exception as exc:
            self._logger.warning(
                f"[WanDaemon] Pipeline import failed ({exc}). "
                "Falling back to subprocess mode."
            )
            self._loaded = False

    def generate(self, prompt: str, image_path: Optional[Path],
                 output_path: Path, per_cycle_flags: Dict[str, Any]) -> Tuple[bool, str]:
        """Run one generation cycle. Thread-safe via lock."""
        with self._lock:
            if not self._loaded:
                self._load_pipeline()
            if not self._loaded:
                return False, "WanDaemon pipeline not available"
            # TODO: call the pipeline's generate method with the merged flags.
            # Return (True, str(output_path)) on success or (False, error) on failure.
            raise NotImplementedError("WanDaemon.generate() — see implementation note below")

    def shutdown(self):
        """Release pipeline resources."""
        self._pipeline = None
        self._loaded = False
        self._logger.info("[WanDaemon] Shut down.")
```

---

### IMPORTANT: Implementation discovery step

Before implementing `WanDaemon.generate()` fully, the agent MUST first introspect
the installed mlx-gen package to find the correct Python API:

```bash
python3 - <<'PY'
import importlib, pkgutil, mflux
print("mflux location:", mflux.__file__)
# List all submodules
for info in pkgutil.walk_packages(mflux.__path__, prefix="mflux."):
    print(info.name)
PY
```

And:

```bash
python3 -c "
from mflux.cli import mlx_gen
import inspect
print(inspect.getsource(mlx_gen))
"
```

Use the output to find:
- The correct pipeline class name and import path
- The constructor signature (what args does it take for model path, quantize, etc.)
- The generate/run method signature (prompt, image, output, steps, etc.)
- How `--keep-text-encoder` maps to a Python API argument

Then implement `WanDaemon._load_pipeline()` and `WanDaemon.generate()` using
the actual API. Do NOT guess — read the source first.

---

### 2. Integrate `WanDaemon` into `WanRunner`

Modify `WanRunner` to optionally use `WanDaemon` instead of subprocess:

```python
class WanRunner(LocalRunner):
    name = "wan"

    def __init__(self, config, local_cfg=None):
        super().__init__(config, local_cfg)
        self._daemon: Optional[WanDaemon] = None
        # Use daemon mode when keep_text_encoder is explicitly enabled
        self._use_daemon = bool(config.get("keep_text_encoder") or config.get("keep-text-encoder"))

    def _get_or_create_daemon(self, model_spec: str, base_flags: Dict[str, Any]) -> WanDaemon:
        if self._daemon is None:
            self._daemon = WanDaemon(model_spec, base_flags)
        return self._daemon

    def shutdown_daemon(self):
        if self._daemon:
            self._daemon.shutdown()
            self._daemon = None
```

When `self._use_daemon` is True and the Python API is available, route
`build_video_command` calls through `WanDaemon.generate()` directly instead of
returning a subprocess command list.

If the daemon's pipeline import fails, **fall back silently to subprocess mode**
and log a warning. Never crash the pipeline over this.

---

### 3. Daemon lifecycle in `LocalBackend`

`LocalBackend` already has `self._runner_cache`. Add daemon cleanup to it:

```python
def shutdown(self):
    """Call this when the vfaq session ends to release daemon resources."""
    for runner in self._runner_cache.values():
        if hasattr(runner, "shutdown_daemon"):
            runner.shutdown_daemon()
    self._runner_cache.clear()
```

Wire `LocalBackend.shutdown()` into `vfaq_cli.py`'s main loop teardown
(the `finally` block or `atexit` handler).

---

### 4. Fallback behaviour (CRITICAL)

The daemon mode MUST be transparent. If anything goes wrong:
- Pipeline import fails → fall back to subprocess, log WARNING not ERROR
- Generation raises exception → fall back to subprocess for that cycle
- Output file missing → same error handling as current subprocess path

The existing `_run_with_live_progress` and `_run_captured` methods stay
untouched as the fallback path.

---

### 5. Config flag

Enable daemon mode via config:

```yaml
local:
  wan:
    keep_text_encoder: true   # enables both --keep-text-encoder flag AND daemon mode
```

No new config keys needed — `keep_text_encoder: true` is already supported in
`WanRunner._common()` for the subprocess flag. We just also use it as the signal
to enable daemon mode.

---

### 6. Tests to add in `test_wan_runner_config.py`

```python
def test_daemon_mode_enabled_when_keep_text_encoder_true():
    runner = WanRunner({"keep_text_encoder": True}, local_cfg={})
    assert runner._use_daemon is True

def test_daemon_mode_disabled_by_default():
    runner = WanRunner({}, local_cfg={})
    assert runner._use_daemon is False

def test_daemon_shutdown_is_idempotent():
    runner = WanRunner({"keep_text_encoder": True}, local_cfg={})
    runner.shutdown_daemon()  # should not raise even if daemon never started
    runner.shutdown_daemon()  # second call also safe
```

---

### 7. Version bump

- `vfaq/version.py`: `_DEFAULT_VERSION = "v0.9.9-beta"`
- `VERSION` file: `v0.9.9-beta`
- `RELEASE-NOTES.md`: add section:

```markdown
## v0.9.9-beta

### Persistent Wan daemon mode
- `WanRunner` now supports in-process Wan pipeline execution via `WanDaemon`
  when `local.wan.keep_text_encoder: true` is set in config.
- T5 text encoder loads once per `vfaq_cli.py` session and is reused across
  all cycles, saving 5-10s per cycle.
- Falls back transparently to subprocess mode if mlx-gen Python API is
  unavailable or pipeline initialisation fails.
- `LocalBackend.shutdown()` added for clean daemon teardown at session end.
- Version bump: v0.9.8-beta → v0.9.9-beta.
```

---

### 8. Test run

After implementation, run:

```bash
# Unit tests first
cd ~/x/visual-faqtory
python -m pytest tests/ -x -q

# Then a real generation test with current story.txt
./vfaq_cli.py --config config.yaml
```

Watch the logs for:
```
[WanDaemon] mlx-gen module loaded: ...
```

If you see that, daemon mode is active. If you see:
```
[WanDaemon] Pipeline import failed ... Falling back to subprocess mode.
```
That's fine too — it means the mlx-gen Python API wasn't importable in the
expected way, and subprocess fallback is active. In that case, report the
exact import error and the output of the introspection step so we can
adjust the import path.

---

## Summary of what NOT to do

- Do NOT modify `_run_with_live_progress` or `_run_captured` — they stay as fallback
- Do NOT add new config keys beyond `keep_text_encoder`
- Do NOT crash if mlx-gen Python API is unavailable — always fall back
- Do NOT remove `--keep-text-encoder` from the subprocess command builder —
  it stays for the fallback path
- Do NOT bump version until ALL tests pass
