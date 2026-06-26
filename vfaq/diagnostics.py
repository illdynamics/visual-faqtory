#!/usr/bin/env python3
"""
diagnostics.py — Run Diagnostics & Crash Recovery Artifacts (v0.9.3+)
═══════════════════════════════════════════════════════════════════════════════

Writes structured diagnostics summaries alongside each run. These artifacts
make post-stream debugging fast and auditable.

Key outputs:
  - run/.diagnostics_summary.json — overall run health summary
  - run/.diagnostics_events.jsonl  — per-cycle event log

Also provides a context manager `CYCLE_GUARD` that wraps cycle execution
and logs failures without crashing the whole app.

Part of Visual FaQtory v0.9.3-beta
"""
from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DiagnosticsWriter:
    """Writes structured diagnostics artifacts into the run directory."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.events_path = self.run_dir / ".diagnostics_events.jsonl"
        self.summary_path = self.run_dir / ".diagnostics_summary.json"
        self._events: List[Dict[str, Any]] = []
        self._counters = {
            "cycles_total": 0,
            "cycles_failed": 0,
            "generation_failures": 0,
            "playback_switch_failures": 0,
            "fallback_activations": 0,
            "crowd_prompts_received": 0,
            "crowd_prompts_applied": 0,
            "exceptions_caught": 0,
        }

    def log_event(self, event: str, **kwargs) -> None:
        """Append a timestamped event to the in-memory buffer and flush."""
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **kwargs,
        }
        self._events.append(record)
        # Atomic append to JSONL
        try:
            with open(self.events_path, 'a') as f:
                f.write(json.dumps(record, default=str) + '\n')
        except OSError:
            pass

    def increment(self, counter: str, delta: int = 1) -> None:
        """Increment a named counter."""
        if counter in self._counters:
            self._counters[counter] += delta

    def cycle_start(self, cycle_idx: int, **kwargs) -> None:
        """Log cycle start."""
        self.increment("cycles_total")
        self.log_event("cycle_start", cycle=cycle_idx, **kwargs)

    def cycle_end(self, cycle_idx: int, success: bool = True, **kwargs) -> None:
        """Log cycle end."""
        self.log_event(
            "cycle_end", cycle=cycle_idx, success=success, **kwargs
        )
        if not success:
            self.increment("cycles_failed")

    def cycle_crashed(self, cycle_idx: int, error: str, tb: str = "", **kwargs) -> None:
        """Log a cycle crash with full diagnostics."""
        self.increment("cycles_failed")
        self.increment("exceptions_caught")
        self.log_event(
            "cycle_crashed",
            cycle=cycle_idx,
            error=str(error)[:500],
            traceback=tb[:2000] if tb else "",
            **kwargs,
        )

    def write_summary(self, **extra) -> Path:
        """Write the final diagnostics summary JSON file."""
        summary = {
            "written_at": datetime.now(timezone.utc).isoformat(),
            **self._counters,
            **extra,
        }
        try:
            self.summary_path.write_text(
                json.dumps(summary, indent=2, default=str)
            )
        except OSError:
            pass
        logger.info(
            f"[Diagnostics] Summary written to {self.summary_path}: "
            f"cycles={self._counters['cycles_total']}, "
            f"failed={self._counters['cycles_failed']}, "
            f"exceptions={self._counters['exceptions_caught']}"
        )
        return self.summary_path


class CycleGuard:
    """Context manager that wraps a single cycle's execution.

    On exception, logs the full traceback, writes a diagnostics event,
    and suppresses the exception so the main loop can continue.

    Usage:
        with CycleGuard(diag, cycle_idx, reraise=False):
            # cycle body
    """

    def __init__(
        self,
        diag: DiagnosticsWriter,
        cycle_idx: int,
        *,
        reraise: bool = False,
    ):
        self.diag = diag
        self.cycle_idx = cycle_idx
        self.reraise = reraise
        self.exception: Optional[Exception] = None

    def __enter__(self):
        self.diag.cycle_start(self.cycle_idx)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            tb_str = ''.join(traceback.format_exception(
                exc_type, exc_val, exc_tb
            )) if exc_tb else str(exc_val)
            self.diag.cycle_crashed(
                self.cycle_idx,
                error=str(exc_val),
                tb=tb_str,
            )
            logger.error(
                f"[CycleGuard] Cycle {self.cycle_idx} crashed: "
                f"{exc_val}\n{tb_str}"
            )
            self.exception = exc_val
            if self.reraise:
                return False  # Re-raise
            return True  # Suppress — continue to next cycle
        self.diag.cycle_end(self.cycle_idx, success=True)
        return False  # No exception

    @property
    def crashed(self) -> bool:
        return self.exception is not None


__all__ = ['DiagnosticsWriter', 'CycleGuard']
