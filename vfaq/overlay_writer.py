#!/usr/bin/env python3
"""
overlay_writer.py - Atomic OBS / TouchDesigner Overlay Writer
═══════════════════════════════════════════════════════════════════════════════

Writes text overlay files for OBS "Text from File" sources:
  - now.txt: Current generation status
  - next.txt: Next crowd prompt preview
  - queue.txt: Top N queued prompts
  - toast.txt: Latest accepted prompt notification

All writes are atomic (write to .tmp, then os.replace).

Part of QonQrete Visual FaQtory v0.3.5-beta
"""
import os
import logging
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class OverlayWriter:
    """
    Writes OBS-compatible text overlay files atomically.

    Usage:
        writer = OverlayWriter("live_output")
        writer.write_now("Generating: neon skull dreamscape")
        writer.write_next("Next: @Dave: 'melting crystals'")
    """

    def __init__(self, out_dir: str | Path = "live_output", config: Optional[Dict] = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        cfg = config or {}
        self.now_file = cfg.get('now_file', 'now.txt')
        self.next_file = cfg.get('next_file', 'next.txt')
        self.queue_file = cfg.get('queue_file', 'queue.txt')
        self.toast_file = 'toast.txt'
        self.atomic = cfg.get('atomic', True)

        # Format templates
        fmt = cfg.get('format', {})
        self.now_fmt = fmt.get('now', 'NOW: {prompt}')
        self.next_fmt = fmt.get('next', 'NEXT: {next_crowd}')
        self.queue_fmt = fmt.get('queue', '{queue_lines}')

        # Toast ring buffer
        self._toast_ring: List[str] = []
        self._toast_max = 5

    def _write_atomic(self, filename: str, content: str) -> None:
        """Write content to file atomically."""
        target = self.out_dir / filename
        if self.atomic:
            tmp_path = target.with_suffix('.tmp')
            try:
                tmp_path.write_text(content, encoding='utf-8')
                os.replace(str(tmp_path), str(target))
            except OSError as e:
                logger.debug(f"[Overlay] Atomic write failed, direct fallback: {e}")
                target.write_text(content, encoding='utf-8')
        else:
            target.write_text(content, encoding='utf-8')

    def write_now(self, prompt: str = "", macro: str = "", crowd_status: str = "") -> None:
        """Write current generation status."""
        content = self.now_fmt.format(
            prompt=prompt,
            base_title=prompt[:60],
            macro=f" [{macro}]" if macro else "",
            crowd_status=crowd_status or "—",
        )
        self._write_atomic(self.now_file, content)

    def write_next(self, next_prompt: str = "") -> None:
        """Write next crowd prompt preview."""
        content = self.next_fmt.format(next_crowd=next_prompt or "—")
        self._write_atomic(self.next_file, content)

    def write_queue(self, items: List[Dict[str, Any]]) -> None:
        """Write queue listing."""
        if not items:
            self._write_atomic(self.queue_file, "(empty)")
            return

        lines = []
        for i, item in enumerate(items[:10], 1):
            name = item.get('name', 'anon')
            prompt = item.get('prompt', '')[:60]
            lines.append(f"{i}) @{name}: {prompt}")

        content = self.queue_fmt.format(queue_lines='\n'.join(lines))
        self._write_atomic(self.queue_file, content)

    def write_toast(self, name: str, prompt: str) -> None:
        """Write a toast notification for accepted prompt."""
        toast = f"✅ @{name} fed: '{prompt[:50]}' (queued)"
        self._toast_ring.append(toast)
        if len(self._toast_ring) > self._toast_max:
            self._toast_ring.pop(0)
        self._write_atomic(self.toast_file, toast)

    def write_frame_jpg(self, frame_bytes: bytes, filename: str = "current_frame.jpg") -> None:
        """Write frame bytes atomically (for OBS image source)."""
        target = self.out_dir / filename
        if self.atomic:
            tmp_path = target.with_suffix('.tmp')
            try:
                tmp_path.write_bytes(frame_bytes)
                os.replace(str(tmp_path), str(target))
            except OSError:
                target.write_bytes(frame_bytes)
        else:
            target.write_bytes(frame_bytes)

    def write_macro_status(self, macro: str) -> None:
        """Write current macro state for debug."""
        self._write_atomic('macro.txt', macro or 'CHILL')

    def clear_all(self) -> None:
        """Clear all overlay files."""
        for f in [self.now_file, self.next_file, self.queue_file, self.toast_file, 'macro.txt']:
            path = self.out_dir / f
            if path.exists():
                path.write_text('')


__all__ = ['OverlayWriter']
