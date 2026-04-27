#!/usr/bin/env python3
"""
run_state.py — Run State Persistence & Artifact Discovery
═══════════════════════════════════════════════════════════════════════════════

Manages checkpoint state for resumable runs:
  - Atomic JSON state writes (write-to-temp + rename)
  - Artifact discovery from disk (videos, frames, briqs)
  - Resume context reconstruction
  - Per-cycle checkpoint updates

Part of Visual FaQtory v0.9.0-beta
"""
import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .version import __version__

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# RUN STATE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

VALID_STATUSES = ("starting", "running", "failed", "interrupted", "completed")


@dataclass
class RunState:
    """Persistent run state, written after every cycle for crash recovery."""
    run_id: str = ""
    version: str = __version__
    status: str = "starting"
    backend_type: str = "mock"
    mode: str = "text"
    reinject: bool = True
    story_path: str = ""
    cycles_planned: int = 0
    cycles_completed: int = 0
    next_cycle_index: int = 1
    last_completed_cycle: int = 0
    last_frame_path: str = ""
    anchor_frame_path: str = ""
    final_video_paths: List[str] = field(default_factory=list)
    base_image: str = ""
    base_video: str = ""
    base_audio: str = ""
    start_time: str = ""
    end_time: str = ""
    error_message: str = ""
    saved_to: str = ""
    resume_enabled: bool = False
    completed_cycle_indices: List[int] = field(default_factory=list)
    config_snapshot: str = "meta/config.yaml"
    story_snapshot: str = "meta/story.txt"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunState":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    @classmethod
    def load(cls, state_path: Path) -> Optional["RunState"]:
        """Load state from JSON file, or None if missing/corrupt."""
        if not state_path.exists():
            return None
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            return cls.from_dict(data)
        except Exception as e:
            logger.warning(f"[State] Failed to load {state_path}: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# ATOMIC STATE WRITES
# ═══════════════════════════════════════════════════════════════════════════════

def write_state_atomic(state: RunState, state_path: Path) -> None:
    """Write state JSON atomically (temp file + rename)."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(state.to_dict(), indent=2, default=str)
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=str(state_path.parent),
            prefix=".faqtory_state_",
            suffix=".tmp",
        )
        with os.fdopen(fd, "w") as f:
            f.write(data)
        os.replace(tmp_path, str(state_path))
    except Exception as e:
        logger.error(f"[State] Atomic write failed: {e}")
        # Fallback: direct write (better than losing state entirely)
        try:
            state_path.write_text(data, encoding="utf-8")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# ARTIFACT DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

_CYCLE_VIDEO_RE = re.compile(r"video_(\d{3})\.mp4$")
_CYCLE_LOOP_RE = re.compile(r"video_loop_(\d{3})\.mp4$")
_CYCLE_LASTFRAME_RE = re.compile(r"lastframe_(\d{3})\.png$")
_CYCLE_BRIQ_RE = re.compile(r"cycle_(\d{3})\.json$")


@dataclass
class DiscoveredProgress:
    """Result of scanning run/ for existing cycle artifacts."""
    completed_cycles: List[int] = field(default_factory=list)
    last_completed_cycle: int = 0
    last_frame_path: Optional[Path] = None
    anchor_frame_path: Optional[Path] = None
    final_video_paths: List[Path] = field(default_factory=list)
    loop_closure_paths: List[Path] = field(default_factory=list)
    next_cycle_index: int = 1
    needs_finalization_only: bool = False
    already_finalized: bool = False


def discover_progress(run_dir: Path, total_cycles: int = 0) -> DiscoveredProgress:
    """Scan run/ for existing cycle artifacts and reconstruct progress.

    Trust order:
      1. Existing cycle videos in run/videos/video_NNN.mp4
      2. Existing last frames in run/frames/lastframe_NNN.png
      3. Briq JSON as supplemental

    A cycle is "complete" if it has BOTH a video and a lastframe.
    If video exists without lastframe, attempt to regenerate it.

    CONTIGUOUS CHAIN REQUIRED: cycles must be 1, 2, 3, ... with no gaps.
    If cycle 3 is missing but cycle 4 exists, progress stops at cycle 2.
    """
    result = DiscoveredProgress()
    videos_dir = run_dir / "videos"
    frames_dir = run_dir / "frames"

    # Check if already finalized
    final_output = run_dir / "final_output.mp4"
    if final_output.exists() and final_output.stat().st_size > 0:
        result.already_finalized = True

    if not videos_dir.exists():
        return result

    # Discover videos
    video_cycles: Dict[int, Path] = {}
    for f in sorted(videos_dir.iterdir()):
        m = _CYCLE_VIDEO_RE.match(f.name)
        if m and f.stat().st_size > 0:
            video_cycles[int(m.group(1))] = f

    if not video_cycles:
        return result

    # Discover loop-closure clips
    loop_clips: Dict[int, Path] = {}
    for f in sorted(videos_dir.iterdir()):
        m = _CYCLE_LOOP_RE.match(f.name)
        if m and f.stat().st_size > 0:
            loop_clips[int(m.group(1))] = f

    # Discover lastframes
    lastframe_cycles: Dict[int, Path] = {}
    if frames_dir.exists():
        for f in sorted(frames_dir.iterdir()):
            m = _CYCLE_LASTFRAME_RE.match(f.name)
            if m and f.stat().st_size > 0:
                lastframe_cycles[int(m.group(1))] = f

    # ── Contiguous chain validation ───────────────────────────────────────
    # Walk from cycle 1 upward. Stop at the first gap or incomplete cycle.
    completed = []
    ordered_videos = []
    expected_cycle = 1

    for cyc in sorted(video_cycles.keys()):
        if cyc != expected_cycle:
            # Gap detected — stop here
            logger.warning(
                f"[Resume] Gap detected: expected cycle {expected_cycle}, "
                f"found cycle {cyc}. Stopping at cycle {expected_cycle - 1}."
            )
            break

        if cyc in lastframe_cycles:
            completed.append(cyc)
            ordered_videos.append(video_cycles[cyc])
            expected_cycle = cyc + 1
        else:
            # Try to regenerate lastframe from video
            video_path = video_cycles[cyc]
            lastframe_path = frames_dir / f"lastframe_{cyc:03d}.png"
            if _try_extract_last_frame(video_path, lastframe_path):
                completed.append(cyc)
                ordered_videos.append(video_path)
                lastframe_cycles[cyc] = lastframe_path
                expected_cycle = cyc + 1
                logger.info(f"[Resume] Recovered lastframe for cycle {cyc} from video")
            else:
                logger.warning(
                    f"[Resume] Cycle {cyc} has video but lastframe extraction failed — "
                    f"stopping here (will retry this cycle)"
                )
                break

    if not completed:
        return result

    result.completed_cycles = completed
    result.last_completed_cycle = completed[-1]
    result.last_frame_path = lastframe_cycles.get(completed[-1])
    result.final_video_paths = ordered_videos
    result.next_cycle_index = completed[-1] + 1

    # Include loop-closure clips in recovery
    if loop_clips:
        result.loop_closure_paths = list(loop_clips.values())
        logger.info(f"[Resume] Found {len(loop_clips)} loop-closure clip(s)")

    # Recover anchor frame (cycle 1 first frame)
    anchor = frames_dir / "anchor_frame_001.png"
    if anchor.exists() and anchor.stat().st_size > 0:
        result.anchor_frame_path = anchor
    elif 1 in video_cycles:
        if _try_extract_first_frame(video_cycles[1], anchor):
            result.anchor_frame_path = anchor
            logger.info("[Resume] Recovered anchor frame from cycle 1 video")

    # Check if all cycles are done but finalization is missing
    if total_cycles > 0 and result.last_completed_cycle >= total_cycles:
        if not result.already_finalized:
            result.needs_finalization_only = True
            logger.info("[Resume] All cycles complete — needs finalization only")

    logger.info(
        f"[Resume] Discovered {len(completed)} contiguous completed cycle(s), "
        f"last={result.last_completed_cycle}, next={result.next_cycle_index}"
    )

    return result


def check_needs_finalization(run_dir: Path, total_cycles: int) -> bool:
    """Check if all cycles are done but final output is missing."""
    progress = discover_progress(run_dir, total_cycles=total_cycles)
    return progress.needs_finalization_only


# ═══════════════════════════════════════════════════════════════════════════════
# FRAME EXTRACTION HELPERS (reuse ffmpeg patterns from sliding_story_engine)
# ═══════════════════════════════════════════════════════════════════════════════

def _try_extract_last_frame(video_path: Path, output_path: Path) -> bool:
    """Extract last frame from video. Returns True on success."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Get duration
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, check=True,
        )
        duration = float(probe.stdout.strip())
        seek = max(0, duration - 0.5)
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(seek), "-i", str(video_path),
             "-vframes", "1", "-q:v", "2", str(output_path)],
            capture_output=True, text=True, check=True,
        )
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception as e:
        logger.warning(f"[Resume] Last frame extraction failed for {video_path}: {e}")
        return False


def _try_extract_first_frame(video_path: Path, output_path: Path) -> bool:
    """Extract first frame from video. Returns True on success."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path),
             "-vframes", "1", "-q:v", "2", str(output_path)],
            capture_output=True, text=True, check=True,
        )
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception as e:
        logger.warning(f"[Resume] First frame extraction failed for {video_path}: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# RESUME STATUS REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

def format_status_report(run_dir: Path) -> str:
    """Generate a human-readable status report for a run directory."""
    lines = []
    state_path = run_dir / "faqtory_state.json"
    state = RunState.load(state_path)

    progress = discover_progress(run_dir)

    lines.append(f"  Run directory: {run_dir}")

    if state:
        lines.append(f"  Run ID: {state.run_id}")
        lines.append(f"  Status: {state.status}")
        lines.append(f"  Backend: {state.backend_type}")
        lines.append(f"  Mode: {state.mode}")
        lines.append(f"  Cycles planned: {state.cycles_planned}")
        lines.append(f"  Cycles completed: {state.cycles_completed}")
        lines.append(f"  Next cycle: {state.next_cycle_index}")
        if state.error_message:
            lines.append(f"  Last error: {state.error_message[:200]}")
        if state.start_time:
            lines.append(f"  Started: {state.start_time}")
        if state.end_time:
            lines.append(f"  Ended: {state.end_time}")
    else:
        lines.append("  State file: missing or corrupt")

    # Artifact-based progress (always show, even without state file)
    lines.append(f"  Artifacts found: {len(progress.completed_cycles)} completed cycle(s)")
    if progress.completed_cycles:
        lines.append(f"  Last completed: cycle {progress.last_completed_cycle}")
        lines.append(f"  Next to run: cycle {progress.next_cycle_index}")
    lines.append(f"  Last frame: {'yes' if progress.last_frame_path else 'no'}")
    lines.append(f"  Anchor frame: {'yes' if progress.anchor_frame_path else 'no'}")

    # Resumability
    final_output = run_dir / "final_output.mp4"
    if progress.already_finalized:
        lines.append("  Resumable: no (run already finalized)")
    elif progress.needs_finalization_only:
        lines.append("  Resumable: YES (finalization-only — all cycles complete)")
    elif progress.completed_cycles:
        lines.append("  Resumable: YES")
    else:
        lines.append("  Resumable: no (no completed cycles found)")
    if progress.loop_closure_paths:
        lines.append(f"  Loop closure clips: {len(progress.loop_closure_paths)}")

    return "\n".join(lines)


__all__ = [
    "RunState",
    "write_state_atomic",
    "discover_progress",
    "check_needs_finalization",
    "format_status_report",
    "DiscoveredProgress",
]
