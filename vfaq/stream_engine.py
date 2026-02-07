#!/usr/bin/env python3
"""
stream_engine.py - Stream Continuation Engine
═══════════════════════════════════════════════════════════════════════════════

Utility functions for stream continuation modes:
  - Extract the last portion of a video as a context clip (for longcat and sliding‑window modes)
  - Extract the last frame as a fallback for img2vid recovery
  - Compute beat‑aligned generation lengths to align cuts to the musical grid
  - Provide configuration helpers for both legacy `stream_mode` and new `stream` (longcat) sections

This module is used by the backends to implement both the older sliding window continuation (v0.2.x) and the new longcat autoregressive continuation (v0.3.x). It does not perform generation itself; instead it prepares context clips and durations in a VRAM‑safe manner.

Part of QonQrete Visual FaQtory v0.3.5-beta
"""
import math
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


def extract_video_context(
    video_path: Path,
    duration_sec: float,
    out_path: Path,
    reencode_if_needed: bool = True,
) -> Path:
    """
    Extract the last `duration_sec` seconds from a video.

    Uses -sseof for tail extraction. Tries stream copy first,
    falls back to re-encode if keyframe alignment fails.

    Args:
        video_path: Source video
        duration_sec: Seconds to extract from the end
        out_path: Destination path for context clip
        reencode_if_needed: Allow re-encode fallback

    Returns:
        Path to extracted context video

    Raises:
        RuntimeError: If extraction fails
    """
    if not video_path or not video_path.exists():
        raise FileNotFoundError(f"Source video not found: {video_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Try stream copy first (fast, preserves quality)
    cmd_copy = [
        'ffmpeg', '-y',
        '-sseof', f'-{duration_sec}',
        '-i', str(video_path),
        '-c', 'copy',
        str(out_path)
    ]
    try:
        result = subprocess.run(cmd_copy, capture_output=True, text=True)
        if result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 100:
            logger.info(f"[StreamEngine] Extracted context (copy): {duration_sec}s → {out_path.name}")
            return out_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    if not reencode_if_needed:
        raise RuntimeError(f"Stream copy failed and re-encode disabled for {video_path}")

    # Fallback: deterministic re-encode
    cmd_encode = [
        'ffmpeg', '-y',
        '-sseof', f'-{duration_sec}',
        '-i', str(video_path),
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18',
        '-pix_fmt', 'yuv420p',
        '-an',
        str(out_path)
    ]
    try:
        subprocess.run(cmd_encode, capture_output=True, check=True)
        if out_path.exists() and out_path.stat().st_size > 100:
            logger.info(f"[StreamEngine] Extracted context (re-encode): {duration_sec}s → {out_path.name}")
            return out_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Context extraction failed: {e}")

    raise RuntimeError(f"Context extraction produced empty output for {video_path}")


def extract_last_frame(video_path: Path, out_png: Path) -> Path:
    """
    Extract the very last frame from a video as PNG.
    Used for fallback and debugging.
    """
    if not video_path or not video_path.exists():
        raise FileNotFoundError(f"Source video not found: {video_path}")

    out_png.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'ffmpeg', '-y',
        '-sseof', '-0.05',
        '-i', str(video_path),
        '-vframes', '1',
        '-q:v', '2',
        str(out_png)
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        if out_png.exists():
            logger.debug(f"[StreamEngine] Extracted last frame: {out_png.name}")
            return out_png
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Last frame extraction failed: {e}")

    raise RuntimeError(f"Last frame extraction produced no output for {video_path}")


def compute_beat_aligned_generation_length(
    generation_frames: int,
    fps: int,
    bpm: float,
    max_adjustment_pct: float = 0.10,
) -> int:
    """
    Adjust generation_length so the stitch point aligns to nearest 1/4 bar.

    Drop protection: ensures context→generation boundary lands on a beat.

    Args:
        generation_frames: Requested generation length in frames
        fps: Video FPS
        bpm: Beats per minute
        max_adjustment_pct: Maximum adjustment (±%)

    Returns:
        Adjusted generation frame count
    """
    if bpm <= 0 or fps <= 0:
        return generation_frames

    seconds_per_beat = 60.0 / bpm
    generation_sec = generation_frames / fps

    # Find nearest beat boundary
    beats = generation_sec / seconds_per_beat
    rounded_beats = round(beats)
    if rounded_beats < 1:
        rounded_beats = 1

    target_sec = rounded_beats * seconds_per_beat
    target_frames = round(target_sec * fps)

    # Clamp adjustment
    min_frames = int(generation_frames * (1 - max_adjustment_pct))
    max_frames = int(generation_frames * (1 + max_adjustment_pct))
    target_frames = max(min_frames, min(max_frames, target_frames))

    if target_frames != generation_frames:
        logger.info(
            f"[StreamEngine] Beat-aligned generation: "
            f"{generation_frames}→{target_frames} frames "
            f"({rounded_beats} beats @ {bpm} BPM)"
        )

    return target_frames


def get_stream_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and validate stream configuration with defaults.

    Supports both legacy ``stream_mode`` (sliding window) and new ``stream`` (longcat)
    sections. When ``stream`` is present it takes precedence and maps
    ``context_frames`` → ``context_length`` and ``generate_frames`` → ``generation_length``
    for backward compatibility. Additional parameters such as ``max_iterations``,
    ``checkpoint`` and ``overlap_strategy`` are preserved to be consumed by the
    backend implementation.
    """
    # New longcat configuration
    stream = config.get('stream', {}) or {}
    if stream:
        vram = stream.get('vram_safety', {}) or {}
        fps = int(stream.get('fps', 8) or 8)
        gen_frames = int(stream.get('generate_frames', 16) or 16)
        max_iter = int(stream.get('max_iterations', 999) or 999)

        # ── Target duration computation (v0.3.5-beta) ────────────────
        # Priority:
        #   1. target_seconds * fps  (explicit duration)
        #   2. target_frames         (explicit frame count)
        #   3. generate_frames * max_iterations (fallback)
        #
        # This fixes the v0.3.4 bug where default runs stopped after one
        # iteration because target_frames fell through to request.video_frames.
        target_seconds = stream.get('target_seconds', None)
        target_frames_cfg = stream.get('target_frames', None)

        if target_seconds is not None:
            computed_target = int(float(target_seconds) * fps)
            logger.info(
                f"[StreamConfig] Target: {target_seconds}s × {fps}fps = "
                f"{computed_target} frames"
            )
        elif target_frames_cfg is not None:
            computed_target = int(target_frames_cfg)
            logger.info(f"[StreamConfig] Target: {computed_target} frames (explicit)")
        else:
            computed_target = gen_frames * max_iter
            logger.info(
                f"[StreamConfig] Target: {gen_frames} × {max_iter} = "
                f"{computed_target} frames (generate_frames × max_iterations)"
            )

        return {
            'enabled': stream.get('enabled', False),
            'method': stream.get('mode', 'longcat'),
            # Normalise names for legacy code
            'context_length': stream.get('context_frames', 16),
            'generation_length': gen_frames,
            'max_iterations': max_iter,
            'target_frames': computed_target,
            'checkpoint': stream.get('checkpoint', 'svd_xt.safetensors'),
            'overlap_strategy': stream.get('overlap_strategy', 'tail'),
            'fps': fps,
            'workflow': 'worqspace/workflows/stream_continuation.json',
            'vram_safety': {
                'max_context_frames': vram.get('max_context_frames', 24),
                'max_generate_frames': vram.get('max_generate_frames', 24),
                'oom_retry': vram.get('oom_retry', True),
            },
            # Pass parent config for stability controller access
            '_parent_config': config,
        }
    # Legacy sliding window configuration
    sm = config.get('stream_mode', {})
    return {
        'enabled': sm.get('enabled', False),
        'method': sm.get('method', 'sliding_window'),
        'context_length': sm.get('context_length', 24),
        'generation_length': sm.get('generation_length', 72),
        'overlap': sm.get('overlap', 8),
        'context_duration_sec': sm.get('context_duration_sec', 1.5),
        'workflow': sm.get('workflow', 'worqspace/workflows/stream_continuation.json'),
        'vram_safe': sm.get('vram_safe', True),
        'seed_mode': sm.get('seed_mode', 'locked'),
        'base_seed': sm.get('base_seed', 1337),
    }


def prepare_stream_cycle(
    cycle_index: int,
    previous_video: Optional[Path],
    output_dir: Path,
    stream_config: Dict[str, Any],
    fps: int = 8,
    bpm: float = 0.0,
) -> Dict[str, Any]:
    """
    Prepare a stream-mode cycle:
      - Extract context tail from previous video
      - Extract last frame (debug/fallback)
      - Compute beat-aligned generation length
      - Return prepared context info

    Returns dict with:
        context_video_path: Path or None
        last_frame_path: Path or None
        generation_frames: int
        context_frames: int
    """
    result = {
        'context_video_path': None,
        'last_frame_path': None,
        'generation_frames': stream_config['generation_length'],
        'context_frames': stream_config['context_length'],
    }

    if cycle_index == 0 or not previous_video:
        logger.info("[StreamEngine] Cycle 0 or no previous video — no context extraction")
        return result

    if not previous_video.exists():
        logger.warning(f"[StreamEngine] Previous video missing: {previous_video}")
        return result

    # Compute context duration
    context_dur = stream_config.get('context_duration_sec')
    if not context_dur and fps > 0:
        context_dur = stream_config['context_length'] / fps
    context_dur = context_dur or 1.5

    # Create context directory
    ctx_dir = output_dir / f"cycle_{cycle_index:04d}_context"
    ctx_dir.mkdir(parents=True, exist_ok=True)

    # Extract context tail
    try:
        ctx_path = ctx_dir / "context_tail.mp4"
        extract_video_context(previous_video, context_dur, ctx_path)
        result['context_video_path'] = ctx_path
    except Exception as e:
        logger.warning(f"[StreamEngine] Context extraction failed: {e}")

    # Extract last frame (always, for debug/fallback)
    try:
        frame_path = ctx_dir / "last.png"
        extract_last_frame(previous_video, frame_path)
        result['last_frame_path'] = frame_path
    except Exception as e:
        logger.debug(f"[StreamEngine] Last frame extraction failed: {e}")

    # Beat-aligned generation length
    if bpm > 0:
        result['generation_frames'] = compute_beat_aligned_generation_length(
            stream_config['generation_length'], fps, bpm
        )

    return result


__all__ = [
    'extract_video_context', 'extract_last_frame',
    'compute_beat_aligned_generation_length',
    'get_stream_config', 'prepare_stream_cycle',
]
