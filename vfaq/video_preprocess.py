#!/usr/bin/env python3
"""
video_preprocess.py - Video2Video Preprocessing Pipeline
═══════════════════════════════════════════════════════════════════════════════

MANDATORY preprocessing for Video2Video input mode:
  - Normalize any input video to a consistent format
  - Resolution: ≤ 1024×576 (configurable)
  - FPS: 8 (configurable)
  - Duration: 4 seconds (configurable)
  - Pixel format: yuv420p
  - Output: deterministic, reproducible

If preprocessing fails → abort cycle cleanly with error.

Also provides audio extraction for audio-reactive features.

Part of QonQrete Visual FaQtory v0.3.5-beta
"""
import json
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


def preprocess_video(
    input_path: Path,
    output_path: Path,
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Preprocess base_video for Video2Video generation.

    MANDATORY step before any V2V workflow. Normalizes:
      - Resolution (scale down to fit max, preserving aspect)
      - FPS
      - Duration
      - Pixel format

    Args:
        input_path:  Source video path
        output_path: Destination for preprocessed video
        config:      video2video.preprocess config section

    Returns:
        Path to preprocessed video

    Raises:
        RuntimeError: If ffmpeg preprocessing fails
    """
    if not input_path.exists():
        raise FileNotFoundError(f"[VideoPreprocess] Input video not found: {input_path}")

    cfg = config or {}
    width = cfg.get("width", 1024)
    height = cfg.get("height", 576)
    fps = cfg.get("fps", 8)
    duration_sec = cfg.get("duration_sec", 4)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Scale filter: fit within max dimensions, preserving aspect ratio
    scale_filter = (
        f"scale={width}:{height}:"
        f"force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
        f"fps={fps}"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", scale_filter,
        "-t", str(duration_sec),
        "-pix_fmt", "yuv420p",
        "-an",  # Strip audio (handled separately)
        str(output_path),
    ]

    logger.info(
        f"[VideoPreprocess] Normalizing: {input_path.name} → "
        f"{width}x{height}@{fps}fps, {duration_sec}s"
    )

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"[VideoPreprocess] ffmpeg failed (exit {result.returncode}): "
            f"{result.stderr[:500]}"
        )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(
            f"[VideoPreprocess] Output file missing or empty: {output_path}"
        )

    logger.info(f"[VideoPreprocess] Preprocessed: {output_path} ({output_path.stat().st_size} bytes)")
    return output_path


def extract_audio_from_video(
    video_path: Path,
    output_path: Path,
    sample_rate: int = 44100,
) -> Optional[Path]:
    """
    Extract audio track from video file.

    Returns None if video has no audio stream (not an error).
    """
    if not video_path.exists():
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # First check if video has audio
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_type",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)

    if not probe_result.stdout.strip():
        logger.info(f"[VideoPreprocess] No audio stream in {video_path.name}")
        return None

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",  # No video
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",  # Mono
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0 and output_path.exists():
        logger.info(f"[VideoPreprocess] Extracted audio: {output_path}")
        return output_path

    logger.warning(f"[VideoPreprocess] Audio extraction failed: {result.stderr[:200]}")
    return None


def get_video_info(video_path: Path) -> Dict[str, Any]:
    """
    Probe video file for metadata (resolution, fps, duration, codec).
    """
    if not video_path.exists():
        return {}

    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,duration,codec_name",
        "-of", "json",
        str(video_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return {}

    try:
        data = json.loads(result.stdout)
        stream = data.get("streams", [{}])[0]

        # Parse frame rate fraction
        fps_str = stream.get("r_frame_rate", "8/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)

        return {
            "width": int(stream.get("width", 0)),
            "height": int(stream.get("height", 0)),
            "fps": fps,
            "duration": float(stream.get("duration", 0)),
            "codec": stream.get("codec_name", "unknown"),
        }
    except (json.JSONDecodeError, ValueError, IndexError) as e:
        logger.warning(f"[VideoPreprocess] Probe parse error: {e}")
        return {}


def validate_preprocessed_video(
    video_path: Path,
    expected_width: int = 1024,
    expected_height: int = 576,
    expected_fps: int = 8,
    max_duration: float = 5.0,
) -> Tuple[bool, str]:
    """
    Validate that a preprocessed video meets requirements.

    Returns (is_valid, reason_if_invalid).
    """
    if not video_path.exists():
        return False, f"File not found: {video_path}"

    info = get_video_info(video_path)
    if not info:
        return False, "Could not probe video"

    issues = []
    if info.get("width", 0) > expected_width:
        issues.append(f"width {info['width']} > {expected_width}")
    if info.get("height", 0) > expected_height:
        issues.append(f"height {info['height']} > {expected_height}")
    if info.get("duration", 0) > max_duration:
        issues.append(f"duration {info['duration']:.1f}s > {max_duration}s")

    if issues:
        return False, "; ".join(issues)
    return True, "OK"


__all__ = [
    "preprocess_video",
    "extract_audio_from_video",
    "get_video_info",
    "validate_preprocessed_video",
]
