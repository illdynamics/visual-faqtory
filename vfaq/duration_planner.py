#!/usr/bin/env python3
"""
duration_planner.py - Auto-Duration + Audio Match Planning
═══════════════════════════════════════════════════════════════════════════════

Computes required cycle count from:
  - Audio file duration (ffprobe)
  - BPM + bars_per_cycle grid logic
  - Config overrides (fixed seconds, match_audio)

Also handles post-render trim + audio mux via ffmpeg.

Part of QonQrete Visual FaQtory v0.3.5-beta
"""
import math
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


def detect_audio_duration(audio_path: Path) -> float:
    """
    Detect audio file duration in seconds using ffprobe.

    Args:
        audio_path: Path to audio file

    Returns:
        Duration in seconds

    Raises:
        RuntimeError: If ffprobe fails or returns invalid data
    """
    if not audio_path or not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(audio_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        logger.info(f"[DurationPlanner] Audio duration: {duration:.2f}s ({audio_path.name})")
        return duration
    except (subprocess.CalledProcessError, ValueError) as e:
        raise RuntimeError(f"Failed to detect audio duration: {e}")


def compute_cycle_duration(bpm: float, bars_per_cycle: int = 8) -> float:
    """
    Compute cycle duration from BPM and bars_per_cycle.

    Formula: cycle_duration = bars_per_cycle × 4 × (60 / BPM)
    """
    if bpm <= 0:
        raise ValueError(f"Invalid BPM: {bpm}")
    beat_dur = 60.0 / bpm
    bar_dur = beat_dur * 4  # 4/4 time
    cycle_dur = bar_dur * bars_per_cycle
    return cycle_dur


def plan_duration(
    config: Dict[str, Any],
    audio_path: Optional[Path] = None,
    bpm: float = 0.0,
    bars_per_cycle: int = 8,
    requested_cycles: int = 0,
    clip_seconds: float = 8.0,
) -> Dict[str, Any]:
    """
    Plan the run duration based on config, audio, and BPM.

    Returns dict with:
        required_cycles: int
        cycle_duration: float
        audio_duration: float or None
        override_reason: str or None
        match_audio: bool
        mux_audio: bool
        trim_to: float or None  (seconds to trim final video)
    """
    dur_config = config.get('duration', {})
    mode = dur_config.get('mode', 'auto')
    match_audio = dur_config.get('match_audio', False)
    mux_audio = dur_config.get('mux_audio', True)
    fixed_seconds = dur_config.get('seconds')

    result = {
        'required_cycles': requested_cycles,
        'cycle_duration': clip_seconds * 2,  # default: loop duration
        'audio_duration': None,
        'override_reason': None,
        'match_audio': match_audio,
        'mux_audio': mux_audio,
        'trim_to': None,
    }

    # Compute cycle duration from BPM if available
    if bpm > 0:
        try:
            result['cycle_duration'] = compute_cycle_duration(bpm, bars_per_cycle)
        except ValueError:
            pass

    # Detect audio duration if available
    if audio_path and audio_path.exists():
        try:
            result['audio_duration'] = detect_audio_duration(audio_path)
        except Exception as e:
            logger.warning(f"[DurationPlanner] Could not detect audio duration: {e}")

    # Mode: fixed
    if mode == 'fixed' and fixed_seconds and fixed_seconds > 0:
        required = math.ceil(fixed_seconds / result['cycle_duration'])
        result['required_cycles'] = max(1, required)
        result['override_reason'] = f"fixed duration: {fixed_seconds}s → {required} cycles"
        result['trim_to'] = fixed_seconds
        logger.info(f"[DurationPlanner] Fixed mode: {fixed_seconds}s → {required} cycles")
        return result

    # Mode: auto + match_audio
    if mode == 'auto' and match_audio and result['audio_duration']:
        audio_dur = result['audio_duration']
        required = math.ceil(audio_dur / result['cycle_duration'])
        result['required_cycles'] = max(1, required)
        result['override_reason'] = (
            f"match_audio: {audio_dur:.1f}s audio ÷ "
            f"{result['cycle_duration']:.1f}s/cycle = {required} cycles"
        )
        result['trim_to'] = audio_dur
        logger.info(
            f"[DurationPlanner] Auto-duration override: "
            f"{audio_dur:.1f}s audio → {required} cycles "
            f"(cycle_dur={result['cycle_duration']:.1f}s)"
        )
        return result

    # Mode: unlimited or default
    return result


def trim_video(
    video_path: Path,
    output_path: Path,
    trim_seconds: float,
    preferred_codec: str = 'h264_nvenc',
) -> Path:
    """
    Trim video to exact duration using ffmpeg.
    Tries stream copy first, falls back to re-encode.
    """
    # Try stream copy (fast, no quality loss)
    cmd_copy = [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-t', str(trim_seconds),
        '-c', 'copy',
        str(output_path)
    ]
    try:
        subprocess.run(cmd_copy, capture_output=True, check=True)
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info(f"[DurationPlanner] Trimmed (copy): {output_path} to {trim_seconds:.2f}s")
            return output_path
    except subprocess.CalledProcessError:
        logger.debug("[DurationPlanner] Stream copy trim failed, falling back to re-encode")

    # Fallback: deterministic re-encode
    for codec in [preferred_codec, 'libx264']:
        cmd_encode = [
            'ffmpeg', '-y',
            '-i', str(video_path),
            '-t', str(trim_seconds),
            '-c:v', codec, '-preset', 'fast', '-crf', '18',
            '-pix_fmt', 'yuv420p',
            str(output_path)
        ]
        try:
            subprocess.run(cmd_encode, capture_output=True, check=True)
            if output_path.exists() and output_path.stat().st_size > 0:
                logger.info(f"[DurationPlanner] Trimmed (re-encode {codec}): {trim_seconds:.2f}s")
                return output_path
        except subprocess.CalledProcessError:
            continue

    raise RuntimeError(f"Failed to trim video to {trim_seconds}s")


def mux_audio_video(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
) -> Path:
    """
    Mux audio and video into a single MP4.
    Audio starts at t=0, video duration matches audio.
    """
    cmd = [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-i', str(audio_path),
        '-c:v', 'copy',
        '-c:a', 'aac', '-b:a', '192k',
        '-shortest',
        '-movflags', '+faststart',
        str(output_path)
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info(f"[DurationPlanner] Muxed audio+video: {output_path}")
            return output_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Audio mux failed: {e}")

    raise RuntimeError("Audio mux produced empty output")


def post_finalize_trim_and_mux(
    final_video: Path,
    audio_path: Optional[Path],
    plan: Dict[str, Any],
    preferred_codec: str = 'h264_nvenc',
) -> Path:
    """
    Run trim + optional mux after finalizer completes.

    Returns path to the final output (trimmed, or muxed, or original).
    """
    if not plan.get('trim_to'):
        return final_video

    trim_seconds = plan['trim_to']

    # Trim
    trimmed_path = final_video.parent / f"{final_video.stem}_trimmed.mp4"
    try:
        trim_video(final_video, trimmed_path, trim_seconds, preferred_codec)
    except RuntimeError as e:
        logger.error(f"[DurationPlanner] Trim failed: {e}")
        return final_video

    # Mux audio if enabled
    if plan.get('mux_audio') and audio_path and audio_path.exists():
        muxed_path = final_video.parent / f"{final_video.stem}_muxed.mp4"
        try:
            mux_audio_video(trimmed_path, audio_path, muxed_path)
            # Replace original final with muxed version
            import shutil
            shutil.move(str(muxed_path), str(final_video))
            trimmed_path.unlink(missing_ok=True)
            logger.info(f"[DurationPlanner] Final output (muxed): {final_video}")
            return final_video
        except RuntimeError as e:
            logger.error(f"[DurationPlanner] Mux failed: {e}")
            # Fall through to trimmed-only path

    # Replace original with trimmed version
    import shutil
    shutil.move(str(trimmed_path), str(final_video))
    logger.info(f"[DurationPlanner] Final output (trimmed): {final_video}")
    return final_video


__all__ = [
    'detect_audio_duration', 'compute_cycle_duration', 'plan_duration',
    'trim_video', 'mux_audio_video', 'post_finalize_trim_and_mux',
]
