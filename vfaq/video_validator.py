#!/usr/bin/env python3
"""
video_validator.py — Video File Validation Utilities (v0.9.3+)
═══════════════════════════════════════════════════════════════════════════════

Provides helpers for validating video files before they are handed to
media players, OBS sources, or the A/B swap system. Prevents the
"swapped but not playing" class of bugs by ensuring the file is:
  1. Present and non-empty
  2. Readable (basic I/O check)
  3. Stable (not still being written)
  4. Has valid video metadata (optional ffprobe probe)

Also provides a lightweight function to wait for a file to stop growing.

Part of Visual FaQtory v0.9.3-beta
"""
from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def validate_video_file(
    path: Path,
    *,
    check_metadata: bool = False,
    min_size_bytes: int = 1024,
) -> bool:
    """Validate that a video file is present, non-empty, and readable.

    Args:
        path: Path to the video file.
        check_metadata: If True, also run ffprobe to check for a video stream.
        min_size_bytes: Minimum file size in bytes.

    Returns:
        True if the file passes all checks.
    """
    if not path.exists():
        logger.warning(f"[validate_video_file] File does not exist: {path}")
        return False

    if not path.is_file():
        logger.warning(f"[validate_video_file] Path is not a regular file: {path}")
        return False

    try:
        size = path.stat().st_size
    except OSError as e:
        logger.warning(f"[validate_video_file] Cannot stat file {path}: {e}")
        return False

    if size < min_size_bytes:
        logger.warning(
            f"[validate_video_file] File too small: {path} ({size} bytes, "
            f"min={min_size_bytes})"
        )
        return False

    # Basic readability: can we open and read the first byte?
    try:
        with open(path, 'rb') as f:
            f.read(1)
    except OSError as e:
        logger.warning(f"[validate_video_file] File not readable: {path}: {e}")
        return False

    if check_metadata:
        if not _probe_video_stream(path):
            logger.warning(
                f"[validate_video_file] No video stream found in: {path}"
            )
            return False

    return True


def _probe_video_stream(path: Path) -> bool:
    """Check if a file has a valid video stream via ffprobe."""
    try:
        result = subprocess.run(
            [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=codec_type',
                '-of', 'default=nokey=1:noprint_wrappers=1',
                str(path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0 and 'video' in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.debug(f"[validate_video_file] ffprobe failed (non-fatal): {e}")
        return True  # Fail-open: ffprobe might not be installed
    except Exception as e:
        logger.warning(f"[validate_video_file] ffprobe error: {e}")
        return True  # Fail-open


def await_file_stable(
    path: Path,
    *,
    stable_checks: int = 3,
    poll_interval: float = 0.3,
    timeout_sec: float = 60.0,
) -> bool:
    """Wait for a file to stop growing (stable size across N checks).

    Useful before handing a freshly-written video to a media player.
    Prevents the "played a half-written file" class of bugs.

    Args:
        path: Path to the file.
        stable_checks: Number of consecutive stable-size polls required.
        poll_interval: Seconds between size checks.
        timeout_sec: Maximum total wait time.

    Returns:
        True if the file stabilized within the timeout.
    """
    if not path.exists():
        logger.warning(f"[await_file_stable] File does not exist: {path}")
        return False

    deadline = time.monotonic() + timeout_sec
    last_size = -1
    stable_count = 0

    while time.monotonic() < deadline:
        try:
            size = path.stat().st_size
        except OSError:
            time.sleep(poll_interval)
            continue

        if size > 0 and size == last_size:
            stable_count += 1
        else:
            stable_count = 0

        last_size = size

        if stable_count >= stable_checks:
            logger.debug(
                f"[await_file_stable] File stabilized at {size} bytes "
                f"after {stable_count} checks: {path.name}"
            )
            return True

        time.sleep(poll_interval)

    logger.warning(
        f"[await_file_stable] File did not stabilize within {timeout_sec}s: "
        f"{path.name} (last_size={last_size})"
    )
    return False


__all__ = ['validate_video_file', 'await_file_stable']
