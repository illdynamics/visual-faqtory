#!/usr/bin/env python3
"""
finalizer.py - Video Stitching & Final Output
═══════════════════════════════════════════════════════════════════════════════

Finalizer is responsible for stitching all per-cycle MP4s into a single
final_output.mp4 when ALL cycles are complete.

Behavior:
  1. Collect all per-cycle MP4s in chronological order
  2. Concatenate using ffmpeg (stream copy / no re-encoding when possible)
  3. Produce final_output.mp4
  4. Log final duration, fps, resolution

If any cycle failed:
  - Abort finalization
  - Report which cycle failed and why

Supports NVENC-based encoding (h264_nvenc preferred, libx264 fallback).

Part of QonQrete Visual FaQtory v0.0.5-alpha
"""
import logging
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class FinalizerError(Exception):
    """Raised when finalization cannot proceed."""
    pass


class Finalizer:
    """
    Stitches per-cycle looped MP4s into a single final output video.
    """

    def __init__(
        self,
        project_dir: Path,
        preferred_codec: str = "h264_nvenc",
        output_quality: int = 18
    ):
        self.project_dir = Path(project_dir)
        self.preferred_codec = preferred_codec
        self.output_quality = output_quality
        self.videos_dir = self.project_dir / "videos"
        self.final_output_path = self.project_dir / "final_output.mp4"

    def finalize(
        self,
        cycle_video_paths: Optional[List[Path]] = None,
        failed_cycles: Optional[List[int]] = None
    ) -> Path:
        """
        Stitch all per-cycle videos into final_output.mp4.

        Args:
            cycle_video_paths: Explicit list of video paths (chronological order).
                               If None, auto-discovers from videos_dir.
            failed_cycles: List of cycle indices that failed.

        Returns:
            Path to final_output.mp4

        Raises:
            FinalizerError: If any cycles failed or no videos found.
        """
        # Check for failed cycles
        if failed_cycles:
            raise FinalizerError(
                f"[Finalizer] Cannot finalize: cycles {failed_cycles} failed. "
                f"Fix or re-run failed cycles before finalizing."
            )

        # Collect videos
        if cycle_video_paths:
            videos = [Path(p) for p in cycle_video_paths if Path(p).exists()]
        else:
            videos = self._discover_videos()

        if not videos:
            raise FinalizerError(
                "[Finalizer] No per-cycle videos found to stitch. "
                "Run the pipeline first."
            )

        # Validate all videos exist
        missing = [v for v in videos if not v.exists()]
        if missing:
            raise FinalizerError(
                f"[Finalizer] Missing video files: {[str(m) for m in missing]}"
            )

        logger.info(f"[Finalizer] Stitching {len(videos)} videos into final_output.mp4")

        # Try stream-copy concat first (fastest, no re-encoding)
        success = self._concat_stream_copy(videos)

        if not success:
            # Fallback: re-encode concat
            logger.info("[Finalizer] Stream copy failed, falling back to re-encode concat")
            success = self._concat_reencode(videos)

        if not success:
            raise FinalizerError("[Finalizer] All concatenation methods failed.")

        # Log final video info
        self._log_video_info(self.final_output_path)

        return self.final_output_path

    def _discover_videos(self) -> List[Path]:
        """Auto-discover per-cycle videos in chronological order."""
        # Look in videos_dir first, then project_dir root
        videos = sorted(self.videos_dir.glob("cycle*_video.mp4")) if self.videos_dir.exists() else []
        if not videos:
            videos = sorted(self.project_dir.glob("cycle*_video.mp4"))
        return videos

    def _concat_stream_copy(self, videos: List[Path]) -> bool:
        """Concatenate using stream copy (no re-encoding). Fastest method."""
        concat_file = self.project_dir / "_concat_list.txt"

        try:
            with open(concat_file, 'w') as f:
                for vid in videos:
                    f.write(f"file '{vid.resolve()}'\n")

            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c', 'copy',
                str(self.final_output_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0 and self.final_output_path.exists():
                logger.info("[Finalizer] Stream-copy concat succeeded")
                return True

            logger.warning(f"[Finalizer] Stream-copy failed: {result.stderr[:200]}")
            return False

        finally:
            if concat_file.exists():
                concat_file.unlink()

    def _concat_reencode(self, videos: List[Path]) -> bool:
        """Concatenate with re-encoding. Slower but handles mixed formats."""
        concat_file = self.project_dir / "_concat_list.txt"

        try:
            with open(concat_file, 'w') as f:
                for vid in videos:
                    f.write(f"file '{vid.resolve()}'\n")

            # Try preferred codec, then fallback
            for codec in [self.preferred_codec, 'libx264']:
                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', str(concat_file),
                    '-c:v', codec,
                    '-crf', str(self.output_quality),
                    '-pix_fmt', 'yuv420p',
                    str(self.final_output_path)
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0 and self.final_output_path.exists():
                    logger.info(f"[Finalizer] Re-encode concat succeeded with {codec}")
                    return True

                logger.warning(f"[Finalizer] Re-encode with {codec} failed: {result.stderr[:200]}")

            return False

        finally:
            if concat_file.exists():
                concat_file.unlink()

    def _log_video_info(self, video_path: Path) -> None:
        """Log final video duration, fps, resolution."""
        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,r_frame_rate,duration',
                '-show_entries', 'format=duration',
                '-of', 'json',
                str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                import json
                info = json.loads(result.stdout)

                stream = info.get('streams', [{}])[0] if info.get('streams') else {}
                fmt = info.get('format', {})

                width = stream.get('width', '?')
                height = stream.get('height', '?')
                fps_str = stream.get('r_frame_rate', '?')
                duration = float(fmt.get('duration', 0) or stream.get('duration', 0))

                # Parse fps fraction
                fps = '?'
                if '/' in str(fps_str):
                    num, den = fps_str.split('/')
                    if int(den) > 0:
                        fps = f"{int(num) / int(den):.1f}"

                file_size_mb = video_path.stat().st_size / (1024 * 1024)

                logger.info("=" * 50)
                logger.info("[Finalizer] FINAL OUTPUT READY")
                logger.info(f"  File: {video_path}")
                logger.info(f"  Resolution: {width}x{height}")
                logger.info(f"  FPS: {fps}")
                logger.info(f"  Duration: {duration:.1f}s ({duration/60:.1f} min)")
                logger.info(f"  Size: {file_size_mb:.1f} MB")
                logger.info("=" * 50)

        except Exception as e:
            logger.warning(f"[Finalizer] Could not probe final video info: {e}")
            logger.info(f"[Finalizer] Final output saved to: {video_path}")


__all__ = ['Finalizer', 'FinalizerError']
