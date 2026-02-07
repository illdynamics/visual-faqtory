#!/usr/bin/env python3
"""
finalizer.py - Video Stitching, Interpolation & Upscale Pipeline
═══════════════════════════════════════════════════════════════════════════════

Finalizer is responsible for:
  1. Stitching all per-cycle MP4s into a single final_output.mp4 (BASE MASTER)
  2. POST-STITCH FINALIZER (runs ONCE after all cycles):
     → Interpolate to 60fps (minterpolate)
     → Upscale to 1920×1080 (bicubic)
     → Encode with h264_nvenc (GPU) or libx264 (CPU fallback)
     → Produce final_60fps_1080p.mp4 (FINAL DELIVERABLE)

Pipeline position (MANDATORY):
  cycle generation
  → cycle stitching (crossfade / etc)
  → final_output.mp4 (BASE MASTER — 8fps, 1024×576)
  → FINALIZER:
       → interpolation to 60fps
       → upscale to 1920×1080
       → encode final output
  → final_60fps_1080p.mp4 (FINAL DELIVERABLE)
  → pipeline exit

NON-NEGOTIABLE RULES:
  ❌ MUST NOT run per cycle
  ❌ MUST NOT run before final stitching
  ❌ MUST NOT modify the raw stitched master
  ❌ MUST NOT double-run if pipeline is resumed or re-entered
  ✅ MUST run once, after final stitching, before pipeline exits

Supports NVENC-based encoding (h264_nvenc preferred, libx264 fallback).

Part of QonQrete Visual FaQtory v0.3.5-beta
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
    Stitches per-cycle looped MP4s into a single final output video,
    then optionally runs post-stitch interpolation + upscale.
    """

    def __init__(
        self,
        project_dir: Path,
        preferred_codec: str = "h264_nvenc",
        output_quality: int = 18,
        finalizer_config: Optional[Dict[str, Any]] = None
    ):
        self.project_dir = Path(project_dir)
        self.preferred_codec = preferred_codec
        self.output_quality = output_quality
        self.videos_dir = self.project_dir / "videos"
        self.final_output_path = self.project_dir / "final_output.mp4"

        # Post-stitch finalizer config
        cfg = finalizer_config or {}
        self.finalizer_enabled = cfg.get('enabled', False)
        self.interpolate_fps = cfg.get('interpolate_fps', 60)
        self.upscale_width = 1920
        self.upscale_height = 1080
        upscale_res = cfg.get('upscale_resolution', '1920x1080')
        if isinstance(upscale_res, str) and 'x' in upscale_res:
            parts = upscale_res.split('x')
            self.upscale_width = int(parts[0])
            self.upscale_height = int(parts[1])
        self.scale_algo = cfg.get('scale_algo', 'bicubic')
        self.finalizer_crf = cfg.get('quality', {}).get('crf', 16) if isinstance(cfg.get('quality'), dict) else cfg.get('crf', 16)
        self.encoder_preference = cfg.get('encoder_preference', ['h264_nvenc', 'libx264'])

        # Deliverable paths
        self.final_deliverable_path = self.project_dir / "final_60fps_1080p.mp4"
        self._interpolated_temp_path = self.project_dir / "_temp_interpolated_60fps.mp4"

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

        # Log stitched video info
        self._log_video_info(self.final_output_path, label="STITCHED BASE MASTER")

        return self.final_output_path

    def run_post_stitch_finalizer(self) -> Optional[Path]:
        """
        Run the post-stitch finalizer: interpolation → upscale → encode.

        This MUST only be called ONCE, AFTER final stitching is complete.
        It NEVER modifies the base stitched master (final_output.mp4).

        Returns:
            Path to final_60fps_1080p.mp4 or None if skipped/failed.
        """
        # ── Guard: finalizer disabled ──
        if not self.finalizer_enabled:
            logger.info("[Finalizer] Post-stitch finalizer is DISABLED in config. Skipping.")
            return None

        # ── Guard: base master must exist ──
        if not self.final_output_path.exists():
            logger.warning(
                "[Finalizer] Cannot run post-stitch finalizer: "
                f"base master not found at {self.final_output_path}"
            )
            return None

        # ── Guard: already completed (idempotent / no double-run) ──
        if self.final_deliverable_path.exists():
            logger.info(
                f"[Finalizer] Post-stitch deliverable already exists: "
                f"{self.final_deliverable_path} — skipping finalizer."
            )
            return self.final_deliverable_path

        logger.info("=" * 60)
        logger.info("[Finalizer] POST-STITCH FINALIZER STARTING")
        logger.info(f"  Input:  {self.final_output_path}")
        logger.info(f"  Target: {self.interpolate_fps}fps → "
                     f"{self.upscale_width}x{self.upscale_height}")
        logger.info("=" * 60)

        # ── Detect best encoder ──
        encoder, encoder_args = self._detect_encoder()
        logger.info(f"[Finalizer] Encoder selected: {encoder}")

        # ── STEP 1: Interpolation (8fps → 60fps) ──
        logger.info(f"[Finalizer] STEP 1/2: Interpolating to {self.interpolate_fps}fps...")
        interp_success = self._interpolate(
            input_path=self.final_output_path,
            output_path=self._interpolated_temp_path,
            encoder=encoder,
            encoder_args=encoder_args
        )

        if not interp_success:
            logger.error("[Finalizer] Interpolation FAILED. Post-stitch finalizer aborted.")
            self._cleanup_temp()
            return None

        logger.info("[Finalizer] Interpolation SUCCESS ✓")

        # ── STEP 2: Upscale (1024×576 → 1920×1080) ──
        logger.info(
            f"[Finalizer] STEP 2/2: Upscaling to "
            f"{self.upscale_width}x{self.upscale_height}..."
        )
        upscale_success = self._upscale(
            input_path=self._interpolated_temp_path,
            output_path=self.final_deliverable_path,
            encoder=encoder,
            encoder_args=encoder_args
        )

        if not upscale_success:
            logger.error("[Finalizer] Upscale FAILED. Post-stitch finalizer aborted.")
            self._cleanup_temp()
            return None

        logger.info("[Finalizer] Upscale SUCCESS ✓")

        # ── Cleanup temp interpolated file ──
        self._cleanup_temp()

        # ── Log deliverable info ──
        self._log_video_info(self.final_deliverable_path, label="FINAL DELIVERABLE")

        logger.info("=" * 60)
        logger.info("[Finalizer] POST-STITCH FINALIZER COMPLETE")
        logger.info(f"  Base master:  {self.final_output_path}")
        logger.info(f"  Deliverable:  {self.final_deliverable_path}")
        logger.info("=" * 60)

        return self.final_deliverable_path

    # ──────────────────────────────────────────────────────────────────────────
    # Post-stitch internal methods
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_encoder(self) -> tuple:
        """
        Detect the best available encoder.
        Tries h264_nvenc first, falls back to libx264.

        Returns:
            (encoder_name, encoder_args_list)
        """
        for encoder in self.encoder_preference:
            if encoder == 'h264_nvenc':
                # Test NVENC availability with a minimal encode
                test_cmd = [
                    'ffmpeg', '-y',
                    '-f', 'lavfi', '-i', 'nullsrc=s=64x64:d=0.1',
                    '-c:v', 'h264_nvenc',
                    '-f', 'null', '-'
                ]
                try:
                    result = subprocess.run(
                        test_cmd, capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0:
                        logger.info("[Finalizer] NVENC (h264_nvenc) detected and available")
                        return 'h264_nvenc', [
                            '-c:v', 'h264_nvenc',
                            '-cq', str(self.finalizer_crf),
                            '-preset', 'p5',
                            '-pix_fmt', 'yuv420p'
                        ]
                    else:
                        logger.info(
                            f"[Finalizer] h264_nvenc not available: "
                            f"{result.stderr[:200]}"
                        )
                except Exception as e:
                    logger.info(f"[Finalizer] h264_nvenc test failed: {e}")

            elif encoder == 'libx264':
                logger.info("[Finalizer] Using libx264 (CPU fallback)")
                return 'libx264', [
                    '-c:v', 'libx264',
                    '-crf', str(self.finalizer_crf),
                    '-preset', 'slow',
                    '-pix_fmt', 'yuv420p'
                ]
            else:
                logger.warning(f"[Finalizer] Unknown encoder in preference list: {encoder}")

        # If ALL encoders in preference list failed, hard fail
        raise FinalizerError(
            "[Finalizer] FATAL: No usable video encoder found. "
            f"Tried: {self.encoder_preference}. "
            "Ensure ffmpeg is installed with h264_nvenc or libx264 support."
        )

    def _interpolate(
        self,
        input_path: Path,
        output_path: Path,
        encoder: str,
        encoder_args: list
    ) -> bool:
        """
        Interpolate video to target fps using minterpolate filter.
        This MUST run before upscaling.
        """
        cmd = [
            'ffmpeg', '-y',
            '-i', str(input_path),
            '-vf', (
                f'minterpolate=fps={self.interpolate_fps}'
                f':mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1'
            ),
            '-r', str(self.interpolate_fps),
            *encoder_args,
            str(output_path)
        ]

        logger.info(f"[Finalizer] Interpolation cmd: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=7200
            )
            if result.returncode == 0 and output_path.exists():
                size_mb = output_path.stat().st_size / (1024 * 1024)
                logger.info(
                    f"[Finalizer] Interpolated file: {output_path} "
                    f"({size_mb:.1f} MB)"
                )
                return True
            else:
                logger.error(
                    f"[Finalizer] Interpolation failed (rc={result.returncode}): "
                    f"{result.stderr[:500]}"
                )
                return False
        except subprocess.TimeoutExpired:
            logger.error("[Finalizer] Interpolation timed out (2h limit)")
            return False
        except Exception as e:
            logger.error(f"[Finalizer] Interpolation error: {e}")
            return False

    def _upscale(
        self,
        input_path: Path,
        output_path: Path,
        encoder: str,
        encoder_args: list
    ) -> bool:
        """
        Upscale video to target resolution using bicubic scaling.
        This MUST run after interpolation.
        """
        cmd = [
            'ffmpeg', '-y',
            '-i', str(input_path),
            '-vf', (
                f'scale={self.upscale_width}:{self.upscale_height}'
                f':flags={self.scale_algo}'
            ),
            '-r', str(self.interpolate_fps),
            *encoder_args,
            str(output_path)
        ]

        logger.info(f"[Finalizer] Upscale cmd: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=7200
            )
            if result.returncode == 0 and output_path.exists():
                size_mb = output_path.stat().st_size / (1024 * 1024)
                logger.info(
                    f"[Finalizer] Upscaled file: {output_path} "
                    f"({size_mb:.1f} MB)"
                )
                return True
            else:
                logger.error(
                    f"[Finalizer] Upscale failed (rc={result.returncode}): "
                    f"{result.stderr[:500]}"
                )
                return False
        except subprocess.TimeoutExpired:
            logger.error("[Finalizer] Upscale timed out (2h limit)")
            return False
        except Exception as e:
            logger.error(f"[Finalizer] Upscale error: {e}")
            return False

    def _cleanup_temp(self) -> None:
        """Remove temporary intermediate files."""
        try:
            if self._interpolated_temp_path.exists():
                self._interpolated_temp_path.unlink()
                logger.info(
                    f"[Finalizer] Cleaned up temp: {self._interpolated_temp_path}"
                )
        except Exception as e:
            logger.warning(f"[Finalizer] Temp cleanup failed: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # Stitching methods (unchanged from v0.0.5)
    # ──────────────────────────────────────────────────────────────────────────

    def _discover_videos(self) -> List[Path]:
        """Auto-discover per-cycle videos in chronological order."""
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

    def _log_video_info(self, video_path: Path, label: str = "FINAL OUTPUT") -> None:
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

                fps = '?'
                if '/' in str(fps_str):
                    num, den = fps_str.split('/')
                    if int(den) > 0:
                        fps = f"{int(num) / int(den):.1f}"

                file_size_mb = video_path.stat().st_size / (1024 * 1024)

                logger.info("=" * 50)
                logger.info(f"[Finalizer] {label} READY")
                logger.info(f"  File: {video_path}")
                logger.info(f"  Resolution: {width}x{height}")
                logger.info(f"  FPS: {fps}")
                logger.info(f"  Duration: {duration:.1f}s ({duration/60:.1f} min)")
                logger.info(f"  Size: {file_size_mb:.1f} MB")
                logger.info("=" * 50)

        except Exception as e:
            logger.warning(f"[Finalizer] Could not probe video info: {e}")
            logger.info(f"[Finalizer] {label} saved to: {video_path}")


__all__ = ['Finalizer', 'FinalizerError']
