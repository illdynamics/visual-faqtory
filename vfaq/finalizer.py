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

Part of Visual FaQtory v0.5.6-beta
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

        # Ensure cfg is defined from finalizer_config
        cfg = finalizer_config or {}

        # Per-cycle interpolation config
        self.per_cycle_interpolation = cfg.get('per_cycle_interpolation', False)
        self.per_cycle_pingpong = cfg.get('per_cycle_pingpong', False)  # New config for pingpong

        # Backwards-compatible FPS knobs:
        # - Prefer explicit per-cycle key
        # - Fall back to legacy interpolate_fps for per-cycle ONLY
        self.per_cycle_interpolate_fps = cfg.get('per_cycle_interpolate_fps')
        if self.per_cycle_interpolate_fps is None:
            self.per_cycle_interpolate_fps = cfg.get('interpolate_fps', 30)

        if self.per_cycle_interpolation:
            self.videos_interpolated_dir = self.project_dir / "videos_interpolated"
            self.videos_interpolated_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[Finalizer] Per-cycle interpolation enabled. Output dir: {self.videos_interpolated_dir}")
        else:
            self.videos_interpolated_dir = None

        # Post-stitch finalizer config
        self.finalizer_enabled = cfg.get('enabled', False)

        # Keep post-stitch FPS independent from per-cycle interpolation.
        # Users can override with post_interpolate_fps.
        self.interpolate_fps = cfg.get('post_interpolate_fps', 60)

        # Backwards compat: if per-cycle interpolation is OFF and user only set interpolate_fps,
        # treat interpolate_fps as post-stitch fps.
        if (not self.per_cycle_interpolation) and ('post_interpolate_fps' not in cfg) and ('interpolate_fps' in cfg):
            self.interpolate_fps = cfg.get('interpolate_fps', 60)
        self.upscale_width = 1920
        self.upscale_height = 1080
        upscale_res = cfg.get('upscale_resolution', '1920x1080')
        if isinstance(upscale_res, str) and 'x' in upscale_res:
            parts = upscale_res.split('x')
            self.upscale_width = int(parts[0])
            self.upscale_height = int(parts[1])
        self.scale_algo = cfg.get('scale_algo', 'bicubic')
        
        # Quality parsing: can be int or {crf: int}
        raw_quality = cfg.get('quality')
        if isinstance(raw_quality, dict):
            self.per_cycle_quality = raw_quality.get('crf', 18) # Default 18 for per-cycle
            self.finalizer_crf = raw_quality.get('crf', 16) # Default 16 for post-stitch
        elif isinstance(raw_quality, int):
            self.per_cycle_quality = raw_quality
            self.finalizer_crf = raw_quality
        else:
            self.per_cycle_quality = 18
            self.finalizer_crf = 16

        self.encoder_preference = cfg.get('encoder_preference', ['h264_nvenc', 'libx264'])

        # Deliverable paths
        self.final_deliverable_path = self.project_dir / "final_60fps_1080p.mp4"
        self._interpolated_temp_path = self.project_dir / "_temp_interpolated_60fps.mp4"

        self._encoder: Optional[str] = None
        # _encoder_args will no longer be cached here, generated per-use dynamically

    def _get_encoder_args(self, encoder_name: str, quality: int) -> list:
        """
        Generate FFmpeg encoder arguments based on encoder name and quality.
        """
        if encoder_name == 'h264_nvenc':
            return [
                '-c:v', 'h264_nvenc',
                '-cq', str(quality),
                '-preset', 'p5',
                '-pix_fmt', 'yuv420p'
            ]
        elif encoder_name == 'libx264':
            return [
                '-c:v', 'libx264',
                '-crf', str(quality),
                '-preset', 'slow',
                '-pix_fmt', 'yuv420p'
            ]
        else:
            raise FinalizerError(f"Unknown encoder_name: {encoder_name}")

    def _detect_encoder(self) -> str:
        """
        Detect the best available encoder based on preference and system capabilities.
        Tries encoders in self.encoder_preference.

        Returns:
            str: The name of the detected encoder ('h264_nvenc' or 'libx264').
        Raises:
            FinalizerError: If no usable encoder is found after all attempts.
        """
        for encoder_pref in self.encoder_preference:
            if encoder_pref == 'auto': # 'auto' is a special keyword, handled implicitly by trying specific encoders
                continue

            if encoder_pref == 'h264_nvenc':
                # Test NVENC availability with a minimal encode
                test_cmd = [
                    'ffmpeg', '-y',
                    '-f', 'lavfi', '-i', 'nullsrc=s=64x64:d=0.1', # Small dummy input
                    '-c:v', 'h264_nvenc',
                    '-f', 'null', '-' # Encode to null output
                ]
                try:
                    result = subprocess.run(
                        test_cmd, capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0:
                        logger.info("[Finalizer] NVENC (h264_nvenc) detected and available.")
                        return 'h264_nvenc'
                    else:
                        logger.info(
                            f"[Finalizer] h264_nvenc not available or failed: "
                            f"Exit code {result.returncode}. Stderr: {result.stderr[:200]}"
                        )
                except Exception as e:
                    logger.info(f"[Finalizer] h264_nvenc test failed unexpectedly: {e}")

            elif encoder_pref == 'libx264':
                logger.info("[Finalizer] Using libx264 (CPU fallback) as preferred.")
                return 'libx264'

            else:
                logger.warning(f"[Finalizer] Unknown encoder in preference list, skipping: {encoder_pref}")

        # If we reach here, none of the explicitly preferred encoders were found or worked.
        # Default to libx264 if it wasn't already picked up or specified.
        if 'libx264' in self.encoder_preference or 'auto' in self.encoder_preference: # Explicit or implicit fallback
            logger.info("[Finalizer] Falling back to libx264 as no other preferred encoder is available.")
            return 'libx264'
        
        # If no encoders available at all, raise a fatal error.
        raise FinalizerError(
            "[Finalizer] FATAL: No usable video encoder found after checking preferences. "
            f"Tried: {self.encoder_preference}. "
            "Ensure ffmpeg is installed with h264_nvenc or libx264 support."
        )


    def _get_or_detect_encoder(self) -> str:
        """Detect encoder once and cache its name."""
        if self._encoder:
            return self._encoder
        
        self._encoder = self._detect_encoder()
        return self._encoder

    @staticmethod
    def _probe_video_metadata(video_path: Path) -> Dict[str, Any]:
        meta: Dict[str, Any] = {'duration': 0.0, 'frames': None, 'fps': 0.0}
        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-count_frames',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=nb_read_frames,nb_frames,avg_frame_rate,r_frame_rate,duration',
                '-show_entries', 'format=duration',
                '-of', 'json',
                str(video_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            import json
            payload = json.loads(result.stdout or '{}')
            stream = (payload.get('streams') or [{}])[0]
            fmt = payload.get('format') or {}

            def _to_float(value):
                if value in (None, '', 'N/A'):
                    return None
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None

            fps_raw = stream.get('avg_frame_rate') or stream.get('r_frame_rate')
            if isinstance(fps_raw, str) and '/' in fps_raw:
                num, den = fps_raw.split('/', 1)
                try:
                    den_f = float(den)
                    if den_f:
                        meta['fps'] = float(num) / den_f
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
            else:
                fps_val = _to_float(fps_raw)
                if fps_val:
                    meta['fps'] = fps_val

            frames_raw = stream.get('nb_read_frames') or stream.get('nb_frames')
            if frames_raw not in (None, '', 'N/A'):
                try:
                    meta['frames'] = int(frames_raw)
                except (TypeError, ValueError):
                    meta['frames'] = None

            duration = _to_float(fmt.get('duration')) or _to_float(stream.get('duration'))
            if duration is None and meta['frames'] and meta['fps'] > 0:
                duration = meta['frames'] / meta['fps']
            meta['duration'] = float(duration or 0.0)
        except Exception as e:
            logger.warning(f"[Finalizer] Could not probe video metadata for {video_path}: {e}")
        return meta

    def _processed_video_is_usable(self, input_path: Path, output_path: Path, target_fps: float) -> bool:
        if not output_path.exists() or output_path.stat().st_size <= 0:
            return False
        in_meta = self._probe_video_metadata(input_path)
        out_meta = self._probe_video_metadata(output_path)
        out_duration = float(out_meta.get('duration') or 0.0)
        out_frames = out_meta.get('frames')
        if out_duration <= 0.0 and not out_frames:
            logger.error(f"[Finalizer] Processed video has no usable duration/frame metadata: {output_path}")
            return False
        if out_frames is not None and out_frames <= 1:
            logger.error(f"[Finalizer] Processed video collapsed to {out_frames} frame: {output_path}")
            return False
        in_duration = float(in_meta.get('duration') or 0.0)
        if in_duration > 0 and out_duration > 0 and out_duration < max(0.20, in_duration * 0.20):
            logger.error(
                f"[Finalizer] Processed video duration shrank too far ({out_duration:.3f}s from {in_duration:.3f}s): {output_path}"
            )
            return False
        out_fps = float(out_meta.get('fps') or 0.0)
        if target_fps > 0 and out_fps > 0 and out_fps < max(1.0, target_fps * 0.5):
            logger.error(
                f"[Finalizer] Processed video fps looks wrong ({out_fps:.2f} < expected around {target_fps:.2f}): {output_path}"
            )
            return False
        return True
        
    def _process_cycle_video(self, input_path: Path, input_fps: float) -> Optional[Path]:
        """
        Performs per-cycle video processing (interpolation, optionally with pingpong).

        Args:
            input_path: Path to the raw SVD video file.
            input_fps: The original FPS of the input video (resolved_fps from GenerationRequest).

        Returns:
            Path to the processed video, or None if processing fails.
        """
        if not self.per_cycle_interpolation:
            return input_path # Skip processing if disabled

        if not input_path.exists():
            logger.error(f"[Finalizer] Cannot process cycle: Input video not found at {input_path}")
            return None

        output_dir = self.videos_interpolated_dir
        if not output_dir: # Should not happen if per_cycle_interpolation is true
            logger.error("[Finalizer] videos_interpolated_dir is not set for per-cycle interpolation.")
            return None
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Output path: run/videos_interpolated/<same base name>.mp4
        output_path = output_dir / input_path.name
        temp_output_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
        if temp_output_path.exists():
            temp_output_path.unlink()

        encoder = self._get_or_detect_encoder()
        encoder_args = self._get_encoder_args(encoder, self.per_cycle_quality)

        filter_complex_str = ""
        target_fps = self.per_cycle_interpolate_fps

        if self.per_cycle_pingpong:
            logger.info(f"[Finalizer] Performing per-cycle PINGPONG + INTERPOLATION on {input_path.name} to {target_fps}fps...")
            # Normalize input timestamp before pingpong and interpolation
            filter_complex_str = (
                f"[0:v]setpts=PTS-STARTPTS[v0];"
                f"[v0]split=2[vf][vr];"
                f"[vr]reverse,select='not(eq(n,0))'[vr2];"
                f"[vf][vr2]concat=n=2:v=1:a=0,"
                f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
            )
        else: # Only interpolation
            logger.info(f"[Finalizer] Performing per-cycle INTERPOLATION on {input_path.name} to {target_fps}fps...")
            filter_complex_str = (
                f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
            )
        
        cmd = [
            'ffmpeg', '-y',
            '-fflags', '+genpts',
            '-i', str(input_path),
            '-filter_complex', filter_complex_str,
            '-an',
            '-fps_mode', 'cfr',
            '-r', str(target_fps),
            *encoder_args,
            '-movflags', '+faststart',
            str(temp_output_path)
        ]
        
        logger.debug(f"[Finalizer] Per-cycle Processing cmd: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0 and self._processed_video_is_usable(input_path, temp_output_path, float(target_fps)):
                temp_output_path.replace(output_path)
                logger.info(f"[Finalizer] Cycle processing SUCCESS: {output_path}")
                return output_path

            logger.error(
                f"[Finalizer] Cycle processing FAILED (rc={result.returncode}) for {input_path.name}: "
                f"{result.stderr}"
            )
            if result.stdout:
                logger.debug(f"[Finalizer] FFmpeg stdout: {result.stdout}")
            return None
        except subprocess.TimeoutExpired:
            logger.error(f"[Finalizer] Cycle processing timed out for {input_path.name}")
            return None
        except Exception as e:
            logger.error(f"[Finalizer] Cycle processing error for {input_path.name}: {e}")
            return None
        finally:
            if temp_output_path.exists():
                temp_output_path.unlink(missing_ok=True)

    def finalize(
        self,
        cycle_video_paths: Optional[List[Path]] = None,
        failed_cycles: Optional[List[int]] = None
    ) -> Path:
        """
        Stitch all per-cycle videos into final_output.mp4.

        v0.5.6-beta: Failed cycles are SKIPPED, not fatal. The finalizer
        stitches whatever videos exist. cycle_video_paths only contains
        successful cycles anyway, so failed_cycles is informational.

        Args:
            cycle_video_paths: Explicit list of video paths (chronological order).
                               If None, auto-discovers from videos_dir.
            failed_cycles: List of cycle indices that failed (logged, not fatal).

        Returns:
            Path to final_output.mp4

        Raises:
            FinalizerError: If no videos found at all (zero successful cycles).
        """
        # v0.5.6-beta: Failed cycles are SKIPPED, not fatal.
        # The finalizer stitches whatever videos exist.
        # cycle_video_paths only contains successful cycles anyway,
        # so failed_cycles is just informational logging.
        if failed_cycles:
            logger.warning(
                f"[Finalizer] {len(failed_cycles)} cycle(s) failed: {failed_cycles}. "
                f"Skipping them — stitching remaining videos."
            )

        # Collect videos
        if cycle_video_paths:
            videos = [Path(p) for p in cycle_video_paths if Path(p).exists()]
        else:
            # If per_cycle_interpolation is enabled and no explicit paths are given,
            # discover from the interpolated videos directory.
            if self.per_cycle_interpolation and self.videos_interpolated_dir:
                logger.info("[Finalizer] Discovering videos from interpolated directory.")
                videos = sorted(self.videos_interpolated_dir.glob("processed_cycle*_video.mp4"))
            else:
                videos = self._discover_videos() # Fallback to original videos_dir

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
        encoder = self._get_or_detect_encoder()
        encoder_args = self._get_encoder_args(encoder, self.finalizer_crf)
        logger.info(f"[Finalizer] Encoder selected: {encoder}")

        # ── STEP 1: Interpolation (8fps → 60fps) ──
        logger.info(f"[Finalizer] STEP 1/2: Interpolating to {self.interpolate_fps}fps...")
        interp_success = self._interpolate(
            input_path=self.final_output_path,
            output_path=self._interpolated_temp_path,
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

    def _interpolate(
        self,
        input_path: Path,
        output_path: Path,
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

            encoder = self._get_or_detect_encoder()
            encoder_args = self._get_encoder_args(encoder, self.output_quality)

            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                *encoder_args,
                str(self.final_output_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0 and self.final_output_path.exists():
                logger.info(f"[Finalizer] Re-encode concat succeeded with {encoder}")
                return True

            logger.warning(f"[Finalizer] Re-encode with {encoder} failed: {result.stderr[:200]}")
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
