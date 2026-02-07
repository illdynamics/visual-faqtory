#!/usr/bin/env python3
"""
construqtor.py - Visual Construction Agent
═══════════════════════════════════════════════════════════════════════════════

ConstruQtor is the second agent in the Visual FaQtory pipeline.

Responsibilities:
  1. Receive VisualBriq from InstruQtor
  2. VALIDATE required inputs per mode (fail-fast)
  3. Call configured backend (mock/comfyui/diffusers/replicate)
  4. Generate image from prompt (txt2img or img2img)
  5. Generate video from image (img2vid)
  6. Save raw video to project directory as cycleN_raw.mp4

Mode handling (v0.3.x):
  - TEXT mode: txt2img → img2vid (no base inputs required)
  - IMAGE mode: requires base_image_path, skip image gen, feed to video pipeline
  - VIDEO mode: requires base_video_path, preprocess → video2video (NO image fallback)
  - STREAM mode: context tail from previous cycle → continuation generation

Part of QonQrete Visual FaQtory v0.3.5-beta
"""
import os
import time
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

from .visual_briq import VisualBriq, BriqStatus, InputMode
from .backends import (
    create_backend, GenerationRequest, GenerationResult,
    GeneratorBackend, FatalConfigError
)
from .video_preprocess import preprocess_video, get_video_info
from .stream_engine import (
    extract_video_context, extract_last_frame,
    get_stream_config, prepare_stream_cycle,
    compute_beat_aligned_generation_length,
)

logger = logging.getLogger(__name__)


class ConstruQtor:
    """
    The construction agent that generates visuals from briqs.

    Validates inputs per mode and handles backend communication.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        qodeyard_dir: Path,
        backend: Optional[GeneratorBackend] = None
    ):
        self.config = config
        self.qodeyard_dir = Path(qodeyard_dir)
        self.qodeyard_dir.mkdir(parents=True, exist_ok=True)

        # Video2Video config (v0.1.0)
        self.v2v_config = config.get('input', {}).get('video2video', {})
        self.v2v_enabled = self.v2v_config.get('enabled', True)

        # Stream mode config (v0.2.0-beta)
        self.stream_config = get_stream_config(config)
        self.stream_enabled = self.stream_config.get('enabled', False)

        # Initialize backend
        if backend:
            self.backend = backend
        else:
            backend_config = config.get('backend', {'type': 'mock'})
            self.backend = create_backend(backend_config)

        # Wire V2V workflow path into backend config so generate_video2video() can find it
        v2v_workflow = self.v2v_config.get('comfyui', {}).get('workflow')
        if v2v_workflow:
            self.backend.config['v2v_workflow'] = v2v_workflow
            # Also propagate to inner backends in SplitBackend
            if hasattr(self.backend, 'video_backend'):
                self.backend.video_backend.config['v2v_workflow'] = v2v_workflow

        # Check backend availability
        available, msg = self.backend.check_availability()
        if not available:
            logger.warning(f"Backend not fully available: {msg}")
        else:
            logger.info(f"[ConstruQtor] Backend ready: {msg}")

        # ─── Color Stability Controller (v0.3.5-beta) ────────────────
        # Applied to normal offline runs and stream/longcat output.
        # Must be optional via config, never blocks rendering.
        from .color_stability import create_stability_controller
        self.stability = create_stability_controller(config)
        if self.stability:
            logger.info("[ConstruQtor] Color stability controller enabled")

    def _validate_briq(self, briq: VisualBriq) -> None:
        """
        Validate required inputs per mode. Fail fast with clear errors.
        """
        if briq.mode == InputMode.TEXT:
            if not briq.prompt or not briq.prompt.strip():
                raise ValueError(
                    "[ConstruQtor] TEXT mode requires a non-empty prompt. "
                    "Check your tasq.md."
                )
            # TEXT mode must NOT attempt direct text→video
            # It always goes text→image→video

        elif briq.mode == InputMode.IMAGE:
            if not briq.base_image_path:
                raise ValueError(
                    "[ConstruQtor] IMAGE mode requires base_image_path. "
                    "Set 'input_image' or 'base_image' in tasq.md."
                )
            if not briq.base_image_path.exists():
                raise FileNotFoundError(
                    f"[ConstruQtor] IMAGE mode base image not found: {briq.base_image_path}"
                )

        elif briq.mode == InputMode.VIDEO:
            if not briq.base_video_path:
                raise ValueError(
                    "[ConstruQtor] VIDEO mode requires base_video_path (previous cycle output). "
                    "This should be set automatically by the pipeline for cycle N>0."
                )
            if not briq.base_video_path.exists():
                raise FileNotFoundError(
                    f"[ConstruQtor] VIDEO mode base video not found: {briq.base_video_path}"
                )
        else:
            raise ValueError(f"[ConstruQtor] Unknown mode: {briq.mode}")

    def construct(self, briq: VisualBriq) -> VisualBriq:
        """
        Construct visual content from a VisualBriq.

        Pipeline per mode (v0.1.1-alpha):
          TEXT:  txt2img → img2vid → save
          IMAGE: (skip img gen) → img2vid → save
          VIDEO: preprocess → video2video → save  (NO image fallback)

        Args:
            briq: The VisualBriq with generation instructions

        Returns:
            Updated briq with raw_video_path set
        """
        logger.info(f"[ConstruQtor] Starting construction for briq {briq.briq_id}")

        # VALIDATE before doing any work
        self._validate_briq(briq)

        briq.status = BriqStatus.CONSTRUCTING
        briq.backend_used = self.backend.name

        start_time = time.time()

        try:
            # ── STREAM MODE ROUTING (v0.2.0-beta) ─────────────────────
            if self.stream_enabled and briq.context_video_path:
                self._construct_stream_video(briq)

            elif briq.mode == InputMode.VIDEO:
                # ── VIDEO MODE: true video2video (v0.1.1) ────────────────
                # NEVER falls back to image pipeline. NEVER extracts frames.
                self._construct_video2video(briq)

            elif briq.mode == InputMode.IMAGE:
                # IMAGE mode: skip image generation, feed base image directly to video
                briq.source_image_path = briq.base_image_path
                logger.info(f"[ConstruQtor] IMAGE mode: using base image directly: {briq.base_image_path}")

                # Generate video from image
                video_result = self._generate_video(briq, briq.source_image_path)
                if not video_result.success:
                    raise RuntimeError(f"Video generation failed: {video_result.error}")
                # Apply stability correction (v0.3.5-beta)
                corrected = self._apply_stability_to_video(video_result.video_path)
                raw_video_path = self._save_to_qodeyard(briq, corrected)
                briq.raw_video_path = raw_video_path

            else:
                # TEXT mode: generate/transform image, then img2vid
                image_result = self._generate_image(briq)
                if not image_result.success:
                    raise RuntimeError(f"Image generation failed: {image_result.error}")
                briq.source_image_path = image_result.image_path
                logger.info(f"[ConstruQtor] Image generated: {image_result.image_path}")

                # Generate video from image
                video_result = self._generate_video(briq, briq.source_image_path)
                if not video_result.success:
                    raise RuntimeError(f"Video generation failed: {video_result.error}")
                # Apply stability correction (v0.3.5-beta)
                corrected = self._apply_stability_to_video(video_result.video_path)
                raw_video_path = self._save_to_qodeyard(briq, corrected)
                briq.raw_video_path = raw_video_path

            briq.generation_time = time.time() - start_time
            briq.status = BriqStatus.CONSTRUCTED

            logger.info(f"[ConstruQtor] Construction complete: {briq.raw_video_path} "
                       f"({briq.generation_time:.1f}s)")

        except FatalConfigError:
            # Config errors bubble up immediately — no recovery
            briq.status = BriqStatus.FAILED
            raise
        except Exception as e:
            briq.status = BriqStatus.FAILED
            briq.error_message = str(e)
            logger.error(f"[ConstruQtor] Construction failed: {e}")
            raise

        return briq

    def _construct_video2video(self, briq: VisualBriq) -> None:
        """
        True Video2Video pipeline (v0.1.1).

        Steps:
          1. Preprocess video (MANDATORY — abort if fails)
          2. Call backend.generate_video2video() (MANDATORY — no fallback)
          3. Save result

        STRICTLY FORBIDDEN:
          ❌ Falling back to frame extraction
          ❌ Falling back to image→video
          ❌ Silent behavior changes
        """
        if not self.v2v_enabled:
            raise RuntimeError(
                "[ConstruQtor] VIDEO mode requires video2video to be enabled. "
                "Set input.video2video.enabled: true in config.yaml"
            )

        # ── STEP 1: Preprocess (MANDATORY) ──────────────────────────────
        preprocess_cfg = self.v2v_config.get('preprocess', {})
        precond_path = self.qodeyard_dir / f"precond_{briq.briq_id}.mp4"

        try:
            preprocess_video(briq.base_video_path, precond_path, preprocess_cfg)
            briq.v2v_preprocessed_path = precond_path
            logger.info(f"[ConstruQtor] V2V preprocessed: {precond_path}")
        except Exception as e:
            # Preprocessing failure → abort THIS cycle (no fallback)
            raise RuntimeError(
                f"[ConstruQtor] V2V preprocessing FAILED for {briq.base_video_path}: {e}. "
                f"Aborting cycle (no image fallback)."
            )

        # ── STEP 2: Build V2V request ───────────────────────────────────
        v2v_comfyui = self.v2v_config.get('comfyui', {})
        v2v_sampler = v2v_comfyui.get('sampler', {})
        denoise = v2v_sampler.get('denoise', 0.35)

        request = GenerationRequest(
            prompt=briq.get_full_prompt(),
            negative_prompt=briq.negative_prompt,
            seed=briq.seed,
            mode=briq.mode,
            base_video_path=precond_path,
            width=briq.spec.width,
            height=briq.spec.height,
            cfg_scale=v2v_sampler.get('cfg', briq.spec.cfg_scale),
            steps=v2v_sampler.get('steps', briq.spec.steps),
            sampler=briq.spec.sampler,
            denoise_strength=denoise,
            video_frames=briq.spec.video_frames,
            video_fps=briq.spec.video_fps,
            output_dir=self.qodeyard_dir,
            atom_id=briq.briq_id,
            video_prompt=getattr(briq, 'video_prompt', None) or None,
            motion_prompt=getattr(briq, 'motion_prompt', None) or None,
        )

        # ── STEP 3: Call V2V backend (MANDATORY — no fallback) ──────────
        result = self.backend.generate_video2video(request)
        if not result.success:
            raise RuntimeError(
                f"[ConstruQtor] Video2Video generation FAILED: {result.error}. "
                f"No image-pipeline fallback."
            )

        # ── STEP 4: Save ────────────────────────────────────────────────
        # Apply stability correction (v0.3.5-beta)
        corrected = self._apply_stability_to_video(result.video_path)
        raw_video_path = self._save_to_qodeyard(briq, corrected)
        briq.raw_video_path = raw_video_path
        logger.info(f"[ConstruQtor] V2V output saved: {raw_video_path}")

    def _construct_stream_video(self, briq: VisualBriq) -> None:
        """
        Stream continuation pipeline (v0.2.0-beta).

        Uses context_video (tail of previous cycle) for autoregressive continuation.
        Falls back to V2V/img2vid if stream workflow unavailable.
        """
        logger.info(f"[ConstruQtor] STREAM mode: using context from {briq.context_video_path}")

        # Build generation request with context video
        video_prompt = briq.get_video_prompt()
        gen_frames = briq.spec.generation_frames or self.stream_config.get('generation_length', 72)

        request = GenerationRequest(
            prompt=video_prompt,
            negative_prompt=briq.negative_prompt,
            seed=briq.seed,
            mode=briq.mode,
            base_video_path=briq.context_video_path,
            width=briq.spec.width,
            height=briq.spec.height,
            cfg_scale=briq.spec.cfg_scale,
            steps=briq.spec.steps,
            sampler=briq.spec.sampler,
            denoise_strength=briq.spec.denoise_strength,
            video_frames=gen_frames,
            video_fps=briq.spec.video_fps,
            output_dir=self.qodeyard_dir,
            atom_id=briq.briq_id,
            video_prompt=getattr(briq, 'video_prompt', None) or None,
            motion_prompt=getattr(briq, 'motion_prompt', None) or None,
        )

        # Try stream-capable backend first
        try:
            if hasattr(self.backend, 'generate_stream_video'):
                result = self.backend.generate_stream_video(request, self.stream_config)
                if result.success:
                    # Apply stability correction (v0.3.5-beta)
                    corrected = self._apply_stability_to_video(result.video_path)
                    stream_path = self.qodeyard_dir / f"cycle{briq.cycle_index:04d}_stream.mp4"
                    import shutil
                    if corrected and Path(corrected).exists():
                        shutil.copy2(corrected, stream_path)
                    briq.stream_video_path = stream_path
                    briq.raw_video_path = self._save_to_qodeyard(briq, result.video_path)
                    logger.info(f"[ConstruQtor] Stream output: {stream_path}")
                    return
                else:
                    logger.warning(f"[ConstruQtor] Stream generation failed: {result.error}, trying fallback")
        except NotImplementedError:
            logger.info("[ConstruQtor] Backend doesn't support stream continuation, using fallback")

        # Fallback: use context video as V2V input with locked seed
        logger.info("[ConstruQtor] Stream fallback: using context as V2V input")
        request.denoise_strength = 0.35  # Conservative for continuity
        try:
            result = self.backend.generate_video2video(request)
            if result.success:
                stream_path = self.qodeyard_dir / f"cycle{briq.cycle_index:04d}_stream.mp4"
                import shutil
                if result.video_path and result.video_path.exists():
                    shutil.copy2(result.video_path, stream_path)
                briq.stream_video_path = stream_path
                briq.raw_video_path = self._save_to_qodeyard(briq, result.video_path)
                return
        except Exception as e:
            logger.warning(f"[ConstruQtor] V2V fallback failed: {e}")

        # Final fallback: use last frame as img2vid base
        logger.warning("[ConstruQtor] All stream paths failed, falling back to img2vid from last frame")
        last_frame = self.qodeyard_dir / f"cycle{briq.cycle_index:04d}_last_frame.png"
        try:
            extract_last_frame(briq.context_video_path, last_frame)
            briq.source_image_path = last_frame
            video_result = self._generate_video(briq, last_frame)
            if video_result.success:
                briq.raw_video_path = self._save_to_qodeyard(briq, video_result.video_path)
                return
        except Exception as e:
            raise RuntimeError(f"[ConstruQtor] All stream generation paths failed: {e}")

    def _apply_stability_to_video(self, video_path: Path) -> Path:
        """Apply color stability correction to a generated video (v0.3.5-beta).

        Decodes frames, runs each through the stability controller, and
        re-encodes. If stability is disabled, numpy is missing, or any
        error occurs, returns the original video path unchanged (never
        blocks rendering).

        Args:
            video_path: Path to the raw generated video

        Returns:
            Path to corrected video (may be same as input if skipped)
        """
        if not self.stability:
            return video_path

        try:
            import numpy as _np
            from PIL import Image as _PIL
        except ImportError:
            logger.debug("[ConstruQtor] numpy/PIL not available, skipping stability")
            return video_path

        try:
            import subprocess as sp
            import io
            import tempfile

            # Probe video info
            probe = sp.run(
                ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                 '-show_entries', 'stream=width,height,r_frame_rate',
                 '-of', 'csv=p=0', str(video_path)],
                capture_output=True, text=True
            )
            if probe.returncode != 0:
                return video_path

            parts = probe.stdout.strip().split(',')
            if len(parts) < 3:
                return video_path
            w, h = int(parts[0]), int(parts[1])
            fps_str = parts[2]
            # Parse rational fps (e.g. "8/1")
            if '/' in fps_str:
                num, den = fps_str.split('/')
                fps = float(num) / float(den)
            else:
                fps = float(fps_str)

            # Extract frames as raw RGB
            extract = sp.run(
                ['ffmpeg', '-y', '-i', str(video_path),
                 '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                capture_output=True
            )
            if extract.returncode != 0:
                return video_path

            raw = extract.stdout
            frame_size = w * h * 3
            n_frames = len(raw) // frame_size
            if n_frames < 2:
                return video_path

            # Process each frame through stability controller
            corrected_frames = bytearray()
            for i in range(n_frames):
                frame_data = raw[i * frame_size:(i + 1) * frame_size]
                frame_np = _np.frombuffer(frame_data, dtype=_np.uint8).reshape(h, w, 3).copy()
                corrected = self.stability.process_frame(frame_np)
                corrected_frames.extend(corrected.tobytes())

            # Re-encode corrected frames
            corrected_path = video_path.parent / f"{video_path.stem}_stable{video_path.suffix}"
            encode = sp.run(
                ['ffmpeg', '-y',
                 '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                 '-s', f'{w}x{h}', '-r', str(fps),
                 '-i', '-',
                 '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18',
                 '-pix_fmt', 'yuv420p',
                 str(corrected_path)],
                input=bytes(corrected_frames), capture_output=True
            )
            if encode.returncode == 0 and corrected_path.exists():
                logger.info(
                    f"[ConstruQtor] Stability correction applied: "
                    f"{n_frames} frames processed"
                )
                return corrected_path
            else:
                return video_path

        except Exception as e:
            # Never block rendering on stability failure
            logger.debug(f"[ConstruQtor] Stability post-process error: {e}")
            return video_path

    def _generate_image(self, briq: VisualBriq) -> GenerationResult:
        """Generate source image based on briq mode."""
        request = GenerationRequest(
            prompt=briq.get_full_prompt(),
            negative_prompt=briq.negative_prompt,
            seed=briq.seed,
            mode=briq.mode,
            width=briq.spec.width,
            height=briq.spec.height,
            cfg_scale=briq.spec.cfg_scale,
            steps=briq.spec.steps,
            sampler=briq.spec.sampler,
            denoise_strength=briq.spec.denoise_strength,
            video_frames=briq.spec.video_frames,
            video_fps=briq.spec.video_fps,
            motion_bucket_id=briq.spec.motion_bucket_id,
            noise_aug_strength=briq.spec.noise_aug_strength,
            output_dir=self.qodeyard_dir,
            atom_id=briq.briq_id
        )

        if briq.mode == InputMode.IMAGE and briq.base_image_path:
            request.base_image_path = briq.base_image_path
            request.init_image_path = briq.base_image_path
        elif briq.mode == InputMode.VIDEO and briq.base_video_path:
            init_frame = self._extract_frame_from_video(briq.base_video_path)
            if init_frame:
                request.init_image_path = init_frame
                request.base_image_path = init_frame
                request.mode = InputMode.IMAGE
            else:
                raise RuntimeError(
                    f"[ConstruQtor] Failed to extract frame from video: {briq.base_video_path}"
                )

        return self.backend.generate_image(request)

    def _generate_video(self, briq: VisualBriq, source_image: Path) -> GenerationResult:
        """Generate video from source image. Uses briq.video_prompt if available."""
        # For video generation, prefer video_prompt over image prompt
        video_prompt = briq.get_video_prompt()

        request = GenerationRequest(
            prompt=video_prompt,
            negative_prompt=briq.negative_prompt,
            seed=briq.seed,
            mode=briq.mode,
            width=briq.spec.width,
            height=briq.spec.height,
            video_frames=briq.spec.video_frames,
            video_fps=briq.spec.video_fps,
            motion_bucket_id=briq.spec.motion_bucket_id,
            noise_aug_strength=briq.spec.noise_aug_strength,
            output_dir=self.qodeyard_dir,
            atom_id=briq.briq_id,
            # Prompt Bundle extensions (v0.0.7)
            video_prompt=getattr(briq, 'video_prompt', None) or None,
            motion_prompt=getattr(briq, 'motion_prompt', None) or None,
        )
        return self.backend.generate_video(request, source_image)

    def _extract_frame_from_video(self, video_path: Path) -> Optional[Path]:
        """Extract middle frame from video for img2img base."""
        if not video_path or not video_path.exists():
            return None

        output_path = self.qodeyard_dir / f"{video_path.stem}_frame.png"

        try:
            probe_cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(video_path)
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            duration = float(result.stdout.strip() or 5.0)

            timestamp = duration / 2
            extract_cmd = [
                'ffmpeg', '-y',
                '-ss', str(timestamp),
                '-i', str(video_path),
                '-vframes', '1',
                '-q:v', '2',
                str(output_path)
            ]
            subprocess.run(extract_cmd, capture_output=True, check=True)

            if output_path.exists():
                logger.info(f"[ConstruQtor] Extracted frame from video: {output_path}")
                return output_path

        except Exception as e:
            logger.warning(f"Failed to extract frame from video: {e}")

        return None

    def _save_to_qodeyard(self, briq: VisualBriq, video_path: Path) -> Path:
        """Save/move video to project directory with standardized naming."""
        target_name = f"cycle{briq.cycle_index:04d}_raw.mp4"
        target_path = self.qodeyard_dir / target_name

        if video_path == target_path:
            return target_path

        if video_path.exists():
            import shutil
            shutil.copy2(video_path, target_path)
            logger.info(f"[ConstruQtor] Saved to project: {target_path}")
        else:
            logger.warning(f"Video not found at {video_path}, using {target_path}")

        return target_path


__all__ = ['ConstruQtor']
