#!/usr/bin/env python3
"""
construqtor.py - Visual Construction Agent
═══════════════════════════════════════════════════════════════════════════════

ConstruQtor is the second agent in the Visual FaQtory pipeline.

Responsibilities:
  1. Receive VisualBriq from InstruQtor
  2. VALIDATE required inputs per mode (fail-fast)
  3. Call configured backend (mock/comfyui)
  4. Generate image from prompt (txt2img or img2img)
  5. Generate video from image (img2vid)
  6. Save raw video to project directory as cycleN_raw.mp4

Mode handling:
  - TEXT mode: txt2img → img2vid (no base inputs required)
  - IMAGE mode: requires base_image_path, feed to img2img → img2vid
  - VIDEO mode: extract frame from video → treat as IMAGE mode

Part of QonQrete Visual FaQtory v0.5.6-beta
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

        # Initialize backend
        if backend:
            self.backend = backend
        else:
            # Merge lora config into backend config if present at top-level
            backend_config = config.get('backend', {'type': 'mock'}).copy()
            lora_cfg = config.get('lora')
            if lora_cfg:
                backend_config['lora'] = lora_cfg
                try:
                    enabled = bool(lora_cfg.get('enabled', False))
                except Exception:
                    enabled = False
                if enabled:
                    if backend_config.get('type', '').lower() != 'comfyui':
                        raise FatalConfigError(
                            "LoRA enabled but backend type is not 'comfyui'. "
                            "Set backend: comfyui or disable lora.enabled."
                        )
            self.backend = create_backend(backend_config)

        # Check backend availability
        available, msg = self.backend.check_availability()
        if not available:
            logger.warning(f"Backend not fully available: {msg}")
        else:
            logger.info(f"[ConstruQtor] Backend ready: {msg}")

    def _validate_briq(self, briq: VisualBriq) -> None:
        """Validate required inputs per mode. Fail fast with clear errors."""
        if briq.mode == InputMode.TEXT:
            if not briq.prompt or not briq.prompt.strip():
                raise ValueError(
                    "[ConstruQtor] TEXT mode requires a non-empty prompt. "
                    "Check your story.txt."
                )

        elif briq.mode == InputMode.IMAGE:
            if not briq.base_image_path:
                raise ValueError(
                    "[ConstruQtor] IMAGE mode requires base_image_path. "
                    "Place an image in worqspace/base_images/."
                )
            if not briq.base_image_path.exists():
                raise FileNotFoundError(
                    f"[ConstruQtor] IMAGE mode base image not found: {briq.base_image_path}"
                )

        elif briq.mode == InputMode.VIDEO:
            if not briq.base_video_path:
                raise ValueError(
                    "[ConstruQtor] VIDEO mode requires base_video_path. "
                    "Place a video in worqspace/base_video/."
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

        Pipeline per mode:
          TEXT:  txt2img → img2vid → save
          IMAGE: img2img → img2vid → save
          VIDEO: extract frame → img2img → img2vid → save

        Args:
            briq: The VisualBriq with generation instructions

        Returns:
            Updated briq with raw_video_path set
        """
        logger.info(f"[ConstruQtor] Starting construction for briq {briq.briq_id}")
        self._validate_briq(briq)

        briq.status = BriqStatus.CONSTRUCTING
        briq.backend_used = self.backend.name
        start_time = time.time()

        try:
            if briq.mode == InputMode.VIDEO:
                # VIDEO mode: extract frame, then use as image base
                init_frame = self._extract_frame_from_video(briq.base_video_path)
                if not init_frame:
                    raise RuntimeError(
                        f"[ConstruQtor] Failed to extract frame from video: {briq.base_video_path}"
                    )
                briq.source_image_path = init_frame
                logger.info(f"[ConstruQtor] VIDEO mode: extracted frame → {init_frame}")

            if briq.mode in (InputMode.IMAGE, InputMode.VIDEO):
                # IMAGE/VIDEO mode: use base/extracted image directly for video gen
                if briq.mode == InputMode.IMAGE:
                    briq.source_image_path = briq.base_image_path
                    logger.info(f"[ConstruQtor] IMAGE mode: using base image: {briq.base_image_path}")

                video_result = self._generate_video(briq, briq.source_image_path)
                if not video_result.success:
                    raise RuntimeError(f"Video generation failed: {video_result.error}")
                briq.raw_video_path = self._save_to_qodeyard(briq, video_result.video_path)

            else:
                # TEXT mode: txt2img → img2vid
                image_result = self._generate_image(briq)
                if not image_result.success:
                    raise RuntimeError(f"Image generation failed: {image_result.error}")
                briq.source_image_path = image_result.image_path
                logger.info(f"[ConstruQtor] Image generated: {image_result.image_path}")

                video_result = self._generate_video(briq, briq.source_image_path)
                if not video_result.success:
                    raise RuntimeError(f"Video generation failed: {video_result.error}")
                briq.raw_video_path = self._save_to_qodeyard(briq, video_result.video_path)

            briq.generation_time = time.time() - start_time
            briq.status = BriqStatus.CONSTRUCTED
            logger.info(f"[ConstruQtor] Construction complete: {briq.raw_video_path} "
                       f"({briq.generation_time:.1f}s)")

        except FatalConfigError:
            briq.status = BriqStatus.FAILED
            raise
        except Exception as e:
            briq.status = BriqStatus.FAILED
            briq.error_message = str(e)
            logger.error(f"[ConstruQtor] Construction failed: {e}")
            raise

        return briq

    def _generate_image(self, briq: VisualBriq) -> GenerationResult:
        """Generate source image based on briq mode."""
        prompt = briq.get_full_prompt()
        negative = briq.negative_prompt

        request = GenerationRequest(
            prompt=prompt,
            negative_prompt=negative,
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
        """Generate video from source image.

        Uses briq.video_prompt if available. Guarantees non-empty prompt
        for text-conditioned video models.
        """
        video_prompt = briq.get_video_prompt()
        negative = briq.negative_prompt
        motion = getattr(briq, 'motion_prompt', None) or None

        if not video_prompt or not video_prompt.strip():
            video_prompt = briq.prompt or "cinematic motion, smooth transition"

        request = GenerationRequest(
            prompt=video_prompt,
            negative_prompt=negative,
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
            video_prompt=video_prompt,
            motion_prompt=motion,
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
