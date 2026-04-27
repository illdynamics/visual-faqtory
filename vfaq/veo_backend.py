#!/usr/bin/env python3
"""
veo_backend.py — Google Veo Video Generation Backend
═══════════════════════════════════════════════════════════════════════════════

First-class Veo backend for Visual FaQtory, supporting:
  - Google Gemini Developer API (provider=gemini)
  - Google Vertex AI (provider=vertex)
  - text_to_video, image_to_video, first_last_frame, extend_video modes
  - Immediate file download / durable persistence
  - Retry with exponential backoff + jitter
  - Full observability logging

Uses the official Google Gen AI SDK (google-genai).

Part of Visual FaQtory v0.6.0-beta
"""
import base64
import json
import logging
import os
import random
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# VEO ENUMS & DATA
# ═══════════════════════════════════════════════════════════════════════════════


class VeoProvider(Enum):
    """Supported Veo API providers."""
    GEMINI = "gemini"
    VERTEX = "vertex"


class VeoMode(Enum):
    """Veo generation modes."""
    TEXT_TO_VIDEO = "text_to_video"
    IMAGE_TO_VIDEO = "image_to_video"
    FIRST_LAST_FRAME = "first_last_frame"
    EXTEND_VIDEO = "extend_video"


# Allowed Veo 3.1 durations (seconds) — 4s, 6s, or 8s.
# 8s is required for 1080p, 4K, and reference-image usage.
# Output framerate is 24fps natively.
VEO_ALLOWED_DURATIONS = [4, 6, 8]

# Maximum retries for transient failures
VEO_MAX_RETRIES = 3
VEO_BASE_BACKOFF = 2.0
VEO_MAX_BACKOFF = 60.0
VEO_POLL_INTERVAL = 10.0
VEO_POLL_TIMEOUT = 600.0  # 10 minutes max poll


def _clamp_duration(requested: float) -> int:
    """Clamp requested duration to nearest allowed Veo duration."""
    if requested <= 0:
        return 8
    closest = min(VEO_ALLOWED_DURATIONS, key=lambda d: abs(d - requested))
    if closest != int(requested):
        logger.info(f"[Veo] Clamped duration {requested}s → {closest}s (nearest allowed)")
    return closest


def _load_image_bytes(image_path: Path) -> Tuple[bytes, str]:
    """Load image from disk and return (bytes, mime_type)."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    suffix = image_path.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    mime_type = mime_map.get(suffix, "image/png")
    return image_path.read_bytes(), mime_type


# ═══════════════════════════════════════════════════════════════════════════════
# VEO CONFIG DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class VeoConfig:
    """Parsed Veo configuration from YAML veo: section."""
    provider: str = "gemini"
    model: str = "veo-3.1-generate-preview"
    fast_model: str = "veo-3.1-fast-generate-preview"
    api_version: str = "v1beta"
    default_mode: str = "image_to_video"
    duration_seconds: int = 8
    aspect_ratio: str = "16:9"
    resolution: str = "720p"
    sample_count: int = 1
    person_generation: str = "allow_adult"
    generate_audio: bool = False
    storage_uri: Optional[str] = None
    resize_mode: str = "PAD"
    compression_quality: str = "optimized"
    enable_reference_images: bool = True
    enable_last_frame: bool = True
    enable_extension: bool = False
    continuity_strength: float = 0.85
    mutation_strength: float = 0.25
    identity_lock_strength: float = 0.80
    loop_closure_strength: float = 0.90
    prefer_fast_model: bool = False
    download_immediately: bool = True
    # Polling
    poll_interval: float = VEO_POLL_INTERVAL
    poll_timeout: float = VEO_POLL_TIMEOUT
    # Retry
    max_retries: int = VEO_MAX_RETRIES

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VeoConfig":
        """Create VeoConfig from a dictionary (e.g. YAML veo: section)."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


# ═══════════════════════════════════════════════════════════════════════════════
# AUTH RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════════


def _resolve_auth(provider: VeoProvider) -> Dict[str, Any]:
    """Resolve authentication for the given provider.

    Returns a dict with keys needed for google.genai.Client() initialization.

    For gemini:
      Tries GOOGLE_API_KEY → GEMINI_API_KEY → GOOGLE_API_TOKEN → GEMINI_API_TOKEN

    For vertex:
      Uses ADC (Application Default Credentials) with optional
      GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION.
    """
    if provider == VeoProvider.GEMINI:
        api_key = (
            os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_TOKEN")
            or os.environ.get("GEMINI_API_TOKEN")
        )
        if not api_key:
            raise RuntimeError(
                "[Veo Auth] No API key found for provider=gemini. "
                "Set one of: GOOGLE_API_KEY, GEMINI_API_KEY, "
                "GOOGLE_API_TOKEN, GEMINI_API_TOKEN"
            )
        logger.info("[Veo Auth] Using Gemini Developer API (API key)")
        return {"api_key": api_key}

    elif provider == VeoProvider.VERTEX:
        # Vertex AI uses ADC — set env flags
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        if not project:
            raise RuntimeError(
                "[Veo Auth] No project found for provider=vertex. "
                "Set GOOGLE_CLOUD_PROJECT environment variable."
            )
        logger.info(f"[Veo Auth] Using Vertex AI (ADC) project={project} location={location}")
        return {"vertexai": True, "project": project, "location": location}

    raise RuntimeError(f"[Veo Auth] Unknown provider: {provider}")


# ═══════════════════════════════════════════════════════════════════════════════
# VEO BACKEND
# ═══════════════════════════════════════════════════════════════════════════════


class VeoBackend:
    """
    Google Veo video generation backend for Visual FaQtory.

    Implements the GeneratorBackend interface (generate_image, generate_video,
    generate_morph_video, check_availability) using the Google Gen AI SDK.

    Unlike ComfyUI, Veo generates video directly from text or image — there is
    no separate txt2img step. The backend handles this by:
      - generate_image: returns a still frame extracted from a short Veo clip
        (or uses the input image directly for image modes)
      - generate_video: full Veo video generation
      - generate_morph_video: uses first_last_frame mode
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize VeoBackend from config dict.

        Args:
            config: Backend config dict. Expected keys:
                - veo: dict with VeoConfig fields
                - width/height: resolution hints (mapped to Veo aspect_ratio/resolution)
        """
        self.config = config
        self.name = "veo"

        # Parse Veo-specific config
        veo_dict = config.get("veo", {})
        self.veo_cfg = VeoConfig.from_dict(veo_dict)

        # Resolve provider
        self.provider = VeoProvider(self.veo_cfg.provider.lower())

        # Resolve authentication
        self._auth_kwargs = _resolve_auth(self.provider)
        self._auth_mode = "api_key" if "api_key" in self._auth_kwargs else "adc"

        # Initialize Google GenAI Client
        try:
            from google import genai
            from google.genai import types as genai_types
            self._genai = genai
            self._genai_types = genai_types

            # Pass api_version via http_options if configured
            client_kwargs = dict(self._auth_kwargs)
            if self.veo_cfg.api_version:
                client_kwargs["http_options"] = genai_types.HttpOptions(
                    api_version=self.veo_cfg.api_version
                )

            self._client = genai.Client(**client_kwargs)
            logger.info(
                f"[Veo] Client initialized: provider={self.provider.value} "
                f"model={self._effective_model()} auth={self._auth_mode} "
                f"api_version={self.veo_cfg.api_version}"
            )
        except Exception as e:
            raise RuntimeError(f"[Veo] Failed to initialize Google GenAI client: {e}")

    def _effective_model(self) -> str:
        """Return the effective model ID based on prefer_fast_model config."""
        if self.veo_cfg.prefer_fast_model:
            return self.veo_cfg.fast_model
        return self.veo_cfg.model

    # ─── Core generation method ───────────────────────────────────────────

    def _generate_veo_video(
        self,
        prompt: str,
        mode: VeoMode,
        output_dir: Path,
        atom_id: str,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        duration_seconds: Optional[int] = None,
        aspect_ratio: Optional[str] = None,
        resolution: Optional[str] = None,
        image_path: Optional[Path] = None,
        last_frame_path: Optional[Path] = None,
        input_video_path: Optional[Path] = None,
        reference_image_paths: Optional[List[Path]] = None,
        reference_image_types: Optional[List[str]] = None,
        generate_audio: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Core Veo generation dispatcher. Returns metadata dict with output path.

        Args:
            prompt: Text prompt for generation.
            mode: VeoMode enum.
            output_dir: Where to save the output video.
            atom_id: Unique identifier for this generation.
            seed: Random seed.
            negative_prompt: Negative prompt (if supported).
            duration_seconds: Video duration in seconds.
            aspect_ratio: Aspect ratio string (e.g. "16:9").
            resolution: Resolution string (e.g. "720p").
            image_path: Starting image for image_to_video / first_last_frame.
            last_frame_path: Ending image for first_last_frame.
            input_video_path: Input video for extend_video.
            reference_image_paths: Reference images for style/subject.
            reference_image_types: Corresponding reference types ("STYLE"/"ASSET").
            generate_audio: Whether to generate audio track.

        Returns:
            Dict with keys: success, video_path, metadata, error
        """
        from google.genai import types

        start_time = time.time()

        # Resolve defaults from config
        aspect = aspect_ratio or self.veo_cfg.aspect_ratio
        res = resolution or self.veo_cfg.resolution
        model_id = self._effective_model()
        gen_audio = generate_audio if generate_audio is not None else self.veo_cfg.generate_audio

        # ── Duration handling ─────────────────────────────────────────────
        # Extension output is fixed ~7s by Veo; don't send duration_seconds.
        # For other modes, clamp to [4, 6, 8].
        if mode == VeoMode.EXTEND_VIDEO:
            duration = None  # Veo controls extension output duration
            # Extension output is fixed at 720p/24fps per Veo docs.
            # Force resolution regardless of global config.
            if res and res.lower() != "720p":
                logger.info(
                    f"[Veo] Extension mode: forcing resolution 720p "
                    f"(was '{res}' — extension output is always 720p/24fps/~7s)"
                )
            res = "720p"
            logger.info("[Veo] Extension mode: duration/resolution controlled by Veo (720p, ~7s, 24fps)")
        else:
            duration = _clamp_duration(duration_seconds or self.veo_cfg.duration_seconds)

        # ── Capability guards (Veo 3.1) ───────────────────────────────────
        # Validate aspect ratio: Veo 3.1 only supports 16:9 and 9:16
        is_veo3 = '3.' in model_id or '3-' in model_id
        if is_veo3 and aspect not in ("16:9", "9:16"):
            logger.warning(
                f"[Veo] Aspect ratio '{aspect}' not supported on Veo 3.1 "
                f"(only 16:9, 9:16). Forcing 16:9."
            )
            aspect = "16:9"

        # 8s required for 1080p, 4K, or reference images on Veo 3.1
        if is_veo3 and duration is not None:
            needs_8s = (
                (res and res.lower() in ("1080p", "4k"))
                or (reference_image_paths and len(reference_image_paths) > 0)
            )
            if needs_8s and duration < 8:
                logger.info(
                    f"[Veo] Forcing duration to 8s (required for "
                    f"{'resolution=' + res if res else 'reference images'})"
                )
                duration = 8

        # Build source
        source_kwargs: Dict[str, Any] = {"prompt": prompt}

        if mode in (VeoMode.IMAGE_TO_VIDEO, VeoMode.FIRST_LAST_FRAME):
            if image_path and image_path.exists():
                img_bytes, img_mime = _load_image_bytes(image_path)
                source_kwargs["image"] = types.Image(
                    image_bytes=img_bytes, mime_type=img_mime
                )
            else:
                logger.warning(f"[Veo] Mode {mode.value} but no valid image_path — falling back to text_to_video")
                mode = VeoMode.TEXT_TO_VIDEO

        if mode == VeoMode.EXTEND_VIDEO:
            if input_video_path and input_video_path.exists():
                if self.provider == VeoProvider.VERTEX and self.veo_cfg.storage_uri:
                    # Vertex: upload to GCS and use URI (preferred path)
                    import uuid
                    gcs_dest = f"{self.veo_cfg.storage_uri.rstrip('/')}/extend_input_{uuid.uuid4().hex[:8]}.mp4"
                    logger.info(f"[Veo] Uploading extension input to GCS: {gcs_dest}")
                    try:
                        subprocess.run(
                            ["gsutil", "cp", str(input_video_path), gcs_dest],
                            capture_output=True, text=True, check=True, timeout=120,
                        )
                        source_kwargs["video"] = types.Video(
                            uri=gcs_dest, mime_type="video/mp4"
                        )
                    except Exception as e:
                        logger.warning(f"[Veo] GCS upload failed, falling back to video_bytes: {e}")
                        vid_bytes = input_video_path.read_bytes()
                        source_kwargs["video"] = types.Video(
                            video_bytes=vid_bytes, mime_type="video/mp4"
                        )
                elif self.provider == VeoProvider.VERTEX:
                    # Vertex without storage_uri: use video_bytes
                    vid_bytes = input_video_path.read_bytes()
                    source_kwargs["video"] = types.Video(
                        video_bytes=vid_bytes, mime_type="video/mp4"
                    )
                else:
                    # Gemini Developer API: extension support is limited.
                    # The SDK docs note video bytes are not supported on the
                    # developer API for extension. Try anyway but warn.
                    logger.warning(
                        "[Veo] Video extension on Gemini Developer API has limited support. "
                        "Consider using provider=vertex for reliable extension."
                    )
                    vid_bytes = input_video_path.read_bytes()
                    source_kwargs["video"] = types.Video(
                        video_bytes=vid_bytes, mime_type="video/mp4"
                    )
            else:
                logger.warning("[Veo] extend_video but no valid input_video — falling back to text_to_video")
                mode = VeoMode.TEXT_TO_VIDEO

        source = types.GenerateVideosSource(**source_kwargs)

        # ══════════════════════════════════════════════════════════════════
        # Build GenerateVideosConfig — provider-aware parameter gating
        # ══════════════════════════════════════════════════════════════════
        # The Gemini Developer API accepts a much smaller parameter set than
        # Vertex AI. Sending unsupported params causes immediate rejection.
        #
        # Gemini Developer API proven working:
        #   prompt (in source), image (in source), aspect_ratio,
        #   negative_prompt, duration_seconds, last_frame, enhance_prompt
        #
        # Vertex AI additional (all rejected by Gemini):
        #   seed, generate_audio, person_generation, compression_quality,
        #   number_of_videos, resolution, output_gcs_uri, pubsub_topic,
        #   reference_images
        # ══════════════════════════════════════════════════════════════════
        is_vertex = (self.provider == VeoProvider.VERTEX)

        # Start with params supported on BOTH providers
        config_kwargs: Dict[str, Any] = {
            "aspect_ratio": aspect,
        }

        # Duration: set for non-extension modes (both providers)
        if duration is not None:
            config_kwargs["duration_seconds"] = duration

        # Negative prompt (both providers)
        if negative_prompt:
            config_kwargs["negative_prompt"] = negative_prompt

        # Last frame for first_last_frame mode (both providers)
        if mode == VeoMode.FIRST_LAST_FRAME and last_frame_path and last_frame_path.exists():
            lf_bytes, lf_mime = _load_image_bytes(last_frame_path)
            config_kwargs["last_frame"] = types.Image(
                image_bytes=lf_bytes, mime_type=lf_mime
            )

        # Reference images — Vertex only.
        # Despite docs suggesting Gemini support, the Gemini Developer API
        # rejects reference_images with INVALID_ARGUMENT in practice.
        if (
            is_vertex
            and self.veo_cfg.enable_reference_images
            and reference_image_paths
            and reference_image_types
        ):
            ref_images = []
            for rpath, rtype in zip(reference_image_paths, reference_image_types):
                if rpath.exists():
                    rb, rm = _load_image_bytes(rpath)
                    ref_images.append(types.VideoGenerationReferenceImage(
                        image=types.Image(image_bytes=rb, mime_type=rm),
                        reference_type=rtype.upper(),
                    ))
            if ref_images:
                config_kwargs["reference_images"] = ref_images
        elif (
            not is_vertex
            and reference_image_paths
            and reference_image_types
        ):
            logger.info(
                f"[Veo] Gemini API: skipping {len(reference_image_paths)} reference image(s) "
                f"(not supported on Gemini Developer API)"
            )

        # ── Vertex-only parameters ────────────────────────────────────────
        # These are silently skipped on Gemini with an info log.
        vertex_only_params = {}
        if seed is not None:
            vertex_only_params["seed"] = seed
        if gen_audio:
            vertex_only_params["generate_audio"] = gen_audio
        if self.veo_cfg.person_generation:
            vertex_only_params["person_generation"] = self.veo_cfg.person_generation
        cq = self.veo_cfg.compression_quality.upper()
        if cq in ("LOSSLESS", "OPTIMIZED"):
            vertex_only_params["compression_quality"] = cq
        if res:
            vertex_only_params["resolution"] = res
        if self.veo_cfg.sample_count and self.veo_cfg.sample_count > 1:
            vertex_only_params["number_of_videos"] = self.veo_cfg.sample_count
        if self.veo_cfg.storage_uri:
            vertex_only_params["output_gcs_uri"] = self.veo_cfg.storage_uri

        if is_vertex:
            config_kwargs.update(vertex_only_params)
        else:
            # Log which params are being skipped on Gemini
            skipped = [k for k, v in vertex_only_params.items() if v is not None and v != ""]
            if skipped:
                logger.info(
                    f"[Veo] Gemini API: skipping Vertex-only params: {', '.join(skipped)}"
                )

        # sample_count warning (Vertex: downloads all but pipeline uses first)
        if is_vertex and self.veo_cfg.sample_count > 1:
            logger.warning(
                f"[Veo] sample_count={self.veo_cfg.sample_count} — all videos downloaded "
                f"but pipeline uses only the first."
            )

        # resize_mode: not in the Python SDK at all (REST API only)
        if self.veo_cfg.resize_mode and self.veo_cfg.resize_mode.upper() != "PAD":
            logger.warning(
                f"[Veo] resize_mode='{self.veo_cfg.resize_mode}' is set but the "
                f"google-genai Python SDK does not expose this parameter."
            )

        # Log exactly what config keys are being sent (debug)
        config_keys = sorted(config_kwargs.keys())
        logger.info(f"[Veo] Config params being sent: {', '.join(config_keys)}")

        veo_config = types.GenerateVideosConfig(**config_kwargs)

        # Log request
        logger.info(
            f"[Veo] Generating video: model={model_id} mode={mode.value} "
            f"duration={duration}s aspect={aspect} resolution={res} "
            f"seed={seed} audio={gen_audio} provider={self.provider.value}"
        )

        # ── Execute with retry ─────────────────────────────────────────────
        last_error = None
        for attempt in range(1, self.veo_cfg.max_retries + 1):
            try:
                operation = self._client.models.generate_videos(
                    model=model_id,
                    source=source,
                    config=veo_config,
                )
                logger.info(f"[Veo] Operation started: {operation.name} (attempt {attempt})")

                # Poll until done
                video_path = self._poll_and_download(
                    operation, output_dir, atom_id, model_id, mode
                )

                elapsed = time.time() - start_time
                logger.info(
                    f"[Veo] Generation complete: {video_path} "
                    f"({elapsed:.1f}s total, model={model_id})"
                )

                return {
                    "success": True,
                    "video_path": video_path,
                    "error": None,
                    "metadata": {
                        "backend": "veo",
                        "provider": self.provider.value,
                        "model": model_id,
                        "mode": mode.value,
                        "duration_seconds": duration,
                        "aspect_ratio": aspect,
                        "resolution": res,
                        "seed": seed,
                        "generation_time": elapsed,
                        "auth_mode": self._auth_mode,
                        "atom_id": atom_id,
                        "attempt": attempt,
                    },
                }

            except Exception as e:
                last_error = e
                err_str = str(e).lower()

                # Non-retryable errors
                if any(k in err_str for k in ["auth", "permission", "forbidden", "invalid_api_key"]):
                    logger.error(f"[Veo] Auth failure (non-retryable): {e}")
                    break
                if any(k in err_str for k in ["unsupported", "invalid", "not supported", "not available", "does not support"]):
                    logger.error(f"[Veo] Parameter error (non-retryable): {e}")
                    break

                # Retryable: quota, timeout, server error
                if attempt < self.veo_cfg.max_retries:
                    backoff = min(
                        VEO_BASE_BACKOFF * (2 ** (attempt - 1)) + random.uniform(0, 1),
                        VEO_MAX_BACKOFF,
                    )
                    logger.warning(
                        f"[Veo] Transient failure (attempt {attempt}/{self.veo_cfg.max_retries}): {e}. "
                        f"Retrying in {backoff:.1f}s..."
                    )
                    time.sleep(backoff)
                else:
                    logger.error(f"[Veo] All {self.veo_cfg.max_retries} attempts failed: {e}")

        return {
            "success": False,
            "video_path": None,
            "error": str(last_error),
            "metadata": {
                "backend": "veo",
                "provider": self.provider.value,
                "model": model_id,
                "mode": mode.value,
                "attempts": self.veo_cfg.max_retries,
            },
        }

    def _poll_and_download(
        self,
        operation,
        output_dir: Path,
        atom_id: str,
        model_id: str,
        mode: VeoMode,
    ) -> Path:
        """Poll a Veo long-running operation and download the result.

        Args:
            operation: GenerateVideosOperation from the SDK.
            output_dir: Directory to save the output.
            atom_id: Unique atom identifier.
            model_id: Model used for generation.
            mode: VeoMode used.

        Returns:
            Path to the downloaded video file.

        Raises:
            RuntimeError: If polling times out or operation fails.
        """
        poll_start = time.time()
        poll_count = 0

        while not operation.done:
            elapsed = time.time() - poll_start
            if elapsed > self.veo_cfg.poll_timeout:
                raise RuntimeError(
                    f"[Veo] Operation timed out after {elapsed:.0f}s "
                    f"(limit: {self.veo_cfg.poll_timeout}s)"
                )

            poll_count += 1
            if poll_count % 3 == 0:
                logger.info(
                    f"[Veo] Polling operation {operation.name} "
                    f"({elapsed:.0f}s elapsed, poll #{poll_count})"
                )

            time.sleep(self.veo_cfg.poll_interval)

            try:
                operation = self._client.operations.get(operation)
            except Exception as e:
                logger.warning(f"[Veo] Poll error (continuing): {e}")
                time.sleep(5)

        # Check for error
        if operation.error:
            raise RuntimeError(f"[Veo] Operation failed: {operation.error}")

        # Extract result
        response = operation.result
        if not response or not response.generated_videos:
            raise RuntimeError("[Veo] Operation completed but no videos in response")

        if response.rai_media_filtered_count and response.rai_media_filtered_count > 0:
            logger.warning(
                f"[Veo] {response.rai_media_filtered_count} video(s) filtered by safety. "
                f"Reasons: {response.rai_media_filtered_reasons}"
            )

        # Download generated video(s)
        total_generated = len(response.generated_videos)
        if total_generated > 1:
            logger.info(
                f"[Veo] {total_generated} video(s) generated. "
                f"Downloading all; pipeline uses the first."
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        primary_path = None

        for vid_idx, generated in enumerate(response.generated_videos):
            video_obj = generated.video
            suffix = "" if vid_idx == 0 else f"_alt{vid_idx}"
            output_path = output_dir / f"{atom_id}_veo{suffix}.mp4"

            if video_obj.video_bytes:
                output_path.write_bytes(video_obj.video_bytes)
                logger.info(f"[Veo] Downloaded video {vid_idx + 1}/{total_generated} → {output_path} ({len(video_obj.video_bytes)} bytes)")
            elif video_obj.uri:
                self._download_uri(video_obj.uri, output_path)
            else:
                logger.warning(f"[Veo] Video {vid_idx + 1} has neither bytes nor URI — skipped")
                continue

            if vid_idx == 0:
                primary_path = output_path

        if not primary_path or not primary_path.exists():
            raise RuntimeError("[Veo] Failed to download primary generated video")

        return primary_path

    def _download_uri(self, uri: str, output_path: Path) -> None:
        """Download a video from a URI — handles both HTTPS and GCS paths.

        Gemini Developer API returns HTTPS download URLs like:
            https://generativelanguage.googleapis.com/v1beta/files/xxx:download?alt=media
        Vertex AI may return GCS URIs like:
            gs://bucket/path/video.mp4
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if uri.startswith("gs://"):
            # GCS URI → use gsutil
            logger.info(f"[Veo] Downloading from GCS: {uri} → {output_path}")
            try:
                cmd = ["gsutil", "cp", uri, str(output_path)]
                subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=120)
                if output_path.exists() and output_path.stat().st_size > 0:
                    logger.info(f"[Veo] GCS download complete: {output_path}")
                    return
                raise RuntimeError(f"Downloaded file is empty: {output_path}")
            except FileNotFoundError:
                raise RuntimeError(
                    "[Veo] gsutil not found. Install Google Cloud SDK for GCS downloads."
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"[Veo] GCS download failed: {e.stderr}")

        elif uri.startswith("http://") or uri.startswith("https://"):
            # HTTPS URL → download with httpx (already a google-genai dependency)
            # For Gemini API URLs, we need to pass the API key.
            logger.info(f"[Veo] Downloading from HTTPS: {uri} → {output_path}")
            try:
                import httpx

                # Build download URL with API key for Gemini endpoints
                download_url = uri
                api_key = self._auth_kwargs.get("api_key")
                if api_key and "googleapis.com" in uri:
                    separator = "&" if "?" in uri else "?"
                    download_url = f"{uri}{separator}key={api_key}"

                with httpx.Client(timeout=120.0, follow_redirects=True) as client:
                    response = client.get(download_url)
                    response.raise_for_status()

                    output_path.write_bytes(response.content)
                    size_mb = len(response.content) / (1024 * 1024)
                    logger.info(
                        f"[Veo] HTTPS download complete: {output_path} ({size_mb:.1f}MB)"
                    )
                    return

            except Exception as e:
                raise RuntimeError(f"[Veo] HTTPS download failed: {e}")

        else:
            raise RuntimeError(f"[Veo] Unknown URI scheme: {uri}")

    # ─── GeneratorBackend interface ───────────────────────────────────────

    def generate_image(self, request) -> "GenerationResult":
        """Generate a keyframe image via Veo.

        Veo has no native txt2img or img2img. Instead:
          - TEXT mode: generate a short text_to_video clip, extract first frame.
          - IMAGE mode: generate a short image_to_video clip from the init image
            + new prompt, extract the LAST frame as the evolved target keyframe.
            This produces genuine visual evolution — the denoise_strength on the
            request controls clip duration (more denoise → longer clip → more
            evolution from the source image).
        """
        from .backends import GenerationResult, InputMode

        start_time = time.time()

        if request.mode == InputMode.IMAGE and request.init_image_path and request.init_image_path.exists():
            # ── Evolved keyframe via image_to_video + last-frame extraction ──
            # Map denoise_strength to evolution: higher denoise → use longer clip
            # for more visual drift from the source image.
            denoise = getattr(request, 'denoise_strength', 0.4)
            # Scale denoise (0.0–1.0) to Veo duration: low denoise → 4s (mild),
            # high denoise → 8s (strong evolution).
            evo_duration = 4 if denoise < 0.35 else (6 if denoise < 0.6 else 8)

            logger.info(
                f"[Veo] Evolving keyframe: image_to_video({evo_duration}s) "
                f"from {request.init_image_path.name} (denoise={denoise:.3f})"
            )

            result = self._generate_veo_video(
                prompt=request.prompt,
                mode=VeoMode.IMAGE_TO_VIDEO,
                output_dir=request.output_dir,
                atom_id=f"{request.atom_id}_evo",
                seed=request.seed,
                negative_prompt=request.negative_prompt,
                duration_seconds=evo_duration,
                aspect_ratio=self.veo_cfg.aspect_ratio,
                resolution=self.veo_cfg.resolution,
                image_path=request.init_image_path,
            )

            if not result["success"]:
                logger.warning(
                    f"[Veo] Evolved keyframe failed: {result['error']} — "
                    f"falling back to text_to_video"
                )
                # Fall through to text_to_video below
            else:
                # Extract LAST frame as the evolved target
                video_path = result["video_path"]
                frame_path = request.output_dir / f"{request.atom_id}_image.png"
                self._extract_frame(video_path, frame_path, position="last")

                return GenerationResult(
                    success=True,
                    image_path=frame_path,
                    video_path=video_path,
                    generation_time=time.time() - start_time,
                    metadata={
                        **result["metadata"],
                        "keyframe_mode": "evolved_last_frame",
                        "evolution_duration": evo_duration,
                        "denoise_mapped": denoise,
                    },
                )

        # Text mode (or fallback): generate a short clip and extract first frame
        logger.info("[Veo] Text-to-video for keyframe extraction (4s clip)")
        result = self._generate_veo_video(
            prompt=request.prompt,
            mode=VeoMode.TEXT_TO_VIDEO,
            output_dir=request.output_dir,
            atom_id=f"{request.atom_id}_kf",
            seed=request.seed,
            negative_prompt=request.negative_prompt,
            duration_seconds=4,  # Shortest Veo 3.1 duration
            aspect_ratio=self.veo_cfg.aspect_ratio,
            resolution=self.veo_cfg.resolution,
        )

        if not result["success"]:
            return GenerationResult(
                success=False,
                error=f"Veo keyframe generation failed: {result['error']}",
                generation_time=time.time() - start_time,
            )

        # Extract first frame
        video_path = result["video_path"]
        frame_path = request.output_dir / f"{request.atom_id}_image.png"
        self._extract_frame(video_path, frame_path, position="first")

        return GenerationResult(
            success=True,
            image_path=frame_path,
            video_path=video_path,  # Also keep the short clip
            generation_time=time.time() - start_time,
            metadata=result["metadata"],
        )

    def generate_video(self, request, source_image: Optional[Path] = None) -> "GenerationResult":
        """Generate a video from a source image + prompt.

        Maps to Veo image_to_video or text_to_video depending on image availability.
        source_image can be None for text_to_video mode.
        """
        from .backends import GenerationResult

        start_time = time.time()

        # Determine mode
        if source_image and source_image.exists():
            mode = VeoMode.IMAGE_TO_VIDEO
        else:
            mode = VeoMode.TEXT_TO_VIDEO

        # Map duration from request
        duration = request.duration_seconds or self.veo_cfg.duration_seconds

        result = self._generate_veo_video(
            prompt=request.video_prompt or request.prompt,
            mode=mode,
            output_dir=request.output_dir,
            atom_id=request.atom_id,
            seed=request.seed,
            negative_prompt=request.negative_prompt,
            duration_seconds=int(duration),
            image_path=source_image if mode == VeoMode.IMAGE_TO_VIDEO else None,
            reference_image_paths=getattr(request, 'reference_image_paths', None),
            reference_image_types=getattr(request, 'reference_image_types', None),
            generate_audio=getattr(request, 'generate_audio', None),
        )

        if not result["success"]:
            return GenerationResult(
                success=False,
                error=result["error"],
                generation_time=time.time() - start_time,
                metadata=result["metadata"],
            )

        return GenerationResult(
            success=True,
            video_path=result["video_path"],
            generation_time=time.time() - start_time,
            metadata=result["metadata"],
        )

    def generate_morph_video(
        self, request, start_image_path: Path, end_image_path: Path
    ) -> "GenerationResult":
        """Generate a morphing transition using Veo first_last_frame mode.

        This is the Veo replacement for the ComfyUI morph workflow.
        Uses start_image as the first frame and end_image as the last frame.
        """
        from .backends import GenerationResult

        start_time = time.time()

        if not self.veo_cfg.enable_last_frame:
            return GenerationResult(
                success=False,
                error="first_last_frame mode disabled in config (enable_last_frame=false)",
            )

        duration = request.duration_seconds or self.veo_cfg.duration_seconds

        result = self._generate_veo_video(
            prompt=request.video_prompt or request.prompt,
            mode=VeoMode.FIRST_LAST_FRAME,
            output_dir=request.output_dir,
            atom_id=request.atom_id,
            seed=request.seed,
            negative_prompt=request.negative_prompt,
            duration_seconds=int(duration),
            image_path=start_image_path,
            last_frame_path=end_image_path,
        )

        if not result["success"]:
            return GenerationResult(
                success=False,
                error=result["error"],
                generation_time=time.time() - start_time,
                metadata=result["metadata"],
            )

        return GenerationResult(
            success=True,
            video_path=result["video_path"],
            generation_time=time.time() - start_time,
            metadata=result["metadata"],
        )

    def generate_extension(self, request, input_video_path: Path) -> "GenerationResult":
        """Extend an existing Veo-generated video.

        Only supported on Vertex AI provider. Input video must be a prior
        Veo-generated clip within the retention window (1–30s duration).
        Extension output is ~7s, controlled by Veo.
        """
        from .backends import GenerationResult

        start_time = time.time()

        if not self.veo_cfg.enable_extension:
            return GenerationResult(
                success=False,
                error="Video extension disabled in config (enable_extension=false)",
            )

        # Extension is Vertex-only in the current SDK
        if self.provider != VeoProvider.VERTEX:
            return GenerationResult(
                success=False,
                error=(
                    f"Video extension requires provider=vertex "
                    f"(current: {self.provider.value}). "
                    f"Set provider: vertex in your veo: config."
                ),
            )

        if not input_video_path or not input_video_path.exists():
            return GenerationResult(
                success=False,
                error="extend_video requires a valid input video path",
            )

        # Extension does NOT send duration_seconds — output is ~7s by Veo
        result = self._generate_veo_video(
            prompt=request.video_prompt or request.prompt,
            mode=VeoMode.EXTEND_VIDEO,
            output_dir=request.output_dir,
            atom_id=request.atom_id,
            seed=request.seed,
            negative_prompt=request.negative_prompt,
            duration_seconds=None,  # Veo controls extension duration
            input_video_path=input_video_path,
        )

        if not result["success"]:
            return GenerationResult(
                success=False,
                error=result["error"],
                generation_time=time.time() - start_time,
                metadata=result["metadata"],
            )

        return GenerationResult(
            success=True,
            video_path=result["video_path"],
            generation_time=time.time() - start_time,
            metadata=result["metadata"],
        )

    def check_availability(self) -> tuple:
        """Check if Veo backend is reachable and authenticated."""
        try:
            # Lightweight check: list models to verify credentials
            models = list(self._client.models.list())
            veo_models = [m.name for m in models if "veo" in m.name.lower()]
            if veo_models:
                return True, f"Veo available ({len(veo_models)} model(s): {', '.join(veo_models[:3])})"
            return True, "Connected but no Veo models found in listing"
        except Exception as e:
            return False, f"Veo unavailable: {e}"

    def supports_mode(self, mode) -> bool:
        """Check if a specific input mode is supported."""
        from .backends import InputMode
        if mode == InputMode.TEXT:
            return True
        if mode == InputMode.IMAGE:
            return True
        if mode == InputMode.VIDEO:
            return self.veo_cfg.enable_extension
        return False

    # ─── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_frame(video_path: Path, output_path: Path, position: str = "first") -> None:
        """Extract first or last frame from a video using ffmpeg.

        Args:
            video_path: Input video.
            output_path: Output image path.
            position: "first" or "last".
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if position == "last":
            # Get duration first
            probe_cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ]
            try:
                dur_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
                duration = float(dur_result.stdout.strip())
                seek_time = max(0, duration - 0.5)
            except Exception:
                seek_time = 0

            cmd = [
                "ffmpeg", "-y",
                "-ss", str(seek_time),
                "-i", str(video_path),
                "-vframes", "1",
                "-q:v", "2",
                str(output_path),
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vframes", "1",
                "-q:v", "2",
                str(output_path),
            ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise RuntimeError(f"Frame extraction produced empty file: {output_path}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Frame extraction failed: {e.stderr}")


__all__ = [
    "VeoBackend",
    "VeoConfig",
    "VeoProvider",
    "VeoMode",
]
