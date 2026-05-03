#!/usr/bin/env python3
"""
backends.py - AI Generation Backend Abstraction
═══════════════════════════════════════════════════════════════════════════════

Pluggable backends for image and video generation:
  - MockBackend: Testing without GPU (fully functional)
  - ComfyUIBackend: ComfyUI API (production backend)
  - QwenImageComfyUIBackend: Qwen still-image generation via ComfyUI workflows
  - QwenImagePythonBackend: Native local Python inference for Qwen still images
  - VeoBackend: Google Veo via Gen AI SDK (see veo_backend.py)
  - LTXVideoBackend: LTX-Video self-hosted local inference (see ltx_video_backend.py)

Each backend implements the GeneratorBackend interface.

Part of Visual FaQtory v0.9.3-beta
"""
import os
import io
import json
import copy
import time
import random
import logging
import hashlib
import subprocess
import urllib.request
import re
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """Available backend types."""
    MOCK = "mock"
    COMFYUI = "comfyui"
    VEO = "veo"
    VENICE = "venice"
    DELEGATING = "delegating"


class InputMode(Enum):
    """Visual generation input modes."""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"


@dataclass
class GenerationRequest:
    """Unified generation request for all backends."""
    prompt: str
    negative_prompt: str = ""
    seed: int = 42
    mode: InputMode = InputMode.TEXT
    base_image_path: Optional[Path] = None
    base_video_path: Optional[Path] = None
    width: int = 1024
    height: int = 576
    cfg_scale: float = 7.0
    steps: int = 30
    sampler: str = "euler_ancestral"
    scheduler: str = "normal"
    denoise_strength: float = 0.4
    init_image_path: Optional[Path] = None
    video_frames: Optional[int] = None
    video_fps: float = 8.0
    motion_bucket_id: int = 127
    noise_aug_strength: float = 0.02
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    atom_id: str = ""
    # Prompt Bundle extensions (v0.1.0-alpha)
    video_prompt: Optional[str] = None       # Dedicated prompt for video stage
    motion_prompt: Optional[str] = None      # Raw motion intent from motion_prompt.md
    # Duration authority (v0.5.9-beta)
    duration_seconds: Optional[float] = None  # Explicit duration in seconds (authoritative)

    # ── Veo-specific fields (v0.6.0-beta) ─────────────────────────────────
    # These fields are used exclusively by the VeoBackend and are ignored
    # by ComfyUI and Mock backends.  They are part of the unified request
    # to avoid backend-specific request subclasses.
    veo_mode: Optional[str] = None            # text_to_video | image_to_video | first_last_frame | extend_video
    last_frame_path: Optional[Path] = None    # End frame for first_last_frame mode
    input_video_path: Optional[Path] = None   # Input video for extend_video mode
    reference_image_paths: Optional[List[Path]] = None    # Reference images
    reference_image_types: Optional[List[str]] = None     # Reference types (STYLE|ASSET)
    generate_audio: Optional[bool] = None     # Veo audio generation
    aspect_ratio: Optional[str] = None        # Veo aspect ratio (e.g. "16:9")
    resolution: Optional[str] = None          # Veo resolution (e.g. "720p")
    person_generation: Optional[str] = None   # Veo person generation policy
    sample_count: Optional[int] = None        # Number of videos to generate
    storage_uri: Optional[str] = None         # GCS URI for Vertex output
    compression_quality: Optional[str] = None # Veo compression quality

    # ── AnimateDiff-specific fields (v0.7.0-beta) ─────────────────────────
    checkpoint: Optional[str] = None
    motion_model: Optional[str] = None
    beta_schedule: Optional[str] = None
    output_format: Optional[str] = None
    save_output: Optional[bool] = None
    context_length: Optional[int] = None
    context_stride: Optional[int] = None
    context_overlap: Optional[int] = None
    closed_loop: Optional[bool] = None
    motion_loras: Optional[List[Dict[str, Any]]] = None
    prompt_schedule: Optional[Any] = None
    pingpong: Optional[bool] = None
    # Orchestration params (not native Veo — used by story engine scheduling)
    continuity_strength: Optional[float] = None
    mutation_strength: Optional[float] = None
    identity_lock_strength: Optional[float] = None
    loop_closure_strength: Optional[float] = None

    @property
    def effective_frames(self) -> int:
        """Derive frame count from duration_seconds × fps if duration is set,
        otherwise fall back to video_frames."""
        if self.duration_seconds is not None and self.duration_seconds > 0:
            return max(1, int(self.duration_seconds * self.video_fps))
        return self.video_frames if self.video_frames is not None else 25 # Fallback to 25


CAPABILITY_KEYS = ("image", "video", "morph")
CAPABILITY_CONFIG_MAP = {
    "image": "image_backend",
    "video": "video_backend",
    "morph": "morph_backend",
}

SHARED_BACKEND_SECTION_KEYS = ("lora", "comfyui", "veo", "venice")
FACADE_BACKEND_TYPES = {"hybrid", "delegating"}


def extract_backend_config(source: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normalize either a backend-only config dict or a full root config dict into
    the backend config shape expected by the backend factory.

    Supported input shapes:
      1. legacy / current: {"backend": {...}, "lora": {...}, ...}
      2. backend-only:      {"type": "comfyui", ...}
      3. split top-level:   {"backend": {"type": "hybrid"}, "image_backend": {...}, ...}
    """
    raw = copy.deepcopy(source or {})

    if not isinstance(raw, dict):
        return {"type": "mock"}

    looks_like_root_cfg = (
        "backend" in raw
        or any(key in raw for key in CAPABILITY_CONFIG_MAP.values())
        or (
            any(key in raw for key in SHARED_BACKEND_SECTION_KEYS)
            and "type" not in raw
        )
    )

    if looks_like_root_cfg:
        backend_cfg = copy.deepcopy(raw.get("backend") or {})
        for section_name in CAPABILITY_CONFIG_MAP.values():
            if raw.get(section_name) is not None and backend_cfg.get(section_name) is None:
                backend_cfg[section_name] = copy.deepcopy(raw[section_name])
        for section_name in SHARED_BACKEND_SECTION_KEYS:
            if raw.get(section_name) is not None and backend_cfg.get(section_name) is None:
                backend_cfg[section_name] = copy.deepcopy(raw[section_name])
        return backend_cfg or {"type": "mock"}

    return raw or {"type": "mock"}


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two config dictionaries."""
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def has_split_backend_config(config: Optional[Dict[str, Any]]) -> bool:
    """Return True when capability-specific backend sections are present."""
    cfg = extract_backend_config(config)
    return any(cfg.get(section) is not None for section in CAPABILITY_CONFIG_MAP.values())


def resolve_capability_backend_configs(config: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Normalize legacy and split backend config into per-capability configs."""
    raw_cfg = extract_backend_config(config)

    if not has_split_backend_config(raw_cfg):
        normalized = copy.deepcopy(raw_cfg)
        normalized["type"] = str(normalized.get("type", "mock") or "mock").lower()
        return {cap: copy.deepcopy(normalized) for cap in CAPABILITY_KEYS}

    shared_cfg = copy.deepcopy(raw_cfg)
    explicit_cfgs = {
        cap: shared_cfg.pop(section, None)
        for cap, section in CAPABILITY_CONFIG_MAP.items()
    }

    default_cfg = copy.deepcopy(shared_cfg)
    default_type = str(default_cfg.get("type", "") or "").lower()
    if default_type in FACADE_BACKEND_TYPES:
        default_type = ""
        default_cfg.pop("type", None)

    resolved: Dict[str, Optional[Dict[str, Any]]] = {}
    for cap in CAPABILITY_KEYS:
        explicit = explicit_cfgs[cap]
        if explicit is not None:
            cap_cfg = _deep_merge_dicts(default_cfg, explicit)
        elif default_type:
            cap_cfg = copy.deepcopy(default_cfg)
        else:
            cap_cfg = None

        if cap_cfg is not None:
            cap_cfg["type"] = str(cap_cfg.get("type", default_type or "mock") or (default_type or "mock")).lower()
        resolved[cap] = cap_cfg

    if resolved["video"] is None:
        fallback = resolved["morph"] or resolved["image"] or {"type": default_type or "mock"}
        resolved["video"] = copy.deepcopy(fallback)
    if resolved["image"] is None:
        fallback = resolved["video"] or resolved["morph"] or {"type": default_type or "mock"}
        resolved["image"] = copy.deepcopy(fallback)
    if resolved["morph"] is None:
        fallback = resolved["video"] or resolved["image"] or {"type": default_type or "mock"}
        resolved["morph"] = copy.deepcopy(fallback)

    return {cap: copy.deepcopy(cfg) for cap, cfg in resolved.items() if cfg is not None}


def get_backend_type_for_capability(config: Optional[Dict[str, Any]], capability: str) -> str:
    """Return the normalized backend type for one capability."""
    capability_key = capability.lower()
    if capability_key not in CAPABILITY_KEYS:
        raise ValueError(f"Unknown backend capability: {capability}")
    cfg = resolve_capability_backend_configs(config).get(capability_key, {"type": "mock"})
    return str(cfg.get("type", "mock") or "mock").lower()


def describe_backend_config(config: Optional[Dict[str, Any]]) -> str:
    """Create a compact human-readable backend summary string."""
    cfg = config or {}
    if not has_split_backend_config(cfg):
        return get_backend_type_for_capability(cfg, "video")

    resolved = resolve_capability_backend_configs(cfg)
    return (
        "split("
        f"image={resolved['image'].get('type', 'mock')}, "
        f"video={resolved['video'].get('type', 'mock')}, "
        f"morph={resolved['morph'].get('type', 'mock')}"
        ")"
    )


@dataclass
class GenerationResult:
    """Result from a generation request."""
    success: bool
    image_path: Optional[Path] = None
    video_path: Optional[Path] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation_time: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# BASE BACKEND INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

class FatalConfigError(RuntimeError):
    """Raised when a configuration value is unsafe or invalid. Aborts the cycle."""
    pass


class GeneratorBackend(ABC):
    """Abstract interface for image/video generation backends."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = "base"
    
    @abstractmethod
    def generate_image(self, request: GenerationRequest) -> GenerationResult:
        pass
    
    @abstractmethod
    def generate_video(self, request: GenerationRequest, source_image: Path) -> GenerationResult:
        pass
    
    @abstractmethod
    def check_availability(self) -> tuple:
        pass
    
    def supports_mode(self, mode: InputMode) -> bool:
        return True

    def generate_morph_video(self, request: GenerationRequest, start_image_path: Path, end_image_path: Path) -> GenerationResult:
        """
        Generate a morphing video between two images.  Implementations must
        produce a smooth transition from the start image to the end image
        using the specified duration and frame rate in `request`.  The
        default implementation does not support morph video and will raise
        NotImplementedError.  Backends capable of morphing must override
        this method.

        Args:
            request: GenerationRequest containing prompt and video settings.
            start_image_path: Path to the starting image (last frame).
            end_image_path: Path to the ending image (new keyframe).

        Returns:
            GenerationResult with video_path pointing to the generated
            morph video.
        """
        return GenerationResult(success=False, error=f"Backend '{self.name}' does not support morph video generation")


# ═══════════════════════════════════════════════════════════════════════════════
# MOCK BACKEND (Testing)
# ═══════════════════════════════════════════════════════════════════════════════

class MockBackend(GeneratorBackend):
    """Mock backend for testing without GPU."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "mock"
        self.delay = config.get('mock_delay', 0.5)
    
    def generate_image(self, request: GenerationRequest) -> GenerationResult:
        start_time = time.time()
        time.sleep(self.delay)
        
        output_path = request.output_dir / f"{request.atom_id}_image.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._create_placeholder_image(output_path, request)
        logger.info(f"[MOCK] Generated image: {output_path}")
        
        return GenerationResult(
            success=True,
            image_path=output_path,
            generation_time=time.time() - start_time,
            metadata={"backend": "mock", "seed": request.seed}
        )
    
    def generate_video(self, request: GenerationRequest, source_image: Path) -> GenerationResult:
        start_time = time.time()
        time.sleep(self.delay * 2)
        
        output_path = request.output_dir / f"{request.atom_id}_video.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._create_placeholder_video(output_path, source_image, request)
        logger.info(f"[MOCK] Generated video: {output_path}")
        
        return GenerationResult(
            success=True,
            video_path=output_path,
            generation_time=time.time() - start_time,
            metadata={"backend": "mock", "frames": request.video_frames}
        )

    def generate_morph_video(self, request: GenerationRequest, start_image_path: Path, end_image_path: Path) -> GenerationResult:
        """
        Generate a morphing video between two images for the mock backend.

        The mock backend cannot perform true latent interpolation.  To
        approximate a morph, this implementation simply delegates to
        ``generate_video`` using the end image.  The resulting video is a
        static loop of the final keyframe for the requested duration.  No
        external dependencies such as OpenCV are used.

        Args:
            request: GenerationRequest containing duration and fps.
            start_image_path: Path to starting image (ignored for mock morph).
            end_image_path: Path to ending image used as the source frame.

        Returns:
            GenerationResult with path to the mock morph video.
        """
        # The mock morph is implemented by reusing the standard video
        # generator with the ending image as the source.  This produces a
        # simple looping video of the keyframe without any interpolation.
        return self.generate_video(request, end_image_path)

    def check_availability(self) -> tuple:
        return True, "Mock backend always available"
    
    def _create_placeholder_image(self, path: Path, request: GenerationRequest):
        try:
            from PIL import Image, ImageDraw, ImageFont
            random.seed(request.seed)
            r, g, b = random.randint(30, 80), random.randint(30, 80), random.randint(80, 150)
            img = Image.new('RGB', (request.width, request.height), (r, g, b))
            draw = ImageDraw.Draw(img)
            
            # Add some visual elements
            for _ in range(20):
                x1 = random.randint(0, request.width)
                y1 = random.randint(0, request.height)
                x2 = x1 + random.randint(50, 200)
                y2 = y1 + random.randint(50, 200)
                color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255), 128)
                draw.ellipse([x1, y1, x2, y2], fill=color[:3])
            
            # Add text
            try:
                draw.text((20, 20), f"MOCK: {request.atom_id}", fill=(255, 255, 255))
                draw.text((20, 50), f"Seed: {request.seed}", fill=(200, 200, 200))
            except:
                pass
            
            img.save(path)
        except ImportError:
            self._create_minimal_png(path)
    
    def _create_minimal_png(self, path: Path):
        png_data = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xFF, 0xFF, 0x3F,
            0x00, 0x05, 0xFE, 0x02, 0xFE, 0xDC, 0xCC, 0x59,
            0xE7, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
            0x44, 0xAE, 0x42, 0x60, 0x82
        ])
        path.write_bytes(png_data)
    
    def _create_placeholder_video(self, path: Path, source_image: Path, request: GenerationRequest):
        duration = request.effective_frames / request.video_fps
        # Try h264_nvenc first (preferred), then libx264 fallback
        for codec in ['h264_nvenc', 'libx264']:
            try:
                cmd = [
                    'ffmpeg', '-y',
                    '-loop', '1',
                    '-i', str(source_image),
                    '-c:v', codec,
                    '-t', str(duration),
                    '-pix_fmt', 'yuv420p',
                    '-vf', f'scale={request.width}:{request.height}',
                    '-r', str(request.video_fps),
                    str(path)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    return
                logger.warning(f"[MOCK] {codec} failed, trying next...")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                logger.warning(f"FFmpeg {codec} failed: {e}")
        logger.warning("All codecs failed, creating placeholder file")
        path.write_bytes(b'mock_video_placeholder')


class ComfyUIBackend(GeneratorBackend):
    """
    ComfyUI API backend - fully functional with proper output handling.
    
    Config:
        api_url: http://localhost:8188
        workflow_image: path/to/image_workflow.json (optional)
        workflow_video: path/to/video_workflow.json (optional)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "comfyui"
        self.api_url = config.get('api_url', 'http://localhost:8188')
        self.workflow_image = config.get('workflow_image')
        self.workflow_img2img = config.get('workflow_img2img') or config.get('workflow_i2i')
        self.workflow_video = config.get('workflow_video')
        # Morph workflow for image-to-video morphing (two images)
        self.workflow_morph = config.get('workflow_morph')
        self.timeout = config.get('timeout', 300)
        self._comfyui_object_info_cache: Optional[Dict] = None

        # ── LoRA Configuration ──────────────────────────────────────────────
        # LoRA support is optional but must be explicitly enabled via config.
        # The lora configuration should live at the same level as the backend
        # config passed into the backend factory.  It supports these keys:
        #   enabled (bool): whether to apply the LoRA
        #   name    (str): path to a .safetensors file on the local filesystem
        #   strength (float): the weight applied to both model and clip
        #   backend  (str): must equal "comfyui" when enabled
        # If lora.enabled is True but backend != "comfyui", the run fails.
        # If the file specified by lora.name does not exist, the run fails.
        # This configuration is captured at construction time and persisted
        # throughout the backend instance.  The LoRA loader will be injected
        # into every workflow built by this backend.
        self.lora_config = None
        raw_lora_cfg = config.get('lora')
        if raw_lora_cfg:
            # Normalize booleans and defaults
            enabled = bool(raw_lora_cfg.get('enabled', False))
            backend = raw_lora_cfg.get('backend', 'comfyui')
            if enabled:
                # Enforce backend match
                if backend.lower() != 'comfyui':
                    raise FatalConfigError(
                        f"LoRA backend mismatch: lora.backend={backend!r} but backend type is 'comfyui'"
                    )
                lora_path_str = raw_lora_cfg.get('path')
                if not lora_path_str:
                    raise FatalConfigError(
                        "LoRA enabled but no lora.path specified in config"
                    )
                strength = float(raw_lora_cfg.get('strength', 1.0))

                # --- LoRA Path Validation ---
                lora_path = Path(lora_path_str).expanduser().resolve()

                if not lora_path.exists():
                    raise FatalConfigError(
                        f"LoRA file not found: {lora_path}"
                    )
                if not lora_path.is_file():
                    raise FatalConfigError(
                        f"LoRA path is not a file: {lora_path}"
                    )
                if lora_path.suffix.lower() != '.safetensors':
                    raise FatalConfigError(
                        f"LoRA file must be a .safetensors file, got: {lora_path}"
                    )

                try:
                    # Resolve ComfyUI-style relative name from absolute path
                    lora_name = self._resolve_lora_name_from_path(lora_path)
                    if not lora_name: # Handle case where it's directly in models/loras
                         lora_name = lora_path.name
                except FatalConfigError as e:
                    raise e # Re-raise if our resolver fails
                except Exception:
                    # Generic fallback for unexpected Path parsing issues
                    raise FatalConfigError(
                        f"Could not resolve relative LoRA name from '{lora_path}'"
                    )

                # Store both the absolute path and the ComfyUI-style relative name
                self.lora_config = {
                    'enabled': True,
                    'path': str(lora_path), # Store as string for Dict serialization
                    'name': lora_name,
                    'strength': strength
                }

        self.repo_root = Path(__file__).resolve().parents[1]
        self.workflow_image_path = self._resolve_configured_path(self.workflow_image, 'workflow_image')
        self.workflow_img2img_path = self._resolve_configured_path(self.workflow_img2img, 'workflow_img2img')
        self.workflow_video_path = self._resolve_configured_path(self.workflow_video, 'workflow_video')
        self.workflow_morph_path = self._resolve_configured_path(self.workflow_morph, 'workflow_morph')

    def _resolve_configured_path(self, raw_path: Optional[str], setting_name: str) -> Optional[Path]:
        """Resolve a configured path relative to CWD or repo root and fail loudly when missing."""
        if not raw_path:
            return None

        raw = Path(str(raw_path)).expanduser()
        candidates = []
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.extend([Path.cwd() / raw, self.repo_root / raw])

        for candidate in candidates:
            if candidate.exists():
                if not candidate.is_file():
                    raise FatalConfigError(f"Configured {setting_name} is not a file: {candidate}")
                return candidate.resolve()

        searched = ", ".join(str(p) for p in candidates) or str(raw)
        raise FatalConfigError(
            f"Configured {setting_name} was not found: {raw_path!r}. Searched: {searched}"
        )

    def _load_workflow_json(self, workflow_path: Path, setting_name: str) -> Dict:
        try:
            return json.loads(workflow_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError as e:
            raise FatalConfigError(f"Invalid JSON in {setting_name} ({workflow_path}): {e}") from e
        except OSError as e:
            raise FatalConfigError(f"Could not read {setting_name} ({workflow_path}): {e}") from e

    @staticmethod
    def _workflow_has_class(workflow: Dict, class_type: str) -> bool:
        return any(node.get('class_type') == class_type for node in workflow.values())

    @staticmethod
    def _node_accepts_input(node_info: Dict[str, Any], param_name: str) -> bool:
        inputs = node_info.get("input", {}) if isinstance(node_info, dict) else {}
        for category in ("required", "optional", "hidden"):
            if param_name in inputs.get(category, {}):
                return True
        return False

    @classmethod
    def _first_supported_input_name(cls, node_info: Dict[str, Any], candidates: List[str]) -> Optional[str]:
        for candidate in candidates:
            if cls._node_accepts_input(node_info, candidate):
                return candidate
        return None

    @staticmethod
    def _get_allowed_option_values(node_info: Dict[str, Any], input_name: str) -> Optional[List[Any]]:
        """Return enum-like allowed values from ComfyUI object_info when available."""
        if not isinstance(node_info, dict):
            return None
        inputs = node_info.get("input", {})
        for category in ("required", "optional", "hidden"):
            raw = inputs.get(category, {}).get(input_name)
            if not isinstance(raw, list) or not raw:
                continue
            first = raw[0]
            if isinstance(first, list):
                return first
        return None

    @classmethod
    def _validate_named_model_available(
        cls,
        object_info: Dict[str, Any],
        node_class: str,
        input_name: str,
        configured_value: Optional[str],
        label: str,
    ) -> None:
        """Fail fast when ComfyUI advertises allowed model options and configured value is missing."""
        if not configured_value:
            return
        allowed = cls._get_allowed_option_values(object_info.get(node_class, {}), input_name)
        if allowed is not None and configured_value not in allowed:
            raise FatalConfigError(
                f"Configured {label} '{configured_value}' was not found by ComfyUI for {node_class}.{input_name}. "
                f"ComfyUI sees: {allowed}"
            )

    @staticmethod
    def _sorted_workflow_node_ids(workflow: Dict[str, Any]) -> List[str]:
        def _sort_key(item: str):
            item = str(item)
            return (0, int(item)) if item.isdigit() else (1, item)
        return sorted((str(node_id) for node_id in workflow.keys()), key=_sort_key)

    @classmethod
    def _next_workflow_node_id(cls, workflow: Dict[str, Any]) -> str:
        numeric_ids = []
        for node_id in cls._sorted_workflow_node_ids(workflow):
            if str(node_id).isdigit():
                numeric_ids.append(int(node_id))
        return str((max(numeric_ids) + 1) if numeric_ids else 1)

    def _inject_denoise(self, workflow: Dict, denoise: float) -> Dict:
        for node in workflow.values():
            if node.get('class_type') == 'KSampler' and 'denoise' in node.get('inputs', {}):
                node['inputs']['denoise'] = denoise
        return workflow

    @staticmethod
    def _resolve_first_vae_ref(workflow: Dict) -> Optional[List[Any]]:
        for node in workflow.values():
            vae_ref = node.get('inputs', {}).get('vae')
            if isinstance(vae_ref, list) and len(vae_ref) == 2:
                return [vae_ref[0], vae_ref[1]]
        return None

    def _get_comfyui_object_info(self) -> Dict:
        """
        Fetches and caches ComfyUI's object info (node schema).
        """
        if self._comfyui_object_info_cache:
            return self._comfyui_object_info_cache

        try:
            import requests
            response = requests.get(f"{self.api_url}/object_info", timeout=10)
            response.raise_for_status()
            self._comfyui_object_info_cache = response.json()
            logger.info("[ComfyUI] Fetched object_info from API.")
            return self._comfyui_object_info_cache
        except ImportError:
            raise RuntimeError("requests package not installed, cannot fetch ComfyUI object info.")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to fetch ComfyUI object info from {self.api_url}: {e}")
    
    def generate_image(self, request: GenerationRequest) -> GenerationResult:
        try:
            import requests
        except ImportError:
            return GenerationResult(success=False, error="requests package not installed")
        
        start_time = time.time()
        
        # Build workflow
        try:
            workflow = self._build_image_workflow(request)
        except FatalConfigError as e:
            return GenerationResult(success=False, error=str(e))
        except Exception as e:
            return GenerationResult(success=False, error=f"Failed to prepare image workflow: {e}")

        # Inject LoRA if enabled in config
        if self.lora_config and self.lora_config.get('enabled'):
            try:
                workflow = self._inject_lora(workflow)
            except FatalConfigError:
                # Bubble up fatal config errors
                raise
            except Exception as e:
                return GenerationResult(success=False, error=f"Failed to apply LoRA: {e}")
        
        # Handle img2img if init image provided
        if request.init_image_path and request.init_image_path.exists():
            image_name = self._upload_image(request.init_image_path)
            if image_name:
                workflow = self._inject_init_image(workflow, image_name, request.denoise_strength)
        
        # Queue and wait
        result = self._queue_and_wait(workflow, request, is_video=False)
        result.generation_time = time.time() - start_time
        return result
    
    def generate_video(self, request: GenerationRequest, source_image: Path) -> GenerationResult:
        try:
            import requests
        except ImportError:
            return GenerationResult(success=False, error="requests package not installed")
        
        start_time = time.time()
        
        # Upload source image
        image_name = self._upload_image(source_image)
        if not image_name:
            return GenerationResult(success=False, error="Failed to upload source image")
        
        # Build video workflow
        try:
            workflow = self._build_video_workflow(request, image_name)
        except FatalConfigError as e:
            return GenerationResult(success=False, error=str(e))
        except Exception as e:
            return GenerationResult(success=False, error=f"Failed to prepare video workflow: {e}")

        # Note: LoRA injection is intentionally skipped for video workflows in this release.
        # Video generation should not be influenced by LoRA.  If LoRA is enabled in the
        # configuration, it will still apply to image generation, but not here.

        # Log motion_prompt warning if workflow lacks text conditioning (Fix 4)
        self._warn_motion_prompt_if_ignored(workflow, request)
        
        # Queue and wait
        result = self._queue_and_wait(workflow, request, is_video=True)
        result.generation_time = time.time() - start_time
        return result

    def generate_morph_video(self, request: GenerationRequest, start_image_path: Path, end_image_path: Path) -> GenerationResult:
        """Generate a two-image morph video using an explicit ComfyUI workflow."""
        try:
            import requests  # noqa: F401
        except ImportError:
            return GenerationResult(success=False, error="requests package not installed")

        if not self.workflow_morph_path:
            return GenerationResult(
                success=False,
                error=(
                    "ComfyUI morph generation requires backend.workflow_morph to point to "
                    "a real two-image workflow JSON. No default morph workflow ships with this repo."
                ),
            )

        morph_path = self.workflow_morph_path

        start_time = time.time()

        start_image_name = self._upload_image(start_image_path)
        if not start_image_name:
            return GenerationResult(success=False, error="Failed to upload start morph image")

        end_image_name = self._upload_image(end_image_path)
        if not end_image_name:
            return GenerationResult(success=False, error="Failed to upload end morph image")

        try:
            workflow = self._load_workflow_json(morph_path, "workflow_morph")
            workflow = self._customize_workflow(workflow, request, is_video=True)
            workflow = self._inject_loaded_images(workflow, [start_image_name, end_image_name])
        except FatalConfigError as e:
            return GenerationResult(success=False, error=str(e))
        except Exception as e:
            return GenerationResult(success=False, error=f"Failed to prepare morph workflow: {e}")

        result = self._queue_and_wait(workflow, request, is_video=True)
        result.generation_time = time.time() - start_time
        return result

    def check_availability(self) -> tuple:
        try:
            import requests
            resp = requests.get(f"{self.api_url}/system_stats", timeout=5)
            if resp.status_code == 200:
                return True, f"ComfyUI available at {self.api_url}"
            return False, f"ComfyUI returned status {resp.status_code}"
        except ImportError:
            return False, "requests package not installed"
        except Exception as e:
            return False, f"ComfyUI not reachable: {e}"
    
    def _build_image_workflow(self, request: GenerationRequest) -> Dict:
        """Build SDXL image generation workflow."""
        if self.workflow_image_path:
            workflow = self._load_workflow_json(self.workflow_image_path, 'workflow_image')
            return self._customize_workflow(workflow, request, is_video=False)
        
        # Default SDXL workflow
        sdxl_ckpt = self.config.get("comfyui", {}).get("sdxl_ckpt", "sd_xl_base_1.0.safetensors")
        logger.info(f"[ComfyUI] Using SDXL checkpoint: {sdxl_ckpt}")

        # Validate SDXL checkpoint availability via /object_info
        try:
            object_info = self._get_comfyui_object_info()
            ckpt_loader_info = object_info.get("CheckpointLoaderSimple", {})
            allowed_ckpts = ckpt_loader_info.get("input", {}).get("required", {}).get("ckpt_name", [["STRING"]])[0]
            if isinstance(allowed_ckpts, list) and sdxl_ckpt not in allowed_ckpts:
                raise RuntimeError(
                    f"Configured SDXL checkpoint '{sdxl_ckpt}' not found by ComfyUI. "
                    f"ComfyUI sees: {allowed_ckpts}. "
                    f"Place '{sdxl_ckpt}' in ComfyUI's checkpoint directory and restart."
                )
        except RuntimeError:
            raise
        except Exception as e:
            logger.warning(f"[ComfyUI] Could not validate SDXL checkpoint: {e}")

        return {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": request.seed,
                    "steps": request.steps,
                    "cfg": request.cfg_scale,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0]
                }
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": sdxl_ckpt}
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": request.width, "height": request.height, "batch_size": 1}
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": request.prompt, "clip": ["4", 1]}
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": request.negative_prompt or "low quality, blurry", "clip": ["4", 1]}
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["3", 0], "vae": ["4", 2]}
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": request.atom_id or "vfaq", "images": ["8", 0]}
            }
        }
    
    def _build_video_workflow(self, request: GenerationRequest, image_name: str) -> Dict:
        """Build SVD video generation workflow."""
        if self.workflow_video_path:
            workflow = self._load_workflow_json(self.workflow_video_path, 'workflow_video')
            workflow = self._customize_workflow(workflow, request, is_video=True)
            workflow = self._inject_loaded_image(workflow, image_name)
            return workflow
        
        # Default SVD workflow
        svd_ckpt = self.config.get("comfyui", {}).get("svd_ckpt", "svd_xt.safetensors")
        logger.info(f"[ComfyUI] Using SVD checkpoint: {svd_ckpt}")

        # Get ComfyUI object info for dynamic input handling
        object_info = self._get_comfyui_object_info()
        image_only_ckpt_loader_info = object_info.get("ImageOnlyCheckpointLoader", {})
        vhs_video_combine_info = object_info.get("VHS_VideoCombine", {})

        # Check for SVD model existence (D)
        allowed_ckpts = image_only_ckpt_loader_info.get("input", {}).get("required", {}).get("ckpt_name", [["STRING"]])[0]
        if isinstance(allowed_ckpts, list) and svd_ckpt not in allowed_ckpts:
             raise RuntimeError(
                f"Configured SVD checkpoint '{svd_ckpt}' not found by ComfyUI. "
                f"ComfyUI sees: {allowed_ckpts}. "
                f"Please place '{svd_ckpt}' in a ComfyUI checkpoint directory (e.g., ComfyUI/models/checkpoints/) "
                f"and restart ComfyUI or click 'Reload models'."
            )
        # Determine video_frames for workflow injection, preferring request.video_frames if set
        workflow_video_frames = request.video_frames if request.video_frames is not None else request.effective_frames

        return {
            "1": {
                "class_type": "ImageOnlyCheckpointLoader",
                "inputs": {"ckpt_name": svd_ckpt}
            },
            "2": {
                "class_type": "LoadImage",
                "inputs": {"image": image_name}
            },
            "3": {
                "class_type": "SVD_img2vid_Conditioning",
                "inputs": {
                    "width": request.width,
                    "height": request.height,
                    "video_frames": workflow_video_frames,
                    "motion_bucket_id": request.motion_bucket_id,
                    "fps": request.video_fps,
                    "augmentation_level": request.noise_aug_strength,
                    "clip_vision": ["1", 1],
                    "init_image": ["2", 0],
                    "vae": ["1", 2]
                }
            },
            "4": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": request.seed,
                    "steps": request.steps,
                    "cfg": request.cfg_scale,
                    "sampler_name": request.sampler,
                    "scheduler": request.scheduler,
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["3", 0],
                    "negative": ["3", 1],
                    "latent_image": ["3", 2]
                }
            },
            "5": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["4", 0], "vae": ["1", 2]}
            },
            "6": {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "frame_rate": request.video_fps,
                    "loop_count": 0,
                    "filename_prefix": f"{request.atom_id}_video",
                    "format": "video/h264-mp4",
                    "images": ["5", 0],
                    **({"save_output": True} if self._vhs_has_input(vhs_video_combine_info, "save_output") else {}),
                    **({"pingpong": False} if self._vhs_has_input(vhs_video_combine_info, "pingpong") else {})
                }
            }
        }
    
    def _customize_workflow(self, workflow: Dict, request: GenerationRequest, is_video: bool = False) -> Dict:
        """Inject parameters into loaded workflow using graph-aware, workflow-aware rules."""
        effective_prompt = request.prompt
        if is_video and request.video_prompt:
            effective_prompt = request.video_prompt
            logger.info(f"[ComfyUI] Using video_prompt for video workflow injection")

        workflow = self._inject_prompts_graph_based(
            workflow, effective_prompt, request.negative_prompt
        )

        workflow_video_frames = request.video_frames if request.video_frames is not None else request.effective_frames
        megapixels = max(0.01, round((request.width * request.height) / 1_000_000, 4))

        for node_id, node in workflow.items():
            inputs = node.get('inputs', {})
            class_type = node.get('class_type', '')

            if 'seed' in inputs and not isinstance(inputs.get('seed'), list):
                inputs['seed'] = request.seed
            if 'noise_seed' in inputs and not isinstance(inputs.get('noise_seed'), list):
                inputs['noise_seed'] = request.seed

            if class_type == 'KSampler':
                inputs['seed'] = request.seed
                if 'steps' in inputs:
                    inputs['steps'] = request.steps
                if 'cfg' in inputs:
                    inputs['cfg'] = request.cfg_scale
                if 'sampler_name' in inputs and not isinstance(inputs.get('sampler_name'), list):
                    inputs['sampler_name'] = request.sampler
                if 'scheduler' in inputs and not isinstance(inputs.get('scheduler'), list):
                    inputs['scheduler'] = request.scheduler

            if 'width' in inputs and 'height' in inputs:
                if not isinstance(inputs.get('width'), list):
                    inputs['width'] = request.width
                if not isinstance(inputs.get('height'), list):
                    inputs['height'] = request.height
            elif class_type == 'ImageScaleToTotalPixels' and 'megapixels' in inputs and not isinstance(inputs.get('megapixels'), list):
                inputs['megapixels'] = megapixels

            if class_type == 'SaveImage' and 'filename_prefix' in inputs:
                inputs['filename_prefix'] = request.atom_id or 'vfaq'

            if is_video:
                if class_type == 'SVD_img2vid_Conditioning':
                    inputs['width'] = request.width
                    inputs['height'] = request.height
                    inputs['video_frames'] = workflow_video_frames
                    inputs['fps'] = request.video_fps
                    if 'motion_bucket_id' in inputs:
                        inputs['motion_bucket_id'] = request.motion_bucket_id
                    if 'augmentation_level' in inputs:
                        inputs['augmentation_level'] = request.noise_aug_strength
                elif class_type == 'VHS_VideoCombine':
                    inputs['frame_rate'] = request.video_fps
                    if 'filename_prefix' in inputs:
                        inputs['filename_prefix'] = f"{request.atom_id}_video"

        return workflow

    # ────────────────────────────────────────────────────────────────────────
    # LoRA Injection Utilities
    #
    # The ComfyUI backend optionally supports injecting a LoRA loader into
    # every workflow.  This is done by locating the checkpoint loader node
    # (CheckpointLoaderSimple or ImageOnlyCheckpointLoader), inserting a
    # LoraLoader node wired to its outputs and then redirecting all
    # downstream references from the loader to the new LoRA node for model
    # (index 0) and clip (index 1).  VAE outputs (index 2) remain untouched.
    # If no loader is found, a FatalConfigError is raised.  If a LoRA
    # already exists in the workflow, the original workflow is returned.

    def _inject_lora(self, workflow: Dict) -> Dict:
        """Inject a LoraLoader node into the workflow if enabled.

        Args:
            workflow: The ComfyUI workflow graph to modify.

        Returns:
            Modified workflow with a new LoraLoader node and updated
            connections.  If a LoRA node is already present, the workflow
            is returned unchanged.

        Raises:
            FatalConfigError: If no suitable loader node is found for
                              injection.
        """
        # If no lora_config or not enabled, pass through
        if not self.lora_config or not self.lora_config.get('enabled'):
            return workflow

        configured_lora_name = self.lora_config['name']
        configured_strength = float(self.lora_config['strength'])

        # Detect existing LoRA loader; if present validate configuration and wiring
        for lora_id, node in workflow.items():
            if node.get('class_type') == 'LoraLoader':
                # Validate that the existing LoRA matches the configured name and strength
                inputs = node.get('inputs', {})
                # Validate name
                existing_lora_name = inputs.get('lora_name')
                if existing_lora_name != configured_lora_name:
                    raise FatalConfigError(
                        f"Existing LoRALoader uses '{existing_lora_name}', expected '{configured_lora_name}'"
                    )
                # Validate strength for model and clip
                model_strength = float(inputs.get('strength_model', 0.0))
                clip_strength = float(inputs.get('strength_clip', 0.0))
                
                if not (abs(model_strength - configured_strength) < 1e-6 and
                        abs(clip_strength - configured_strength) < 1e-6):
                    raise FatalConfigError(
                        f"Existing LoRALoader strength mismatch: model={model_strength}, clip={clip_strength}, expected={configured_strength}"
                    )
                # Ensure the LoRA is wired into at least one downstream input for model or clip
                used = False
                for other_id, other_node in workflow.items():
                    if other_id == lora_id:
                        continue
                    other_inputs = other_node.get('inputs', {})
                    for key, val in other_inputs.items():
                        if isinstance(val, list) and len(val) == 2:
                            ref_id, idx = val
                            # Check if it refers to the LoraLoader and its model (0) or clip (1) output
                            if ref_id == lora_id and idx in (0, 1):
                                used = True
                                break
                    if used:
                        break
                if not used:
                    raise FatalConfigError("Existing LoRALoader is not wired into model/clip outputs")
                
                # If validation passes, return original workflow
                return workflow

        # Identify the base model loader node; only inject into CheckpointLoaderSimple
        loader_id: Optional[str] = None
        for nid, node in workflow.items():
            ct = node.get('class_type')
            # LoRA should only be injected after CheckpointLoaderSimple
            if ct == 'CheckpointLoaderSimple':
                loader_id = nid
                break
        if not loader_id:
            raise FatalConfigError("No CheckpointLoaderSimple found for LoRA injection; cannot apply LoRA.")

        # Compute new node id (string) by incrementing the highest numeric id
        existing_ids: List[int] = []
        for key in workflow.keys():
            try:
                existing_ids.append(int(key))
            except ValueError:
                continue
        new_id_int = max(existing_ids) + 1 if existing_ids else 1
        new_id = str(new_id_int)

        # Build LoRA loader node
        lora_node = {
            'class_type': 'LoraLoader',
            'inputs': {
                'lora_name': configured_lora_name,
                'strength_model': configured_strength,
                'strength_clip': configured_strength,
                # Connect to model (index 0) and clip (index 1) of loader
                'model': [loader_id, 0],
                'clip': [loader_id, 1],
            }
        }

        # Insert LoRA node into workflow
        workflow[new_id] = lora_node

        # Redirect downstream connections referencing loader model (0) or clip (1)
        for nid, node in workflow.items():
            if nid == new_id:
                continue
            inputs = node.get('inputs', {})
            for key, value in list(inputs.items()):
                if isinstance(value, list) and len(value) == 2:
                    ref_id, idx = value
                    # Only update model (0) and clip (1) references that point to the original loader
                    if ref_id == loader_id and idx in (0, 1):
                        inputs[key] = [new_id, idx]
        return workflow

    @staticmethod
    def _resolve_conditioning_nodes_from_graph(workflow: Dict) -> Dict[str, List[str]]:
        """Resolve the nodes feeding KSampler positive/negative conditioning inputs."""
        positive_ids: List[str] = []
        negative_ids: List[str] = []

        for node in workflow.values():
            if node.get('class_type') != 'KSampler':
                continue

            inputs = node.get('inputs', {})
            for polarity, target_ids in (("positive", positive_ids), ("negative", negative_ids)):
                ref = inputs.get(polarity)
                if isinstance(ref, list) and len(ref) >= 1:
                    target_id = str(ref[0])
                    if target_id in workflow and target_id not in target_ids:
                        target_ids.append(target_id)

        return {"positive": positive_ids, "negative": negative_ids}

    @staticmethod
    def _set_prompt_text_on_node(node: Dict[str, Any], prompt_text: str, polarity: str) -> Optional[str]:
        inputs = node.setdefault('inputs', {})
        preferred_keys = ['text', 'prompt']
        if polarity == 'positive':
            preferred_keys.extend(['positive', 'caption'])
        else:
            preferred_keys.extend(['negative'])

        for key in preferred_keys:
            if key in inputs and not isinstance(inputs.get(key), list):
                inputs[key] = prompt_text
                return key
        return None

    def _inject_prompts_graph_based(
        self, workflow: Dict, prompt: str, negative_prompt: str
    ) -> Dict:
        """Inject prompts into whichever conditioning nodes are actually wired into KSampler."""
        resolved = self._resolve_conditioning_nodes_from_graph(workflow)

        injected_positive = []
        injected_negative = []

        for nid in resolved["positive"]:
            target_node = workflow.get(nid, {})
            used_key = self._set_prompt_text_on_node(target_node, prompt, 'positive')
            if used_key:
                injected_positive.append(f"{nid}:{used_key}")

        for nid in resolved["negative"]:
            target_node = workflow.get(nid, {})
            used_key = self._set_prompt_text_on_node(target_node, negative_prompt or "low quality, blurry", 'negative')
            if used_key:
                injected_negative.append(f"{nid}:{used_key}")

        if injected_positive or injected_negative:
            logger.info(
                f"[ComfyUI] Graph-based prompt injection: "
                f"positive={injected_positive}, negative={injected_negative}"
            )
        else:
            logger.warning(
                "[ComfyUI] No graph-wired prompt inputs were found on KSampler conditioning nodes. "
                "Workflow may not support runtime prompt injection."
            )

        return workflow

    def _warn_motion_prompt_if_ignored(self, workflow: Dict, request: GenerationRequest) -> None:
        """
        Log a warning when motion_prompt is provided but the workflow has
        no text conditioning path (no CLIPTextEncode wired to KSampler).

        The motion_prompt is still preserved in the briq JSON for audit.
        """
        if not request.motion_prompt:
            return

        resolved = self._resolve_conditioning_nodes_from_graph(workflow)
        has_text_conditioning = bool(resolved["positive"] or resolved["negative"])

        if not has_text_conditioning:
            logger.warning(
                "WARNING: motion_prompt provided but ignored by backend "
                "(workflow does not support text conditioning)"
            )
    
    def _inject_init_image(self, workflow: Dict, image_name: str, denoise: float) -> Dict:
        """Convert txt2img workflow to img2img by adding LoadImage and VAEEncode nodes."""
        vae_ref = self._resolve_first_vae_ref(workflow)
        if vae_ref is None:
            raise FatalConfigError(
                "Could not convert custom workflow into img2img: no VAE input reference was found."
            )

        workflow["load_init"] = {
            "class_type": "LoadImage",
            "inputs": {"image": image_name}
        }

        workflow["vae_encode_init"] = {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["load_init", 0],
                "vae": vae_ref
            }
        }

        for node in workflow.values():
            if node.get('class_type') == 'KSampler':
                node['inputs']['denoise'] = denoise
                node['inputs']['latent_image'] = ["vae_encode_init", 0]

        return workflow
    
    def _inject_loaded_image(self, workflow: Dict, image_name: str) -> Dict:
        """Inject one image name into every LoadImage node."""
        for node_id, node in workflow.items():
            if node.get('class_type') == 'LoadImage':
                node.setdefault('inputs', {})['image'] = image_name
        return workflow

    def _inject_loaded_images(self, workflow: Dict, image_names: List[str]) -> Dict:
        """Inject multiple images into LoadImage nodes in stable node-id order."""
        load_nodes = [
            (node_id, node)
            for node_id, node in workflow.items()
            if node.get('class_type') == 'LoadImage'
        ]
        if len(load_nodes) < len(image_names):
            raise ValueError(
                f"Morph workflow requires at least {len(image_names)} LoadImage nodes, found {len(load_nodes)}"
            )

        def _sort_key(item):
            node_id = str(item[0])
            return (0, int(node_id)) if node_id.isdigit() else (1, node_id)

        for (_, node), image_name in zip(sorted(load_nodes, key=_sort_key), image_names):
            node.setdefault('inputs', {})['image'] = image_name

        return workflow

    @staticmethod
    def _vhs_has_input(node_info: Dict, param_name: str) -> bool:
        """
        Check if a VHS node accepts a given input parameter.
        Checks required, optional, and hidden input categories.
        """
        inputs = node_info.get("input", {})
        for category in ("required", "optional", "hidden"):
            if param_name in inputs.get(category, {}):
                return True
        return False

    @staticmethod
    def _resolve_lora_name_from_path(lora_path: Path) -> str:
        """
        Resolves the ComfyUI-style relative LoRA name from an absolute filesystem path.
        Assumes the path is validated to exist and end in .safetensors.
        The lora_path must live under a directory structure ending in 'models/loras'.
        Example: /x/ComfyUI/models/loras/psy/glitch/brainmelt.safetensors -> psy/glitch/brainmelt.safetensors
        """
        # Search for 'loras' in the path components, then 'models' immediately before it
        parts = lora_path.parts
        try:
            # Iterate backwards to find the last 'loras' in the path, ensuring it's for models/loras
            loras_idx = -1
            for i in range(len(parts) -1, -1, -1):
                if parts[i] == 'loras':
                    if i > 0 and parts[i-1] == 'models':
                        loras_idx = i
                        break
            
            if loras_idx == -1:
                raise ValueError("Path structure 'models/loras' not found within the LoRA path.")

            # The relative path starts from the element AFTER 'loras'
            relative_parts = parts[loras_idx + 1:]
            
            # If the lora file is directly in models/loras/, relative_parts will be empty
            # In this case, the lora_name is just the file name
            if not relative_parts:
                return lora_path.name

            return str(Path(*relative_parts))
        except ValueError as e:
            raise FatalConfigError(
                f"LoRA path '{lora_path}' is not within a 'models/loras' directory structure: {e}"
            )

    
    def _upload_image(self, image_path: Path) -> Optional[str]:
        """Upload image to ComfyUI and return the filename."""
        try:
            import requests
            
            with open(image_path, 'rb') as f:
                files = {'image': (image_path.name, f, 'image/png')}
                resp = requests.post(f"{self.api_url}/upload/image", files=files)
            
            if resp.status_code == 200:
                return resp.json().get('name')
            logger.error(f"Upload failed: {resp.status_code} {resp.text}")
            return None
        except Exception as e:
            logger.error(f"Image upload failed: {e}")
            return None
    
    def _queue_and_wait(self, workflow: Dict, request: GenerationRequest, is_video: bool) -> GenerationResult:
        """Queue workflow and wait for completion with proper output downloading."""
        import requests
        
        # Generate client ID for websocket
        client_id = hashlib.sha256(f"{time.time()}_{request.atom_id}".encode()).hexdigest()[:16]
        
        # Queue the prompt
        resp = requests.post(
            f"{self.api_url}/prompt",
            json={"prompt": workflow, "client_id": client_id}
        )
        
        if resp.status_code != 200:
            return GenerationResult(success=False, error=f"Queue failed: {resp.text}")
        
        prompt_id = resp.json().get('prompt_id')
        logger.info(f"[ComfyUI] Queued prompt {prompt_id}")
        
        # Poll for completion
        start = time.time()
        while time.time() - start < self.timeout:
            try:
                history_resp = requests.get(f"{self.api_url}/history/{prompt_id}")
                history = history_resp.json()
                
                if prompt_id in history:
                    entry = history[prompt_id]

                    # Check execution status FIRST
                    status_info = entry.get('status', {})
                    status_str = status_info.get('status_str', '')
                    if status_str == 'error':
                        # Collect error details from node outputs
                        error_msgs = []
                        for node_id, node_output in entry.get('outputs', {}).items():
                            if 'errors' in node_output:
                                error_msgs.append(f"Node {node_id}: {node_output['errors']}")
                        error_detail = '; '.join(error_msgs) if error_msgs else 'unknown error'
                        return GenerationResult(
                            success=False,
                            error=f"ComfyUI execution error: {error_detail}"
)

                    outputs = entry.get('outputs', {})

                    # For video jobs, prefer real downloadable video outputs first.
                    # Some ComfyUI histories also expose preview images from
                    # intermediate nodes, which can otherwise mask the actual VHS
                    # output and make debugging absolute soup.
                    if is_video:
                        ordered_video_outputs = []
                        fallback_image_outputs = []
                        for node_id, output in outputs.items():
                            vid_list = (
                                output.get('videos')
                                or output.get('gifs')
                                or output.get('video')
                                or []
                            )
                            if isinstance(vid_list, dict):
                                vid_list = [vid_list]
                            if vid_list:
                                ordered_video_outputs.append((node_id, vid_list))
                            elif 'images' in output:
                                fallback_image_outputs.append((node_id, output['images']))

                        for _node_id, vid_list in ordered_video_outputs:
                            for vid_data in vid_list:
                                result = self._download_output(vid_data, request, is_image=False)
                                if result.success:
                                    return result

                        for _node_id, img_list in fallback_image_outputs:
                            for img_data in img_list:
                                result = self._download_output(img_data, request, is_image=True)
                                if result.success:
                                    return result
                    else:
                        # Find and download image outputs first for still-image jobs
                        for node_id, output in outputs.items():
                            if 'images' in output:
                                for img_data in output['images']:
                                    result = self._download_output(img_data, request, is_image=True)
                                    if result.success:
                                        return result

                            vid_list = (
                                output.get('videos')
                                or output.get('gifs')
                                or output.get('video')
                                or []
                            )
                            if isinstance(vid_list, dict):
                                vid_list = [vid_list]
                            if vid_list:
                                for vid_data in vid_list:
                                    result = self._download_output(vid_data, request, is_image=False)
                                    if result.success:
                                        return result
                    # If we got here, outputs were found but no media downloaded

                    # Log what we actually got for debugging
                    output_keys = {
                        nid: list(out.keys()) for nid, out in outputs.items()
                    }
                    logger.error(
                        f"[ComfyUI] Prompt completed but no downloadable media found. "
                        f"Output structure: {output_keys}"
                    )
                    return GenerationResult(
                        success=False,
                        error=f"Could not download outputs. Node output keys: {output_keys}"
                    )
                
            except Exception as e:
                logger.warning(f"Polling error: {e}")
            
            time.sleep(2)
        
        return GenerationResult(success=False, error=f"Generation timed out after {self.timeout}s")
    
    def _download_output(self, output_data: Dict, request: GenerationRequest, is_image: bool) -> GenerationResult:
        """Download generated output from ComfyUI."""
        import requests
        
        filename = output_data.get('filename')
        subfolder = output_data.get('subfolder', '')
        output_type = output_data.get('type', 'output')
        
        if not filename:
            logger.warning(f"[ComfyUI] Output entry has no filename: {output_data}")
            return GenerationResult(success=False, error="No filename in output")
        
        # Build download URL - use urllib for proper encoding
        from urllib.parse import urlencode
        params = urlencode({
            'filename': filename,
            'subfolder': subfolder,
            'type': output_type
        })
        url = f"{self.api_url}/view?{params}"
        
        try:
            logger.debug(f"[ComfyUI] Downloading from: {url}")
            resp = requests.get(url, timeout=60)
            if resp.status_code != 200:
                logger.warning(
                    f"[ComfyUI] Download failed for {filename}: "
                    f"HTTP {resp.status_code} - {resp.text[:200]}"
                )
                return GenerationResult(success=False, error=f"Download failed: HTTP {resp.status_code}")
            
            if len(resp.content) == 0:
                logger.warning(f"[ComfyUI] Downloaded empty file for {filename}")
                return GenerationResult(success=False, error="Downloaded empty file")
            
            # Save to output directory
            ext = '.png' if is_image else (Path(filename).suffix or '.mp4')
            if is_image:
                output_path = request.output_dir / f"{request.atom_id}_image{ext}"
            else:
                output_path = request.output_dir / f"{request.atom_id}_video{ext}"
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(resp.content)
            
            logger.info(f"[ComfyUI] Downloaded: {output_path} ({len(resp.content)} bytes)")
            
            return GenerationResult(
                success=True,
                image_path=output_path if is_image else None,
                video_path=output_path if not is_image else None,
                metadata={"comfyui_filename": filename}
            )
            
        except Exception as e:
            logger.warning(f"[ComfyUI] Download error for {filename}: {e}")
            return GenerationResult(success=False, error=f"Download error: {e}")


class AnimateDiffBackend(ComfyUIBackend):
    """ComfyUI-backed AnimateDiff video backend."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "animatediff"

    def check_availability(self) -> tuple:
        if not self.workflow_video_path:
            return False, "AnimateDiff requires workflow_video in config"
        return super().check_availability()


class QwenImageComfyUIBackend(ComfyUIBackend):
    """ComfyUI-backed Qwen image backend."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "qwen_image_comfyui"

    def check_availability(self) -> tuple:
        if not self.workflow_image_path:
            return False, "Qwen image backend requires workflow_image in config"
        return super().check_availability()


class QwenImageBackend(QwenImageComfyUIBackend):
    """
    Legacy alias for Qwen image backend.

    Keeps compatibility with older configs/tests that reference `qwen_image`.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "qwen_image"


class DelegatingBackend(GeneratorBackend):
    """Facade backend that routes image/video/morph calls by capability."""

    def __init__(
        self,
        config: Dict[str, Any],
        image_backend: GeneratorBackend,
        video_backend: GeneratorBackend,
        morph_backend: GeneratorBackend,
        capability_configs: Dict[str, Dict[str, Any]],
    ):
        super().__init__(config)
        self.name = "delegating"
        self.image_backend = image_backend
        self.video_backend = video_backend
        self.morph_backend = morph_backend
        self.capability_configs = copy.deepcopy(capability_configs)
        self.capability_types = {
            cap: str(cfg.get('type', 'mock')).lower()
            for cap, cfg in capability_configs.items()
        }

    def generate_image(self, request: GenerationRequest) -> GenerationResult:
        return self.image_backend.generate_image(request)

    def generate_video(self, request: GenerationRequest, source_image: Path) -> GenerationResult:
        return self.video_backend.generate_video(request, source_image)

    def generate_morph_video(self, request: GenerationRequest, start_image_path: Path, end_image_path: Path) -> GenerationResult:
        return self.morph_backend.generate_morph_video(request, start_image_path, end_image_path)

    def generate_extension(self, request: GenerationRequest, source_video: Path) -> GenerationResult:
        if not hasattr(self.video_backend, 'generate_extension'):
            return GenerationResult(
                success=False,
                error=(
                    f"Configured video backend '{self.video_backend.name}' does not support video extension"
                ),
            )
        return self.video_backend.generate_extension(request, source_video)

    def __getattr__(self, item):
        """Forward backend-specific helpers to the most likely capability backend.

        This keeps legacy call sites like backend.generate_extension() working when
        the public backend instance is a capability router.
        """
        for candidate in (self.video_backend, self.image_backend, self.morph_backend):
            if hasattr(candidate, item):
                return getattr(candidate, item)
        raise AttributeError(item)

    def check_availability(self) -> tuple:
        checks = []
        available = True
        for capability, backend in (
            ('image', self.image_backend),
            ('video', self.video_backend),
            ('morph', self.morph_backend),
        ):
            ok, msg = backend.check_availability()
            checks.append(f"{capability}={backend.name}: {msg}")
            available = available and ok
        return available, " | ".join(checks)


def _create_single_backend(config: Dict[str, Any]) -> GeneratorBackend:
    """Instantiate one concrete backend from a normalized config."""
    backend_type = str(config.get('type', 'mock') or 'mock').lower()

    backends = {
        'mock': MockBackend,
        'comfyui': ComfyUIBackend,
        'animatediff': AnimateDiffBackend,
        'qwen_image_comfyui': QwenImageComfyUIBackend,
        'qwen_image': QwenImageBackend,
        'venice': None,
        }

    if backend_type == 'venice':
        try:
            from .venice_backend import VeniceBackend
            return VeniceBackend(config)
        except ImportError as e:
            raise RuntimeError(
                f"Venice backend import failed: {e}\n"
                f"Ensure venice_backend.py is present and requests is installed."
            )

    if backend_type == 'veo':
        try:
            from .veo_backend import VeoBackend
            return VeoBackend(config)
        except ImportError as e:
            raise RuntimeError(
                f"Veo backend requires google-genai SDK: pip install google-genai\n"
                f"Import error: {e}"
            )

    backends.pop('venice', None)

    if backend_type not in backends:
        logger.warning(f"Unknown backend '{backend_type}', using mock")
        backend_type = 'mock'

    return backends[backend_type](config)


def create_backend(config: Dict[str, Any]) -> GeneratorBackend:
    """Factory function to create a legacy or capability-split backend."""
    resolved = resolve_capability_backend_configs(config)
    if has_split_backend_config(config):
        return DelegatingBackend(
            config=config,
            image_backend=_create_single_backend(resolved['image']),
            video_backend=_create_single_backend(resolved['video']),
            morph_backend=_create_single_backend(resolved['morph']),
            capability_configs=resolved,
        )

    return _create_single_backend(resolved['video'])


def list_available_backends() -> Dict[str, tuple]:
    """Check all backend availability."""
    results = {}
    for name, cls in [('mock', MockBackend), ('comfyui', ComfyUIBackend), ('animatediff', AnimateDiffBackend), ('qwen_image_comfyui', QwenImageComfyUIBackend), ('qwen_image', QwenImageBackend)]:
        try:
            backend = cls({})
            results[name] = backend.check_availability()
        except Exception as e:
            results[name] = (False, str(e))

    # Check Venice availability
    try:
        from .venice_backend import VeniceBackend  # noqa: F401
        import os
        api_key = os.environ.get('VENICE_API_KEY') or os.environ.get('VENICE_API_TOKEN')
        if api_key:
            results['venice'] = (True, 'Venice backend importable, credentials detected')
        else:
            results['venice'] = (True, 'Venice backend importable (set VENICE_API_KEY to use)')
    except ImportError as e:
        results['venice'] = (False, f'Venice: import failed ({e})')
    except Exception as e:
        results['venice'] = (False, f'Venice: {e}')

    # Check Veo availability: import check + auth hint (don't require live auth)
    try:
        from .veo_backend import VeoBackend  # noqa: F401
        import importlib
        importlib.import_module("google.genai")
        # SDK importable — check if any auth env vars are set
        import os
        has_auth = any(os.environ.get(k) for k in [
            "GOOGLE_API_KEY", "GEMINI_API_KEY",
            "GOOGLE_API_TOKEN", "GEMINI_API_TOKEN",
            "GOOGLE_CLOUD_PROJECT",
        ])
        if has_auth:
            results['veo'] = (True, "Veo SDK installed, credentials detected")
        else:
            results['veo'] = (True, "Veo SDK installed (no credentials set — set GEMINI_API_TOKEN to use)")
    except ImportError:
        results['veo'] = (False, "Veo: google-genai not installed (pip install google-genai)")
    except Exception as e:
        results['veo'] = (False, f"Veo: {e}")

    # Check LTX-Video availability

    return results


__all__ = [
    'BackendType', 'InputMode', 'GenerationRequest', 'GenerationResult',
    'FatalConfigError',
    'extract_backend_config', 'has_split_backend_config', 'resolve_capability_backend_configs',
    'get_backend_type_for_capability', 'describe_backend_config',
    'GeneratorBackend', 'MockBackend', 'ComfyUIBackend', 'AnimateDiffBackend',
    'QwenImageComfyUIBackend', 'QwenImageBackend', 'DelegatingBackend',
    'create_backend', 'list_available_backends',
    # LTXVideoBackend is lazily imported via create_backend() — not in __all__
]
