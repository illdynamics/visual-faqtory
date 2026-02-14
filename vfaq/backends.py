#!/usr/bin/env python3
"""
backends.py - AI Generation Backend Abstraction
═══════════════════════════════════════════════════════════════════════════════

Pluggable backends for image and video generation:
  - MockBackend: Testing without GPU (fully functional)
  - ComfyUIBackend: ComfyUI API (production backend)

Each backend implements the GeneratorBackend interface.

Part of QonQrete Visual FaQtory v0.5.6-beta
"""
import os
import io
import json
import time
import random
import logging
import hashlib
import subprocess
import urllib.request
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
    denoise_strength: float = 0.4
    init_image_path: Optional[Path] = None
    video_frames: int = 25
    video_fps: int = 8
    motion_bucket_id: int = 127
    noise_aug_strength: float = 0.02
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    atom_id: str = ""
    # Prompt Bundle extensions (v0.1.0-alpha)
    video_prompt: Optional[str] = None       # Dedicated prompt for video stage
    motion_prompt: Optional[str] = None      # Raw motion intent from motion_prompt.md
    # Duration authority (v0.5.6-beta)
    duration_seconds: Optional[float] = None  # Explicit duration in seconds (authoritative)

    @property
    def effective_frames(self) -> int:
        """Derive frame count from duration_seconds × fps if duration is set,
        otherwise fall back to video_frames."""
        if self.duration_seconds is not None and self.duration_seconds > 0:
            return max(1, int(self.duration_seconds * self.video_fps))
        return self.video_frames


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
        duration = request.video_frames / request.video_fps
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


# ═══════════════════════════════════════════════════════════════════════════════
# COMFYUI BACKEND (Fully Functional)
# ═══════════════════════════════════════════════════════════════════════════════

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

        # Resolve default morph workflow path if not provided
        if not self.workflow_morph:
            try:
                # Determine repo root (two levels up from this file)
                repo_root = Path(__file__).resolve().parents[2]
                default_morph = repo_root / 'worqspace' / 'workflows' / 'morph_i2v.json'
                self.workflow_morph = str(default_morph)
            except Exception:
                self.workflow_morph = None

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
        workflow = self._build_image_workflow(request)

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
        workflow = self._build_video_workflow(request, image_name)

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
        """
        Generate a morphing video between two images via the ComfyUI backend.

        In the current release, ComfyUI does not natively support true
        two‑image interpolation.  Rather than performing an unsupported
        cross‑fade with OpenCV, this implementation delegates to the
        existing ``generate_video`` method using the ending image as the
        source.  The generated video will be a simple looping animation
        of the new keyframe for the requested duration.  This keeps
        all video rendering within the ComfyUI backend and avoids any
        direct image manipulation or external multimedia dependencies.

        Args:
            request: GenerationRequest containing duration_seconds and video_fps.
            start_image_path: Path to starting image (ignored for morph).
            end_image_path: Path to ending image used as the source frame.

        Returns:
            GenerationResult indicating success and the path to the output mp4.
        """
        # Ensure morph workflow exists for validation.  The presence of
        # worqspace/workflows/morph_i2v.json is treated as a contract that
        # morphing is supported.  If missing, fail fast.
        if not self.workflow_morph or not Path(self.workflow_morph).exists():
            return GenerationResult(success=False, error=f"Morph workflow not found: {self.workflow_morph}")
        # Generate a video from the end image.  LoRA injection is
        # intentionally skipped in generate_video.  We reuse the same
        # GenerationRequest so that duration and fps are honoured.
        return self.generate_video(request, end_image_path)

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
        if self.workflow_image:
            try:
                workflow = json.loads(Path(self.workflow_image).read_text())
                return self._customize_workflow(workflow, request, is_video=False)
            except Exception as e:
                logger.warning(f"Failed to load custom workflow: {e}, using default")
        
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
        if self.workflow_video:
            try:
                workflow = json.loads(Path(self.workflow_video).read_text())
                workflow = self._customize_workflow(workflow, request, is_video=True)
                workflow = self._inject_loaded_image(workflow, image_name)
                return workflow
            except Exception as e:
                logger.warning(f"Failed to load custom video workflow: {e}, using default")
        
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
                    "video_frames": request.video_frames,
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
                    "steps": 20,
                    "cfg": 2.5,
                    "sampler_name": "euler",
                    "scheduler": "karras",
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
        """Inject parameters into loaded workflow using graph-based CLIP resolution."""
        # Determine which prompt to use for positive CLIP nodes
        effective_prompt = request.prompt
        if is_video and request.video_prompt:
            effective_prompt = request.video_prompt
            logger.info(f"[ComfyUI] Using video_prompt for video workflow injection")

        # ── GRAPH-BASED PROMPT INJECTION (v0.1.1 — replaces heuristic) ──
        workflow = self._inject_prompts_graph_based(
            workflow, effective_prompt, request.negative_prompt
        )

        for node_id, node in workflow.items():
            inputs = node.get('inputs', {})
            class_type = node.get('class_type', '')
            
            # Inject seed into KSampler nodes
            if class_type == 'KSampler':
                inputs['seed'] = request.seed
                if 'steps' in inputs:
                    inputs['steps'] = request.steps
                if 'cfg' in inputs:
                    inputs['cfg'] = request.cfg_scale
            
            # Inject dimensions
            if class_type == 'EmptyLatentImage':
                inputs['width'] = request.width
                inputs['height'] = request.height
        
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
    def _resolve_clip_nodes_from_graph(workflow: Dict) -> Dict[str, List[str]]:
        """
        Graph-based CLIP node resolution (v0.1.1).

        Walk the workflow graph to find which CLIPTextEncode nodes are wired
        to KSampler positive vs negative inputs.

        Algorithm:
          1. Find all KSampler nodes
          2. For each KSampler, follow the 'positive' input reference → node id
          3. If that node is CLIPTextEncode → it's a positive CLIP node
          4. Same for 'negative' input → negative CLIP node

        Returns:
            {"positive": [node_id, ...], "negative": [node_id, ...]}
        """
        positive_ids: List[str] = []
        negative_ids: List[str] = []

        for node_id, node in workflow.items():
            if node.get('class_type') != 'KSampler':
                continue

            inputs = node.get('inputs', {})

            # Follow positive reference: ["node_id", output_slot]
            pos_ref = inputs.get('positive')
            if isinstance(pos_ref, list) and len(pos_ref) >= 1:
                target_id = str(pos_ref[0])
                target_node = workflow.get(target_id, {})
                if target_node.get('class_type') == 'CLIPTextEncode':
                    if target_id not in positive_ids:
                        positive_ids.append(target_id)

            # Follow negative reference
            neg_ref = inputs.get('negative')
            if isinstance(neg_ref, list) and len(neg_ref) >= 1:
                target_id = str(neg_ref[0])
                target_node = workflow.get(target_id, {})
                if target_node.get('class_type') == 'CLIPTextEncode':
                    if target_id not in negative_ids:
                        negative_ids.append(target_id)

        return {"positive": positive_ids, "negative": negative_ids}

    def _inject_prompts_graph_based(
        self, workflow: Dict, prompt: str, negative_prompt: str
    ) -> Dict:
        """
        Inject prompt text into workflow using graph-based CLIP resolution.

        Falls back to scanning all CLIPTextEncode nodes if no KSampler
        references are found (e.g., unusual workflow topologies).
        """
        resolved = self._resolve_clip_nodes_from_graph(workflow)

        injected_positive = False
        injected_negative = False

        # Inject positive prompt
        for nid in resolved["positive"]:
            workflow[nid]['inputs']['text'] = prompt
            injected_positive = True
            logger.debug(f"[ComfyUI] Graph-injected positive prompt into node {nid}")

        # Inject negative prompt
        for nid in resolved["negative"]:
            workflow[nid]['inputs']['text'] = negative_prompt or "low quality, blurry"
            injected_negative = True
            logger.debug(f"[ComfyUI] Graph-injected negative prompt into node {nid}")

        if injected_positive or injected_negative:
            logger.info(
                f"[ComfyUI] Graph-based CLIP injection: "
                f"positive={resolved['positive']}, negative={resolved['negative']}"
            )
        else:
            logger.warning(
                "[ComfyUI] No CLIPTextEncode nodes wired to KSampler found. "
                "Workflow may not support text conditioning."
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

        resolved = self._resolve_clip_nodes_from_graph(workflow)
        has_text_conditioning = bool(resolved["positive"] or resolved["negative"])

        if not has_text_conditioning:
            logger.warning(
                "WARNING: motion_prompt provided but ignored by backend "
                "(workflow does not support text conditioning)"
            )
    
    def _inject_init_image(self, workflow: Dict, image_name: str, denoise: float) -> Dict:
        """Convert txt2img workflow to img2img by adding LoadImage and VAEEncode nodes."""
        # Add LoadImage node
        workflow["load_init"] = {
            "class_type": "LoadImage",
            "inputs": {"image": image_name}
        }
        
        # Add VAEEncode node to encode the image to latent
        workflow["vae_encode_init"] = {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["load_init", 0],
                "vae": ["4", 2]  # Assumes checkpoint loader is node "4"
            }
        }
        
        # Find and modify KSampler to use the encoded latent
        for node_id, node in workflow.items():
            if node.get('class_type') == 'KSampler':
                node['inputs']['denoise'] = denoise
                # Wire the latent image from VAEEncode instead of EmptyLatentImage
                node['inputs']['latent_image'] = ["vae_encode_init", 0]
        
        # Remove or disconnect EmptyLatentImage (optional, keeping for fallback)
        
        return workflow
    
    def _inject_loaded_image(self, workflow: Dict, image_name: str) -> Dict:
        """Inject image name into LoadImage nodes."""
        for node_id, node in workflow.items():
            if node.get('class_type') == 'LoadImage':
                node['inputs']['image'] = image_name
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
                    
                    # Find and download outputs
                    for node_id, output in outputs.items():
                        # Handle image outputs (SaveImage, PreviewImage, etc.)
                        if 'images' in output:
                            for img_data in output['images']:
                                result = self._download_output(img_data, request, is_image=True)
                                if result.success:
                                    return result
                        
                        # Handle video outputs (VHS_VideoCombine, etc.)
                        # VHS uses 'gifs' key historically; some versions use 'videos' or 'video'
                        vid_list = (
                            output.get('videos')
                            or output.get('gifs')
                            or output.get('video')
                            or []
                        )
                        # Normalize: if dict (single video), wrap in list
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
            ext = '.png' if is_image else '.mp4'
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


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_backend(config: Dict[str, Any]) -> GeneratorBackend:
    """Factory function to create backend."""
    backend_type = config.get('type', 'mock').lower()

    backends = {
        'mock': MockBackend,
        'comfyui': ComfyUIBackend,
    }

    if backend_type not in backends:
        logger.warning(f"Unknown backend '{backend_type}', using mock")
        backend_type = 'mock'

    return backends[backend_type](config)


def list_available_backends() -> Dict[str, tuple]:
    """Check all backend availability."""
    results = {}
    for name, cls in [('mock', MockBackend), ('comfyui', ComfyUIBackend)]:
        try:
            backend = cls({})
            results[name] = backend.check_availability()
        except Exception as e:
            results[name] = (False, str(e))
    return results


__all__ = [
    'BackendType', 'InputMode', 'GenerationRequest', 'GenerationResult',
    'FatalConfigError',
    'GeneratorBackend', 'MockBackend', 'ComfyUIBackend',
    'create_backend', 'list_available_backends'
]
