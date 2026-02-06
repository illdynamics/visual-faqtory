#!/usr/bin/env python3
"""
backends.py - AI Generation Backend Abstraction
═══════════════════════════════════════════════════════════════════════════════

Pluggable backends for image and video generation:
  - MockBackend: Testing without GPU (fully functional)
  - ComfyUIBackend: ComfyUI API (fully functional)
  - DiffusersBackend: Local HuggingFace diffusers
  - ReplicateBackend: Replicate.com API

Each backend implements the GeneratorBackend interface.

Part of QonQrete Visual FaQtory v0.0.7-alpha
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
    DIFFUSERS = "diffusers"
    REPLICATE = "replicate"


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
    # Prompt Bundle extensions (v0.0.7-alpha)
    video_prompt: Optional[str] = None       # Dedicated prompt for video stage
    motion_prompt: Optional[str] = None      # Raw motion intent from motion_prompt.md


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
        self.timeout = config.get('timeout', 300)
        self._comfyui_object_info_cache: Optional[Dict] = None

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
        
        # Queue and wait
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
        """Inject parameters into loaded workflow. For video, prefers video_prompt."""
        # Determine which prompt to use for positive CLIP nodes
        effective_prompt = request.prompt
        if is_video and request.video_prompt:
            effective_prompt = request.video_prompt
            logger.info(f"[ComfyUI] Using video_prompt for video workflow injection")

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
            
            # Inject prompt into CLIP nodes
            if class_type == 'CLIPTextEncode':
                if 'positive' in node_id.lower() or inputs.get('text', '').startswith('masterpiece'):
                    inputs['text'] = effective_prompt
                elif 'negative' in node_id.lower():
                    inputs['text'] = request.negative_prompt
            
            # Inject dimensions
            if class_type == 'EmptyLatentImage':
                inputs['width'] = request.width
                inputs['height'] = request.height
        
        return workflow
    
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
# DIFFUSERS BACKEND (Local GPU)
# ═══════════════════════════════════════════════════════════════════════════════

class DiffusersBackend(GeneratorBackend):
    """Local HuggingFace diffusers backend."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "diffusers"
        self.model_id = config.get('model_id', 'stabilityai/stable-diffusion-xl-base-1.0')
        self.video_model_id = config.get('video_model_id', 'stabilityai/stable-video-diffusion-img2vid')
        self.device = config.get('device', 'cuda')
        self.dtype = config.get('dtype', 'float16')
        self._image_pipe = None
        self._video_pipe = None
    
    def generate_image(self, request: GenerationRequest) -> GenerationResult:
        start_time = time.time()
        
        try:
            pipe = self._get_image_pipeline()
            import torch
            from PIL import Image
            
            generator = torch.Generator(device=self.device).manual_seed(request.seed)
            
            if request.init_image_path and request.init_image_path.exists():
                # img2img
                init_image = Image.open(request.init_image_path).convert('RGB')
                init_image = init_image.resize((request.width, request.height))
                image = pipe(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    image=init_image,
                    strength=request.denoise_strength,
                    num_inference_steps=request.steps,
                    guidance_scale=request.cfg_scale,
                    generator=generator
                ).images[0]
            else:
                # txt2img
                image = pipe(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    width=request.width,
                    height=request.height,
                    num_inference_steps=request.steps,
                    guidance_scale=request.cfg_scale,
                    generator=generator
                ).images[0]
            
            output_path = request.output_dir / f"{request.atom_id}_image.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
            
            return GenerationResult(
                success=True,
                image_path=output_path,
                generation_time=time.time() - start_time
            )
        except Exception as e:
            return GenerationResult(success=False, error=str(e))
    
    def generate_video(self, request: GenerationRequest, source_image: Path) -> GenerationResult:
        start_time = time.time()
        
        try:
            pipe = self._get_video_pipeline()
            import torch
            from PIL import Image
            
            image = Image.open(source_image).convert('RGB').resize((request.width, request.height))
            generator = torch.Generator(device=self.device).manual_seed(request.seed)
            
            frames = pipe(
                image,
                num_frames=request.video_frames,
                motion_bucket_id=request.motion_bucket_id,
                noise_aug_strength=request.noise_aug_strength,
                generator=generator
            ).frames[0]
            
            output_path = request.output_dir / f"{request.atom_id}_video.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._frames_to_video(frames, output_path, request.video_fps)
            
            return GenerationResult(
                success=True,
                video_path=output_path,
                generation_time=time.time() - start_time
            )
        except Exception as e:
            return GenerationResult(success=False, error=str(e))
    
    def check_availability(self) -> tuple:
        try:
            import torch
            if not torch.cuda.is_available():
                return False, "CUDA not available"
            import diffusers
            return True, f"Diffusers {diffusers.__version__} with CUDA"
        except ImportError as e:
            return False, f"Missing package: {e}"
    
    def _get_image_pipeline(self):
        if self._image_pipe is None:
            import torch
            from diffusers import AutoPipelineForText2Image
            dtype = torch.float16 if self.dtype == 'float16' else torch.float32
            self._image_pipe = AutoPipelineForText2Image.from_pretrained(
                self.model_id, torch_dtype=dtype, variant="fp16"
            ).to(self.device)
        return self._image_pipe
    
    def _get_video_pipeline(self):
        if self._video_pipe is None:
            import torch
            from diffusers import StableVideoDiffusionPipeline
            dtype = torch.float16 if self.dtype == 'float16' else torch.float32
            self._video_pipe = StableVideoDiffusionPipeline.from_pretrained(
                self.video_model_id, torch_dtype=dtype, variant="fp16"
            ).to(self.device)
        return self._video_pipe
    
    def _frames_to_video(self, frames: List, output_path: Path, fps: int):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, frame in enumerate(frames):
                frame.save(f"{tmpdir}/frame_{i:04d}.png")
            # Try h264_nvenc first, then libx264 fallback
            for codec in ['h264_nvenc', 'libx264']:
                result = subprocess.run([
                    'ffmpeg', '-y', '-framerate', str(fps),
                    '-i', f'{tmpdir}/frame_%04d.png',
                    '-c:v', codec, '-pix_fmt', 'yuv420p',
                    str(output_path)
                ], capture_output=True, text=True)
                if result.returncode == 0:
                    return
            raise RuntimeError("All video encoders failed")


# ═══════════════════════════════════════════════════════════════════════════════
# REPLICATE BACKEND (Cloud API)
# ═══════════════════════════════════════════════════════════════════════════════

class ReplicateBackend(GeneratorBackend):
    """Replicate.com API backend."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "replicate"
        self.api_token = config.get('api_token') or os.environ.get('REPLICATE_API_TOKEN')
        self.image_model = config.get('image_model', 'stability-ai/sdxl')
        self.video_model = config.get('video_model', 'stability-ai/stable-video-diffusion')
    
    def generate_image(self, request: GenerationRequest) -> GenerationResult:
        try:
            import replicate
        except ImportError:
            return GenerationResult(success=False, error="replicate package not installed")
        
        start_time = time.time()
        try:
            input_params = {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "width": request.width,
                "height": request.height,
                "num_inference_steps": request.steps,
                "guidance_scale": request.cfg_scale,
                "seed": request.seed
            }
            
            if request.init_image_path and request.init_image_path.exists():
                input_params["image"] = open(request.init_image_path, 'rb')
                input_params["prompt_strength"] = request.denoise_strength
            
            output = replicate.run(self.image_model, input=input_params)
            
            output_path = request.output_dir / f"{request.atom_id}_image.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(output[0], output_path)
            
            return GenerationResult(
                success=True,
                image_path=output_path,
                generation_time=time.time() - start_time
            )
        except Exception as e:
            return GenerationResult(success=False, error=str(e))
    
    def generate_video(self, request: GenerationRequest, source_image: Path) -> GenerationResult:
        try:
            import replicate
        except ImportError:
            return GenerationResult(success=False, error="replicate package not installed")
        
        start_time = time.time()
        try:
            with open(source_image, 'rb') as f:
                output = replicate.run(
                    self.video_model,
                    input={
                        "input_image": f,
                        "motion_bucket_id": request.motion_bucket_id,
                        "fps": request.video_fps,
                        "seed": request.seed
                    }
                )
            
            output_path = request.output_dir / f"{request.atom_id}_video.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(output, output_path)
            
            return GenerationResult(
                success=True,
                video_path=output_path,
                generation_time=time.time() - start_time
            )
        except Exception as e:
            return GenerationResult(success=False, error=str(e))
    
    def check_availability(self) -> tuple:
        if not self.api_token:
            return False, "REPLICATE_API_TOKEN not set"
        try:
            import replicate
            return True, "Replicate API available"
        except ImportError:
            return False, "replicate package not installed"


# ═══════════════════════════════════════════════════════════════════════════════
# SPLIT BACKEND (v0.0.7-alpha — separate image and video backends)
# ═══════════════════════════════════════════════════════════════════════════════

class SplitBackend(GeneratorBackend):
    """
    Wrapper that delegates image and video generation to separate backends.

    Config (v0.0.7-alpha):
      backends:
        image:
          type: comfyui
          ...
        video:
          type: comfyui
          ...

    Falls back to single backend if both are the same instance.
    """

    def __init__(self, image_backend: GeneratorBackend, video_backend: GeneratorBackend):
        # No config needed at wrapper level
        super().__init__({})
        self.name = f"split({image_backend.name}/{video_backend.name})"
        self.image_backend = image_backend
        self.video_backend = video_backend

    def generate_image(self, request: GenerationRequest) -> GenerationResult:
        """Delegate image generation to image backend."""
        return self.image_backend.generate_image(request)

    def generate_video(self, request: GenerationRequest, source_image: Path) -> GenerationResult:
        """Delegate video generation to video backend."""
        return self.video_backend.generate_video(request, source_image)

    def check_availability(self) -> tuple:
        """Combined status of both backends."""
        img_ok, img_msg = self.image_backend.check_availability()
        vid_ok, vid_msg = self.video_backend.check_availability()

        if img_ok and vid_ok:
            return True, f"Image: {img_msg} | Video: {vid_msg}"
        elif img_ok:
            return False, f"Image OK ({img_msg}) but Video failed: {vid_msg}"
        elif vid_ok:
            return False, f"Video OK ({vid_msg}) but Image failed: {img_msg}"
        else:
            return False, f"Both failed — Image: {img_msg} | Video: {vid_msg}"

    def supports_mode(self, mode: InputMode) -> bool:
        return self.image_backend.supports_mode(mode) and self.video_backend.supports_mode(mode)


def create_split_backend(config: Dict[str, Any]) -> GeneratorBackend:
    """
    Create backend(s) from config, supporting both legacy single-backend
    and new split-backend configurations.

    Legacy (v0.0.6):
      backend:
        type: comfyui

    New (v0.0.7):
      backends:
        image:
          type: comfyui
        video:
          type: comfyui

    Falls back gracefully: if 'backends' not present, uses legacy 'backend'.
    """
    backends_config = config.get('backends')

    if backends_config:
        # New split-backend config
        image_cfg = backends_config.get('image', {})
        video_cfg = backends_config.get('video', image_cfg)  # default to image if video not set

        # Merge top-level comfyui section into each backend for ckpt visibility
        comfyui_global = config.get('comfyui', {})
        if comfyui_global:
            if 'comfyui' not in image_cfg:
                image_cfg = {**image_cfg, 'comfyui': comfyui_global}
            if 'comfyui' not in video_cfg:
                video_cfg = {**video_cfg, 'comfyui': comfyui_global}

        image_backend = create_backend(image_cfg)
        video_backend = create_backend(video_cfg)

        logger.info(
            f"[SplitBackend] Image: {image_backend.name}, Video: {video_backend.name}"
        )
        return SplitBackend(image_backend, video_backend)

    # Legacy single-backend config
    backend_config = config.get('backend', {'type': 'mock'})
    return create_backend(backend_config)


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def create_backend(config: Dict[str, Any]) -> GeneratorBackend:
    """Factory function to create backend."""
    backend_type = config.get('type', 'mock').lower()
    
    backends = {
        'mock': MockBackend,
        'comfyui': ComfyUIBackend,
        'diffusers': DiffusersBackend,
        'replicate': ReplicateBackend,
    }
    
    if backend_type not in backends:
        logger.warning(f"Unknown backend '{backend_type}', using mock")
        backend_type = 'mock'
    
    return backends[backend_type](config)


def list_available_backends() -> Dict[str, tuple]:
    """Check all backend availability."""
    results = {}
    for name, cls in [('mock', MockBackend), ('comfyui', ComfyUIBackend), 
                      ('diffusers', DiffusersBackend), ('replicate', ReplicateBackend)]:
        try:
            backend = cls({})
            results[name] = backend.check_availability()
        except Exception as e:
            results[name] = (False, str(e))
    return results


__all__ = [
    'BackendType', 'InputMode', 'GenerationRequest', 'GenerationResult',
    'GeneratorBackend', 'MockBackend', 'ComfyUIBackend', 'DiffusersBackend',
    'ReplicateBackend', 'SplitBackend',
    'create_backend', 'create_split_backend', 'list_available_backends'
]
