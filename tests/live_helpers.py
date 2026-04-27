from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

_MINIMAL_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR42mP4////PwAF/gL+3MxZ5wAAAABJRU5ErkJggg=="
)


def env_flag(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def skip_unless_flag(name: str, description: str) -> None:
    if not env_flag(name):
        pytest.skip(f"{name}=1 not set; skipping {description}")


def require_env(name: str, description: str) -> str:
    value = os.getenv(name)
    if value is None or not str(value).strip():
        pytest.skip(f"{name} is required for {description}")
    return str(value).strip()


def optional_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None or not str(value).strip():
        return default
    return str(value).strip()


def require_existing_file_env(name: str, description: str) -> Path:
    path = Path(require_env(name, description)).expanduser()
    if not path.is_file():
        pytest.skip(f"{name} points to a missing file for {description}: {path}")
    return path


def int_env(name: str, default: int) -> int:
    value = optional_env(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        pytest.skip(f"{name} must be an integer, got: {value!r}")


def float_env(name: str, default: float) -> float:
    value = optional_env(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        pytest.skip(f"{name} must be a float, got: {value!r}")


def create_placeholder_png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(_MINIMAL_PNG_B64))
    return path


def detect_media_kind(path: Path) -> str:
    header = path.read_bytes()[:64]
    if header.startswith(bytes.fromhex("89504E470D0A1A0A")):
        return "image/png"
    if header.startswith(bytes.fromhex("FFD8FF")):
        return "image/jpeg"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "image/webp"
    if header.startswith(b"GIF8"):
        return "image/gif"
    if b"ftyp" in header[:32]:
        return "video/mp4"
    if header.startswith(bytes.fromhex("1A45DFA3")):
        return "video/webm"
    return "unknown"

def assert_valid_image(path: Path) -> str:
    assert path is not None and path.exists(), f"Expected image output, got missing path: {path}"
    assert path.stat().st_size > 0, f"Image output is empty: {path}"
    kind = detect_media_kind(path)
    assert kind.startswith("image/"), f"Expected image media, got {kind} for {path}"
    return kind


def assert_valid_video(path: Path) -> str:
    assert path is not None and path.exists(), f"Expected video output, got missing path: {path}"
    assert path.stat().st_size > 0, f"Video output is empty: {path}"
    kind = detect_media_kind(path)
    assert kind.startswith("video/"), f"Expected video media, got {kind} for {path}"
    return kind


def comfy_api_url() -> str:
    return optional_env("VF_COMFYUI_API_URL", "http://127.0.0.1:8188")


def comfy_timeout() -> int:
    return int_env("VF_COMFY_TIMEOUT", 300)


def build_venice_config() -> Dict[str, Any]:
    api_key_env = optional_env("VF_VENICE_API_KEY_ENV", "VENICE_API_KEY")
    require_env(api_key_env, "Venice live integration tests")
    return {
        "venice": {
            "api_key_env": api_key_env,
            "base_url": optional_env("VF_VENICE_BASE_URL", "https://api.venice.ai/api/v1"),
            "validate_models": True,
            "strict_model_validation": False,
            "cleanup_after_download": True,
            "poll_interval": float_env("VF_VENICE_POLL_INTERVAL", 5.0),
            "poll_timeout": float_env("VF_VENICE_POLL_TIMEOUT", 900.0),
            "models": {
                "text2img": optional_env("VF_VENICE_MODEL_TEXT2IMG", "z-image-turbo"),
                "img2img": optional_env("VF_VENICE_MODEL_IMG2IMG", "qwen-edit"),
                "text2vid": optional_env("VF_VENICE_MODEL_TEXT2VID", "wan-2.5-preview-text-to-video"),
                "img2vid": optional_env("VF_VENICE_MODEL_IMG2VID", "wan-2.5-preview-image-to-video"),
            },
            "image": {
                "width": int_env("VF_VENICE_IMAGE_WIDTH", 512),
                "height": int_env("VF_VENICE_IMAGE_HEIGHT", 512),
                "cfg_scale": float_env("VF_VENICE_IMAGE_CFG_SCALE", 7.0),
                "steps": int_env("VF_VENICE_IMAGE_STEPS", 8),
            },
            "video": {
                "duration_seconds": float_env("VF_VENICE_VIDEO_DURATION_SECONDS", 5.0),
                "resolution": optional_env("VF_VENICE_VIDEO_RESOLUTION", "720p"),
                "aspect_ratio": optional_env("VF_VENICE_VIDEO_ASPECT_RATIO", "16:9"),
                "negative_prompt": optional_env("VF_VENICE_VIDEO_NEGATIVE_PROMPT", "low quality, blurry, distorted"),
                "audio": env_flag("VF_VENICE_VIDEO_AUDIO"),
            },
            "text_to_video_first_cycle": True,
            "enable_end_frame_morph": env_flag("VF_VENICE_ENABLE_END_FRAME_MORPH"),
        }
    }
