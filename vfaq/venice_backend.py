#!/usr/bin/env python3
"""
venice_backend.py — Native Venice.ai multi-flavour backend
═══════════════════════════════════════════════════════════════════════════════

Native Venice backend for Visual FaQtory, supporting:
  - text2img via /image/generate
  - img2img via /image/edit
  - text2vid via /video/queue + /video/retrieve
  - img2vid via /video/queue with image_url
  - first/last-frame style morphing via end_image_url when supported

The backend is intentionally native HTTP rather than a ComfyUI shim because
Venice already exposes first-class image and video endpoints with distinct
async semantics for video jobs.

Part of Visual FaQtory v0.9.0-beta
"""
from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from .backends import GenerationRequest, GenerationResult, GeneratorBackend, InputMode

logger = logging.getLogger(__name__)

_VENICE_DEFAULT_BASE_URL = "https://api.venice.ai/api/v1"
_VENICE_DEFAULT_VIDEO_NEGATIVE = "low resolution, error, worst quality, low quality, defects"
_VENICE_RETRYABLE_STATUS_CODES = (429, 500, 503)
_VENICE_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"  # braille dot spinner — 10 frames
_VENICE_JOB_TIMINGS_FILE = Path(__file__).parent.parent / "run" / ".venice_job_timings.json"


def _load_job_timings() -> Dict[str, List[float]]:
    """Load per-model rolling job timing history from disk."""
    try:
        if _VENICE_JOB_TIMINGS_FILE.exists():
            return json.loads(_VENICE_JOB_TIMINGS_FILE.read_text())
    except Exception:
        pass
    return {}


def _save_job_timing(model: str, elapsed: float) -> None:
    """Append a completed job's elapsed time to the per-model rolling history."""
    try:
        _VENICE_JOB_TIMINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        timings = _load_job_timings()
        history = timings.get(model, [])
        history.append(round(elapsed, 1))
        timings[model] = history[-20:]  # keep last 20 samples
        _VENICE_JOB_TIMINGS_FILE.write_text(json.dumps(timings, indent=2))
    except Exception:
        pass


def _eta_seconds(model: str, elapsed: float) -> Optional[float]:
    """
    Return estimated seconds remaining based on rolling average of past jobs.
    Returns None if no history exists yet.
    """
    timings = _load_job_timings()
    history = timings.get(model) or []
    if not history:
        return None
    avg = sum(history) / len(history)
    remaining = avg - elapsed
    return max(0.0, remaining)


class _LiveSpinner:
    """
    Background-thread terminal spinner that redraws stderr at 10 fps regardless
    of how slow the main thread is (polling, HTTP blocking, etc.).

    Displays: op type, status, ETA estimate (from rolling job history), elapsed time.
    Progress % shown only when the API actually returns a numeric progress field.

    Usage — video polling loop:
        spinner = _LiveSpinner("img2vid", model)
        spinner.start()
        while True:
            response = retrieve()
            if done:
                spinner.stop("✓ Done")
                break
            spinner.update(status_str, progress_0_to_1, poll_number)
            time.sleep(poll_interval)

    Usage — synchronous HTTP (image generation):
        spinner = _LiveSpinner("text2img", model)
        spinner.start()
        try:
            result = self._request_json(...)
        finally:
            spinner.stop()
    """

    _REDRAW_INTERVAL = 0.1   # seconds between redraws → 10 fps

    def __init__(self, op: str, model: str = "") -> None:
        self.op = op
        self.model = model
        self._started = time.time()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._status: str = "starting"
        self._progress: Optional[float] = None   # 0.0–1.0 or None
        self._poll_n: int = 0
        self._frame: int = 0
        self._thread: Optional[threading.Thread] = None
        self._width = 80

    # ── Public API ────────────────────────────────────────────────────

    def start(self) -> "_LiveSpinner":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def update(
        self,
        status: str = "",
        progress: Optional[float] = None,
        poll_n: int = 0,
    ) -> None:
        with self._lock:
            if status:
                self._status = status
            if progress is not None:
                self._progress = progress
            if poll_n:
                self._poll_n = poll_n

    def stop(self, final_msg: str = "") -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._clear()
        if final_msg:
            logger.info(final_msg)

    # ── Internal ──────────────────────────────────────────────────────

    def _clear(self) -> None:
        sys.stderr.write("\r" + " " * self._width + "\r")
        sys.stderr.flush()

    def _run(self) -> None:
        while not self._stop_event.wait(self._REDRAW_INTERVAL):
            self._redraw()

    def _redraw(self) -> None:
        elapsed = time.time() - self._started
        spin = _VENICE_SPINNER[self._frame % len(_VENICE_SPINNER)]
        self._frame += 1

        with self._lock:
            status = self._status.upper()
            progress = self._progress
            poll_n = self._poll_n

        # Progress bar — only when API gives us a real value
        if progress is not None:
            pct = min(100.0, max(0.0, progress * 100 if progress <= 1.0 else progress))
            filled = int(pct / 5)
            progress_part = f" [{'█' * filled}{'░' * (20 - filled)}] {pct:.0f}%"
        else:
            # ETA from rolling history (video jobs only — image calls are fast)
            eta = _eta_seconds(self.model, elapsed) if self.model else None
            if eta is not None:
                progress_part = f" ~{eta:.0f}s left"
            else:
                progress_part = ""

        poll_part = f" poll#{poll_n}" if poll_n else ""
        line = (
            f"\r  {spin} Venice {self.op} — {status}{progress_part} "
            f"— {elapsed:.1f}s{poll_part}  "
        )
        sys.stderr.write(f"{line:<{self._width}}"[:self._width])
        sys.stderr.flush()
_ENV_VAR_RE = re.compile(r"^\$\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}$|^\$(?P<plain>[A-Za-z_][A-Za-z0-9_]*)$")


def _expand_env_placeholder(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    match = _ENV_VAR_RE.match(value.strip())
    if not match:
        return value
    env_name = match.group("braced") or match.group("plain")
    return os.environ.get(env_name, value)


def _aspect_ratio_from_dims(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        return "16:9"
    g = math.gcd(int(width), int(height)) or 1
    return f"{int(width) // g}:{int(height) // g}"


def _seconds_from_duration_token(token: str) -> Optional[float]:
    if token is None:
        return None
    token = str(token).strip().lower()
    if not token:
        return None
    if token.endswith("s"):
        token = token[:-1]
    try:
        return float(token)
    except ValueError:
        return None


def _duration_token(seconds: float) -> str:
    value = max(1, int(round(float(seconds))))
    return f"{value}s"


@dataclass
class VeniceConfig:
    base_url: str = _VENICE_DEFAULT_BASE_URL
    api_key_env: str = "VENICE_API_KEY"
    api_key: Optional[str] = None
    timeout: float = 120.0
    poll_interval: float = 3.0
    poll_timeout: float = 900.0
    cleanup_after_download: bool = True
    validate_models: bool = True
    strict_model_validation: bool = False
    max_retries: int = 3
    retry_backoff_base: float = 1.5
    retry_backoff_max: float = 20.0

    image_model: str = "z-image-turbo"
    image_edit_model: str = "qwen-edit"
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    image_cfg_scale: float = 7.5
    image_steps: int = 8
    image_style_preset: Optional[str] = None
    image_variants: int = 1
    image_seed: Optional[int] = None
    hide_watermark: bool = True
    safe_mode: bool = True
    image_negative_prompt: str = ""
    image_lora_strength: Optional[int] = None

    video_model_text_to_video: str = "wan-2.5-preview-text-to-video"
    video_model_image_to_video: str = "wan-2.1-pro-image-to-video"
    video_duration: str = "5s"
    video_resolution: str = "720p"
    video_aspect_ratio: str = "16:9"
    video_negative_prompt: str = _VENICE_DEFAULT_VIDEO_NEGATIVE
    video_audio: bool = False
    video_seed: Optional[int] = None
    video_reference_image_urls: List[str] = field(default_factory=list)
    text_to_video_first_cycle: bool = True
    enable_end_frame_morph: bool = True

    # ── Per-operation overrides (text2vid / img2vid) ─────────────────────────
    # When set, these take priority over the global video.* values above.
    # Leave unset to inherit the global default for that field.
    text2vid_duration: Optional[str] = None
    text2vid_aspect_ratio: Optional[str] = None
    text2vid_resolution: Optional[str] = None

    img2vid_duration: Optional[str] = None
    img2vid_aspect_ratio: Optional[str] = None
    img2vid_resolution: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VeniceConfig":
        data = dict(d or {})
        models = dict(data.get("models") or {})
        image = dict(data.get("image") or {})
        video = dict(data.get("video") or {})

        # Friendly aliases from requested clean schema.
        if models:
            data.setdefault("image_model", models.get("text2img") or models.get("image") or models.get("generate"))
            data.setdefault("image_edit_model", models.get("img2img") or models.get("edit") or models.get("image_edit"))
            data.setdefault("video_model_text_to_video", models.get("text2vid") or models.get("text_to_video"))
            data.setdefault("video_model_image_to_video", models.get("img2vid") or models.get("image_to_video"))

        if image:
            data.setdefault("image_width", image.get("width"))
            data.setdefault("image_height", image.get("height"))
            data.setdefault("image_cfg_scale", image.get("cfg_scale"))
            data.setdefault("image_steps", image.get("steps"))
            data.setdefault("image_style_preset", image.get("style_preset"))
            data.setdefault("image_negative_prompt", image.get("negative_prompt"))
            data.setdefault("image_seed", image.get("seed"))
            data.setdefault("image_variants", image.get("variants"))
            data.setdefault("hide_watermark", image.get("hide_watermark"))
            data.setdefault("safe_mode", image.get("safe_mode"))
            data.setdefault("image_lora_strength", image.get("lora_strength"))

        if video:
            if video.get("duration_seconds") is not None:
                data.setdefault("video_duration", _duration_token(float(video.get("duration_seconds"))))
            elif video.get("duration") is not None:
                data.setdefault("video_duration", video.get("duration"))
            data.setdefault("video_aspect_ratio", video.get("aspect_ratio"))
            data.setdefault("video_resolution", video.get("resolution"))
            data.setdefault("video_negative_prompt", video.get("negative_prompt"))
            data.setdefault("video_audio", video.get("audio"))
            data.setdefault("video_seed", video.get("seed"))
            data.setdefault("video_reference_image_urls", video.get("reference_image_urls"))
            data.setdefault("poll_interval", video.get("poll_interval"))
            data.setdefault("poll_timeout", video.get("poll_timeout"))

            # Per-operation overrides: venice.video.text2vid.* / venice.video.img2vid.*
            def _parse_op_overrides(op_key: str, dur_field: str, ar_field: str, res_field: str) -> None:
                op = dict(video.get(op_key) or {})
                if not op:
                    return
                dur = op.get("duration_seconds") or op.get("duration")
                if dur is not None:
                    data.setdefault(dur_field, _duration_token(float(dur)) if isinstance(dur, (int, float)) else str(dur))
                ar = op.get("aspect_ratio")
                if ar is not None:
                    data.setdefault(ar_field, str(ar))
                res = op.get("resolution")
                if res is not None:
                    data.setdefault(res_field, str(res))

            _parse_op_overrides("text2vid", "text2vid_duration", "text2vid_aspect_ratio", "text2vid_resolution")
            _parse_op_overrides("img2vid",  "img2vid_duration",  "img2vid_aspect_ratio",  "img2vid_resolution")

        # Legacy / flat aliases.
        if "model" in data and "image_model" not in data:
            data["image_model"] = data["model"]
        if "video_model" in data:
            data.setdefault("video_model_text_to_video", data["video_model"])
            data.setdefault("video_model_image_to_video", data["video_model"])
        if "cfg_scale" in data and "image_cfg_scale" not in data:
            data["image_cfg_scale"] = data["cfg_scale"]
        if "steps" in data and "image_steps" not in data:
            data["image_steps"] = data["steps"]
        if "duration" in data and "video_duration" not in data:
            data["video_duration"] = data["duration"]
        if "duration_seconds" in data and "video_duration" not in data:
            data["video_duration"] = _duration_token(float(data["duration_seconds"]))
        if "poll_seconds" in data and "poll_interval" not in data:
            data["poll_interval"] = data["poll_seconds"]
        if "aspect_ratio" in data and "video_aspect_ratio" not in data:
            data["video_aspect_ratio"] = data["aspect_ratio"]
        if "resolution" in data and "video_resolution" not in data:
            data["video_resolution"] = data["resolution"]

        for env_key in ("api_key", "base_url"):
            if env_key in data:
                data[env_key] = _expand_env_placeholder(data[env_key])

        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known and v is not None}
        cfg = cls(**filtered)
        return cfg


class VeniceBackend(GeneratorBackend):
    """Native Venice.ai backend covering image and video flavours."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "venice"
        self.venice_cfg = VeniceConfig.from_dict(config.get("venice", {}))
        self.session = requests.Session()
        self._model_cache: Optional[List[Dict[str, Any]]] = None

    # ──────────────────────────────────────────────────────────────────
    # Public backend API
    # ──────────────────────────────────────────────────────────────────
    def check_availability(self) -> Tuple[bool, str]:
        api_key = self._resolve_api_key()
        if not api_key:
            return (
                False,
                f"Venice API key not found. Set {self.venice_cfg.api_key_env} or venice.api_key in config.",
            )
        if not self.venice_cfg.validate_models:
            return True, f"Venice configured ({self.venice_cfg.base_url})"
        try:
            models = self._get_models()
            image_models = [m["id"] for m in models if m.get("type") == "image"]
            video_models = [m["id"] for m in models if m.get("type") == "video"]
            return True, (
                f"Venice reachable ({len(image_models)} image model(s), {len(video_models)} video model(s))"
            )
        except Exception as e:
            return False, f"Venice availability check failed: {e}"

    def supports_mode(self, mode: InputMode) -> bool:
        return mode in {InputMode.TEXT, InputMode.IMAGE}

    def generate_image(self, request: GenerationRequest) -> GenerationResult:
        start_time = time.time()
        try:
            if request.mode == InputMode.IMAGE or request.init_image_path:
                source = request.init_image_path or request.base_image_path
                if not source:
                    raise RuntimeError("Venice img2img requires init_image_path/base_image_path")
                result = self._edit_image(request, source)
            else:
                result = self._generate_image(request)
            result.generation_time = time.time() - start_time
            return result
        except Exception as e:
            logger.error(f"[Venice] Image generation failed: {e}")
            return GenerationResult(success=False, error=str(e), generation_time=time.time() - start_time)

    def generate_video(self, request: GenerationRequest, source_image: Optional[Path]) -> GenerationResult:
        start_time = time.time()
        try:
            model = self._get_video_model(source_image)
            self._validate_model_for_type(model, "video")
            model_info = self._lookup_model(model)
            payload, omitted_fields = self._build_video_payload(
                request=request,
                model=model,
                source_image=source_image,
                end_image_path=None,
                model_info=model_info,
            )

            result = self._run_video_job(
                payload,
                request.output_dir / f"{request.atom_id}_venice.mp4",
                request,
                omitted_fields=omitted_fields,
            )
            result.generation_time = time.time() - start_time
            return result
        except Exception as e:
            logger.error(f"[Venice] Video generation failed: {e}")
            return GenerationResult(success=False, error=str(e), generation_time=time.time() - start_time)

    def generate_morph_video(
        self,
        request: GenerationRequest,
        start_image_path: Path,
        end_image_path: Path,
    ) -> GenerationResult:
        start_time = time.time()
        try:
            if not self.venice_cfg.enable_end_frame_morph:
                return GenerationResult(
                    success=False,
                    error="Venice end-frame morph is disabled in venice.enable_end_frame_morph",
                    generation_time=time.time() - start_time,
                )
            model = self.venice_cfg.video_model_image_to_video
            self._validate_model_for_type(model, "video")
            model_info = self._lookup_model(model)
            payload, omitted_fields = self._build_video_payload(
                request=request,
                model=model,
                source_image=start_image_path,
                end_image_path=end_image_path,
                model_info=model_info,
            )
            if end_image_path is not None and "end_image_url" in omitted_fields:
                raise RuntimeError(
                    f"Configured Venice model '{model}' does not advertise end-frame morph support; "
                    f"choose a model with end_image_url support or disable morph/loop-closure for Venice."
                )
            result = self._run_video_job(
                payload,
                request.output_dir / f"{request.atom_id}_venice.mp4",
                request,
                omitted_fields=omitted_fields,
            )
            result.generation_time = time.time() - start_time
            return result
        except Exception as e:
            logger.error(f"[Venice] Morph video generation failed: {e}")
            return GenerationResult(success=False, error=str(e), generation_time=time.time() - start_time)

    # ──────────────────────────────────────────────────────────────────
    # Image endpoints
    # ──────────────────────────────────────────────────────────────────
    def _generate_image(self, request: GenerationRequest) -> GenerationResult:
        model = self.venice_cfg.image_model
        self._validate_model_for_type(model, "image")
        width, height = self._resolve_image_dimensions(request)
        cfg_scale = float(request.cfg_scale or self.venice_cfg.image_cfg_scale)
        steps = int(request.steps or self.venice_cfg.image_steps)
        seed = int(request.seed if request.seed is not None else (self.venice_cfg.image_seed or 0))
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": request.prompt,
            "width": width,
            "height": height,
            "format": "png",
            "cfg_scale": cfg_scale,
            "steps": steps,
            "negative_prompt": request.negative_prompt or self.venice_cfg.image_negative_prompt,
            "safe_mode": bool(self.venice_cfg.safe_mode),
            "hide_watermark": bool(self.venice_cfg.hide_watermark),
            "variants": max(1, int(self.venice_cfg.image_variants or 1)),
            "return_binary": False,
            "seed": seed,
        }
        if self.venice_cfg.image_style_preset:
            payload["style_preset"] = self.venice_cfg.image_style_preset
        if self.venice_cfg.image_lora_strength is not None:
            payload["lora_strength"] = int(self.venice_cfg.image_lora_strength)

        spinner = _LiveSpinner("text2img", model)
        spinner.update(status="generating")
        spinner.start()
        try:
            resp = self._request_json("POST", "/image/generate", json=payload)
        finally:
            spinner.stop()

        images = resp.get("images") or []
        if not images:
            raise RuntimeError("Venice image/generate returned no images")
        image_bytes = self._decode_base64_image(images[0])
        output_path = request.output_dir / f"{request.atom_id}_venice.png"
        self._write_png(output_path, image_bytes)
        return GenerationResult(
            success=True,
            image_path=output_path,
            metadata={
                "backend": "venice",
                "model": model,
                "operation": "text2img",
                "request": {
                    "width": width,
                    "height": height,
                    "cfg_scale": cfg_scale,
                    "steps": steps,
                    "seed": seed,
                    "style_preset": self.venice_cfg.image_style_preset,
                },
                "response": {
                    "id": resp.get("id"),
                    "timing": resp.get("timing"),
                },
            },
        )

    def _edit_image(self, request: GenerationRequest, source_path: Path) -> GenerationResult:
        model = self.venice_cfg.image_edit_model
        self._validate_model_for_type(model, "inpaint")
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": request.prompt,
            "image": self._raw_base64_for_file(source_path),
            "aspect_ratio": request.aspect_ratio or _aspect_ratio_from_dims(*self._resolve_image_dimensions(request)),
        }

        spinner = _LiveSpinner("img2img", model)
        spinner.update(status="editing")
        spinner.start()
        try:
            response = self._request_raw("POST", "/image/edit", json=payload)
        finally:
            spinner.stop()

        content_type = response.headers.get("Content-Type", "")
        maybe_json = self._safe_json_from_response(response)
        if response.status_code >= 400:
            raise RuntimeError(
                self._format_error_payload(response.status_code, maybe_json)
                if maybe_json is not None
                else f"Venice API error HTTP {response.status_code} from /image/edit (content-type={content_type or 'unknown'})"
            )
        if "image/" not in content_type:
            if maybe_json is not None:
                raise RuntimeError(self._format_error_payload(response.status_code, maybe_json))
            raise RuntimeError(
                f"Venice /image/edit returned unexpected content-type={content_type or 'unknown'}"
            )
        output_path = request.output_dir / f"{request.atom_id}_venice.png"
        self._write_png(output_path, response.content)
        return GenerationResult(
            success=True,
            image_path=output_path,
            metadata={
                "backend": "venice",
                "model": model,
                "operation": "img2img",
                "request": {
                    "source_image": str(source_path),
                    "aspect_ratio": payload["aspect_ratio"],
                },
            },
        )

    # ──────────────────────────────────────────────────────────────────
    # Video endpoints
    # ──────────────────────────────────────────────────────────────────
    def _run_video_job(
        self,
        queue_payload: Dict[str, Any],
        output_path: Path,
        request: GenerationRequest,
        omitted_fields: Optional[List[str]] = None,
    ) -> GenerationResult:
        queue_data, effective_payload, stripped_retry_fields = self._queue_video_request(queue_payload)
        model = queue_data.get("model") or effective_payload["model"]
        queue_id = queue_data.get("queue_id")
        if not queue_id:
            raise RuntimeError(f"Venice video/queue returned no queue_id: {queue_data}")

        op = "img2vid" if queue_payload.get("image_url") else "text2vid"
        duration_hint = effective_payload.get("duration", "?")
        logger.info(
            f"[Venice] Job queued — id={queue_id[:12]}… model={model} op={op} "
            f"duration={duration_hint}"
        )

        started = time.time()
        polls = 0
        last_status: Dict[str, Any] = {}
        cleanup_ok = None
        last_status_name = ""

        spinner = _LiveSpinner(op, model)
        spinner.update(status="queued")
        spinner.start()

        try:
            while True:
                elapsed = time.time() - started
                if elapsed > float(self.venice_cfg.poll_timeout):
                    raise TimeoutError(
                        f"Venice video job timed out after {self.venice_cfg.poll_timeout}s "
                        f"(queue_id={queue_id}, last_status={last_status})"
                    )

                retrieve_payload = {
                    "model": model,
                    "queue_id": queue_id,
                    "delete_media_on_completion": False,
                }
                response = self._request_raw("POST", "/video/retrieve", json=retrieve_payload)
                polls += 1
                content_type = response.headers.get("Content-Type", "")

                if content_type.startswith("video/"):
                    # ── Success ───────────────────────────────────────────
                    _save_job_timing(model, elapsed)
                    spinner.stop(
                        f"[Venice] {op} complete — {elapsed:.1f}s, {polls} poll(s), "
                        f"{len(response.content) / 1024:.0f} KB → {output_path.name}"
                    )
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(response.content)
                    if self.venice_cfg.cleanup_after_download:
                        try:
                            self._request_json("POST", "/video/complete", json={"model": model, "queue_id": queue_id})
                            cleanup_ok = True
                        except Exception as cleanup_error:
                            cleanup_ok = False
                            logger.warning(f"[Venice] Video cleanup failed for {queue_id}: {cleanup_error}")
                    return GenerationResult(
                        success=True,
                        video_path=output_path,
                        metadata={
                            "backend": "venice",
                            "model": model,
                            "operation": op,
                            "queue_id": queue_id,
                            "request": self._summarize_payload(effective_payload),
                            "response": {
                                "queue": queue_data,
                                "retrieve_polls": polls,
                                "last_status": last_status,
                                "cleanup_after_download": cleanup_ok,
                                "omitted_optional_fields": list(omitted_fields or []),
                                "stripped_retry_fields": list(stripped_retry_fields),
                            },
                            "pipeline": {
                                "requested_duration_seconds": request.duration_seconds,
                                "requested_fps_hint": request.video_fps,
                                "requested_frame_hint": request.video_frames,
                                "video_prompt_present": bool(request.video_prompt),
                                "motion_prompt_present": bool(request.motion_prompt),
                            },
                        },
                    )

                last_status = self._safe_json_from_response(response) or {}
                status_name = str(last_status.get("status", "PROCESSING")).upper()

                if response.status_code >= 400:
                    spinner.stop()
                    raise RuntimeError(self._format_error_payload(response.status_code, last_status))
                if status_name in {"FAILED", "ERROR"}:
                    spinner.stop()
                    raise RuntimeError(f"Venice video generation failed: {last_status}")

                # ── Parse progress from any known field Venice might return ─
                raw_prog = (
                    last_status.get("progress")
                    or last_status.get("percentage")
                    or last_status.get("completionPercentage")
                    or last_status.get("completion_percentage")
                )
                progress_0_to_1: Optional[float] = None
                if raw_prog is not None:
                    try:
                        pv = float(raw_prog)
                        progress_0_to_1 = pv if pv <= 1.0 else pv / 100.0
                    except (TypeError, ValueError):
                        pass

                if status_name != last_status_name:
                    logger.debug(
                        f"[Venice] Job {queue_id[:12]}… status → {status_name}"
                        + (f" ({progress_0_to_1*100:.0f}%)" if progress_0_to_1 is not None else "")
                    )
                    last_status_name = status_name

                # Spinner thread redraws every 0.5s on its own; just push new state.
                spinner.update(status=status_name, progress=progress_0_to_1, poll_n=polls)

                # Main thread sleeps the full poll interval — spinner stays animated.
                time.sleep(float(self.venice_cfg.poll_interval))

        except Exception:
            spinner.stop()
            raise

    # ──────────────────────────────────────────────────────────────────
    # Model validation / config helpers
    # ──────────────────────────────────────────────────────────────────
    def _resolve_api_key(self) -> Optional[str]:
        explicit = _expand_env_placeholder(self.venice_cfg.api_key)
        if explicit and explicit != self.venice_cfg.api_key:
            return explicit
        return explicit or os.environ.get(self.venice_cfg.api_key_env)

    def _get_video_model(self, source_image: Optional[Path]) -> str:
        return (
            self.venice_cfg.video_model_image_to_video
            if source_image is not None
            else self.venice_cfg.video_model_text_to_video
        )

    def _resolve_image_dimensions(self, request: GenerationRequest) -> Tuple[int, int]:
        width = int(request.width or 0)
        height = int(request.height or 0)
        if (
            (width <= 0 or width == 1024)
            and self.venice_cfg.image_width
        ):
            width = int(self.venice_cfg.image_width)
        if (
            (height <= 0 or height == 576)
            and self.venice_cfg.image_height
        ):
            height = int(self.venice_cfg.image_height)
        return max(1, width or 1024), max(1, height or 576)

    def _select_video_duration(self, request: GenerationRequest, op: str = "text2vid") -> str:
        # Per-request explicit value always wins.
        if request.duration_seconds and request.duration_seconds > 0:
            return _duration_token(float(request.duration_seconds))
        # Per-op override from config (venice.video.text2vid / img2vid).
        op_override = (
            self.venice_cfg.text2vid_duration if op == "text2vid"
            else self.venice_cfg.img2vid_duration
        )
        if op_override:
            return op_override
        # Global video default.
        default_seconds = _seconds_from_duration_token(self.venice_cfg.video_duration)
        return _duration_token(default_seconds if default_seconds is not None else 5.0)

    def _select_aspect_ratio(self, request: GenerationRequest, op: str = "text2vid") -> str:
        # Per-request explicit value always wins.
        if request.aspect_ratio:
            return str(request.aspect_ratio)
        # Per-op override.
        op_override = (
            self.venice_cfg.text2vid_aspect_ratio if op == "text2vid"
            else self.venice_cfg.img2vid_aspect_ratio
        )
        if op_override:
            return str(op_override)
        # Global default. Always str() — YAML parses unquoted 16:9 as int 969.
        ar = self.venice_cfg.video_aspect_ratio
        if ar:
            return str(ar)
        return _aspect_ratio_from_dims(request.width, request.height)

    def _select_resolution(self, request: GenerationRequest, op: str = "text2vid") -> str:
        if request.resolution:
            return str(request.resolution)
        op_override = (
            self.venice_cfg.text2vid_resolution if op == "text2vid"
            else self.venice_cfg.img2vid_resolution
        )
        if op_override:
            return str(op_override)
        return self.venice_cfg.video_resolution

    def _get_models(self) -> List[Dict[str, Any]]:
        if self._model_cache is None:
            payload = self._request_json("GET", "/models")
            self._model_cache = list(payload.get("data") or [])
        return self._model_cache

    def _validate_model_for_type(self, model_id: str, expected_type: str) -> None:
        if not self.venice_cfg.validate_models:
            return
        try:
            models = self._get_models()
        except Exception as e:
            if self.venice_cfg.strict_model_validation:
                raise RuntimeError(f"Failed to validate Venice models: {e}")
            logger.warning(f"[Venice] Model validation skipped after lookup failure: {e}")
            return
        matching = [m for m in models if m.get("id") == model_id]
        if not matching:
            raise RuntimeError(
                f"Configured Venice model '{model_id}' was not returned by /models. "
                f"Check venice.models / venice.*_model values and your account access."
            )
        declared_type = str(matching[0].get("type", "")).lower()
        if declared_type and declared_type != expected_type:
            raise RuntimeError(
                f"Configured Venice model '{model_id}' is type='{declared_type}', expected '{expected_type}'."
            )

    def _lookup_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        models: Optional[List[Dict[str, Any]]] = self._model_cache
        if models is None and self.venice_cfg.validate_models:
            try:
                models = self._get_models()
            except Exception:
                models = None
        if not models:
            return None
        for model in models:
            if str(model.get("id")) == str(model_id):
                return model
        return None

    @staticmethod
    def _find_nested_value(container: Any, candidate_keys: List[str]) -> Any:
        wanted = {str(key).lower() for key in candidate_keys}
        queue: List[Any] = [container]
        while queue:
            current = queue.pop(0)
            if isinstance(current, dict):
                for key, value in current.items():
                    if str(key).lower() in wanted:
                        return value
                    if isinstance(value, (dict, list, tuple)):
                        queue.append(value)
            elif isinstance(current, (list, tuple)):
                for item in current:
                    if isinstance(item, (dict, list, tuple)):
                        queue.append(item)
        return None

    @staticmethod
    def _coerce_optional_bool(value: Any) -> Optional[bool]:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "1", "supported", "enabled"}:
                return True
            if normalized in {"false", "no", "0", "unsupported", "disabled"}:
                return False
            if normalized.startswith("[") or normalized.startswith("{"):
                return True
        if isinstance(value, (list, tuple, set, dict)):
            return bool(value)
        return None

    def _supports_video_field(self, model_info: Optional[Dict[str, Any]], field_name: str) -> Optional[bool]:
        if not model_info:
            return None

        normalized_field = str(field_name).lower()
        supported_field_lists = [
            "supportedrequestfields", "requestfields", "allowedparameters",
            "supportedparameters", "inputfields", "supportedinputs",
        ]
        listed = self._find_nested_value(model_info, supported_field_lists)
        if isinstance(listed, (list, tuple, set)) and listed:
            listed_normalized = {str(item).strip().lower() for item in listed if item is not None}
            if normalized_field in listed_normalized:
                return True
            aliases = {
                "aspect_ratio": {"aspect ratio", "aspect_ratio", "aspectratio"},
                "audio": {"audio", "audio_enabled"},
                "reference_image_urls": {"reference_image_urls", "reference images", "referenceimages"},
                "end_image_url": {"end_image_url", "end image", "endimage", "last_frame"},
            }.get(normalized_field, {normalized_field})
            if listed_normalized.intersection(aliases):
                return True
            return False

        capability_key_groups = {
            "audio": ["supportsAudioConfig", "supportsAudio", "audioSupported", "audio_config"],
            "aspect_ratio": [
                "supportsAspectRatio", "supportsAspectRatioSelection", "supportsAspectRatios",
                "supportedAspectRatios", "allowedAspectRatios", "aspectRatioOptions",
                "aspect_ratio_options", "aspectRatios",
            ],
            "reference_image_urls": [
                "supportsReferenceImages", "referenceImageLimit", "maxReferenceImages",
                "reference_image_limit", "reference_image_urls",
            ],
            "end_image_url": [
                "supportsEndImage", "supportsEndImages", "supportsLastFrame",
                "supportsFirstLastFrame", "end_image_url",
            ],
            "resolution": [
                "supportsResolution", "supportedResolutions", "allowedResolutions",
                "resolution_options", "resolutionOptions",
            ],
        }
        value = self._find_nested_value(model_info, capability_key_groups.get(normalized_field, [field_name]))
        if value is None:
            return None
        return self._coerce_optional_bool(value)

    def _build_video_payload(
        self,
        request: GenerationRequest,
        model: str,
        source_image: Optional[Path],
        end_image_path: Optional[Path],
        model_info: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], List[str]]:
        omitted_fields: List[str] = []
        # Determine operation type so per-op config overrides are applied correctly.
        op = "img2vid" if source_image is not None else "text2vid"

        payload: Dict[str, Any] = {
            "model": model,
            "prompt": request.video_prompt or request.prompt,
            "duration": self._select_video_duration(request, op),
            "negative_prompt": request.negative_prompt or self.venice_cfg.video_negative_prompt,
        }

        supports_aspect_ratio = self._supports_video_field(model_info, "aspect_ratio")
        if supports_aspect_ratio is not False:
            payload["aspect_ratio"] = self._select_aspect_ratio(request, op)
        else:
            omitted_fields.append("aspect_ratio")

        supports_resolution = self._supports_video_field(model_info, "resolution")
        if supports_resolution is not False:
            payload["resolution"] = self._select_resolution(request, op)
        else:
            omitted_fields.append("resolution")

        supports_audio = self._supports_video_field(model_info, "audio")
        audio_value = bool(request.generate_audio if request.generate_audio is not None else self.venice_cfg.video_audio)
        if supports_audio is not False:
            payload["audio"] = audio_value
        else:
            omitted_fields.append("audio")

        if source_image is not None:
            # ── Pre-resize source image to target output dimensions ────────────
            # Most img2vid models (Seedance, wan, ovi) ignore the `resolution` and
            # `aspect_ratio` fields and instead match their output to the input
            # image dimensions. If we want 480p 16:9, the source image must be
            # 854×480 before it's sent — Venice won't do the resize for us when
            # those params are unsupported.
            resized = self._resize_source_for_img2vid(source_image, request, op)
            payload["image_url"] = self._data_url_for_file(resized)

        supports_reference_images = self._supports_video_field(model_info, "reference_image_urls")
        if self.venice_cfg.video_reference_image_urls:
            if supports_reference_images is not False:
                payload["reference_image_urls"] = list(self.venice_cfg.video_reference_image_urls)
            else:
                omitted_fields.append("reference_image_urls")

        if end_image_path is not None:
            supports_end_image = self._supports_video_field(model_info, "end_image_url")
            if supports_end_image is not False:
                payload["end_image_url"] = self._data_url_for_file(end_image_path)
            else:
                omitted_fields.append("end_image_url")

        return payload, omitted_fields

    @staticmethod
    def _resolution_to_dims(resolution_token: str, aspect_ratio: str) -> Optional[tuple]:
        """
        Convert a Venice resolution token ('480p', '720p', '1080p') + aspect ratio
        string ('16:9', '4:3', '1:1', etc.) into (width, height) pixel dims,
        rounded to nearest multiple of 8.

        Returns None if resolution_token is unrecognised.
        """
        _HEIGHT_MAP = {"360p": 360, "480p": 480, "576p": 576, "720p": 720, "1080p": 1080, "1440p": 1440}
        height = _HEIGHT_MAP.get(resolution_token.lower())
        if height is None:
            return None
        try:
            ar_parts = [float(x) for x in aspect_ratio.split(":")]
            ar = ar_parts[0] / ar_parts[1]
        except Exception:
            ar = 16.0 / 9.0
        raw_width = height * ar
        width = round(raw_width / 8) * 8
        return (width, height)

    def _resize_source_for_img2vid(
        self,
        source_image: Path,
        request: "GenerationRequest",
        op: str,
    ) -> Path:
        """
        Resize source image to the configured img2vid target dimensions.
        Returns the resized image path (written to a sibling .resized.png).
        If no resize is needed (already correct size or no resolution configured),
        returns the original path unchanged.
        """
        # Determine target resolution token and aspect ratio for this op
        res_token = (
            self.venice_cfg.img2vid_resolution if op == "img2vid"
            else self.venice_cfg.text2vid_resolution
        ) or self.venice_cfg.video_resolution

        ar_str = str(
            (self.venice_cfg.img2vid_aspect_ratio if op == "img2vid" else self.venice_cfg.text2vid_aspect_ratio)
            or self.venice_cfg.video_aspect_ratio
            or "16:9"
        )

        dims = self._resolution_to_dims(res_token, ar_str)
        if dims is None:
            return source_image   # unrecognised token — pass through unchanged

        target_w, target_h = dims

        try:
            from PIL import Image
            img = Image.open(source_image)
            src_w, src_h = img.size
            if src_w == target_w and src_h == target_h:
                return source_image   # already correct — nothing to do

            logger.debug(
                f"[Venice] Resizing source image {src_w}×{src_h} → {target_w}×{target_h} "
                f"for {op} ({res_token} {ar_str})"
            )
            resized_img = img.resize((target_w, target_h), Image.LANCZOS)
            resized_path = source_image.parent / (source_image.stem + f".resized_{target_w}x{target_h}.png")
            resized_img.save(str(resized_path), "PNG")
            return resized_path
        except Exception as e:
            logger.warning(f"[Venice] Source image resize failed ({e}), using original")
            return source_image

    @staticmethod
    def _optional_queue_fields() -> List[str]:
        # Fields that some Venice models genuinely don't accept — safe to auto-strip
        # on HTTP 400 when the response indicates "unsupported".
        #
        # Note: resolution can fail in two distinct ways:
        #   1) unsupported field for this model → strip and retry
        #   2) invalid enum value with options[] → snap-to-nearest and retry
        return ["reference_image_urls", "end_image_url", "audio", "aspect_ratio", "resolution"]

    @staticmethod
    def _queue_field_aliases() -> Dict[str, List[str]]:
        return {
            "reference_image_urls": [
                "reference_image_urls", "reference images", "referenceimages", "reference_image_url",
            ],
            "end_image_url": [
                "end_image_url", "end image", "endimage", "last frame", "last_frame",
            ],
            "audio": ["audio", "audio_enabled", "generate_audio"],
            "aspect_ratio": ["aspect_ratio", "aspect ratio", "aspectratio"],
            "resolution": ["resolution", "output_resolution", "video_resolution"],
        }

    @staticmethod
    def _normalize_error_field(value: Any) -> str:
        token = str(value or "").strip().lower()
        token = re.sub(r"[^a-z0-9_]+", "_", token)
        token = re.sub(r"_+", "_", token).strip("_")
        return token

    @classmethod
    def _resolve_optional_field_name(cls, raw_field: Any, candidates: List[str]) -> Optional[str]:
        aliases = cls._queue_field_aliases()
        normalized = cls._normalize_error_field(raw_field)
        if not normalized:
            return None

        probe_tokens = [normalized]
        if "." in normalized:
            probe_tokens.append(normalized.rsplit(".", 1)[-1])
        if "/" in normalized:
            probe_tokens.append(normalized.rsplit("/", 1)[-1])

        for field in candidates:
            field_aliases = aliases.get(field, [field])
            normalized_aliases = {cls._normalize_error_field(alias) for alias in field_aliases}
            normalized_aliases.add(cls._normalize_error_field(field))
            if any(token in normalized_aliases for token in probe_tokens):
                return field
        return None

    @classmethod
    def _fields_mentioned_in_message(cls, message: str, candidates: List[str]) -> List[str]:
        blob = str(message or "").lower()
        aliases = cls._queue_field_aliases()
        mentioned: List[str] = []
        for field in candidates:
            field_aliases = aliases.get(field, [field])
            if any(str(alias).lower() in blob for alias in field_aliases):
                mentioned.append(field)
        return mentioned

    @staticmethod
    def _classify_queue_error_message(message: Any, code: str = "") -> Tuple[bool, bool]:
        text = str(message or "").lower()
        code_l = str(code or "").lower()

        unsupported_tokens = (
            "does not support",
            "doesn't support",
            "not supported",
            "unsupported",
            "not allowed",
            "unknown field",
            "unrecognized",
            "unexpected",
            "not permitted",
        )
        required_tokens = ("required",)

        is_required = any(token in text for token in required_tokens)
        is_unsupported = any(token in text for token in unsupported_tokens)
        if code_l in {"unknown_field", "unrecognized_keys", "unrecognized_key"}:
            is_unsupported = True
        return is_unsupported, is_required

    def _queue_error_field_hints(self, payload: Dict[str, Any], data: Optional[Dict[str, Any]]) -> Tuple[set, set]:
        present_optionals = [field for field in self._optional_queue_fields() if field in payload]
        unsupported_in_error: set = set()
        required_in_error: set = set()
        if not present_optionals or not data:
            return unsupported_in_error, required_in_error

        def _record(field_name: Optional[str], message: Any, code: str = "") -> None:
            if not field_name:
                return
            is_unsupported, is_required = self._classify_queue_error_message(message, code=code)
            if is_required:
                required_in_error.add(field_name)
            if is_unsupported:
                unsupported_in_error.add(field_name)

        # Structured issues[] hints are the most reliable source.
        issues = list((data or {}).get("issues") or [])
        for issue in issues:
            message = str(issue.get("message") or "")
            code = str(issue.get("code") or "")
            path = issue.get("path") or []

            issue_fields: List[str] = []
            if path:
                for raw_path_part in path:
                    resolved = self._resolve_optional_field_name(raw_path_part, present_optionals)
                    if resolved and resolved not in issue_fields:
                        issue_fields.append(resolved)
            for resolved in self._fields_mentioned_in_message(message, present_optionals):
                if resolved not in issue_fields:
                    issue_fields.append(resolved)

            for field_name in issue_fields:
                _record(field_name, message, code=code)

        # details.<field>._errors can carry unsupported-field messaging even when
        # issues[] is sparse or absent.
        details = (data or {}).get("details")
        if isinstance(details, dict):
            for raw_key, value in details.items():
                field_name = self._resolve_optional_field_name(raw_key, present_optionals)
                if not field_name:
                    continue
                detail_messages: List[str] = []
                if isinstance(value, dict):
                    detail_messages.extend(str(msg) for msg in list(value.get("_errors") or []) if msg is not None)
                elif isinstance(value, (list, tuple, set)):
                    detail_messages.extend(str(msg) for msg in value if msg is not None)
                elif isinstance(value, str):
                    detail_messages.append(value)
                for msg in detail_messages:
                    _record(field_name, msg)

        # Top-level message/detail fallback.
        for msg_key in ("message", "detail", "error"):
            raw_message = data.get(msg_key)
            if raw_message is None:
                continue
            for field_name in self._fields_mentioned_in_message(str(raw_message), present_optionals):
                _record(field_name, raw_message)

        return unsupported_in_error, required_in_error

    @staticmethod
    def _queue_payload_signature(payload: Dict[str, Any]) -> str:
        summarized = VeniceBackend._summarize_payload(payload or {})
        return json.dumps(summarized, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def _queue_retry_candidates(self, payload: Dict[str, Any], data: Optional[Dict[str, Any]]) -> List[str]:
        present_optionals = [field for field in self._optional_queue_fields() if field in payload]
        if not present_optionals:
            return []

        unsupported_in_error, required_in_error = self._queue_error_field_hints(payload, data)

        structured_candidates = [
            field for field in present_optionals
            if field in unsupported_in_error and field not in required_in_error
        ]
        if structured_candidates:
            return structured_candidates

        # Fallback: if the response clearly says "unsupported" and names optionals
        # in text form, strip all named non-required fields.
        message_blob = json.dumps(data, ensure_ascii=False).lower() if data else ""
        blob_mentions = self._fields_mentioned_in_message(message_blob, present_optionals)
        blob_unsupported, _ = self._classify_queue_error_message(message_blob)
        if blob_unsupported and blob_mentions:
            return [field for field in blob_mentions if field not in required_in_error]

        return []

    def _queue_video_request(self, queue_payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
        effective_payload = dict(queue_payload)
        stripped_fields: List[str] = []
        # Fields we stripped but Venice later told us are Required — never strip again.
        restored_fields: set = set()
        attempted_payload_signatures = {self._queue_payload_signature(effective_payload)}
        first_400_error: Optional[str] = None
        # Track whether we already snapped the duration to avoid infinite loops.
        duration_snapped: bool = False
        # Track whether we already snapped the resolution to avoid infinite loops.
        resolution_snapped: bool = False

        while True:
            response = self._request_raw("POST", "/video/queue", json=effective_payload)
            data = self._safe_json_from_response(response)
            if response.status_code < 400:
                if data is None:
                    raise RuntimeError(
                        f"Venice endpoint /video/queue returned non-JSON content-type={response.headers.get('Content-Type')}"
                    )
                return data, effective_payload, stripped_fields

            if response.status_code == 400:
                if first_400_error is None:
                    first_400_error = self._format_error_payload(response.status_code, data)
                issues = list((data or {}).get("issues") or [])

                # ── Duration snap-to-nearest ──────────────────────────────────
                # Venice returns "invalid_enum_value" for duration with an
                # `options` list of what the model actually accepts.
                # Snap to nearest valid duration and retry automatically.
                if not duration_snapped and "duration" in effective_payload:
                    for issue in issues:
                        path = issue.get("path") or []
                        code = str(issue.get("code") or "")
                        options = list(issue.get("options") or [])
                        if path and str(path[0]) == "duration" and code == "invalid_enum_value" and options:
                            current_raw = effective_payload.get("duration", "")
                            try:
                                current_s = float(str(current_raw).rstrip("s"))
                            except (TypeError, ValueError):
                                break
                            def _tok_to_s(tok: str) -> float:
                                try:
                                    return float(str(tok).rstrip("s"))
                                except (TypeError, ValueError):
                                    return float("inf")
                            nearest = min(options, key=lambda t: abs(_tok_to_s(t) - current_s))
                            duration_snapped = True
                            logger.warning(
                                f"[Venice] Duration '{current_raw}' not accepted by model "
                                f"(valid: {options}). Snapping to nearest: '{nearest}'"
                            )
                            effective_payload["duration"] = nearest
                            break
                    if duration_snapped:
                        signature = self._queue_payload_signature(effective_payload)
                        if signature not in attempted_payload_signatures:
                            attempted_payload_signatures.add(signature)
                            continue

                # ── Resolution snap-to-nearest ────────────────────────────────
                # Same logic as duration when resolution is supported but value is
                # outside the model's accepted enum set.
                if not resolution_snapped and "resolution" in effective_payload:
                    for issue in issues:
                        path = issue.get("path") or []
                        code = str(issue.get("code") or "")
                        options = list(issue.get("options") or [])
                        if path and str(path[0]) == "resolution" and code == "invalid_enum_value" and options:
                            current_raw = effective_payload.get("resolution", "")
                            # Extract integer height value for nearest-match (e.g. "480p" → 480)
                            def _res_to_int(tok: str) -> int:
                                try:
                                    return int("".join(c for c in str(tok) if c.isdigit()) or "0")
                                except (TypeError, ValueError):
                                    return 0
                            current_px = _res_to_int(current_raw)
                            nearest = min(options, key=lambda t: abs(_res_to_int(t) - current_px))
                            resolution_snapped = True
                            logger.warning(
                                f"[Venice] Resolution '{current_raw}' not accepted by model "
                                f"(valid: {options}). Snapping to nearest: '{nearest}'"
                            )
                            effective_payload["resolution"] = nearest
                            break
                    if resolution_snapped:
                        signature = self._queue_payload_signature(effective_payload)
                        if signature not in attempted_payload_signatures:
                            attempted_payload_signatures.add(signature)
                            continue

                # ── Check if any previously-stripped fields are now Required ────
                # Venice sometimes says field X is "not supported" (→ we strip it),
                # then in the next attempt says X is "Required" (contradictory).
                # Detect this and restore the field so we can keep trying.
                _, required_in_error = self._queue_error_field_hints(effective_payload, data)
                newly_required_stripped = [field for field in list(stripped_fields) if field in required_in_error]

                if newly_required_stripped:
                    for field in newly_required_stripped:
                        if field in queue_payload:
                            effective_payload[field] = queue_payload[field]
                        stripped_fields.remove(field)
                        restored_fields.add(field)
                    logger.warning(
                        f"[Venice] Re-adding field(s) previously stripped but now Required by model: "
                        f"{', '.join(newly_required_stripped)}"
                    )
                    # Now strip whatever else this 400 is complaining about
                    retry_fields = [
                        field for field in self._queue_retry_candidates(effective_payload, data)
                        if field not in stripped_fields and field not in restored_fields
                    ]
                    if retry_fields:
                        for field in retry_fields:
                            effective_payload.pop(field, None)
                            stripped_fields.append(field)
                        logger.warning(
                            f"[Venice] Retrying /video/queue without optional field(s): {', '.join(retry_fields)}"
                        )
                    signature = self._queue_payload_signature(effective_payload)
                    if signature not in attempted_payload_signatures:
                        attempted_payload_signatures.add(signature)
                        continue

                # ── Normal retry: strip confirmed-optional unsupported fields ───
                retry_fields = [
                    field for field in self._queue_retry_candidates(effective_payload, data)
                    if field not in stripped_fields and field not in restored_fields
                ]
                if retry_fields:
                    for field in retry_fields:
                        effective_payload.pop(field, None)
                        stripped_fields.append(field)
                    logger.warning(
                        f"[Venice] Retrying /video/queue without optional field(s): {', '.join(retry_fields)}"
                    )
                    signature = self._queue_payload_signature(effective_payload)
                    if signature not in attempted_payload_signatures:
                        attempted_payload_signatures.add(signature)
                        continue

            final_error = self._format_error_payload(response.status_code, data)
            if stripped_fields:
                stripped = ", ".join(stripped_fields)
                if first_400_error and first_400_error != final_error:
                    final_error = (
                        f"{final_error} "
                        f"(after retry without optional field(s): {stripped}; original queue error: {first_400_error})"
                    )
                else:
                    final_error = f"{final_error} (after retry without optional field(s): {stripped})"
            raise RuntimeError(final_error)

    # ──────────────────────────────────────────────────────────────────
    # HTTP helpers
    # ──────────────────────────────────────────────────────────────────
    def _headers(self) -> Dict[str, str]:
        api_key = self._resolve_api_key()
        if not api_key:
            raise RuntimeError(
                f"Venice API key missing. Set {self.venice_cfg.api_key_env} or venice.api_key."
            )
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _request_json(self, method: str, endpoint: str, json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        response = self._request_raw(method, endpoint, json=json)
        data = self._safe_json_from_response(response)
        if response.status_code >= 400:
            raise RuntimeError(self._format_error_payload(response.status_code, data))
        if data is None:
            raise RuntimeError(
                f"Venice endpoint {endpoint} returned non-JSON content-type={response.headers.get('Content-Type')}"
            )
        return data

    def _request_raw(self, method: str, endpoint: str, json: Optional[Dict[str, Any]] = None) -> requests.Response:
        url = self.venice_cfg.base_url.rstrip("/") + endpoint
        max_attempts = max(1, int(self.venice_cfg.max_retries or 1))
        attempt = 0
        last_exception: Optional[Exception] = None

        while attempt < max_attempts:
            attempt += 1
            try:
                response = self.session.request(
                    method=method.upper(),
                    url=url,
                    headers=self._headers(),
                    json=json,
                    timeout=float(self.venice_cfg.timeout),
                )
            except requests.RequestException as exc:
                last_exception = exc
                if attempt >= max_attempts:
                    raise RuntimeError(f"Venice request failed for {endpoint}: {exc}") from exc
                delay = self._retry_delay(None, attempt)
                logger.warning(f"[Venice] Transport error on {endpoint}, retrying in {delay:.2f}s: {exc}")
                time.sleep(delay)
                continue

            if response.status_code not in _VENICE_RETRYABLE_STATUS_CODES or attempt >= max_attempts:
                return response

            delay = self._retry_delay(response, attempt)
            logger.warning(
                f"[Venice] HTTP {response.status_code} from {endpoint}, retrying in {delay:.2f}s "
                f"(attempt {attempt}/{max_attempts})"
            )
            time.sleep(delay)

        if last_exception:
            raise RuntimeError(f"Venice request failed for {endpoint}: {last_exception}") from last_exception
        raise RuntimeError(f"Venice request failed for {endpoint}")

    def _retry_delay(self, response: Optional[requests.Response], attempt: int) -> float:
        if response is not None and response.status_code == 429:
            reset_at = response.headers.get("x-ratelimit-reset-requests")
            if reset_at:
                try:
                    delta = max(0.0, float(reset_at) - time.time())
                    if delta > 0:
                        return min(float(self.venice_cfg.retry_backoff_max), delta)
                except ValueError:
                    pass
        base = max(0.25, float(self.venice_cfg.retry_backoff_base or 1.5))
        return min(float(self.venice_cfg.retry_backoff_max), base * (2 ** max(0, attempt - 1)))

    @staticmethod
    def _safe_json_from_response(response: requests.Response) -> Optional[Dict[str, Any]]:
        try:
            return response.json()
        except Exception:
            return None

    @staticmethod
    def _format_error_payload(status_code: int, data: Optional[Dict[str, Any]]) -> str:
        if not data:
            return f"Venice API error HTTP {status_code}"
        code = data.get("error") or data.get("code") or data.get("error_code")
        message = data.get("message") or data.get("detail") or data
        if isinstance(message, dict):
            message = json.dumps(message, ensure_ascii=False)
        if code:
            return f"Venice API error {code} (HTTP {status_code}): {message}"
        return f"Venice API error HTTP {status_code}: {message}"

    @staticmethod
    def _summarize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        summarized: Dict[str, Any] = {}
        for key, value in (payload or {}).items():
            if isinstance(value, str) and value.startswith("data:"):
                summarized[key] = f"<data-url:{value.split(';', 1)[0][5:] or 'binary'}>"
            elif isinstance(value, list) and value and isinstance(value[0], str) and str(value[0]).startswith("data:"):
                summarized[key] = ["<data-url>"] * len(value)
            else:
                summarized[key] = value
        return summarized

    # ──────────────────────────────────────────────────────────────────
    # File / encoding helpers
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _raw_base64_for_file(path: Path) -> str:
        return base64.b64encode(Path(path).read_bytes()).decode("utf-8")

    def _data_url_for_file(self, path: Path) -> str:
        suffix = Path(path).suffix.lower()
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(suffix, "image/png")
        return f"data:{mime};base64,{self._raw_base64_for_file(path)}"

    @staticmethod
    def _decode_base64_image(payload: str) -> bytes:
        if payload.startswith("data:"):
            payload = payload.split(",", 1)[1]
        return base64.b64decode(payload)

    @staticmethod
    def _write_png(path: Path, image_bytes: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if image_bytes.startswith(b"\x89PNG"):
            path.write_bytes(image_bytes)
            return
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(image_bytes))
            img.save(path, format="PNG")
        except Exception as e:
            raise RuntimeError(f"Failed to normalize Venice image output to PNG: {e}")
