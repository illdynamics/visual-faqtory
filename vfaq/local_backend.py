#!/usr/bin/env python3
"""
local_backend.py — Local MLX image/video generation backend
═══════════════════════════════════════════════════════════════════════════════

Runs local models on Apple Silicon (and any host with a compatible CLI) via a
pluggable *runner*. The primary runner is ``lightning-mlx``, but ``mflux`` and a
fully-custom ``command`` runner are also supported so the backend can be pointed
at whatever local inference tool you already have installed.

Supported models
────────────────
  - ``flux-1-dev``      — FLUX.1-dev text-to-image / img2img
  - ``flux-1-kontext``  — FLUX.1 Kontext image editing / first-last-frame
  - ``wan-2.2``         — Wan 2.2 image-to-video (and text-to-video)
  - ``z-image-turbo``   — Z-Image-Turbo text-to-image

These are selected in config under ``local.models``:

    backend:
      type: local
      runner: lightning-mlx
      models:
        image: z-image-turbo     # or flux-1-dev
        video: wan-2.2
        edit: flux-1-kontext
      model_paths:
        flux-1-dev: /path/on/disk
        flux-1-kontext: /path/on/disk
        wan-2.2: /path/on/disk
        z-image-turbo: /path/on/disk

Because the exact ``lightning-mlx`` CLI surface may vary between installs, every
command template is overridable via ``local.lightning_mlx.*_template`` (and the
equivalent ``local.command.*_template`` / ``local.mflux.*_template``). Defaults
are provided for a reasonable ``lightning-mlx`` interface and are documented in
``worqspace/config-local.example.yaml`` plus ``macbook-finish-vfaq-steps.md``.

This backend implements the standard ``GeneratorBackend`` interface, so the
existing sliding-story engine treats it exactly like ComfyUI/Mock: text2img →
img2img keyframe → img2vid / morph.

Part of Visual FaQtory v0.9.4-beta
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .backends import (
    GeneratorBackend,
    GenerationRequest,
    GenerationResult,
)

logger = logging.getLogger(__name__)

# ── Model registry ──────────────────────────────────────────────────────────
MODEL_FLUX_1_DEV = "flux-1-dev"
MODEL_FLUX_1_KONTEXT = "flux-1-kontext"
MODEL_WAN_2_2 = "wan-2.2"
MODEL_Z_IMAGE_TURBO = "z-image-turbo"

KNOWN_MODELS = (
    MODEL_FLUX_1_DEV,
    MODEL_FLUX_1_KONTEXT,
    MODEL_WAN_2_2,
    MODEL_Z_IMAGE_TURBO,
)

# Human aliases so users can type the names naturally. Every alias maps to the
# canonical id used in config lookups and command rendering.
_MODEL_ALIASES = {
    "flux-1-dev": MODEL_FLUX_1_DEV,
    "flux 1 dev": MODEL_FLUX_1_DEV,
    "flux1dev": MODEL_FLUX_1_DEV,
    "flux.1 dev": MODEL_FLUX_1_DEV,
    "flux.1-dev": MODEL_FLUX_1_DEV,
    "flux-1-kontext": MODEL_FLUX_1_KONTEXT,
    "flux 1 kontext": MODEL_FLUX_1_KONTEXT,
    "flux1kontext": MODEL_FLUX_1_KONTEXT,
    "flux.1 kontext": MODEL_FLUX_1_KONTEXT,
    "flux.1-kontext": MODEL_FLUX_1_KONTEXT,
    "kontext": MODEL_FLUX_1_KONTEXT,
    "wan-2.2": MODEL_WAN_2_2,
    "wan 2.2": MODEL_WAN_2_2,
    "wan2.2": MODEL_WAN_2_2,
    "wan": MODEL_WAN_2_2,
    "z-image-turbo": MODEL_Z_IMAGE_TURBO,
    "z image turbo": MODEL_Z_IMAGE_TURBO,
    "zimage-turbo": MODEL_Z_IMAGE_TURBO,
    "z-image": MODEL_Z_IMAGE_TURBO,
}

MODEL_DISPLAY_NAMES = {
    MODEL_FLUX_1_DEV: "FLUX.1-dev",
    MODEL_FLUX_1_KONTEXT: "FLUX.1 Kontext",
    MODEL_WAN_2_2: "Wan 2.2",
    MODEL_Z_IMAGE_TURBO: "Z-Image-Turbo",
}


def normalize_model_name(raw: Optional[str]) -> Optional[str]:
    """Normalize a user-facing model name to a canonical model id."""
    if not raw:
        return None
    key = str(raw).strip().lower()
    if key in _MODEL_ALIASES:
        return _MODEL_ALIASES[key]
    # Also accept canonical ids without requiring exact registry match.
    if key in KNOWN_MODELS:
        return key
    return None


# ── Default command templates ───────────────────────────────────────────────
# These are intentionally conservative and documented. Override them in config
# if your installed lightning-mlx / mflux / command uses a different flag set.
_DEFAULT_LIGHTNING_MLX_TEMPLATES = {
    "image": (
        "lightning-mlx generate-image --model {model} --model-path {model_path} "
        "--prompt {prompt} --output {output} --width {width} --height {height} "
        "--seed {seed} --steps {steps}"
    ),
    "video": (
        "lightning-mlx generate-video --model {model} --model-path {model_path} "
        "--prompt {prompt} --image {input_image} --output {output} "
        "--width {width} --height {height} --fps {fps} --duration {duration} "
        "--seed {seed} --steps {steps}"
    ),
    "morph": (
        "lightning-mlx generate-video --model {model} --model-path {model_path} "
        "--prompt {prompt} --image {start_image} --end-image {end_image} "
        "--output {output} --width {width} --height {height} --fps {fps} "
        "--duration {duration} --seed {seed}"
    ),
}

_DEFAULT_MFLUX_TEMPLATES = {
    "image": (
        "mflux-generate --model {model} --path {model_path} --prompt {prompt} "
        "--output {output} --width {width} --height {height} --seed {seed} "
        "--steps {steps}"
    ),
    # mflux is image-only. Video requests will fail clearly unless the user
    # supplies a custom command template.
    "video": "",
    "morph": "",
}


class _TemplateRenderer:
    """Render a command template with safe, shell-quoted placeholder values."""

    def __init__(self, template: str):
        self.template = template or ""

    def render(self, values: Dict[str, Any]) -> List[str]:
        if not self.template:
            raise RuntimeError("No command template configured for this operation")
        import shlex
        # Quote each value BEFORE formatting so a prompt with spaces stays a
        # single argv token (shlex.split then unquotes it correctly).
        quoted = {
            k: shlex.quote(str(v)) if v is not None and str(v) != "" else "''"
            for k, v in values.items()
        }
        try:
            rendered = self.template.format(**quoted)
        except KeyError as exc:
            raise RuntimeError(
                f"Command template references unknown placeholder {exc}. "
                "Supported placeholders: model, model_path, prompt, output, width, height, "
                "seed, steps, fps, duration, input_image, start_image, end_image, negative_prompt"
            ) from exc
        try:
            return shlex.split(rendered)
        except ValueError as exc:
            raise RuntimeError(f"Could not parse command template: {exc}") from exc


class LocalRunner:
    """Base class for local generation runners."""

    name = "local"

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}

    def executable(self) -> str:
        raise NotImplementedError

    def _templates(self) -> Dict[str, str]:
        raise NotImplementedError

    def check_availability(self) -> Tuple[bool, str]:
        exe = self.executable()
        if not exe:
            return False, f"Runner '{self.name}' has no executable configured"
        found = shutil.which(exe)
        if not found:
            return False, f"Runner executable not found on PATH: {exe}"
        return True, f"Runner executable found: {found}"

    def build_command(self, operation: str, values: Dict[str, Any]) -> List[str]:
        templates = self._templates()
        template = templates.get(operation)
        if not template:
            raise RuntimeError(
                f"Runner '{self.name}' does not support operation '{operation}'. "
                "Provide a custom template in config or use another runner."
            )
        return _TemplateRenderer(template).render(values)


class LightningMLXRunner(LocalRunner):
    name = "lightning-mlx"

    def executable(self) -> str:
        return str(self.config.get("executable") or "lightning-mlx")

    def _templates(self) -> Dict[str, str]:
        base = self.config.get("templates") or {}
        merged = dict(_DEFAULT_LIGHTNING_MLX_TEMPLATES)
        if isinstance(base, dict):
            merged.update(base)
        # Also allow the flat keys used in config-local.example.yaml.
        for op in ("image", "video", "morph"):
            flat = self.config.get(f"{op}_template")
            if flat:
                merged[op] = str(flat)
        return merged


class MfluxRunner(LocalRunner):
    name = "mflux"

    def executable(self) -> str:
        return str(self.config.get("executable") or "mflux-generate")

    def _templates(self) -> Dict[str, str]:
        base = self.config.get("templates") or {}
        merged = dict(_DEFAULT_MFLUX_TEMPLATES)
        if isinstance(base, dict):
            merged.update(base)
        for op in ("image", "video", "morph"):
            flat = self.config.get(f"{op}_template")
            if flat:
                merged[op] = str(flat)
        return merged


class GenericCommandRunner(LocalRunner):
    name = "command"

    def executable(self) -> str:
        # The command template itself contains the executable; we still use a
        # placeholder for availability checks.
        return str(self.config.get("executable") or "")

    def _templates(self) -> Dict[str, str]:
        base = self.config.get("templates") or {}
        out = {
            "image": base.get("image", self.config.get("image_template", "")),
            "video": base.get("video", self.config.get("video_template", "")),
            "morph": base.get("morph", self.config.get("morph_template", "")),
        }
        for op in ("image", "video", "morph"):
            flat = self.config.get(f"{op}_template")
            if flat:
                out[op] = str(flat)
        return out

    def check_availability(self) -> Tuple[bool, str]:
        exe = self.executable()
        if not exe:
            return False, "Generic command runner has no executable configured"
        # Don't require an executable on PATH; a template can be a full shell
        # command. We only validate that templates exist.
        missing = [op for op in ("image", "video", "morph") if not self._templates().get(op)]
        if missing:
            return False, f"Generic command runner missing templates for: {', '.join(missing)}"
        return True, "Generic command runner templates present"


RUNNER_REGISTRY = {
    "lightning-mlx": LightningMLXRunner,
    "lightning_mlx": LightningMLXRunner,
    "mflux": MfluxRunner,
    "command": GenericCommandRunner,
    "generic": GenericCommandRunner,
}


def _resolve_runner(config: Dict[str, Any]) -> LocalRunner:
    runner_name = str(config.get("runner") or config.get("runner_name") or "lightning-mlx").strip().lower()
    runner_config = config.get(runner_name, config.get(runner_name.replace("_", "-"), config))
    if not isinstance(runner_config, dict):
        runner_config = config
    cls = RUNNER_REGISTRY.get(runner_name)
    if cls is None:
        raise RuntimeError(
            f"Unknown local runner '{runner_name}'. Supported runners: "
            f"{', '.join(sorted(RUNNER_REGISTRY))}"
        )
    return cls(runner_config)


class LocalBackend(GeneratorBackend):
    """Local MLX generation backend.

    The backend is a thin orchestrator around a ``LocalRunner``. It resolves
    the model for each capability (image / video / edit), resolves its on-disk
    path, builds a runner command, executes it, and returns a standard
    ``GenerationResult``.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "local"
        self.local_cfg = self._extract_local_cfg(config)
        self.runner = _resolve_runner(self.local_cfg)
        self.model_paths = self._normalize_model_paths(self.local_cfg.get("model_paths") or {})
        self.models = self._resolve_models(self.local_cfg.get("models") or {})
        self.timeout = float(self.local_cfg.get("timeout", 3600))
        self.default_width = int(self.local_cfg.get("width", 1024))
        self.default_height = int(self.local_cfg.get("height", 576))
        self.default_steps = int(self.local_cfg.get("steps", 30))
        self.default_fps = float(self.local_cfg.get("fps", 8.0))

    @staticmethod
    def _extract_local_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
        nested = config.get("local")
        if isinstance(nested, dict):
            # Merge top-level convenience keys only when the nested section
            # does not already provide them.
            merged = dict(nested)
            for key in ("runner", "runner_name", "models", "model_paths", "timeout", "width", "height", "steps", "fps"):
                if key not in merged and key in config:
                    merged[key] = config[key]
            return merged
        return dict(config)

    @staticmethod
    def _normalize_model_paths(raw: Dict[str, Any]) -> Dict[str, str]:
        normalized: Dict[str, str] = {}
        for key, value in (raw or {}).items():
            model_id = normalize_model_name(key)
            if model_id and value:
                normalized[model_id] = os.path.expanduser(str(value))
        return normalized

    @staticmethod
    def _resolve_models(raw: Dict[str, Any]) -> Dict[str, str]:
        resolved = {
            "image": MODEL_Z_IMAGE_TURBO,
            "video": MODEL_WAN_2_2,
            "edit": MODEL_FLUX_1_KONTEXT,
        }
        for key in ("image", "video", "edit"):
            candidate = raw.get(key)
            model_id = normalize_model_name(candidate)
            if model_id:
                resolved[key] = model_id
        return resolved

    def _model_path(self, model_id: str) -> Optional[Path]:
        raw = self.model_paths.get(model_id)
        if not raw:
            return None
        path = Path(raw).expanduser()
        return path if path.exists() else None

    def _model_for_request(self, request: GenerationRequest, capability: str) -> str:
        # Explicit request-level override wins (when present and valid).
        if capability == "video":
            explicit = getattr(request, "checkpoint", None)
            if explicit and normalize_model_name(explicit):
                return normalize_model_name(explicit)
        return self.models.get(capability, self.models.get("image", MODEL_Z_IMAGE_TURBO))

    def check_availability(self) -> Tuple[bool, str]:
        runner_ok, runner_msg = self.runner.check_availability()
        if not runner_ok:
            return False, runner_msg

        missing = []
        for capability in ("image", "video", "edit"):
            model_id = self.models.get(capability)
            if not model_id:
                continue
            if not self._model_path(model_id):
                missing.append(f"{MODEL_DISPLAY_NAMES.get(model_id, model_id)} ({model_id})")
        if missing:
            return False, (
                "Runner is available but model path(s) not found: " + ", ".join(missing)
            )
        return True, (
            f"Local backend ready via {self.runner.name} — "
            f"image={self.models.get('image')}, video={self.models.get('video')}, "
            f"edit={self.models.get('edit')}"
        )

    # ── Command builders ─────────────────────────────────────────────────
    def _base_values(self, request: GenerationRequest, model_id: str) -> Dict[str, Any]:
        return {
            "model": model_id,
            "model_path": str(self._model_path(model_id) or ""),
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt or "",
            "output": "",
            "width": request.width or self.default_width,
            "height": request.height or self.default_height,
            "seed": request.seed,
            "steps": request.steps or self.default_steps,
        }

    def _run_command(self, cmd: List[str], output_path: Path) -> Tuple[bool, str]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"[Local] Running: {' '.join(str(c) for c in cmd)}")
        try:
            proc = subprocess.run(
                [str(c) for c in cmd],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except FileNotFoundError:
            return False, f"Executable not found. Command: {cmd[0] if cmd else ''}"
        except subprocess.TimeoutExpired:
            return False, f"Generation timed out after {self.timeout:.0f}s"
        except Exception as exc:
            return False, f"Subprocess error: {exc}"

        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-600:]
            return False, f"Exit code {proc.returncode}: {tail}"
        if not output_path.exists() or output_path.stat().st_size == 0:
            return False, f"Command succeeded but no output file found at {output_path}"
        return True, ""

    # ── GeneratorBackend interface ───────────────────────────────────────
    def generate_image(self, request: GenerationRequest) -> GenerationResult:
        start = time.time()
        model_id = self._model_for_request(request, "image")
        model_path = self._model_path(model_id)
        if not model_path:
            return GenerationResult(
                success=False,
                error=f"Local model path not found for image model '{model_id}'. "
                      "Set local.model_paths.{model_id} in config.",
            )

        output_path = request.output_dir / f"{request.atom_id}_image.png"
        values = self._base_values(request, model_id)
        values.update({
            "output": str(output_path),
            "input_image": str(request.init_image_path) if request.init_image_path else "",
            "start_image": str(request.init_image_path) if request.init_image_path else "",
        })

        operation = "image"
        # Kontext is an image-edit model. When the user asks for image
        # generation with a Kontext model and an init image, use the edit
        # template; otherwise use the standard image template.
        if model_id == MODEL_FLUX_1_KONTEXT and request.init_image_path:
            operation = "edit"

        try:
            cmd = self.runner.build_command(operation, values)
            ok, err = self._run_command(cmd, output_path)
        except Exception as exc:
            return GenerationResult(success=False, error=f"Local image command build failed: {exc}")

        if not ok:
            return GenerationResult(success=False, error=f"Local image generation failed: {err}")
        return GenerationResult(
            success=True,
            image_path=output_path,
            generation_time=time.time() - start,
            metadata={"backend": "local", "runner": self.runner.name, "model": model_id},
        )

    def generate_video(self, request: GenerationRequest, source_image: Path) -> GenerationResult:
        start = time.time()
        model_id = self._model_for_request(request, "video")
        model_path = self._model_path(model_id)
        if not model_path:
            return GenerationResult(
                success=False,
                error=f"Local model path not found for video model '{model_id}'. "
                      "Set local.model_paths.{model_id} in config.",
            )

        duration = request.duration_seconds or (request.video_frames / request.video_fps if request.video_frames and request.video_fps else 5.0)
        output_path = request.output_dir / f"{request.atom_id}_video.mp4"
        values = self._base_values(request, model_id)
        values.update({
            "output": str(output_path),
            "input_image": str(source_image) if source_image and source_image.exists() else "",
            "start_image": str(source_image) if source_image and source_image.exists() else "",
            "fps": request.video_fps or self.default_fps,
            "duration": duration,
        })

        try:
            cmd = self.runner.build_command("video", values)
            ok, err = self._run_command(cmd, output_path)
        except Exception as exc:
            return GenerationResult(success=False, error=f"Local video command build failed: {exc}")

        if not ok:
            return GenerationResult(success=False, error=f"Local video generation failed: {err}")
        return GenerationResult(
            success=True,
            video_path=output_path,
            generation_time=time.time() - start,
            metadata={"backend": "local", "runner": self.runner.name, "model": model_id},
        )

    def generate_morph_video(self, request: GenerationRequest, start_image_path: Path, end_image_path: Path) -> GenerationResult:
        start = time.time()
        model_id = self.models.get("video", MODEL_WAN_2_2)
        # Morph/two-image conditioning is best served by the edit/video model.
        edit_model = self.models.get("edit")
        if edit_model and self._model_path(edit_model):
            model_id = edit_model
        model_path = self._model_path(model_id)
        if not model_path:
            return GenerationResult(
                success=False,
                error=f"Local model path not found for morph model '{model_id}'. "
                      "Set local.model_paths.{model_id} in config.",
            )

        duration = request.duration_seconds or 5.0
        output_path = request.output_dir / f"{request.atom_id}_video.mp4"
        values = self._base_values(request, model_id)
        values.update({
            "output": str(output_path),
            "start_image": str(start_image_path),
            "end_image": str(end_image_path),
            "fps": request.video_fps or self.default_fps,
            "duration": duration,
        })

        try:
            cmd = self.runner.build_command("morph", values)
            ok, err = self._run_command(cmd, output_path)
        except Exception as exc:
            return GenerationResult(success=False, error=f"Local morph command build failed: {exc}")

        if not ok:
            return GenerationResult(success=False, error=f"Local morph generation failed: {err}")
        return GenerationResult(
            success=True,
            video_path=output_path,
            generation_time=time.time() - start,
            metadata={"backend": "local", "runner": self.runner.name, "model": model_id},
        )


__all__ = [
    "LocalBackend",
    "LightningMLXRunner",
    "MfluxRunner",
    "GenericCommandRunner",
    "normalize_model_name",
    "KNOWN_MODELS",
    "MODEL_DISPLAY_NAMES",
    "MODEL_FLUX_1_DEV",
    "MODEL_FLUX_1_KONTEXT",
    "MODEL_WAN_2_2",
    "MODEL_Z_IMAGE_TURBO",
]
