#!/usr/bin/env python3
"""
local_backend.py — Local MLX image/video generation backend
═══════════════════════════════════════════════════════════════════════════════

Runs local models on Apple Silicon through a pluggable *runner*. Runners are
thin adapters over real, independently-installable CLI tools:

  - ``mflux``      (recommended) — Z-Image-Turbo, FLUX.1-dev, FLUX.1-Kontext
    image generation via the ``mflux`` project (`pip install mflux`).
  - ``flux-swift`` — the ``flux.swift`` CLI for the ``mzbac/flux1.*.4bit.mlx``
    (flux.swift-format) FLUX.1-dev / FLUX.1-Kontext checkpoints.
  - ``wan``        — ``mlx-gen`` for Wan 2.2 TI2V video (t2v / i2v / ti2v).
  - ``command``    — a fully custom shell command template (escape hatch).

The engine talks to this backend through the standard ``GeneratorBackend``
interface (text2img / img2img / img2vid / morph), so ``local`` behaves exactly
like the ComfyUI / Venice / Veo paths.

Part of Visual FaQtory v0.9.6-beta
"""
from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .backends import (
    GeneratorBackend,
    GenerationRequest,
    GenerationResult,
)

logger = logging.getLogger(__name__)


_OFF_TOKENS = {"", "off", "none", "false", "disabled", "null", "no"}


def _is_off(value: Any) -> bool:
    """Return True for config values that mean "turn this flag off".

    Local runners accept a few spellings for disabling an optional flag so
    users can explicitly switch between mutually-exclusive modes (for example
    ``denoising_step_list`` vs ``steps`` + ``flow_shift``).
    """
    if value is None:
        return True
    if isinstance(value, bool):
        return not value
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0
    if isinstance(value, dict):
        return not value
    if isinstance(value, (int, float)):
        return value == 0
    return str(value).strip().lower() in _OFF_TOKENS


def _find_executable(name: str) -> Optional[str]:
    """Resolve an executable name, including the active Python's venv bin dir."""
    if not name:
        return None
    if os.path.sep in name or (name.startswith("./") or name.startswith("../")):
        return name if os.path.exists(name) else None
    found = shutil.which(name)
    if found:
        return found
    # Running via `.venv/bin/python` does not put `.venv/bin` on PATH; look there.
    for base in (Path(sys.executable).parent, Path(sysconfig.get_path("scripts"))):
        candidate = base / name
        if candidate.exists():
            return str(candidate)
    return None

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

_MODEL_ALIASES = {
    "flux-1-dev": MODEL_FLUX_1_DEV,
    "flux 1 dev": MODEL_FLUX_1_DEV,
    "flux1dev": MODEL_FLUX_1_DEV,
    "flux.1 dev": MODEL_FLUX_1_DEV,
    "flux.1-dev": MODEL_FLUX_1_DEV,
    "flux1.dev": MODEL_FLUX_1_DEV,
    "dev": MODEL_FLUX_1_DEV,
    "flux-1-kontext": MODEL_FLUX_1_KONTEXT,
    "flux 1 kontext": MODEL_FLUX_1_KONTEXT,
    "flux1kontext": MODEL_FLUX_1_KONTEXT,
    "flux.1 kontext": MODEL_FLUX_1_KONTEXT,
    "flux.1-kontext": MODEL_FLUX_1_KONTEXT,
    "flux1.kontext": MODEL_FLUX_1_KONTEXT,
    "kontext": MODEL_FLUX_1_KONTEXT,
    "wan-2.2": MODEL_WAN_2_2,
    "wan 2.2": MODEL_WAN_2_2,
    "wan2.2": MODEL_WAN_2_2,
    "wan": MODEL_WAN_2_2,
    "z-image-turbo": MODEL_Z_IMAGE_TURBO,
    "z image turbo": MODEL_Z_IMAGE_TURBO,
    "zimage-turbo": MODEL_Z_IMAGE_TURBO,
    "z-image": MODEL_Z_IMAGE_TURBO,
    "zimage": MODEL_Z_IMAGE_TURBO,
}

MODEL_DISPLAY_NAMES = {
    MODEL_FLUX_1_DEV: "FLUX.1-dev",
    MODEL_FLUX_1_KONTEXT: "FLUX.1 Kontext",
    MODEL_WAN_2_2: "Wan 2.2",
    MODEL_Z_IMAGE_TURBO: "Z-Image-Turbo",
}

# Sensible default runner per model when `local.model_runners` has no override.
DEFAULT_MODEL_RUNNERS = {
    MODEL_FLUX_1_DEV: "mflux",
    MODEL_FLUX_1_KONTEXT: "mflux",
    MODEL_Z_IMAGE_TURBO: "mflux",
    MODEL_WAN_2_2: "wan",
}


def normalize_model_name(raw: Optional[str]) -> Optional[str]:
    """Normalize a user-facing model name to a canonical model id."""
    if not raw:
        return None
    key = str(raw).strip().lower()
    if key in _MODEL_ALIASES:
        return _MODEL_ALIASES[key]
    if key in KNOWN_MODELS:
        return key
    return None


# ── Generic shell template renderer (used by the ``command`` runner) ────────
class _TemplateRenderer:
    def __init__(self, template: str):
        self.template = template or ""

    def render(self, values: Dict[str, Any]) -> List[str]:
        if not self.template:
            raise RuntimeError("No command template configured for this operation")
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
                "seed, steps, fps, duration, input_image, start_image, end_image, negative_prompt, "
                "denoise_strength, cfg_scale, quantize"
            ) from exc
        try:
            return shlex.split(rendered)
        except ValueError as exc:
            raise RuntimeError(f"Could not parse command template: {exc}") from exc


# ── Runners ─────────────────────────────────────────────────────────────────
class LocalRunner:
    """Base class for local generation runners."""

    name = "local"

    def __init__(self, config: Dict[str, Any], local_cfg: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.local_cfg = local_cfg or self.config

    def executable(self) -> str:
        raise NotImplementedError

    def check_availability(self) -> Tuple[bool, str]:
        exe = self.executable()
        if not exe:
            return False, f"Runner '{self.name}' has no executable configured"
        found = _find_executable(exe)
        if not found:
            return False, f"Runner executable not found on PATH: {exe}"
        return True, f"Runner executable found: {found}"

    def build_image_command(
        self,
        model_id: str,
        model_spec: str,
        request: GenerationRequest,
        output_path: Path,
        operation: str,
        values: Dict[str, Any],
    ) -> List[str]:
        raise NotImplementedError

    def build_video_command(
        self,
        model_id: str,
        model_spec: str,
        request: GenerationRequest,
        output_path: Path,
        source_image: Optional[Path],
        values: Dict[str, Any],
    ) -> List[str]:
        raise NotImplementedError

    def build_morph_command(
        self,
        model_id: str,
        model_spec: str,
        request: GenerationRequest,
        output_path: Path,
        start_image: Path,
        end_image: Path,
        values: Dict[str, Any],
    ) -> List[str]:
        raise NotImplementedError


class MfluxRunner(LocalRunner):
    """mflux runner — image models (Z-Image-Turbo / FLUX.1-dev / FLUX.1-Kontext)."""

    name = "mflux"

    def executable(self) -> str:
        return str(self.config.get("executable") or "mflux-generate")

    def _base(self, model_spec: str) -> str:
        return str(self.config.get("executable") or "mflux-generate")

    def _exe_for(self, model_id: str, operation: str) -> str:
        exe = self.config.get("executable")
        if operation == "edit" and model_id == MODEL_FLUX_1_KONTEXT:
            return str(exe or "mflux-generate-kontext")
        if model_id == MODEL_Z_IMAGE_TURBO:
            return str(exe or "mflux-generate-z-image-turbo")
        if model_id == MODEL_FLUX_1_KONTEXT:
            return str(exe or "mflux-generate-kontext")
        return str(exe or "mflux-generate")

    def _image_steps(self, request: GenerationRequest) -> int:
        # The story engine never sets `steps` on image requests, so
        # GenerationRequest defaults to 30. That would force even the
        # 8-step Z-Image-Turbo to run 30 steps. Prefer the configured
        # local.steps for image models, while still honoring an explicit
        # non-default request.steps from direct callers/tests.
        configured = self.local_cfg.get("steps")
        if request.steps is not None and request.steps != 30:
            return int(request.steps)
        if configured not in (None, "", 0, "0"):
            return int(configured)
        return int(request.steps or 8)

    def _quantize_flag(self) -> List[str]:
        q = self.local_cfg.get("quantize")
        if q in (None, "", 0, "0", "auto"):
            return []
        return ["--quantize", str(q)]

    def _common(
        self,
        exe: str,
        model_spec: str,
        request: GenerationRequest,
        output_path: Path,
        values: Dict[str, Any],
        *,
        image_path: Optional[Path] = None,
    ) -> List[str]:
        cmd = [exe, "--model", model_spec]
        if image_path is not None:
            cmd += ["--image-path", str(image_path)]
            denoise = values.get("denoise_strength") or request.denoise_strength or 0.4
            cmd += ["--image-strength", str(denoise)]
        cmd += [
            "--prompt", request.prompt or "",
            "--output", str(output_path),
            "--width", str(request.width or self.local_cfg.get("width", 1024)),
            "--height", str(request.height or self.local_cfg.get("height", 576)),
            "--seed", str(request.seed if request.seed is not None else 0),
            "--steps", str(self._image_steps(request)),
        ]
        if request.negative_prompt and model_spec not in (MODEL_Z_IMAGE_TURBO,):
            cmd += ["--negative-prompt", request.negative_prompt]
        cmd += self._quantize_flag()
        return cmd

    def build_image_command(self, model_id, model_spec, request, output_path, operation, values):
        exe = self._exe_for(model_id, operation)
        init = None
        if operation in ("img2img", "edit") and request.init_image_path:
            init = request.init_image_path
        return self._common(exe, model_spec, request, output_path, values, image_path=init)

    def build_video_command(self, model_id, model_spec, request, output_path, source_image, values):
        raise RuntimeError(
            "mflux is an image generator and does not support video generation. "
            "Use the 'wan' runner (mlxgen-generate-wan) for Wan 2.2 video, or a "
            "custom 'command' template."
        )

    def build_morph_command(self, model_id, model_spec, request, output_path, start_image, end_image, values):
        raise RuntimeError(
            "mflux does not support first/last-frame morphing. The local backend "
            "falls back to an ffmpeg crossfade morph in this case."
        )


class FluxSwiftRunner(LocalRunner):
    """flux.swift runner — the mzbac/flux1.*.4bit.mlx (flux.swift) checkpoints."""

    name = "flux-swift"

    def executable(self) -> str:
        return str(self.config.get("executable") or "flux.swift.cli")

    def _hf_token_flag(self) -> List[str]:
        env = self.config.get("hf_token_env") or "HF_TOKEN"
        token = os.environ.get(env)
        if not token:
            return []
        return ["--hf-token", token]

    def _common(self, model_spec, request, output_path, values, *, init_image=None):
        exe = self.executable()
        cmd = [exe, "--load-quantized-path", model_spec]
        cmd += self._hf_token_flag()
        if init_image is not None:
            cmd += ["--init-image-path", str(init_image)]
        cmd += [
            "--prompt", request.prompt or "",
            "--output", str(output_path),
            "--width", str(request.width or self.local_cfg.get("width", 1024)),
            "--height", str(request.height or self.local_cfg.get("height", 576)),
            "--seed", str(request.seed if request.seed is not None else 0),
            "--steps", str(request.steps or self.local_cfg.get("steps", 30)),
        ]
        return cmd

    def build_image_command(self, model_id, model_spec, request, output_path, operation, values):
        init = request.init_image_path if operation in ("img2img", "edit") else None
        return self._common(model_spec, request, output_path, values, init_image=init)

    def build_video_command(self, model_id, model_spec, request, output_path, source_image, values):
        raise RuntimeError("flux.swift is image-only; it cannot generate video.")

    def build_morph_command(self, model_id, model_spec, request, output_path, start_image, end_image, values):
        raise RuntimeError("flux.swift is image-only; it cannot generate morph video.")


class WanRunner(LocalRunner):
    """MLX-Gen (``mlxgen-generate-wan``) runner for Wan 2.2 video.

    MLX-Gen supports Wan 2.2 TI2V in two modes:
      * text-to-video (t2v)          - no input image
      * image-to-video (i2v / ti2v)  - first-frame ``--image-path`` + ``--prompt``
    First/last-frame morph is only available on the Wan A14B checkpoints, so
    for the default ``wan2.2-ti2v-5b`` model morph falls back to an ffmpeg
    crossfade in the backend.
    """

    name = "wan"

    def executable(self) -> str:
        return str(self.config.get("executable") or "mlxgen-generate-wan")

    def _common(self, model_spec, request, output_path, source_image, values):
        exe = self.executable()
        duration = float(request.duration_seconds or 5.0)
        fps = float(
            request.video_fps
            or self.config.get("fps")
            or self.local_cfg.get("fps")
            or 12
        )
        if request.video_frames:
            frames = int(request.video_frames)
        else:
            frames = max(1, int(round(duration * fps)))
        # The user's working Wan Turbo recipe generates 65 frames; cap to the
        # configured envelope (configurable via local.wan.max_frames).
        max_frames = int(self.config.get("max_frames") or self.local_cfg.get("max_frames") or 65)
        frames = min(frames, max_frames)

        # Wan supports two mutually-exclusive sampling modes:
        #   * denoising_step_list — explicit Self-Forcing denoise grid
        #   * steps (+ optional flow_shift) — standard step count
        # Any of the three can be set to "off" (off/none/false/disabled/0/[]),
        # so you can switch modes without deleting keys:
        #   * grid mode  -> denoising_step_list: [...] + steps: off + flow_shift: off
        #   * step mode  -> denoising_step_list: off + steps: N (+ flow_shift: X)
        #   * all off    -> let mlxgen-generate-wan use its own defaults
        denoising_step_list = None
        if "denoising_step_list" in self.config:
            grid_raw = self.config["denoising_step_list"]
        elif "denoising-step-list" in self.config:
            grid_raw = self.config["denoising-step-list"]
        else:
            grid_raw = None
        if not _is_off(grid_raw):
            if isinstance(grid_raw, (str, int, float)):
                grid_raw = str(grid_raw).split()
            denoising_step_list = [int(step) for step in grid_raw]

        steps = None
        steps_explicit = "steps" in self.config
        if steps_explicit and not _is_off(self.config["steps"]):
            steps = int(self.config["steps"])

        if "flow_shift" in self.config:
            flow_raw = self.config["flow_shift"]
        elif "flow-shift" in self.config:
            flow_raw = self.config["flow-shift"]
        else:
            flow_raw = None
        flow_shift = None if _is_off(flow_raw) else flow_raw

        grid_on = bool(denoising_step_list)
        steps_on = steps is not None
        flow_on = flow_shift is not None
        if grid_on and (steps_on or flow_on):
            raise ValueError(
                "local.wan sampling-mode conflict: denoising_step_list is "
                "mutually exclusive with steps and flow_shift. Set "
                "denoising_step_list to 'off' (or []) to use steps+flow_shift, "
                "or set steps/flow_shift to 'off' (or 0) to use "
                "denoising_step_list."
            )

        # Mirrors the user's ~/bin/wanturbo-{i2v,t2v}.sh scripts exactly.
        cmd = [exe, "--model", str(model_spec)]
        cmd += ["--prompt", request.prompt or ""]
        if source_image is not None:
            cmd += ["--image-path", str(source_image)]
            cmd += ["--canvas-policy", "exact-resize"]
        cmd += [
            "--output", str(output_path),
            "--width", str(request.width or self.local_cfg.get("width", 640)),
            "--height", str(request.height or self.local_cfg.get("height", 352)),
            "--frames", str(frames),
            "--fps", str(int(fps)),
        ]
        guidance = self.config.get("guidance")
        if guidance is not None:
            cmd += ["--guidance", str(guidance)]
        solver = self.config.get("solver")
        if solver:
            cmd += ["--solver", str(solver)]
        if grid_on:
            cmd += ["--denoising-step-list", *[str(step) for step in denoising_step_list]]
        else:
            if steps_on:
                cmd += ["--steps", str(steps)]
            elif not steps_explicit:
                # No explicit local.wan.steps configured: keep the legacy
                # fallback so existing configs continue to emit `--steps`.
                fallback_steps = request.steps or self.local_cfg.get("steps", 30)
                if not _is_off(fallback_steps):
                    cmd += ["--steps", str(int(fallback_steps))]
            if flow_on:
                cmd += ["--flow-shift", str(flow_shift)]
        cmd += [
            "--seed", str(request.seed if request.seed is not None else 0),
            "--progress",
            "--replace",
            "--no-validate-health",
            "--compile-transformer",
        ]
        cache_limit_gb = self.config.get("mlx_cache_limit_gb")
        if cache_limit_gb not in (None, "", 0, "0", "auto"):
            cmd += ["--mlx-cache-limit-gb", str(cache_limit_gb)]
        return cmd

    def build_image_command(self, model_id, model_spec, request, output_path, operation, values):
        raise RuntimeError("The wan runner is video-only; use an image runner for stills.")

    def build_video_command(self, model_id, model_spec, request, output_path, source_image, values):
        return self._common(model_spec, request, output_path, source_image, values)

    def build_morph_command(self, model_id, model_spec, request, output_path, start_image, end_image, values):
        raise RuntimeError(
            "MLX-Gen Wan 2.2 TI2V-5B does not support first/last-frame morph. "
            "The local backend will fall back to an ffmpeg crossfade morph."
        )


class GenericCommandRunner(LocalRunner):
    """Fully custom shell command runner (image/video/morph templates)."""

    name = "command"

    def executable(self) -> str:
        return str(self.config.get("executable") or "")

    def check_availability(self) -> Tuple[bool, str]:
        missing = [op for op in ("image", "video", "morph") if not self._template(op)]
        if missing:
            return False, f"Generic command runner missing templates for: {', '.join(missing)}"
        return True, "Generic command runner templates present"

    def _template(self, operation: str) -> str:
        return (
            self.config.get(operation)
            or self.config.get(f"{operation}_template")
            or self.config.get("templates", {}).get(operation, "")
        )

    def _render(self, operation: str, values: Dict[str, Any]) -> List[str]:
        return _TemplateRenderer(self._template(operation)).render(values)

    def build_image_command(self, model_id, model_spec, request, output_path, operation, values):
        values["operation"] = operation
        return self._render("image", values)

    def build_video_command(self, model_id, model_spec, request, output_path, source_image, values):
        return self._render("video", values)

    def build_morph_command(self, model_id, model_spec, request, output_path, start_image, end_image, values):
        return self._render("morph", values)


RUNNER_REGISTRY = {
    "mflux": MfluxRunner,
    "flux-swift": FluxSwiftRunner,
    "flux_swift": FluxSwiftRunner,
    "fluxswift": FluxSwiftRunner,
    "wan": WanRunner,
    "mlx-gen": WanRunner,
    "mlxgen": WanRunner,
    "sceneworks": WanRunner,
    "mlx-gen-wan": WanRunner,
    "mlxgen-generate-wan": WanRunner,
    "command": GenericCommandRunner,
    "generic": GenericCommandRunner,
}


def _resolve_runner(name: str, config: Dict[str, Any], local_cfg: Dict[str, Any]) -> LocalRunner:
    key = str(name or "mflux").strip().lower()
    # Runner-specific config block is looked up by several spellings.
    runner_config = (
        config.get(key)
        or config.get(key.replace("_", "-"))
        or config.get(key.replace("-", "_"))
        or {}
    )
    if not isinstance(runner_config, dict):
        runner_config = {}
    cls = RUNNER_REGISTRY.get(key)
    if cls is None:
        raise RuntimeError(
            f"Unknown local runner '{name}'. Supported runners: "
            f"{', '.join(sorted(set(RUNNER_REGISTRY)))}"
        )
    return cls(runner_config, local_cfg=local_cfg)


# ── Backend ─────────────────────────────────────────────────────────────────
class LocalBackend(GeneratorBackend):
    """Local MLX generation backend.

    Resolves the model for each capability (image / video / edit), resolves the
    runner responsible for that model, builds the right command and executes it.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "local"
        self.local_cfg = self._extract_local_cfg(config)
        self.default_runner = str(self.local_cfg.get("runner") or "mflux").strip().lower()
        self.model_runners = self._normalize_model_runners(self.local_cfg.get("model_runners") or {})
        self.model_paths = self._normalize_model_paths(self.local_cfg.get("model_paths") or {})
        self.models = self._resolve_models(self.local_cfg.get("models") or {})
        self.timeout = float(self.local_cfg.get("timeout", 3600))
        self.default_width = int(self.local_cfg.get("width", 1024))
        self.default_height = int(self.local_cfg.get("height", 576))
        self.default_steps = int(self.local_cfg.get("steps", 8))
        self.default_fps = float(self.local_cfg.get("fps", 8.0))
        self._runner_cache: Dict[str, LocalRunner] = {}

    @staticmethod
    def _extract_local_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
        nested = config.get("local")
        if isinstance(nested, dict):
            merged = dict(nested)
            for key in ("runner", "models", "model_paths", "model_runners", "timeout",
                        "width", "height", "steps", "fps", "quantize"):
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
    def _normalize_model_runners(raw: Dict[str, Any]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for key, value in (raw or {}).items():
            model_id = normalize_model_name(key)
            if model_id and value:
                out[model_id] = str(value).strip().lower()
        return out

    @staticmethod
    def _resolve_models(raw: Dict[str, Any]) -> Dict[str, str]:
        resolved = {
            "image": MODEL_Z_IMAGE_TURBO,
            "video": MODEL_WAN_2_2,
            "edit": MODEL_FLUX_1_KONTEXT,
        }
        for key in ("image", "video", "edit"):
            model_id = normalize_model_name(raw.get(key))
            if model_id:
                resolved[key] = model_id
        return resolved

    def _runner_for_model(self, model_id: str) -> LocalRunner:
        if model_id in self._runner_cache:
            return self._runner_cache[model_id]
        runner_name = self.model_runners.get(
            model_id, DEFAULT_MODEL_RUNNERS.get(model_id, self.default_runner)
        )
        runner = _resolve_runner(runner_name, self.local_cfg, self.local_cfg)
        self._runner_cache[model_id] = runner
        return runner

    def _model_spec(self, model_id: str) -> Optional[str]:
        """Return the model spec (path or repo/builtin name) for a model."""
        return self.model_paths.get(model_id)

    def _local_path(self, model_id: str) -> Optional[Path]:
        spec = self._model_spec(model_id)
        if not spec:
            return None
        path = Path(spec).expanduser()
        return path if path.exists() else None

    def _model_for_request(self, request: GenerationRequest, capability: str) -> str:
        if capability == "video":
            explicit = getattr(request, "checkpoint", None)
            if explicit and normalize_model_name(explicit):
                return normalize_model_name(explicit)
        return self.models.get(capability, self.models.get("image", MODEL_Z_IMAGE_TURBO))

    def check_availability(self) -> Tuple[bool, str]:
        failures: List[str] = []
        ready: List[str] = []

        for capability in ("image", "video", "edit"):
            model_id = self.models.get(capability)
            if not model_id:
                continue
            runner = self._runner_for_model(model_id)
            ok, msg = runner.check_availability()
            if not ok:
                failures.append(f"{MODEL_DISPLAY_NAMES.get(model_id, model_id)}: {msg}")
                continue
            # flux-swift / wan require a real on-disk checkpoint; mflux can also
            # resolve a built-in name or HuggingFace repo id at generation time.
            if runner.name != "mflux":
                if not self._local_path(model_id) and not self._model_spec(model_id):
                    failures.append(
                        f"{MODEL_DISPLAY_NAMES.get(model_id, model_id)}: no model_path set"
                    )
                    continue
            ready.append(f"{MODEL_DISPLAY_NAMES.get(model_id, model_id)} ({model_id})")

        if failures:
            return False, (
                f"Local backend partially available — ready: {', '.join(ready) or 'none'}; "
                f"missing: {'; '.join(failures)}"
            )
        return True, (
            f"Local backend ready via {self.default_runner} — "
            f"image={self.models.get('image')}, video={self.models.get('video')}, "
            f"edit={self.models.get('edit')}"
        )

    def _base_values(self, request: GenerationRequest, model_id: str) -> Dict[str, Any]:
        return {
            "model": model_id,
            "model_path": self._model_spec(model_id) or "",
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt or "",
            "output": "",
            "width": request.width or self.default_width,
            "height": request.height or self.default_height,
            "seed": request.seed if request.seed is not None else 0,
            "steps": request.steps or self.default_steps,
            "fps": request.video_fps or self.default_fps,
            "duration": request.duration_seconds or 5.0,
            "denoise_strength": request.denoise_strength or 0.4,
            "cfg_scale": request.cfg_scale or 7.0,
            "quantize": self.local_cfg.get("quantize", ""),
        }

    def _run_command(self, cmd: List[str], output_path: Path) -> Tuple[bool, str]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"[Local] Running: {' '.join(shlex.quote(str(c)) for c in cmd)}")
        argv = [str(c) for c in cmd]
        if argv:
            resolved = _find_executable(argv[0])
            if resolved:
                argv[0] = resolved
            elif os.path.sep in argv[0] or argv[0].startswith(("./", "../")):
                pass  # keep explicit path; subprocess will report a clear error
        try:
            proc = subprocess.run(
                argv,
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
            tail = (proc.stderr or proc.stdout or "").strip()[-900:]
            return False, f"Exit code {proc.returncode}: {tail}"
        # mflux may append `.png` if the path has no extension; accept both.
        candidates = [output_path]
        if output_path.suffix != ".png":
            candidates.append(output_path.with_suffix(".png"))
        for cand in candidates:
            if cand.exists() and cand.stat().st_size > 0:
                return True, str(cand)
        return False, f"Command succeeded but no output file found at {output_path}"

    def _resolve_output(self, output_path: Path, preferred: Path) -> Path:
        # If the runner wrote a `.png`-suffixed file (mflux does when the
        # requested path had no extension), prefer that on disk.
        if not preferred.exists() and output_path.with_suffix(".png").exists():
            return output_path.with_suffix(".png")
        return preferred

    # ── GeneratorBackend interface ─────────────────────────────────────────
    def generate_image(self, request: GenerationRequest) -> GenerationResult:
        start = time.time()
        model_id = self._model_for_request(request, "image")
        model_spec = self._model_spec(model_id)
        if not model_spec:
            return GenerationResult(
                success=False,
                error=f"Local model spec not set for image model '{model_id}'. "
                      "Set local.model_paths.{model_id} in config.",
            )
        runner = self._runner_for_model(model_id)

        if model_id == MODEL_FLUX_1_KONTEXT and request.init_image_path:
            operation = "edit"
        elif request.init_image_path:
            operation = "img2img"
        else:
            operation = "image"

        output_path = request.output_dir / f"{request.atom_id}_image.png"
        values = self._base_values(request, model_id)
        values.update({
            "output": str(output_path),
            "input_image": str(request.init_image_path) if request.init_image_path else "",
            "start_image": str(request.init_image_path) if request.init_image_path else "",
        })

        try:
            cmd = runner.build_image_command(model_id, model_spec, request, output_path, operation, values)
            ok, actual = self._run_command(cmd, output_path)
        except Exception as exc:
            return GenerationResult(success=False, error=f"Local image command build failed: {exc}")

        if not ok:
            return GenerationResult(success=False, error=f"Local image generation failed: {actual}")
        final_path = self._resolve_output(output_path, Path(actual))
        return GenerationResult(
            success=True,
            image_path=final_path,
            generation_time=time.time() - start,
            metadata={"backend": "local", "runner": runner.name, "model": model_id},
        )

    def generate_video(self, request: GenerationRequest, source_image: Path) -> GenerationResult:
        start = time.time()
        model_id = self._model_for_request(request, "video")
        model_spec = self._model_spec(model_id)
        if not model_spec:
            return GenerationResult(
                success=False,
                error=f"Local model spec not set for video model '{model_id}'. "
                      "Set local.model_paths.{model_id} in config.",
            )
        runner = self._runner_for_model(model_id)

        duration = request.duration_seconds or (
            request.video_frames / request.video_fps if request.video_frames and request.video_fps else 5.0
        )
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
            cmd = runner.build_video_command(model_id, model_spec, request, output_path, source_image, values)
            ok, actual = self._run_command(cmd, output_path)
        except Exception as exc:
            return GenerationResult(success=False, error=f"Local video command build failed: {exc}")

        if not ok:
            return GenerationResult(success=False, error=f"Local video generation failed: {actual}")
        return GenerationResult(
            success=True,
            video_path=Path(actual) if actual != str(output_path) else output_path,
            generation_time=time.time() - start,
            metadata={"backend": "local", "runner": runner.name, "model": model_id},
        )

    def generate_morph_video(self, request: GenerationRequest, start_image_path: Path, end_image_path: Path) -> GenerationResult:
        start = time.time()
        model_id = self.models.get("video", MODEL_WAN_2_2)
        edit_model = self.models.get("edit")
        if edit_model and self._model_spec(edit_model):
            model_id = edit_model
        model_spec = self._model_spec(model_id)
        if not model_spec:
            return GenerationResult(success=False, error=f"Local model spec not set for morph model '{model_id}'.")

        runner = self._runner_for_model(model_id)
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
            cmd = runner.build_morph_command(model_id, model_spec, request, output_path, start_image_path, end_image_path, values)
            ok, actual = self._run_command(cmd, output_path)
        except Exception as exc:
            # A runner without native morph support gets a deterministic
            # ffmpeg crossfade fallback so loop closure still works.
            logger.info(f"[Local] Morph not supported by runner '{runner.name}' ({exc}); using ffmpeg crossfade fallback.")
            return self._ffmpeg_crossfade_morph(request, start_image_path, end_image_path, output_path, start)

        if not ok:
            return GenerationResult(success=False, error=f"Local morph generation failed: {actual}")
        return GenerationResult(
            success=True,
            video_path=Path(actual) if actual != str(output_path) else output_path,
            generation_time=time.time() - start,
            metadata={"backend": "local", "runner": runner.name, "model": model_id},
        )

    def _ffmpeg_crossfade_morph(
        self,
        request: GenerationRequest,
        start_image_path: Path,
        end_image_path: Path,
        output_path: Path,
        start_time: float,
    ) -> GenerationResult:
        """Crossfade two stills into a short morph-like video with ffmpeg."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        duration = float(request.duration_seconds or 2.0)
        fps = float(request.video_fps or 8.0)
        # Build a 2-input xfade filter. Use a solid dark background just in case
        # the input sizes differ, then scale/crop to the target size.
        width = int(request.width or self.default_width)
        height = int(request.height or self.default_height)
        filters = (
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1[v0];"
            f"[1:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1[v1];"
            f"[v0][v1]xfade=transition=fade:duration={duration}:offset=0,format=yuv420p[v]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-t", str(duration + 0.05), "-i", str(start_image_path),
            "-loop", "1", "-t", str(duration + 0.05), "-i", str(end_image_path),
            "-filter_complex", filters,
            "-map", "[v]",
            "-r", str(fps),
            "-t", str(duration),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        logger.info(f"[Local] Running ffmpeg morph: {' '.join(shlex.quote(str(c)) for c in cmd)}")
        try:
            proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=self.timeout)
        except Exception as exc:
            return GenerationResult(success=False, error=f"ffmpeg morph failed: {exc}")
        if proc.returncode != 0:
            return GenerationResult(success=False, error=f"ffmpeg morph exit {proc.returncode}: {(proc.stderr or '')[-500:]}")
        if not output_path.exists() or output_path.stat().st_size == 0:
            return GenerationResult(success=False, error="ffmpeg morph produced no output")
        return GenerationResult(
            success=True,
            video_path=output_path,
            generation_time=time.time() - start_time,
            metadata={"backend": "local", "runner": "ffmpeg-morph-fallback", "model": self.models.get("edit")},
        )


__all__ = [
    "LocalBackend",
    "MfluxRunner",
    "FluxSwiftRunner",
    "WanRunner",
    "GenericCommandRunner",
    "normalize_model_name",
    "KNOWN_MODELS",
    "MODEL_DISPLAY_NAMES",
    "MODEL_FLUX_1_DEV",
    "MODEL_FLUX_1_KONTEXT",
    "MODEL_WAN_2_2",
    "MODEL_Z_IMAGE_TURBO",
]
