#!/usr/bin/env python3
"""
Visual FaQtory v0.9.8-beta
═══════════════════════════════════════════════════════════════════════════════

An automated, long-form AI visual generation pipeline for music, DJ sets,
and experimental audiovisual projects.

Pipeline: paragraph_story (sliding window) + Hybrid-capable backends (ComfyUI, Venice, Veo, Local MLX) + Finalizer

v0.9.8-beta — Wan keep-text-encoder toggle
  - NEW: local.wan.keep-text-encoder (true | false) maps to --keep-text-encoder
  - KEPT: local MLX backend for FLUX.1-dev, FLUX.1 Kontext, Wan 2.2, Z-Image-Turbo
  - KEPT: pluggable runners (mflux / flux-swift / wan / command)
  - KEPT: split-capability backend routing + ComfyUI/AnimateDiff/Venice/Veo backends

License: AGPL-3.0
"""

from .version import __version__
__author__ = "Ill Dynamics / WoNQ"
__license__ = "AGPL-3.0"

from .finalizer import Finalizer
from .backends import (
    BackendType, GenerationRequest, GenerationResult,
    GeneratorBackend, MockBackend, ComfyUIBackend, DelegatingBackend,
    extract_backend_config, has_split_backend_config, resolve_capability_backend_configs,
    get_backend_type_for_capability, describe_backend_config,
    create_backend, list_available_backends
)
from .sliding_story_engine import SlidingStoryConfig, run_sliding_story
from .venice_backend import VeniceBackend
from .local_backend import LocalBackend

__all__ = [
    "__version__", "__author__", "__license__",
    "Finalizer",
    "BackendType", "GenerationRequest", "GenerationResult",
    "GeneratorBackend", "MockBackend", "ComfyUIBackend", "DelegatingBackend", "VeniceBackend", "LocalBackend",
    "extract_backend_config", "has_split_backend_config", "resolve_capability_backend_configs",
    "get_backend_type_for_capability", "describe_backend_config",
    "create_backend", "list_available_backends",
    "SlidingStoryConfig", "run_sliding_story",
]
