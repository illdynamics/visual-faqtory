#!/usr/bin/env python3
"""
Visual FaQtory v0.9.0-beta
═══════════════════════════════════════════════════════════════════════════════

An automated, long-form AI visual generation pipeline for music, DJ sets,
and experimental audiovisual projects.

Pipeline: paragraph_story (sliding window) + Hybrid-capable backends (ComfyUI, Venice, Veo, Mock) + Finalizer

v0.9.0-beta — Native Python Qwen image backend
  - KEPT: split-capability backend routing
  - KEPT: Qwen image stage via ComfyUI workflows
  - NEW: image-only qwen_image_python / qwen_python local inference backend
  - KEPT: AnimateDiff video backend and Venice native backend

License: AGPL-3.0
"""

from .version import __version__
__author__ = "Ill Dynamics / WoNQ"
__license__ = "AGPL-3.0"

from .visual_briq import (
    VisualBriq, GenerationSpec, InputMode, BriqStatus,
    CycleState, generate_briq_id
)
from .prompt_bundle import PromptBundle, load_prompt_bundle
from .prompt_synth import (
    synthesize_prompt, synthesize_video_prompt,
    load_evolution_lines, select_evolution_mutations, map_motion_to_bucket_id,
)
from .base_folders import select_base_files
from .instruqtor import InstruQtor
from .construqtor import ConstruQtor
from .inspeqtor import InspeQtor
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

__all__ = [
    "__version__", "__author__", "__license__",
    "VisualBriq", "GenerationSpec", "InputMode", "BriqStatus",
    "CycleState", "generate_briq_id",
    "PromptBundle", "load_prompt_bundle",
    "synthesize_prompt", "synthesize_video_prompt",
    "load_evolution_lines", "select_evolution_mutations", "map_motion_to_bucket_id",
    "select_base_files",
    "InstruQtor", "ConstruQtor", "InspeQtor", "Finalizer",
    "BackendType", "GenerationRequest", "GenerationResult",
    "GeneratorBackend", "MockBackend", "ComfyUIBackend", "DelegatingBackend", "VeniceBackend",
    "extract_backend_config", "has_split_backend_config", "resolve_capability_backend_configs",
    "get_backend_type_for_capability", "describe_backend_config",
    "create_backend", "list_available_backends",
    "SlidingStoryConfig", "run_sliding_story",
]
