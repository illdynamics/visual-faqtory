#!/usr/bin/env python3
"""
QonQrete Visual FaQtory v0.5.6-beta
═══════════════════════════════════════════════════════════════════════════════

An automated, long-form AI visual generation pipeline for music, DJ sets,
and experimental audiovisual projects.

Pipeline: paragraph_story (sliding window) + ComfyUI backend + Finalizer

v0.5.6-beta — Clean Base + Reinject Default + Run/Saved-Runs Refactor
  - Reinject mode ON by default (img2img keyframe restoration every cycle)
  - ComfyUI-only backend
  - Output dir: ./run (current run), worqspace/saved-runs/<name> (archives)
  - Working input modes: text / image / video
  - Base audio muxing + optional auto-cycle count from audio duration
  - Finalizer: stitch → interpolate 60fps → upscale 1080p → optional audio mux
  - Deterministic prompt synthesis (no LLM dependency)

License: AGPL-3.0 (same as QonQrete)
"""

__version__ = "0.5.6-beta"
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
    GeneratorBackend, MockBackend, ComfyUIBackend,
    create_backend, list_available_backends
)
from .sliding_story_engine import SlidingStoryConfig, run_sliding_story

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
    "GeneratorBackend", "MockBackend", "ComfyUIBackend",
    "create_backend", "list_available_backends",
    "SlidingStoryConfig", "run_sliding_story",
]
