#!/usr/bin/env python3
"""
QonQrete Visual FaQtory v0.3.5-beta
═══════════════════════════════════════════════════════════════════════════════

An automated, long-form AI visual generation pipeline for music, DJ sets,
and experimental audiovisual projects.

3-Agent Pipeline:
  - InstruQtor: Creates VisualBriq from Prompt Bundle (LLM-powered or deterministic)
  - ConstruQtor: Calls backend to generate video (supports V2V)
  - InspeQtor: Loops video, suggests evolution (LLM-powered or deterministic)

v0.1.0-alpha features:
  - Deterministic Prompt Synth (NO LLM required)
  - Audio Reactivity (BPM, beat grid, spectral features)
  - Base Folder Ingestion (base_image, base_audio, base_video)
  - Video2Video (safe ComfyUI workflow, video preprocessing)
  - Evolution Lines (deterministic visual mutations)

v0.3.5-beta features:
  - Fixed Stream/Longcat: true autoregressive continuation via SVD temporal diffusion
  - Unified macro control semantics (file presence = state, no auto-delete)
  - Finalised stage-safe audio reactive Turbo with explicit audio-paused state
  - Added long-run stability controller (prevents color collapse / green blob)
  - TouchDesigner integration contract (no .toe shipped, td_setup.py + blueprint provided)
  - Removed fake / misleading behavior from stream mode
  - All previous features: MIDI sidecar, crowd queue, OSC output, etc.

License: AGPL-3.0 (same as QonQrete)
"""

__version__ = "0.3.5-beta"
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
from .visual_faqtory import VisualFaQtory, quick_run
from .backends import (
    BackendType, GenerationRequest, GenerationResult,
    GeneratorBackend, MockBackend, ComfyUIBackend,
    DiffusersBackend, ReplicateBackend, SplitBackend,
    create_backend, create_split_backend, list_available_backends
)
from .color_stability import StabilityController, create_stability_controller

__all__ = [
    "__version__", "__author__", "__license__",
    "VisualBriq", "GenerationSpec", "InputMode", "BriqStatus",
    "CycleState", "generate_briq_id",
    "PromptBundle", "load_prompt_bundle",
    "synthesize_prompt", "synthesize_video_prompt",
    "load_evolution_lines", "select_evolution_mutations", "map_motion_to_bucket_id",
    "select_base_files",
    "InstruQtor", "ConstruQtor", "InspeQtor", "Finalizer",
    "VisualFaQtory", "quick_run",
    "BackendType", "GenerationRequest", "GenerationResult",
    "GeneratorBackend", "MockBackend", "ComfyUIBackend",
    "DiffusersBackend", "ReplicateBackend", "SplitBackend",
    "create_backend", "create_split_backend", "list_available_backends",
    "StabilityController", "create_stability_controller",
]
