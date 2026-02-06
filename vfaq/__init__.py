#!/usr/bin/env python3
"""
QonQrete Visual FaQtory v0.0.7-alpha
═══════════════════════════════════════════════════════════════════════════════

An automated, long-form AI visual generation pipeline for music, DJ sets,
and experimental audiovisual projects.

3-Agent Pipeline:
  - InstruQtor: Creates VisualBriq from Prompt Bundle (LLM-powered)
  - ConstruQtor: Calls backend to generate video
  - InspeQtor: Loops video, suggests evolution (LLM-powered, innovative mode)

Prompt Bundle (v0.0.7):
  - tasq.md, negative_prompt.md, style_hints.md, motion_prompt.md

Finalizer:
  - Stitches all per-cycle MP4s into a single final_output.mp4
  - Post-stitch: interpolation to 60fps + upscale to 1080p

Based on QonQrete's deterministic, state-driven architecture.

Usage:
    from vfaq import VisualFaQtory

    faqtory = VisualFaQtory(worqspace_dir="./worqspace", output_dir="./qodeyard")
    faqtory.run(cycles=100)

License: AGPL-3.0 (same as QonQrete)
"""

__version__ = "0.0.7-alpha"
__author__ = "Ill Dynamics / WoNQ"
__license__ = "AGPL-3.0"

# Core models
from .visual_briq import (
    VisualBriq,
    GenerationSpec,
    InputMode,
    BriqStatus,
    CycleState,
    generate_briq_id
)

# Prompt Bundle (v0.0.7)
from .prompt_bundle import PromptBundle, load_prompt_bundle

# Three agents
from .instruqtor import InstruQtor
from .construqtor import ConstruQtor
from .inspeqtor import InspeQtor

# Finalizer
from .finalizer import Finalizer

# Main pipeline
from .visual_faqtory import VisualFaQtory, quick_run

# Backends
from .backends import (
    BackendType,
    GenerationRequest,
    GenerationResult,
    GeneratorBackend,
    MockBackend,
    ComfyUIBackend,
    DiffusersBackend,
    ReplicateBackend,
    SplitBackend,
    create_backend,
    create_split_backend,
    list_available_backends
)

__all__ = [
    # Version
    "__version__",
    "__author__",
    "__license__",

    # Core models
    "VisualBriq",
    "GenerationSpec",
    "InputMode",
    "BriqStatus",
    "CycleState",
    "generate_briq_id",

    # Prompt Bundle
    "PromptBundle",
    "load_prompt_bundle",

    # Agents
    "InstruQtor",
    "ConstruQtor",
    "InspeQtor",

    # Finalizer
    "Finalizer",

    # Pipeline
    "VisualFaQtory",
    "quick_run",

    # Backends
    "BackendType",
    "GenerationRequest",
    "GenerationResult",
    "GeneratorBackend",
    "MockBackend",
    "ComfyUIBackend",
    "DiffusersBackend",
    "ReplicateBackend",
    "SplitBackend",
    "create_backend",
    "create_split_backend",
    "list_available_backends"
]
