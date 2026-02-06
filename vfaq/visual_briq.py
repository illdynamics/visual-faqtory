#!/usr/bin/env python3
"""
visual_briq.py - Visual Instruction Unit for Agent Pipeline
═══════════════════════════════════════════════════════════════════════════════

A VisualBriq is the standardized instruction packet passed between agents:
  InstruQtor → ConstruQtor → InspeQtor

It contains everything needed to generate one visual atom in the cycle.

Part of QonQrete Visual FaQtory v0.0.5-alpha
"""
import json
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from enum import Enum


class InputMode(Enum):
    """Visual generation input modes."""
    TEXT = "text"       # Prompt only (text → image → video)
    IMAGE = "image"     # Prompt + base image (image → video)
    VIDEO = "video"     # Prompt + base video (video → video evolution)


class BriqStatus(Enum):
    """Processing status of a VisualBriq."""
    PENDING = "pending"           # Created, awaiting ConstruQtor
    CONSTRUCTING = "constructing" # ConstruQtor working
    CONSTRUCTED = "constructed"   # Raw video generated
    INSPECTING = "inspecting"     # InspeQtor working
    COMPLETE = "complete"         # Looped video ready
    FAILED = "failed"             # Error occurred


@dataclass
class GenerationSpec:
    """
    Technical specifications for video generation.
    Filled by InstruQtor based on config.yaml ONLY.
    tasq.md MUST NOT override these mechanical parameters.
    """
    # Image generation
    width: int = 1024
    height: int = 576
    cfg_scale: float = 7.0
    steps: int = 30
    sampler: str = "euler_ancestral"

    # Video generation
    video_frames: int = 25
    video_fps: int = 8
    clip_seconds: float = 8.0
    motion_bucket_id: int = 127
    noise_aug_strength: float = 0.02

    # Chaining (for video2video / img2img)
    denoise_strength: float = 0.4


@dataclass
class VisualBriq:
    """
    The fundamental instruction unit for the Visual FaQtory pipeline.

    Created by InstruQtor, processed by ConstruQtor, finalized by InspeQtor.
    """
    # Identity
    briq_id: str
    cycle_index: int
    created_at: datetime = field(default_factory=datetime.now)

    # Input mode
    mode: InputMode = InputMode.TEXT

    # Prompts (refined by InstruQtor from raw tasq)
    prompt: str = ""
    negative_prompt: str = ""
    style_tags: List[str] = field(default_factory=list)
    quality_tags: List[str] = field(default_factory=list)

    # Seeds for reproducibility
    seed: int = 42

    # Base input (for IMAGE/VIDEO modes or chaining)
    base_image_path: Optional[Path] = None
    base_video_path: Optional[Path] = None

    # Technical specs
    spec: GenerationSpec = field(default_factory=GenerationSpec)

    # Processing status
    status: BriqStatus = BriqStatus.PENDING

    # Outputs (filled during processing)
    raw_video_path: Optional[Path] = None      # From ConstruQtor
    looped_video_path: Optional[Path] = None   # From InspeQtor
    source_image_path: Optional[Path] = None   # Intermediate image (for img2vid)

    # Metadata
    generation_time: float = 0.0
    backend_used: str = ""
    error_message: Optional[str] = None

    # Evolution suggestion (from InspeQtor for next cycle)
    evolution_suggestion: Optional[str] = None
    suggested_prompt_delta: Optional[str] = None

    def __post_init__(self):
        # Convert string paths to Path objects
        if isinstance(self.base_image_path, str):
            self.base_image_path = Path(self.base_image_path)
        if isinstance(self.base_video_path, str):
            self.base_video_path = Path(self.base_video_path)
        if isinstance(self.mode, str):
            self.mode = InputMode(self.mode)
        if isinstance(self.status, str):
            self.status = BriqStatus(self.status)

    def get_full_prompt(self) -> str:
        """Combine prompt with quality/style tags."""
        parts = []
        if self.quality_tags:
            parts.append(", ".join(self.quality_tags))
        parts.append(self.prompt)
        if self.style_tags:
            parts.append(", ".join(self.style_tags))
        return ", ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        d = asdict(self)
        d['mode'] = self.mode.value
        d['status'] = self.status.value
        d['created_at'] = self.created_at.isoformat()
        d['base_image_path'] = str(self.base_image_path) if self.base_image_path else None
        d['base_video_path'] = str(self.base_video_path) if self.base_video_path else None
        d['raw_video_path'] = str(self.raw_video_path) if self.raw_video_path else None
        d['looped_video_path'] = str(self.looped_video_path) if self.looped_video_path else None
        d['source_image_path'] = str(self.source_image_path) if self.source_image_path else None
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'VisualBriq':
        """Deserialize from dictionary."""
        d = d.copy()
        d['mode'] = InputMode(d['mode'])
        d['status'] = BriqStatus(d['status'])
        d['created_at'] = datetime.fromisoformat(d['created_at'])
        d['spec'] = GenerationSpec(**d['spec'])
        return cls(**d)

    def save(self, path: Path) -> None:
        """Save briq to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> 'VisualBriq':
        """Load briq from JSON file."""
        return cls.from_dict(json.loads(path.read_text()))


def generate_briq_id(cycle_index: int, seed: int) -> str:
    """Generate unique briq ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hash_input = f"{cycle_index}_{seed}_{timestamp}"
    short_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
    return f"briq_{cycle_index:04d}_{short_hash}"


@dataclass
class CycleState:
    """
    Persistent state for the visual generation cycle.
    Tracks progress across multiple cycles.
    """
    session_id: str
    started_at: datetime

    # Cycle tracking
    current_cycle: int = 0
    total_cycles_requested: int = 0  # 0 = unlimited

    # Completed briqs
    completed_briqs: List[str] = field(default_factory=list)  # briq_ids

    # Current working briq
    active_briq_id: Optional[str] = None

    # Evolution chain (prompt evolution history)
    prompt_history: List[str] = field(default_factory=list)

    # Paths
    qodeyard_path: Path = Path("./qodeyard")

    # Stats
    total_generation_time: float = 0.0
    total_video_duration: float = 0.0

    # Track per-cycle video paths for finalizer
    cycle_video_paths: List[str] = field(default_factory=list)

    # Track any failed cycles
    failed_cycles: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['started_at'] = self.started_at.isoformat()
        d['qodeyard_path'] = str(self.qodeyard_path)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CycleState':
        d = d.copy()
        d['started_at'] = datetime.fromisoformat(d['started_at'])
        d['qodeyard_path'] = Path(d['qodeyard_path'])
        # Handle legacy state files missing new fields
        if 'cycle_video_paths' not in d:
            d['cycle_video_paths'] = []
        if 'failed_cycles' not in d:
            d['failed_cycles'] = []
        return cls(**d)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> 'CycleState':
        return cls.from_dict(json.loads(path.read_text()))


__all__ = [
    'InputMode',
    'BriqStatus',
    'GenerationSpec',
    'VisualBriq',
    'generate_briq_id',
    'CycleState'
]
