#!/usr/bin/env python3
"""
visual_faqtory.py — Main Pipeline Orchestrator
═══════════════════════════════════════════════════════════════════════════════

Thin orchestrator that wires config → sliding_story_engine → finalizer → save.

Pipeline flow:
  1. Load config, detect inputs (base image/video/audio)
  2. Run sliding_story_engine (paragraph_story with reinject default ON)
  3. Finalizer: stitch → interpolate 60fps → upscale 1080p
  4. Audio mux (if base audio present)
  5. Save run to worqspace/saved-runs/<project-name>

Part of QonQrete Visual FaQtory v0.5.6-beta
"""
import json
import logging
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import yaml

from .backends import create_backend, list_available_backends
from .finalizer import Finalizer, FinalizerError
from .sliding_story_engine import SlidingStoryConfig, run_sliding_story

logger = logging.getLogger(__name__)

# ── File detection extensions ─────────────────────────────────────────────────
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm"}
_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".aac", ".m4a", ".ogg"}

# ── Prompt files to copy into run/meta ────────────────────────────────────────
_PROMPT_FILES = [
    "config.yaml",
    "story.txt",
    "motion_prompt.md",
    "evolution_lines.md",
    "style_hints.md",
    "negative_prompt.md",
    "transient_tasq.md",
]


def _detect_newest_file(directory: Path, extensions: set) -> Optional[Path]:
    """Find the newest file matching given extensions in a directory."""
    if not directory.exists():
        return None
    candidates = [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in extensions and f.name != ".gitkeep"
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return candidates[0]


def _get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.error(f"Failed to get audio duration: {e}")
        return 0.0


def _extract_video_frame(video_path: Path, output_path: Path, width: int = 1024, height: int = 576) -> bool:
    """Extract first frame from video and resize/pad to target dimensions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vframes", "1",
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        "-q:v", "2",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return output_path.exists() and output_path.stat().st_size > 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Frame extraction failed: {e.stderr[:300]}")
        return False


def _sanitize_project_name(name: str) -> str:
    """Sanitize project name for safe filesystem use."""
    import re
    name = name.strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    name = re.sub(r'_+', '_', name).strip('_. ')
    return name or "unnamed"


def _mux_audio(video_path: Path, audio_path: Path, output_path: Path) -> bool:
    """Mux audio into video, trimming video to audio duration."""
    # Get audio duration
    dur = _get_audio_duration(audio_path)
    if dur <= 0:
        logger.error("[AudioMux] Could not determine audio duration; skipping mux")
        return False

    cmd = [
        "ffmpeg", "-y",
        "-i", str(audio_path),
        "-i", str(video_path),
        "-map", "1:v:0",
        "-map", "0:a:0",
        "-t", str(dur),
        "-c:v", "copy",
        "-c:a", "aac",
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0 and output_path.exists():
            logger.info(f"[AudioMux] Muxed audio successfully → {output_path}")
            return True
        logger.error(f"[AudioMux] ffmpeg failed: {result.stderr[:300]}")
        return False
    except Exception as e:
        logger.error(f"[AudioMux] Error: {e}")
        return False


class VisualFaQtory:
    """
    Main orchestrator for the Visual FaQtory v0.5.6-beta pipeline.

    Wires config loading, input detection, sliding story engine,
    finalizer, audio mux, and project saving.
    """

    def __init__(
        self,
        worqspace_dir: str | Path = "./worqspace",
        run_dir: str | Path = "./run",
        config_override: Optional[Dict[str, Any]] = None,
        project_name: Optional[str] = None,
        reinject: bool = True,
        mode_override: Optional[str] = None,
        dry_run: bool = False,
    ):
        self.worqspace_dir = Path(worqspace_dir).resolve()
        self.run_dir = Path(run_dir).resolve()
        self.project_name = project_name
        self.reinject = reinject
        self.mode_override = mode_override
        self.dry_run = dry_run
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"

        # Load config
        self.config = self._load_config(config_override or {})

        # Detect inputs
        self.base_image = None
        self.base_video = None
        self.base_audio = None
        self._detect_inputs()

        # Determine effective mode
        self.mode = self._determine_mode()

    def _load_config(self, override: Dict[str, Any]) -> Dict[str, Any]:
        """Load config.yaml and apply overrides."""
        config_path = self.worqspace_dir / "config.yaml"
        config = {}
        if config_path.exists():
            try:
                config = yaml.safe_load(config_path.read_text()) or {}
            except Exception as e:
                logger.warning(f"Failed to parse config.yaml: {e}")
        # Deep merge override
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key].update(value)
            else:
                config[key] = value
        return config

    def _detect_inputs(self):
        """Auto-detect base image, video, and audio files."""
        self.base_image = _detect_newest_file(
            self.worqspace_dir / "base_images", _IMAGE_EXTS
        )
        self.base_video = _detect_newest_file(
            self.worqspace_dir / "base_video", _VIDEO_EXTS
        )
        self.base_audio = _detect_newest_file(
            self.worqspace_dir / "base_audio", _AUDIO_EXTS
        )
        if self.base_image:
            logger.info(f"[Detect] Base image: {self.base_image.name}")
        if self.base_video:
            logger.info(f"[Detect] Base video: {self.base_video.name}")
        if self.base_audio:
            logger.info(f"[Detect] Base audio: {self.base_audio.name}")

    def _determine_mode(self) -> str:
        """Determine input mode: text, image, or video."""
        if self.mode_override:
            return self.mode_override
        cfg_mode = self.config.get("input", {}).get("mode", "text")
        if cfg_mode == "auto":
            if self.base_video:
                return "video"
            elif self.base_image:
                return "image"
            return "text"
        return cfg_mode

    def _setup_run_dirs(self):
        """Create run directory structure."""
        for subdir in ["videos", "frames", "briqs", "meta"]:
            (self.run_dir / subdir).mkdir(parents=True, exist_ok=True)

    def _copy_inputs_to_meta(self):
        """Copy prompt files and inputs into run/meta/ for reproducibility."""
        meta_dir = self.run_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        for fname in _PROMPT_FILES:
            src = self.worqspace_dir / fname
            if src.exists():
                shutil.copy2(src, meta_dir / fname)

        if self.base_image:
            shutil.copy2(self.base_image, meta_dir / self.base_image.name)
        if self.base_video:
            shutil.copy2(self.base_video, meta_dir / self.base_video.name)
        if self.base_audio:
            shutil.copy2(self.base_audio, meta_dir / self.base_audio.name)

    def _build_story_config(self) -> SlidingStoryConfig:
        """Build SlidingStoryConfig from config.yaml values."""
        ps = self.config.get("paragraph_story", {})
        bc = dict(self.config.get("backend", {}))

        # Inject LoRA config into backend config if present
        lora_cfg = self.config.get("lora", {})
        if lora_cfg:
            bc["lora"] = lora_cfg

        # Get comfyui section for checkpoint names
        comfyui_section = self.config.get("comfyui", {})
        if comfyui_section:
            bc["comfyui"] = comfyui_section

        return SlidingStoryConfig(
            max_paragraphs=ps.get("max_paragraphs", 4),
            img2vid_duration_sec=ps.get("img2vid_duration_sec", 3.0),
            img2img_denoise_min=ps.get("img2img_denoise_min", 0.25),
            img2img_denoise_max=ps.get("img2img_denoise_max", 0.45),
            rolling_window_mode=ps.get("rolling_window", True),
            require_morph=ps.get("require_morph", False),
            seed_base=ps.get("seed_base", 42),
            video_fps=ps.get("video_fps", 8),
            backend_config=bc,
            reinject=self.reinject,
        )

    def _compute_cycle_count(self, config: SlidingStoryConfig) -> Optional[int]:
        """If audio sync enabled and audio exists, compute cycles from duration."""
        audio_cfg = self.config.get("audio", {})
        if not audio_cfg.get("sync_video_audio", False):
            return None
        if not self.base_audio:
            return None

        duration = _get_audio_duration(self.base_audio)
        if duration <= 0:
            logger.warning("[AudioSync] Could not determine audio duration")
            return None

        cycle_sec = audio_cfg.get("cycle_seconds", config.img2vid_duration_sec)
        import math
        cycles = math.ceil(duration / cycle_sec)
        logger.info(
            f"[AudioSync] Audio: {duration:.1f}s / {cycle_sec}s per cycle = {cycles} cycles"
        )
        return cycles

    def _run_finalizer(self) -> Dict[str, Optional[str]]:
        """Run the full finalizer pipeline: stitch → interpolate → upscale → audio mux."""
        fc = self.config.get("finalizer", {})
        result = {
            "final_video": None,
            "final_video_60fps": None,
            "final_video_60fps_1080p": None,
            "final_video_60fps_1080p_audio": None,
        }

        # Discover cycle videos
        videos_dir = self.run_dir / "videos"
        videos = sorted(videos_dir.glob("video_*.mp4"))
        if not videos:
            logger.warning("[Finalizer] No cycle videos found to stitch")
            return result

        # Step 1: Stitch
        finalizer = Finalizer(
            project_dir=self.run_dir,
            preferred_codec=fc.get("encoder_preference", ["h264_nvenc", "libx264"])[0]
                if isinstance(fc.get("encoder_preference"), list) else "h264_nvenc",
            output_quality=fc.get("quality", {}).get("crf", 16)
                if isinstance(fc.get("quality"), dict) else 16,
            finalizer_config=fc,
        )

        try:
            stitch_path = finalizer.finalize(cycle_video_paths=videos)
            # Rename to spec naming
            final_video = self.run_dir / "final_video.mp4"
            if stitch_path != final_video:
                shutil.move(str(stitch_path), str(final_video))
            result["final_video"] = str(final_video)
            logger.info(f"[Finalizer] Stitched → {final_video}")
        except FinalizerError as e:
            logger.error(f"[Finalizer] Stitch failed: {e}")
            return result

        # Step 2+3: Interpolate + Upscale (via post-stitch finalizer)
        if fc.get("enabled", True):
            # Temporarily set correct paths for the finalizer
            finalizer.final_output_path = final_video
            finalizer.final_deliverable_path = self.run_dir / "final_video_60fps_1080p.mp4"
            finalizer._interpolated_temp_path = self.run_dir / "_temp_60fps.mp4"
            finalizer.finalizer_enabled = True

            deliverable = finalizer.run_post_stitch_finalizer()
            if deliverable:
                # Copy 60fps intermediate before it gets cleaned
                temp_60fps = self.run_dir / "_temp_60fps.mp4"
                final_60fps = self.run_dir / "final_video_60fps.mp4"
                if temp_60fps.exists():
                    shutil.copy2(str(temp_60fps), str(final_60fps))
                    result["final_video_60fps"] = str(final_60fps)
                result["final_video_60fps_1080p"] = str(deliverable)

                # Keep the 60fps intermediate (don't clean up)
                # Step 4: Audio mux
                if self.base_audio and self.config.get("audio", {}).get("enabled", True):
                    audio_output = self.run_dir / "final_video_60fps_1080p_audio.mp4"
                    if _mux_audio(deliverable, self.base_audio, audio_output):
                        result["final_video_60fps_1080p_audio"] = str(audio_output)

        return result

    def _save_run(self, finalizer_result: Dict, project_name: str):
        """Move run/ to worqspace/saved-runs/<project-name>."""
        saved_runs_dir = self.worqspace_dir / "saved-runs"
        saved_runs_dir.mkdir(parents=True, exist_ok=True)

        safe_name = _sanitize_project_name(project_name)
        target = saved_runs_dir / safe_name

        # Auto-suffix if exists
        if target.exists():
            counter = 1
            while (saved_runs_dir / f"{safe_name}-{counter:03d}").exists():
                counter += 1
            target = saved_runs_dir / f"{safe_name}-{counter:03d}"
            logger.info(f"[Save] Name exists, using: {target.name}")

        # Move run dir
        shutil.move(str(self.run_dir), str(target))
        logger.info(f"[Save] Run saved to: {target}")

        # Rename deliverable to <project-name>.mp4
        # Priority: audio mux > 1080p > 60fps > base stitch
        for key in ["final_video_60fps_1080p_audio", "final_video_60fps_1080p",
                     "final_video_60fps", "final_video"]:
            src_str = finalizer_result.get(key)
            if src_str:
                src = Path(src_str)
                # Remap path to new location
                relative = src.name
                moved = target / relative
                if moved.exists():
                    final_name = target / f"{safe_name}.mp4"
                    shutil.copy2(str(moved), str(final_name))
                    logger.info(f"[Save] Deliverable: {final_name}")
                    break

        return target

    def _write_state(self, start_time: datetime, end_time: datetime,
                     cycles_planned: int, cycles_completed: int,
                     finalizer_result: Dict, saved_to: Optional[Path] = None):
        """Write faqtory_state.json."""
        state = {
            "run_id": self.run_id,
            "version": "v0.5.6-beta",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "config_snapshot": "meta/config.yaml",
            "story_snapshot": "meta/story.txt",
            "mode": self.mode,
            "reinject": self.reinject,
            "base_image": self.base_image.name if self.base_image else None,
            "base_video": self.base_video.name if self.base_video else None,
            "base_audio": self.base_audio.name if self.base_audio else None,
            "cycles_planned": cycles_planned,
            "cycles_completed": cycles_completed,
            "finalizer_outputs": finalizer_result,
            "saved_to": str(saved_to) if saved_to else None,
        }
        state_path = self.run_dir / "faqtory_state.json"
        state_path.write_text(json.dumps(state, indent=2))

    def run(self) -> Path:
        """Execute the full pipeline."""
        start_time = datetime.now()
        logger.info(f"[FaQtory] Starting run: {self.run_id}")
        logger.info(f"[FaQtory] Mode: {self.mode} | Reinject: {self.reinject}")

        # Setup directories
        self._setup_run_dirs()
        self._copy_inputs_to_meta()

        # Build story config
        story_config = self._build_story_config()

        # Compute cycle count from audio if applicable
        audio_cycles = self._compute_cycle_count(story_config)

        # Resolve story file
        story_path = self.worqspace_dir / "story.txt"
        if not story_path.exists():
            raise FileNotFoundError(f"Story file not found: {story_path}")

        # Handle base image/video for image/video modes
        base_image_for_run = None
        if self.mode == "image" and self.base_image:
            base_image_for_run = self.base_image
        elif self.mode == "video" and self.base_video:
            # Extract frame from video
            extracted = self.run_dir / "meta" / "extracted_frame.png"
            width = self.config.get("backend", {}).get("width", 1024)
            height = self.config.get("backend", {}).get("height", 576)
            if _extract_video_frame(self.base_video, extracted, width, height):
                base_image_for_run = extracted
                logger.info(f"[FaQtory] Extracted frame from video → {extracted}")
            else:
                logger.warning("[FaQtory] Video frame extraction failed, falling back to text mode")

        if self.dry_run:
            logger.info("[FaQtory] DRY RUN — config loaded, inputs resolved, exiting before generation")
            self._write_state(start_time, datetime.now(), 0, 0, {})
            return self.run_dir

        # Run sliding story engine
        final_video = run_sliding_story(
            story_path=story_path,
            qodeyard_dir=self.run_dir,
            config=story_config,
            max_cycles=audio_cycles,
            base_image_path=base_image_for_run,
        )

        # Count completed cycles
        cycles_completed = len(list((self.run_dir / "videos").glob("video_*.mp4")))

        # Run finalizer
        finalizer_result = self._run_finalizer()

        # Write state
        end_time = datetime.now()
        self._write_state(
            start_time, end_time,
            audio_cycles or cycles_completed,
            cycles_completed,
            finalizer_result,
        )

        # Save run
        if self.project_name:
            name = self.project_name
        else:
            try:
                name = input("\n📁 Project name to save as: ").strip()
            except (EOFError, KeyboardInterrupt):
                name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if name:
            saved_to = self._save_run(finalizer_result, name)
            # Update state in saved location
            state_path = saved_to / "faqtory_state.json"
            if state_path.exists():
                state = json.loads(state_path.read_text())
                state["saved_to"] = str(saved_to)
                state_path.write_text(json.dumps(state, indent=2))

        elapsed = (end_time - start_time).total_seconds()
        logger.info(f"[FaQtory] Run complete in {elapsed:.1f}s — {cycles_completed} cycles")
        return self.run_dir


def quick_run(**kwargs):
    """Convenience function for quick pipeline runs."""
    faqtory = VisualFaQtory(**kwargs)
    return faqtory.run()
