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

Part of Visual FaQtory v0.9.0-beta
"""
import json
import logging
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import yaml

from .backends import (
    create_backend,
    list_available_backends,
    describe_backend_config,
    extract_backend_config,
    get_backend_type_for_capability,
)
from .finalizer import Finalizer, FinalizerError
from .sliding_story_engine import SlidingStoryConfig, run_sliding_story
from .run_state import RunState, write_state_atomic, discover_progress, check_needs_finalization
from .version import __version__

logger = logging.getLogger(__name__)

# ── File detection extensions ─────────────────────────────────────────────────
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm"}
_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".aac", ".m4a", ".ogg"}

# Matches video_001.mp4 but NOT video_loop_001.mp4
_CYCLE_VIDEO_RE = re.compile(r"^video_\d{3}\.mp4$")


def _list_cycle_videos(videos_dir: Path) -> List[Path]:
    """List cycle videos (video_NNN.mp4) excluding loop clips (video_loop_NNN.mp4)."""
    if not videos_dir.exists():
        return []
    return sorted(f for f in videos_dir.iterdir() if _CYCLE_VIDEO_RE.match(f.name))

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
    Main orchestrator for the Visual FaQtory v0.9.0-beta pipeline.

    Wires config loading, input detection, sliding story engine,
    finalizer, audio mux, and project saving.
    """

    def __init__(
        self,
        worqspace_dir: str | Path = "./worqspace",
        run_dir: str | Path = "./run",
        config_override: Optional[Dict[str, Any]] = None,
        project_name: Optional[str] = None,
        reinject: Optional[bool] = None,
        mode_override: Optional[str] = None,
        dry_run: bool = False,
        resume: bool = False,
    ):
        self.worqspace_dir = Path(worqspace_dir).resolve()
        self.run_dir = Path(run_dir).resolve()
        self.project_name = project_name
        self._reinject_override = reinject
        self.reinject = True
        self.mode_override = mode_override
        self.dry_run = dry_run
        self.resume = resume
        self._state_path = self.run_dir / "faqtory_state.json"

        # On resume, load existing run_id from state; otherwise generate new
        if resume:
            existing_state = RunState.load(self._state_path)
            if existing_state and existing_state.run_id:
                self.run_id = existing_state.run_id
            else:
                self.run_id = f"run_resumed_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
        else:
            self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"

        # Load config
        self.config = self._load_config(config_override or {})
        self._resolve_reinject_setting()

        # Detect inputs
        self.base_image = None
        self.base_video = None
        self.base_audio = None
        self._detect_inputs()

        # Determine effective mode
        self.mode = self._determine_mode()

    def _resolve_reinject_setting(self) -> bool:
        """Resolve reinject from config, with optional CLI hard override."""
        ps = self.config.get("paragraph_story", {})
        cfg_reinject = bool(ps.get("reinject", True)) if isinstance(ps, dict) else True
        if self._reinject_override is None:
            self.reinject = cfg_reinject
        else:
            self.reinject = bool(self._reinject_override)
        return self.reinject

    def _load_config(self, override: Dict[str, Any]) -> Dict[str, Any]:
        """Load config.yaml and apply overrides."""
        config_path = self.worqspace_dir / "config.yaml"
        config = {}
        if config_path.exists():
            try:
                config = yaml.safe_load(config_path.read_text()) or {}
            except Exception as e:
                raise RuntimeError(
                    f"Failed to parse {config_path}: {e}. "
                    "Refusing to continue with an empty fallback config because that can silently route to mock backends."
                ) from e
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
        bc = extract_backend_config(self.config)

        lora_cfg = bc.get("lora", {})
        comfyui_section = bc.get("comfyui", {})
        veo_section = bc.get("veo", {})
        venice_section = bc.get("venice", {})

        def _parse_seconds(raw_value: Any) -> Optional[float]:
            if raw_value is None:
                return None
            try:
                token = str(raw_value).strip()
                if token.lower().endswith("s"):
                    token = token[:-1]
                parsed = float(token)
                return parsed if parsed > 0 else None
            except (TypeError, ValueError):
                return None

        # Determine backend-specific overrides for timing
        video_backend_type = get_backend_type_for_capability(bc, "video")
        is_veo = video_backend_type == "veo"
        is_venice = video_backend_type == "venice"
        img2vid_duration = _parse_seconds(ps.get("img2vid_duration_sec"))
        if is_veo and img2vid_duration is None:
            # Default Veo duration from veo section
            img2vid_duration = _parse_seconds(veo_section.get("duration_seconds")) or 8.0
        if is_venice and img2vid_duration is None:
            venice_video_cfg = venice_section.get("video", {})
            if not isinstance(venice_video_cfg, dict):
                venice_video_cfg = {}
            img2vid_duration = (
                _parse_seconds(venice_video_cfg.get("duration_seconds"))
                or _parse_seconds(venice_video_cfg.get("duration"))
                or _parse_seconds(venice_section.get("video_duration"))
                or _parse_seconds(venice_section.get("duration_seconds"))
                or _parse_seconds(venice_section.get("duration"))
            )
        effective_video_fps = ps.get("video_fps")
        effective_video_frames = ps.get("video_frames")
        effective_timing_authority = ps.get("timing_authority")
        if effective_video_frames is not None:
            effective_timing_authority = "frames"
        elif img2vid_duration is not None:
            effective_timing_authority = "duration"
        if effective_timing_authority is None and is_venice and img2vid_duration is not None:
            effective_timing_authority = "duration"

        logger.debug(
            "[FaQtory] Story timing inputs: fps=%s, frames=%s, duration=%s, authority=%s",
            effective_video_fps,
            effective_video_frames,
            img2vid_duration,
            effective_timing_authority,
        )

        return SlidingStoryConfig(
            max_paragraphs=ps.get("max_paragraphs", 4),
            img2vid_duration_sec=img2vid_duration,
            img2img_denoise_min=ps.get("img2img_denoise_min", 0.25),
            img2img_denoise_max=ps.get("img2img_denoise_max", 0.45),
            rolling_window_mode=ps.get("rolling_window", True),
            require_morph=ps.get("require_morph", False),
            seed_base=ps.get("seed_base", 42),
            video_fps=effective_video_fps, # No default, let TimingResolver handle it
            video_frames=effective_video_frames, # New
            timing_authority=effective_timing_authority, # New
            backend_config=bc,
            finalizer_config=self.config.get("finalizer", {}), # New
            reinject=self.reinject,
            smart_reinject_enabled=ps.get("smart_reinject_enabled", False),
            smart_reinject_every_n_cycles=ps.get("smart_reinject_every_n_cycles", 1),
            smart_reinject_use_morph=ps.get("smart_reinject_use_morph", True),
            smart_reinject_similarity_guard_enabled=ps.get("smart_reinject_similarity_guard_enabled", True),
            smart_reinject_similarity_threshold=ps.get("smart_reinject_similarity_threshold", 0.42),
            smart_reinject_wait_timeout_sec=ps.get("smart_reinject_wait_timeout_sec", 0),
            smart_reinject_sync_fallback=ps.get("smart_reinject_sync_fallback", False),
            smart_reinject_denoise_min=ps.get("smart_reinject_denoise_min"),
            smart_reinject_denoise_max=ps.get("smart_reinject_denoise_max"),
            smart_reinject_prompt_prefix=ps.get(
                "smart_reinject_prompt_prefix",
                "Preserve the source image strongly. Make a subtle evolved keyframe for the next visual beat. Keep composition, identity, palette, lighting, and major shapes stable. Avoid large scene changes.",
            ),
            crowd_control_config=self.config.get("crowd_control", {}),
            veo_config=veo_section,
            venice_config=venice_section,
            continuity_guard_enabled=ps.get("continuity_guard_enabled", True),
            continuity_similarity_threshold=ps.get("continuity_similarity_threshold", 0.42),
            continuity_morph_similarity_threshold=ps.get("continuity_morph_similarity_threshold", 0.86),
            continuity_retry_attempts=ps.get("continuity_retry_attempts", 2),
            continuity_ffmpeg_fallback_enabled=ps.get("continuity_ffmpeg_fallback_enabled", True),
            continuity_ffmpeg_fallback_min_similarity=ps.get("continuity_ffmpeg_fallback_min_similarity", 0.32),
            continuity_force_fallback_for_morph=ps.get("continuity_force_fallback_for_morph", True),
            enable_loop_closure=(
                ps.get("enable_loop_closure", False)
                or ps.get("loop_closure", False)
                or bc.get("enable_loop_closure", False)
                or
                veo_section.get("enable_loop_closure", False)
                or venice_section.get("enable_loop_closure", False)
            ),
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

        # Discover cycle videos (excludes loop clips)
        videos_dir = self.run_dir / "videos"
        videos = _list_cycle_videos(videos_dir)
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


    def _collect_story_outputs(self) -> Dict[str, Optional[str]]:
        """Collect outputs produced by run_sliding_story() + Finalizer inside the story engine.

        This avoids re-stitching cycle videos (which would overwrite per-cycle interpolation/pingpong results).
        """
        result = {
            "final_video": None,
            "final_video_60fps": None,
            "final_video_60fps_1080p": None,
            "final_video_60fps_1080p_audio": None,
        }

        base_master = self.run_dir / "final_output.mp4"
        if base_master.exists():
            result["final_video"] = str(base_master)

        deliverable_1080p = self.run_dir / "final_60fps_1080p.mp4"
        if deliverable_1080p.exists():
            result["final_video_60fps_1080p"] = str(deliverable_1080p)

        # Optional: keep a 60fps intermediate if present (not guaranteed; finalizer cleans temps)
        temp_60 = self.run_dir / "_temp_interpolated_60fps.mp4"
        if temp_60.exists():
            result["final_video_60fps"] = str(temp_60)

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

    def run(self) -> Path:
        """Execute the full pipeline, with optional resume support."""
        start_time = datetime.now()
        logger.info(f"[FaQtory] Starting run — mode: {self.mode} | reinject: {self.reinject} | id: {self.run_id}")

        # ── Resume validation ─────────────────────────────────────────────
        resume_ctx = None  # Will hold DiscoveredProgress if resuming
        if self.resume:
            if not self.run_dir.exists():
                raise RuntimeError(
                    f"[Resume] --resume requires existing run_dir: {self.run_dir} not found"
                )

            # ── Load config/story from run/meta/ snapshot (not current worqspace) ──
            # This ensures resume uses the EXACT same config/story as the original run.
            meta_dir = self.run_dir / "meta"
            meta_config = meta_dir / "config.yaml"
            meta_story = meta_dir / "story.txt"
            if meta_config.exists():
                logger.info("[Resume] Loading config EXCLUSIVELY from run/meta/config.yaml (frozen snapshot)")
                try:
                    self.config = yaml.safe_load(meta_config.read_text()) or {}
                    self._resolve_reinject_setting()
                except Exception as e:
                    logger.warning(f"[Resume] Failed to load meta config: {e} — falling back to worqspace")
            else:
                logger.warning("[Resume] No meta/config.yaml snapshot — using current worqspace config (may drift)")
            if meta_story.exists():
                logger.info("[Resume] Using story from run/meta/story.txt (frozen snapshot)")

            # ── Restore base inputs from run/meta/ (not current worqspace) ──
            # _copy_inputs_to_meta() saved copies during the original run.
            # Use those frozen snapshots instead of current worqspace files.
            meta_image = _detect_newest_file(meta_dir, _IMAGE_EXTS)
            meta_video = _detect_newest_file(meta_dir, _VIDEO_EXTS)
            meta_audio = _detect_newest_file(meta_dir, _AUDIO_EXTS)
            if meta_image or meta_video or meta_audio:
                logger.info("[Resume] Restoring base inputs from run/meta/ (frozen snapshots)")
                if meta_image:
                    self.base_image = meta_image
                    logger.info(f"[Resume] Base image: {meta_image.name}")
                if meta_video:
                    self.base_video = meta_video
                    logger.info(f"[Resume] Base video: {meta_video.name}")
                if meta_audio:
                    self.base_audio = meta_audio
                    logger.info(f"[Resume] Base audio: {meta_audio.name}")
                # Re-determine mode from restored inputs
                self.mode = self._determine_mode()

            # Discover progress from artifacts
            resume_ctx = discover_progress(self.run_dir)

            # Guard: refuse resume when already finalized
            if resume_ctx.already_finalized:
                raise RuntimeError(
                    f"[Resume] Run is already finalized (final_output.mp4 exists). "
                    f"Nothing to resume."
                )

            if not resume_ctx.completed_cycles:
                raise RuntimeError(
                    f"[Resume] No completed cycles found in {self.run_dir} — nothing to resume from"
                )
            logger.info(
                f"[Resume] Found {len(resume_ctx.completed_cycles)} completed cycle(s), "
                f"resuming from cycle {resume_ctx.next_cycle_index}"
            )

        # Setup directories (safe for resume — mkdir is idempotent)
        self._setup_run_dirs()
        if not self.resume:
            self._copy_inputs_to_meta()

        # Build story config
        story_config = self._build_story_config()
        _routing = describe_backend_config(story_config.backend_config or {})
        logger.info(f"[FaQtory] Backend: {_routing}")

        # Compute cycle count from audio if applicable
        audio_cycles = self._compute_cycle_count(story_config)

        # Resolve story file — on resume, prefer the meta snapshot
        if self.resume:
            meta_story = self.run_dir / "meta" / "story.txt"
            if meta_story.exists():
                story_path = meta_story
            else:
                story_path = self.worqspace_dir / "story.txt"
                logger.warning("[Resume] No meta/story.txt snapshot — using current worqspace story")
        else:
            story_path = self.worqspace_dir / "story.txt"
        if not story_path.exists():
            raise FileNotFoundError(f"Story file not found: {story_path}")

        # Handle base image/video for image/video modes
        base_image_for_run = None
        base_video_for_run = None
        backend_cfg = extract_backend_config(self.config)
        is_veo_backend = get_backend_type_for_capability(backend_cfg, "video") == "veo"

        if self.mode == "image" and self.base_image:
            base_image_for_run = self.base_image
        elif self.mode == "video" and self.base_video:
            if is_veo_backend and self.config.get("veo", {}).get("enable_extension", False):
                base_video_for_run = self.base_video
                logger.info(f"[FaQtory] Veo extension bootstrap: passing base video directly")
                extracted = self.run_dir / "meta" / "extracted_frame.png"
                width = backend_cfg.get("width", 1024)
                height = backend_cfg.get("height", 576)
                if _extract_video_frame(self.base_video, extracted, width, height):
                    base_image_for_run = extracted
                    logger.info(f"[FaQtory] Also extracted fallback frame → {extracted}")
            else:
                extracted = self.run_dir / "meta" / "extracted_frame.png"
                width = backend_cfg.get("width", 1024)
                height = backend_cfg.get("height", 576)
                if _extract_video_frame(self.base_video, extracted, width, height):
                    base_image_for_run = extracted
                    logger.info(f"[FaQtory] Extracted frame from video → {extracted}")
                else:
                    logger.warning("[FaQtory] Video frame extraction failed, falling back to text mode")

        if self.dry_run:
            # ── Backend availability check (v0.6.7-beta) ─────────────────
            # For dry-run, actually verify the backend is reachable/configured.
            # If unavailable, raise RuntimeError so the CLI exits non-zero.
            backend_ok = False
            try:
                test_backend = create_backend(story_config.backend_config or {})
                available, msg = test_backend.check_availability()
                if available:
                    logger.info(f"[FaQtory] Backend check: {msg}")
                    backend_ok = True
                else:
                    logger.error(f"[FaQtory] Backend NOT ready: {msg}")
            except Exception as e:
                logger.error(f"[FaQtory] Backend check failed: {e}")

            if not backend_ok:
                raise RuntimeError(
                    "DRY RUN FAILED — backend is not ready. "
                    "Fix the backend configuration and try again."
                )

            logger.info("[FaQtory] DRY RUN — config loaded, inputs resolved, backend ready, exiting before generation")
            self._write_run_state("completed", start_time, cycles_planned=0, cycles_completed=0)
            return self.run_dir

        # ── Write initial state ───────────────────────────────────────────
        self._write_run_state(
            "running",
            start_time,
            cycles_planned=audio_cycles or 0,
            resume_enabled=self.resume,
        )

        # ── Compute total cycles for finalization check ───────────────────
        # Parse story to determine actual cycle count (needed for finalization-only check)
        from .sliding_story_engine import _parse_story_file, _determine_windows
        paragraphs = _parse_story_file(story_path)
        P = len(paragraphs)
        M = story_config.max_paragraphs
        windows = _determine_windows(P, M)
        if audio_cycles and audio_cycles < len(windows):
            total_cycles = audio_cycles
        else:
            total_cycles = len(windows)

        # ── Finalization-only resume path ─────────────────────────────────
        # If all cycles are already done but final output is missing, skip
        # the entire generation engine and go straight to finalization.
        if resume_ctx and resume_ctx.last_completed_cycle >= total_cycles:
            final_output = self.run_dir / "final_output.mp4"
            if not final_output.exists():
                logger.info(
                    f"[Resume] All {total_cycles} cycles already complete — "
                    f"running finalization only"
                )
                # Fall through to finalization below (skip engine call)
                self._cycles_completed = total_cycles
                # Jump to finalization
                cycles_completed = total_cycles
                finalizer_result = self._collect_story_outputs()
                if not finalizer_result.get("final_video"):
                    # Run the finalizer manually
                    from .sliding_story_engine import run_sliding_story as _unused
                    from .finalizer import Finalizer
                    fc = self.config.get("finalizer", {})
                    finalizer = Finalizer(project_dir=self.run_dir, finalizer_config=fc)
                    videos = _list_cycle_videos(self.run_dir / "videos")
                    # Include loop-closure clips (separate from cycle videos)
                    loop_clips = sorted((self.run_dir / "videos").glob("video_loop_*.mp4"))
                    all_videos = list(videos) + list(loop_clips)
                    if all_videos:
                        finalizer.finalize(cycle_video_paths=all_videos)
                        finalizer.run_post_stitch_finalizer()
                        finalizer_result = self._collect_story_outputs()

                # Audio mux
                audio_cfg = self.config.get("audio", {})
                if audio_cfg.get("enabled", False) and self.base_audio:
                    mux_src = None
                    for key in ["final_video_60fps_1080p", "final_video", "final_video_60fps"]:
                        if finalizer_result.get(key):
                            mux_src = Path(finalizer_result[key])
                            break
                    if mux_src and mux_src.exists():
                        audio_output = self.run_dir / "final_video_60fps_1080p_audio.mp4"
                        if _mux_audio(mux_src, self.base_audio, audio_output):
                            finalizer_result["final_video_60fps_1080p_audio"] = str(audio_output)

                self._write_run_state(
                    "completed", start_time,
                    cycles_planned=total_cycles,
                    cycles_completed=total_cycles,
                    resume_enabled=True,
                    completed_indices=list(resume_ctx.completed_cycles),
                    video_paths=[str(p) for p in resume_ctx.final_video_paths],
                )
                logger.info(f"[FaQtory] Finalization-only resume complete")

                # Save run
                if self.project_name:
                    saved_to = self._save_run(finalizer_result, self.project_name)
                    state_path = saved_to / "faqtory_state.json"
                    if state_path.exists():
                        state = json.loads(state_path.read_text())
                        state["saved_to"] = str(saved_to)
                        state_path.write_text(json.dumps(state, indent=2))
                return self.run_dir

        # ── Build resume parameters ───────────────────────────────────────
        engine_kwargs = {}
        if resume_ctx:
            engine_kwargs["start_cycle"] = resume_ctx.next_cycle_index
            engine_kwargs["initial_last_frame_path"] = resume_ctx.last_frame_path
            engine_kwargs["initial_anchor_frame_path"] = resume_ctx.anchor_frame_path
            engine_kwargs["initial_final_video_paths"] = resume_ctx.final_video_paths
            engine_kwargs["initial_completed_cycles"] = set(resume_ctx.completed_cycles)

        # ── Per-cycle checkpoint callback ─────────────────────────────────
        self._cycles_completed = resume_ctx.last_completed_cycle if resume_ctx else 0
        self._completed_indices = list(resume_ctx.completed_cycles) if resume_ctx else []
        self._video_paths = [str(p) for p in resume_ctx.final_video_paths] if resume_ctx else []

        def _checkpoint(cycle_idx, last_frame, video_path, anchor_path):
            self._cycles_completed = cycle_idx
            if cycle_idx not in self._completed_indices:
                self._completed_indices.append(cycle_idx)
            if video_path and str(video_path) not in self._video_paths:
                self._video_paths.append(str(video_path))
            self._write_run_state(
                "running", start_time,
                cycles_planned=total_cycles,
                cycles_completed=cycle_idx,
                next_cycle=cycle_idx + 1,
                last_frame=str(last_frame) if last_frame else "",
                anchor_frame=str(anchor_path) if anchor_path else "",
                resume_enabled=True,
                completed_indices=self._completed_indices,
                video_paths=self._video_paths,
            )

        engine_kwargs["checkpoint_callback"] = _checkpoint

        # ── Run sliding story engine ──────────────────────────────────────
        try:
            final_video = run_sliding_story(
                story_path=story_path,
                qodeyard_dir=self.run_dir,
                config=story_config,
                max_cycles=audio_cycles,
                base_image_path=base_image_for_run,
                base_video_path=base_video_for_run,
                **engine_kwargs,
            )
        except KeyboardInterrupt:
            logger.info("\n[FaQtory] Run interrupted by user")
            self._write_run_state(
                "interrupted", start_time,
                cycles_planned=total_cycles,
                cycles_completed=self._cycles_completed,
                next_cycle=self._cycles_completed + 1,
                error_message="Interrupted by user (KeyboardInterrupt)",
                resume_enabled=True,
                completed_indices=self._completed_indices,
                video_paths=self._video_paths,
            )
            raise
        except Exception as e:
            logger.error(f"[FaQtory] Run failed at cycle ~{self._cycles_completed + 1}: {e}")
            self._write_run_state(
                "failed", start_time,
                cycles_planned=total_cycles,
                cycles_completed=self._cycles_completed,
                next_cycle=self._cycles_completed + 1,
                error_message=str(e),
                resume_enabled=True,
                completed_indices=self._completed_indices,
                video_paths=self._video_paths,
            )
            raise

        # Count completed cycles
        cycles_completed = len(_list_cycle_videos(self.run_dir / "videos"))

        # Collect outputs (story engine already stitched + optionally post-processed)
        finalizer_result = self._collect_story_outputs()

        # Optional Step: Audio mux (only if enabled + audio present)
        audio_cfg = self.config.get("audio", {})
        if audio_cfg.get("enabled", False) and self.base_audio:
            mux_src = None
            for key in ["final_video_60fps_1080p", "final_video", "final_video_60fps"]:
                if finalizer_result.get(key):
                    mux_src = Path(finalizer_result[key])
                    break
            if mux_src and mux_src.exists():
                audio_output = self.run_dir / "final_video_60fps_1080p_audio.mp4"
                if _mux_audio(mux_src, self.base_audio, audio_output):
                    finalizer_result["final_video_60fps_1080p_audio"] = str(audio_output)

        # ── Write completed state ─────────────────────────────────────────
        self._write_run_state(
            "completed", start_time,
            cycles_planned=audio_cycles or cycles_completed,
            cycles_completed=cycles_completed,
            completed_indices=self._completed_indices,
            video_paths=self._video_paths,
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
            state_path = saved_to / "faqtory_state.json"
            if state_path.exists():
                state = json.loads(state_path.read_text())
                state["saved_to"] = str(saved_to)
                state_path.write_text(json.dumps(state, indent=2))

        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        logger.info(f"[FaQtory] Run complete in {elapsed:.1f}s — {cycles_completed} cycles")
        return self.run_dir

    def _write_run_state(
        self,
        status: str,
        start_time: datetime,
        cycles_planned: int = 0,
        cycles_completed: int = 0,
        next_cycle: int = 1,
        last_frame: str = "",
        anchor_frame: str = "",
        error_message: str = "",
        resume_enabled: bool = False,
        completed_indices: Optional[List[int]] = None,
        video_paths: Optional[List[str]] = None,
    ) -> None:
        """Write run state atomically using RunState model."""
        state = RunState(
            run_id=self.run_id,
            version=__version__,
            status=status,
            backend_type=describe_backend_config(extract_backend_config(self.config)),
            mode=self.mode,
            reinject=self.reinject,
            story_path=str(self.worqspace_dir / "story.txt"),
            cycles_planned=cycles_planned,
            cycles_completed=cycles_completed,
            next_cycle_index=next_cycle,
            last_completed_cycle=cycles_completed,
            last_frame_path=last_frame,
            anchor_frame_path=anchor_frame,
            final_video_paths=video_paths or [],
            completed_cycle_indices=completed_indices or [],
            base_image=self.base_image.name if self.base_image else "",
            base_video=self.base_video.name if self.base_video else "",
            base_audio=self.base_audio.name if self.base_audio else "",
            start_time=start_time.isoformat(),
            end_time=datetime.now().isoformat() if status in ("completed", "failed", "interrupted") else "",
            error_message=error_message,
            resume_enabled=resume_enabled,
        )
        write_state_atomic(state, self._state_path)


def quick_run(**kwargs):
    """Convenience function for quick pipeline runs."""
    faqtory = VisualFaQtory(**kwargs)
    return faqtory.run()
