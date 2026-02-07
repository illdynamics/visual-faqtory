#!/usr/bin/env python3
"""
visual_faqtory.py - Main Visual Generation Pipeline
═══════════════════════════════════════════════════════════════════════════════

The Visual FaQtory orchestrates the 3-agent pipeline + finalizer for
automated long-form visual generation:

  InstruQtor → ConstruQtor → InspeQtor → (loop) → Finalizer

Pipeline flow:
  1. InstruQtor reads tasq.md, creates VisualBriq
  2. ConstruQtor calls backend, generates raw video
  3. InspeQtor processes it (passthrough or loop), suggests evolution for next cycle
  4. Repeat with evolved prompt until target duration reached
  5. Finalizer stitches all per-cycle MP4s into final_output.mp4 (BASE MASTER)
  6. Post-stitch Finalizer (if enabled):
     → Interpolate to 60fps (minterpolate)
     → Upscale to 1920×1080 (bicubic)
     → Encode with h264_nvenc / libx264 fallback
     → Produce final_60fps_1080p.mp4 (FINAL DELIVERABLE)

Project-based runs (v0.0.7-alpha):
  - Named projects stored in worqspace/qonstructions/<project-name>/
  - Unnamed runs use temp directory with interactive save prompt
  - Each project has: briqs/, images/, videos/, factory_state.json,
    config_snapshot.yaml, final_output.mp4, final_60fps_1080p.mp4

Part of QonQrete Visual FaQtory v0.3.5-beta
"""
import os
import sys
import json
import yaml
import time
import shutil
import signal
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

from .visual_briq import VisualBriq, CycleState, BriqStatus, InputMode
from .instruqtor import InstruQtor
from .construqtor import ConstruQtor
from .inspeqtor import InspeQtor
from .finalizer import Finalizer, FinalizerError
from .backends import create_backend, create_split_backend, list_available_backends
from .base_folders import select_base_files
from .duration_planner import plan_duration, post_finalize_trim_and_mux
from .stream_engine import get_stream_config, prepare_stream_cycle

logger = logging.getLogger(__name__)


class VisualFaQtory:
    """
    Main orchestrator for the Visual FaQtory pipeline.

    Supports project-based runs with structured archival layout.

    Usage:
        faqtory = VisualFaQtory(worqspace_dir="./worqspace")
        faqtory.run(cycles=100, project_name="my-project")
    """

    def __init__(
        self,
        worqspace_dir: str | Path = "./worqspace",
        output_dir: str | Path = "./qodeyard",
        config_override: Optional[Dict[str, Any]] = None,
        project_name: Optional[str] = None
    ):
        self.worqspace_dir = Path(worqspace_dir).resolve()

        # Load config
        self.config = self._load_config(config_override)

        # Project-based storage
        self.project_name = project_name
        self.qonstructions_base = self.worqspace_dir / "qonstructions"
        self.qonstructions_base.mkdir(parents=True, exist_ok=True)

        if project_name:
            # Named project: store in qonstructions/<name>/
            self.project_dir = self.qonstructions_base / project_name
            self.output_dir = self.project_dir
            self._is_temp = False
        else:
            # Temporary session: use output_dir (qodeyard by default)
            self.output_dir = Path(output_dir).resolve()
            self.project_dir = self.output_dir
            self._is_temp = True

        # Setup project directory structure
        self._setup_project_dirs()

        # Initialize agents
        self._init_agents()

        # State management
        self.state: Optional[CycleState] = None
        self.state_file = self.project_dir / "factory_state.json"

        # Running flag for graceful shutdown
        self._running = False
        self._setup_signal_handlers()

        # Save config snapshot
        self._save_config_snapshot()

        # Base folder file selection (v0.1.0)
        self.base_files = {"base_image": None, "base_audio": None, "base_video": None}
        self._audio_controller = None
        self._audio_analysis = None
        self._beat_grid = None
        self._cycle_timing = None
        self._init_base_folders()

    def _setup_project_dirs(self) -> None:
        """Create the project directory structure."""
        self.project_dir.mkdir(parents=True, exist_ok=True)
        (self.project_dir / "briqs").mkdir(exist_ok=True)
        (self.project_dir / "images").mkdir(exist_ok=True)
        (self.project_dir / "videos").mkdir(exist_ok=True)

    def _save_config_snapshot(self) -> None:
        """Save a snapshot of the config used for this run."""
        snapshot_path = self.project_dir / "config_snapshot.yaml"
        if not snapshot_path.exists():
            snapshot_path.write_text(yaml.dump(self.config, default_flow_style=False))

    def _load_config(self, override: Optional[Dict] = None) -> Dict[str, Any]:
        """Load configuration from config.yaml."""
        config_path = self.worqspace_dir / "config.yaml"

        if config_path.exists():
            config = yaml.safe_load(config_path.read_text())
            logger.info(f"[FaQtory] Loaded config from {config_path}")
        else:
            logger.warning(f"[FaQtory] No config.yaml found, using defaults")
            config = {}

        if override:
            config = self._deep_merge(config, override)

        return config

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _init_agents(self) -> None:
        """Initialize the three agents."""
        logger.info("[FaQtory] Initializing agents...")

        # Create backend(s) for ConstruQtor — supports split config (v0.0.7)
        backend = create_split_backend(self.config)

        # LLM Provider initialization
        llm_provider_instance = None
        llm_config = self.config.get('llm', {})
        if llm_config:
            logger.info(f"[FaQtory] LLM configuration found. Provider: {llm_config.get('provider', 'unknown')}")
            llm_provider_instance = llm_config

        # InstruQtor - instruction creator
        self.instruqtor = InstruQtor(
            config=self.config,
            worqspace_dir=self.worqspace_dir,
            qodeyard_dir=self.project_dir,
            llm_provider=llm_provider_instance
        )

        # ConstruQtor - visual builder (output goes to images/ or videos/ depending on backend)
        self.construqtor = ConstruQtor(
            config=self.config,
            qodeyard_dir=self.project_dir / "images",
            backend=backend
        )

        # InspeQtor - inspector and evolver (output goes to videos/)
        self.inspeqtor = InspeQtor(
            config=self.config,
            qodeyard_dir=self.project_dir / "videos",
            llm_provider=llm_provider_instance,
            mode="innovative"
        )

        logger.info("[FaQtory] Agents initialized: InstruQtor, ConstruQtor, InspeQtor")

    def _init_base_folders(self) -> None:
        """Initialize base folder selection and audio reactivity (v0.1.0)."""
        try:
            self.base_files = select_base_files(
                self.worqspace_dir, self.config, run_id=self.project_name or "default"
            )
        except Exception as e:
            logger.warning(f"[FaQtory] Base folder selection failed: {e}")
            self.base_files = {"base_image": None, "base_audio": None, "base_video": None}

        # Audio reactivity setup
        audio_config = self.config.get("audio_reactivity", {})
        audio_path = self.base_files.get("base_audio")

        if audio_config.get("enabled", False) and audio_path and audio_path.exists():
            try:
                from .audio_reactivity import run_audio_analysis, compute_cycle_timing
                analysis, beat_grid, features, ctrl = run_audio_analysis(
                    audio_path, self.worqspace_dir, audio_config
                )
                self._audio_controller = ctrl
                self._audio_analysis = analysis
                self._beat_grid = beat_grid
                logger.info(
                    f"[FaQtory] Audio reactivity active: "
                    f"bpm={analysis.get('bpm', 0):.1f}, "
                    f"source={analysis.get('source', 'unknown')}"
                )
            except ImportError:
                logger.warning(
                    "[FaQtory] librosa not installed, audio reactivity disabled. "
                    "Install with: pip install librosa"
                )
            except Exception as e:
                logger.warning(f"[FaQtory] Audio analysis failed: {e}")
        elif audio_config.get("enabled", False):
            # Clock-only mode: BPM sync without audio file
            manual_bpm = audio_config.get("bpm_manual")
            if manual_bpm:
                logger.info(f"[FaQtory] Audio clock-only mode: bpm={manual_bpm}")
                self._audio_analysis = {"bpm": manual_bpm, "source": "manual_clock"}
            else:
                logger.info("[FaQtory] Audio reactivity enabled but no audio file found")

    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown on SIGINT/SIGTERM."""
        def handler(signum, frame):
            logger.info("\n[FaQtory] Shutdown requested, completing current cycle...")
            self._running = False

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def run(
        self,
        cycles: int = 0,
        target_hours: float = 0,
        resume: bool = True
    ) -> List[VisualBriq]:
        """
        Run the visual generation pipeline.

        Args:
            cycles: Number of cycles to run (0 = use config or unlimited)
            target_hours: Target total duration (0 = use config)
            resume: Whether to resume from saved state

        Returns:
            List of completed VisualBriqs
        """
        logger.info("=" * 60)
        logger.info("QonQrete Visual FaQtory v0.3.5-beta")
        logger.info("=" * 60)
        if self.project_name:
            logger.info(f"[FaQtory] Project: {self.project_name}")
        logger.info(f"[FaQtory] Output: {self.project_dir}")

        # Determine limits
        max_cycles = cycles or self.config.get('cycle', {}).get('max_cycles', 0)
        target_duration = target_hours or self.config.get('cycle', {}).get('target_duration_hours', 2.0)

        # Calculate cycles needed for target duration
        loop_duration = self.config.get('generation', {}).get('clip_seconds', 8.0) * 2
        if target_duration > 0 and max_cycles == 0:
            max_cycles = int((target_duration * 3600) / loop_duration) + 1
            logger.info(f"[FaQtory] Target {target_duration}h requires ~{max_cycles} cycles")

        # ── AUTO-DURATION PLANNING (v0.1.2 feature) ─────────────────────
        audio_path = self.base_files.get("base_audio")
        audio_config = self.config.get('audio_reactivity', {})
        bpm = audio_config.get('bpm_manual', 0) or getattr(self, '_detected_bpm', 0)
        bars_per = audio_config.get('bars_per_cycle', 8)

        duration_plan = plan_duration(
            config=self.config,
            audio_path=audio_path,
            bpm=bpm,
            bars_per_cycle=bars_per,
            requested_cycles=max_cycles,
            clip_seconds=self.config.get('generation', {}).get('clip_seconds', 8.0),
        )

        if duration_plan.get('override_reason'):
            max_cycles = duration_plan['required_cycles']
            logger.info(f"[FaQtory] AUTO-DURATION: {duration_plan['override_reason']}")
            logger.info(f"[FaQtory] Cycle duration: {duration_plan['cycle_duration']:.1f}s")

        # Stream mode config
        stream_cfg = get_stream_config(self.config)
        if stream_cfg['enabled']:
            logger.info(f"[FaQtory] STREAM MODE: {stream_cfg['method']} "
                       f"(ctx={stream_cfg['context_length']}f, gen={stream_cfg['generation_length']}f)")


        # Load or create state
        if resume and self.state_file.exists():
            self.state = CycleState.load(self.state_file)
            logger.info(f"[FaQtory] Resumed from cycle {self.state.current_cycle}")
        else:
            self.state = self._create_new_state(max_cycles)

        self._running = True
        completed_briqs = []
        previous_briq = None
        evolution_suggestion = None

        # Main generation loop
        while self._running:
            cycle_index = self.state.current_cycle

            if max_cycles > 0 and cycle_index >= max_cycles:
                logger.info(f"[FaQtory] Reached max cycles ({max_cycles})")
                break

            logger.info("-" * 40)
            logger.info(f"[FaQtory] CYCLE {cycle_index}")
            logger.info("-" * 40)

            try:
                # === STAGE 1: InstruQtor ===
                briq = self.instruqtor.create_briq(
                    cycle_index=cycle_index,
                    previous_briq=previous_briq,
                    evolution_suggestion=evolution_suggestion
                )

                # === BASE FOLDERS: inject base_video for V2V if available ===
                if cycle_index == 0 and self.base_files.get("base_video"):
                    base_vid = self.base_files["base_video"]
                    if base_vid.exists() and briq.mode != InputMode.IMAGE:
                        briq.base_video_path = base_vid
                        briq.mode = InputMode.VIDEO
                        logger.info(f"[FaQtory] Using base_video: {base_vid.name}")

                # === AUDIO REACTIVITY: inject features into briq ===
                if self._audio_controller and self._audio_analysis:
                    try:
                        from .audio_reactivity import (
                            compute_cycle_timing, apply_audio_mapping
                        )
                        audio_cfg = self.config.get("audio_reactivity", {})
                        bpm = self._audio_analysis.get("bpm", 120.0)
                        bars_per = audio_cfg.get("beat_grid", {}).get("bars_per_cycle_default", 8)
                        beat_dur = 60.0 / bpm
                        bar_dur = beat_dur * 4
                        cycle_dur = bar_dur * bars_per

                        briq.bpm = bpm
                        briq.cycle_start_time = cycle_index * cycle_dur
                        briq.cycle_end_time = (cycle_index + 1) * cycle_dur

                        segment_stats = self._audio_controller.get_segment_stats(
                            briq.cycle_start_time, briq.cycle_end_time
                        )
                        briq.audio_segment_stats = segment_stats

                        mapping_cfg = audio_cfg.get("mapping", {})
                        if mapping_cfg.get("enabled", True):
                            mapped = apply_audio_mapping(
                                segment_stats, mapping_cfg, cycle_index, self._beat_grid
                            )
                            briq.audio_prompt_additions = mapped.get("prompt_additions", "")
                            briq.seed += mapped.get("seed_offset", 0)

                            if briq.audio_prompt_additions:
                                briq.prompt = f"{briq.prompt}, {briq.audio_prompt_additions}"
                                logger.info(f"[FaQtory] Audio modifiers: {briq.audio_prompt_additions}")
                    except Exception as e:
                        logger.warning(f"[FaQtory] Audio injection failed: {e}")

                self._save_briq(briq)

                # === STREAM MODE: context extraction (v0.2.0-beta) ===
                if stream_cfg['enabled'] and cycle_index > 0 and previous_briq:
                    prev_video = previous_briq.looped_video_path or previous_briq.raw_video_path
                    if prev_video and prev_video.exists():
                        try:
                            ctx = prepare_stream_cycle(
                                cycle_index=cycle_index,
                                previous_video=prev_video,
                                output_dir=self.project_dir / "videos",
                                stream_config=stream_cfg,
                                fps=briq.spec.video_fps,
                                bpm=briq.bpm or bpm,
                            )
                            if ctx.get('context_video_path'):
                                briq.context_video_path = ctx['context_video_path']
                                briq.spec.generation_frames = ctx.get('generation_frames')
                                briq.spec.context_frames = ctx.get('context_frames')
                                logger.info(f"[FaQtory] Stream context: {ctx['context_video_path']}")
                        except Exception as e:
                            logger.warning(f"[FaQtory] Stream context prep failed: {e}")

                # === STAGE 2: ConstruQtor ===
                briq = self.construqtor.construct(briq)
                self._save_briq(briq)

                # === STAGE 3: InspeQtor ===
                briq = self.inspeqtor.inspect(briq)
                self._save_briq(briq)

                # Cycle complete!
                completed_briqs.append(briq)
                self.state.completed_briqs.append(briq.briq_id)
                self.state.total_generation_time += briq.generation_time
                self.state.total_video_duration += loop_duration
                self.state.prompt_history.append(briq.prompt)

                # Track video path for finalizer
                if briq.looped_video_path:
                    self.state.cycle_video_paths.append(str(briq.looped_video_path))

                summary = self.inspeqtor.get_cycle_summary(briq)
                logger.info(f"[FaQtory] Cycle {cycle_index} COMPLETE")
                logger.info(f"  → Looped video: {briq.looped_video_path}")
                logger.info(f"  → Next evolution: {briq.evolution_suggestion[:60]}...")

                # Setup for next cycle
                previous_briq = briq
                evolution_suggestion = briq.evolution_suggestion
                self.state.current_cycle += 1

                # Save state
                self._save_state()

                # Delay between cycles
                delay = self.config.get('cycle', {}).get('delay_seconds', 5.0)
                if delay > 0 and self._running:
                    logger.info(f"[FaQtory] Waiting {delay}s before next cycle...")
                    time.sleep(delay)

            except Exception as e:
                logger.error(f"[FaQtory] Cycle {cycle_index} FAILED: {e}")

                self.state.failed_cycles.append(cycle_index)

                if not self.config.get('cycle', {}).get('continue_on_error', True):
                    raise

                self.state.current_cycle += 1
                self._save_state()

        # === FINALIZATION ===
        final_path = None
        self._deliverable_path = None
        if completed_briqs:
            final_path = self._run_finalizer()

            # ── POST-FINALIZE: TRIM + MUX (v0.1.2 feature) ─────────────
            if final_path and duration_plan.get('trim_to'):
                try:
                    audio_path_for_mux = self.base_files.get("base_audio")
                    codec = self.config.get('looping', {}).get('output_codec', 'h264_nvenc')
                    final_path = post_finalize_trim_and_mux(
                        final_video=final_path,
                        audio_path=audio_path_for_mux,
                        plan=duration_plan,
                        preferred_codec=codec,
                    )
                    logger.info(f"[FaQtory] Post-finalize complete: {final_path}")
                except Exception as e:
                    logger.warning(f"[FaQtory] Post-finalize trim/mux failed: {e}")

        # Final summary
        logger.info("=" * 60)
        logger.info("[FaQtory] Pipeline Complete")
        logger.info(f"  → Cycles completed: {len(completed_briqs)}")
        logger.info(f"  → Total video duration: {self.state.total_video_duration:.1f}s")
        if final_path:
            logger.info(f"  → Stitched master: {final_path}")
        if self._deliverable_path:
            logger.info(f"  → Final deliverable: {self._deliverable_path}")
        logger.info(f"  → Project directory: {self.project_dir}")
        if self.state.failed_cycles:
            logger.warning(f"  → Failed cycles: {self.state.failed_cycles}")
        logger.info("=" * 60)

        # Interactive save prompt for temp runs
        if self._is_temp and completed_briqs:
            self._prompt_save()

        return completed_briqs

    def _run_finalizer(self) -> Optional[Path]:
        """Run the Finalizer to stitch all videos into final_output.mp4,
        then optionally run post-stitch interpolation + upscale."""
        try:
            codec = self.config.get('looping', {}).get('output_codec', 'h264_nvenc')
            quality = self.config.get('looping', {}).get('output_quality', 18)
            finalizer_config = self.config.get('finalizer', {})

            finalizer = Finalizer(
                project_dir=self.project_dir,
                preferred_codec=codec,
                output_quality=quality,
                finalizer_config=finalizer_config
            )

            video_paths = [Path(p) for p in self.state.cycle_video_paths] if self.state.cycle_video_paths else None

            final_path = finalizer.finalize(
                cycle_video_paths=video_paths,
                failed_cycles=self.state.failed_cycles if self.state.failed_cycles else None
            )

            # === POST-STITCH FINALIZER (runs ONCE, after stitching) ===
            deliverable_path = finalizer.run_post_stitch_finalizer()
            if deliverable_path:
                self._deliverable_path = deliverable_path

            return final_path

        except FinalizerError as e:
            logger.warning(f"[FaQtory] Finalization skipped: {e}")
            return None
        except Exception as e:
            logger.error(f"[FaQtory] Finalization failed: {e}")
            return None

    def _prompt_save(self) -> None:
        """Prompt user to save a temporary run to a named project."""
        try:
            if not sys.stdin.isatty():
                return

            print("\n" + "=" * 50)
            response = input("Do you want to save this run? If yes, provide a project name: ").strip()

            if response:
                target_dir = self.qonstructions_base / response
                if target_dir.exists():
                    overwrite = input(f"Project '{response}' already exists. Overwrite? (y/N): ").strip().lower()
                    if overwrite != 'y':
                        logger.info("[FaQtory] Save cancelled.")
                        return
                    shutil.rmtree(target_dir)

                shutil.copytree(self.project_dir, target_dir)
                logger.info(f"[FaQtory] Saved to: {target_dir}")
                self.project_name = response
            else:
                logger.info("[FaQtory] Temporary artifacts will remain in output directory.")

        except (EOFError, KeyboardInterrupt):
            logger.info("\n[FaQtory] Save prompt skipped.")

    def run_single_cycle(self, cycle_index: int = 0) -> VisualBriq:
        """Run a single generation cycle (useful for testing)."""
        logger.info(f"[FaQtory] Running single cycle {cycle_index}")

        briq = self.instruqtor.create_briq(cycle_index=cycle_index)
        briq = self.construqtor.construct(briq)
        briq = self.inspeqtor.inspect(briq)

        self._save_briq(briq)
        return briq

    def _create_new_state(self, max_cycles: int) -> CycleState:
        """Create new session state."""
        session_id = hashlib.sha256(
            f"{datetime.now().isoformat()}_{os.getpid()}".encode()
        ).hexdigest()[:12]

        return CycleState(
            session_id=session_id,
            started_at=datetime.now(),
            total_cycles_requested=max_cycles,
            qodeyard_path=self.project_dir
        )

    def _save_briq(self, briq: VisualBriq) -> None:
        """Save briq to disk."""
        briq_path = self.project_dir / "briqs" / f"{briq.briq_id}.json"
        briq.save(briq_path)

    def _save_state(self) -> None:
        """Save pipeline state."""
        self.state.save(self.state_file)

    def status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        if not self.state_file.exists():
            return {"status": "not_started"}

        state = CycleState.load(self.state_file)
        return {
            "status": "in_progress" if self._running else "paused",
            "session_id": state.session_id,
            "current_cycle": state.current_cycle,
            "total_requested": state.total_cycles_requested,
            "completed_briqs": len(state.completed_briqs),
            "total_video_duration": state.total_video_duration,
            "total_generation_time": state.total_generation_time,
            "failed_cycles": state.failed_cycles,
            "output_dir": str(state.qodeyard_path)
        }

    def list_outputs(self) -> List[Path]:
        """List all generated looped videos."""
        videos_dir = self.project_dir / "videos"
        if videos_dir.exists():
            return sorted(videos_dir.glob("cycle*_video.mp4"))
        return sorted(self.project_dir.glob("cycle*_video.mp4"))

    def check_backends(self) -> Dict[str, tuple]:
        """Check availability of all backends."""
        return list_available_backends()


def quick_run(prompt: str, cycles: int = 5, output_dir: str = "./qodeyard") -> List[Path]:
    """
    Quick run helper - generate visuals from a prompt.

    Creates temporary tasq.md and runs the pipeline.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        worqspace = Path(tmpdir)

        config = {
            'backend': {'type': 'mock'},
            'input': {'tasq_file': 'tasq.md'},
            'generation': {'clip_seconds': 8.0},
            'looping': {'method': 'pingpong'}
        }
        (worqspace / "config.yaml").write_text(yaml.dump(config))
        (worqspace / "tasq.md").write_text(f"---\nmode: text\nseed: 42\n---\n\n{prompt}")

        faqtory = VisualFaQtory(worqspace_dir=worqspace, output_dir=output_dir)
        briqs = faqtory.run(cycles=cycles)

        return [b.looped_video_path for b in briqs if b.looped_video_path]


__all__ = ['VisualFaQtory', 'quick_run']
