#!/usr/bin/env python3
"""
inspeqtor.py - Visual Inspection & Evolution Agent
═══════════════════════════════════════════════════════════════════════════════

InspeQtor is the third agent in the Visual FaQtory pipeline.

Responsibilities:
  1. Receive raw video from ConstruQtor
  2. Use FFmpeg to create loopable version (forward + reverse = ~16s)
  3. Save looped video as cycleN_video.mp4 in project directory
  4. Analyze the visual output using LLM
  5. Suggest subtle creative variations for next cycle

Part of QonQrete Visual FaQtory v0.5.6-beta
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

from .visual_briq import VisualBriq, BriqStatus

# LLM utils are optional — inspeqtor works fully without LLM
try:
    from .llm_utils import create_llm_client, call_llm
except ImportError:
    create_llm_client = None
    call_llm = None

logger = logging.getLogger(__name__)


INSPEQTOR_SYSTEM_PROMPT = """You are InspeQtor, a visual evolution specialist for the QonQrete Visual FaQtory.

You operate in INNOVATIVE mode - thinking outside the box while keeping changes SUBTLE.

Your job is to suggest how to evolve visuals for the next generation cycle.

Guidelines for evolution:
- Keep 80% similar, evolve 20%
- Focus on: color shifts, lighting changes, atmospheric tweaks, energy levels
- Avoid: completely different scenes, drastic style changes, unrelated elements
- Think like a DJ doing a visual mix - smooth transitions, building energy

Output format (JSON):
{
    "quality_assessment": "brief quality notes",
    "evolution_suggestion": "specific prompt modification suggestion",
    "color_evolution": "suggested color shift",
    "energy_evolution": "increase/decrease/maintain",
    "innovative_idea": "one creative but subtle idea",
    "reasoning": "why this evolution works"
}"""

INSPEQTOR_EVOLVE_PROMPT = """Analyze this visual generation cycle and suggest the next evolution:

CURRENT PROMPT:
{current_prompt}

STYLE_HINTS:
{style_hints}

MOTION_PROMPT:
{motion_prompt}

NEGATIVE_PROMPT_SOURCE:
{negative_prompt}

IMPORTANT CONSTRAINTS:
- The negative prompt lists concepts that MUST be avoided.
- You must NOT evolve toward anything listed there.
- You may suggest ADDITIONS ONLY if they remain compatible with the negative prompt.
- Never contradict or weaken the negative prompt constraints.

CYCLE NUMBER: {cycle_index}
TOTAL PLANNED: {target_cycles}

PREVIOUS SUGGESTIONS USED:
{previous_suggestions}

Generate a SUBTLE but INNOVATIVE evolution for the next cycle.
Respect STYLE_HINTS boundaries — evolve WITHIN them, don't break them.
Consider MOTION_PROMPT intent for motion continuity suggestions.
Think drum & bass visuals — keep the energy flowing but evolve the vibe.
Do NOT override the negative prompt — you may suggest ADDITIONS only.

Respond with JSON only."""

class InspeQtor:
    """
    The inspection agent that finalizes visuals and guides evolution.

    Creates loopable videos and generates creative suggestions for next cycle.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        qodeyard_dir: Path,
        llm_provider: Optional[Any] = None,
        mode: str = "innovative",
    ):
        self.config = config
        self.qodeyard_dir = Path(qodeyard_dir)
        self.mode = mode

        self.llm_client = None
        if llm_provider:
            self.llm_client = create_llm_client(llm_provider)

        loop_config = config.get("looping", {})
        self.looping_enabled = loop_config.get("enabled", True)
        self.loop_method = loop_config.get("method", "pingpong")
        self.output_fps = loop_config.get("output_fps", 24)
        self.output_codec = loop_config.get("output_codec", "h264_nvenc")
        self.output_quality = loop_config.get("output_quality", 18)
        self.crossfade_frames = loop_config.get("crossfade_frames", 8)

        self.suggestion_history: list = []

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def inspect(self, briq: VisualBriq) -> VisualBriq:
        """
        Inspect and finalize a constructed briq.

        Pipeline:
          1. Create loopable video (forward + reverse) or crossfade
          2. Save as cycleN_video.mp4
          3. Generate evolution suggestion for next cycle
        """
        logger.info(f"[InspeQtor] Inspecting briq {briq.briq_id}")
        briq.status = BriqStatus.INSPECTING

        try:
            looped_path = self._create_loop(briq)
            briq.looped_video_path = looped_path
            logger.info(f"[InspeQtor] Created loop: {looped_path}")

            suggestion = self._generate_evolution_suggestion(briq)
            briq.evolution_suggestion = suggestion
            briq.suggested_prompt_delta = suggestion

            self.suggestion_history.append({"cycle": briq.cycle_index, "suggestion": suggestion})

            briq.status = BriqStatus.COMPLETE
            logger.info(f"[InspeQtor] Inspection complete for cycle {briq.cycle_index}")

            self._cleanup_temp_files()

        except Exception as e:
            briq.status = BriqStatus.FAILED
            briq.error_message = str(e)
            logger.error(f"[InspeQtor] Inspection failed: {e}")
            raise

        return briq

    # ──────────────────────────────────────────────────────────────────────────
    # Loop creation
    # ──────────────────────────────────────────────────────────────────────────

    def _cleanup_temp_files(self) -> None:
        """Remove temporary files created by InspeQtor."""
        try:
            for pattern in ["*_frame.png", "_temp_*"]:
                for f in self.qodeyard_dir.glob(pattern):
                    if f.is_file():
                        f.unlink()
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

    def _create_loop(self, briq: VisualBriq) -> Path:
        """Create loopable video using FFmpeg, or passthrough if looping disabled."""
        if not briq.raw_video_path or not briq.raw_video_path.exists():
            raise FileNotFoundError(f"Raw video not found: {briq.raw_video_path}")

        input_path = briq.raw_video_path
        output_name = f"cycle{briq.cycle_index:04d}_video.mp4"
        output_path = (self.qodeyard_dir / output_name).resolve()

        if not self.looping_enabled:
            # Passthrough: re-encode raw video with consistent settings
            # No reverse loop — keeps visuals forward-evolving
            self._create_passthrough(input_path, output_path)
        elif self.loop_method == "crossfade":
            self._create_crossfade_loop(input_path, output_path)
        else:
            self._create_pingpong_loop(input_path, output_path)

        return output_path

    def _encode_args(self, codec: str) -> list[str]:
        """Return ffmpeg encoder args for the requested codec."""
        if codec == "h264_nvenc":
            return [
                "-c:v", "h264_nvenc",
                "-rc", "vbr",
                "-cq", str(self.output_quality),
                "-b:v", "0",
                "-profile:v", "high",
                "-preset", "p5",
            ]
        # Default: CRF-capable codecs (libx264, libx265, etc.)
        return ["-c:v", codec, "-crf", str(self.output_quality)]

    def _create_passthrough(self, input_path: Path, output_path: Path) -> None:
        """
        Passthrough: re-encode raw video with consistent codec/fps settings.
        No reverse loop — keeps visuals forward-evolving for the final stitch.
        Used when looping.enabled is false.
        """
        input_path = input_path.resolve()
        logger.info(
            f"[InspeQtor] Looping DISABLED — passthrough mode "
            f"(forward-evolving, no reverse loop)"
        )

        for codec in [self.output_codec, "libx264"]:
            video_args = self._encode_args(codec)

            cmd = [
                "ffmpeg", "-y",
                "-i", str(input_path),
                *video_args,
                "-r", str(self.output_fps),
                "-vsync", "cfr",
                "-pix_fmt", "yuv420p",
                str(output_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"[InspeQtor] Passthrough encode succeeded with {codec}")
                return

            logger.warning(
                f"[InspeQtor] Passthrough with {codec} failed, "
                f"trying next... stderr: {result.stderr[:200]}"
            )

        # Ultimate fallback: raw copy
        import shutil
        shutil.copy2(input_path, output_path)
        logger.warning("[InspeQtor] All codecs failed, raw-copied input as output")

    def _create_pingpong_loop(self, input_path: Path, output_path: Path) -> None:
        """Create ping-pong loop: forward + reverse."""
        input_path = input_path.resolve()
        reversed_path = (self.qodeyard_dir / f"_temp_reversed_{input_path.name}").resolve()
        concat_list = (self.qodeyard_dir / "_temp_concat.txt").resolve()

        try:
            # 1) Reverse video (re-encode to ensure consistent fps/pix_fmt)
            reverse_codec = self.output_codec if self.output_codec else "libx264"
            reverse_cmd = [
                "ffmpeg", "-y",
                "-i", str(input_path),
                "-vf", "reverse",
                "-an",
                *self._encode_args(reverse_codec),
                "-r", str(self.output_fps),
                "-vsync", "cfr",
                "-pix_fmt", "yuv420p",
                str(reversed_path),
            ]

            result = subprocess.run(reverse_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"[InspeQtor] Reverse failed ({reverse_codec}). stderr: {result.stderr}")
                # Fallback: copy original (non-loop) so pipeline can continue
                import shutil
                shutil.copy2(input_path, output_path)
                return

            # 2) Concat forward + reverse into looped output
            concat_list.write_text(
                f"file '{input_path}'\n"
                f"file '{reversed_path}'\n"
            )

            for codec in [self.output_codec, "libx264"]:
                # If the first entry equals "libx264", the list will try it twice; harmless.
                video_args = self._encode_args(codec)

                concat_cmd = [
                    "ffmpeg", "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", str(concat_list),
                    *video_args,
                    "-r", str(self.output_fps),
                    "-vsync", "cfr",
                    "-pix_fmt", "yuv420p",
                    str(output_path),
                ]

                result = subprocess.run(concat_cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    break
                logger.warning(f"[InspeQtor] Concat with {codec} failed, trying next... stderr: {result.stderr}")
            else:
                import shutil
                shutil.copy2(input_path, output_path)

        finally:
            try:
                if reversed_path.exists():
                    reversed_path.unlink()
            except Exception:
                pass
            try:
                if concat_list.exists():
                    concat_list.unlink()
            except Exception:
                pass

    def _create_crossfade_loop(self, input_path: Path, output_path: Path) -> None:
        """Create a loopable clip using a crossfade from end→start (PTS-safe, CFR-safe)."""
        input_path = input_path.resolve()
        duration = self._get_video_duration(input_path)
        fps = float(self.output_fps)
        xfade_duration = self.crossfade_frames / fps
    
        if duration <= xfade_duration * 2:
            logger.warning("[InspeQtor] Video too short for crossfade, using pingpong")
            self._create_pingpong_loop(input_path, output_path)
            return
    
        # PTS + timebase normalization is REQUIRED for xfade
        # This fixes the dreaded: "current rate of 1/0 is invalid"
        filter_complex = (
            f"[0:v]"
            f"settb=AVTB,"
            f"setpts=N/({fps}*TB),"
            f"fps={fps},"
            f"split=2[main][end];"
            f"[end]"
            f"trim=start={duration - xfade_duration},"
            f"setpts=PTS-STARTPTS[endclip];"
            f"[main]"
            f"trim=end={duration - xfade_duration},"
            f"setpts=PTS-STARTPTS[mainclip];"
            f"[mainclip][endclip]"
            f"xfade=transition=fade:duration={xfade_duration}:offset=0[out]"
        )
    
        for codec in [self.output_codec, "libx264"]:
            video_args = self._encode_args(codec)
    
            cmd = [
                "ffmpeg", "-y",
                "-fflags", "+genpts",
                "-i", str(input_path),
                "-filter_complex", filter_complex,
                "-map", "[out]",
                "-an",
                *video_args,
                "-r", str(self.output_fps),
                "-vsync", "cfr",
                "-pix_fmt", "yuv420p",
                str(output_path),
            ]
    
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"[InspeQtor] Crossfade succeeded with {codec}")
                return
    
            logger.warning(
                f"[InspeQtor] Crossfade with {codec} failed, trying next... "
                f"stderr: {result.stderr[:300]}"
            )
    
        logger.warning("[InspeQtor] Crossfade failed for all codecs, falling back to pingpong")
        self._create_pingpong_loop(input_path, output_path)


    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration in seconds."""
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            return float(result.stdout.strip())
        except ValueError:
            return 8.0

    # ──────────────────────────────────────────────────────────────────────────
    # Evolution suggestion
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_evolution_suggestion(self, briq: VisualBriq) -> str:
        """Generate creative evolution suggestion for next cycle."""
        if self.llm_client:
            return self._suggest_with_llm(briq)
        return self._basic_suggestion(briq)

    def _basic_suggestion(self, briq: VisualBriq) -> str:
        """Generate basic evolution suggestion without LLM."""
        evolutions = [
            "shift colors toward cooler blue/purple tones",
            "add more dynamic energy and movement",
            "introduce subtle particle effects",
            "increase atmospheric depth and fog",
            "shift to warmer golden/orange accents",
            "add geometric pattern overlays",
            "increase contrast and intensity",
            "soften edges with dreamlike blur",
            "introduce neon glow highlights",
            "add subtle glitch artifacts",
        ]
        innovative_extras = [
            "experiment with impossible geometry",
            "blend organic and mechanical elements",
            "introduce liquid metal textures",
            "add bioluminescent accents",
            "create depth through layered dimensions",
            "merge abstract with recognizable forms",
            "introduce time-distortion visual effects",
            "blend multiple color temperatures",
        ]

        cycle = briq.cycle_index
        base = evolutions[cycle % len(evolutions)]

        if self.mode == "innovative" and cycle % 3 == 0:
            extra = innovative_extras[cycle % len(innovative_extras)]
            return f"{base}, and {extra}"

        return base

    def _suggest_with_llm(self, briq: VisualBriq) -> str:
        """Generate evolution suggestion using LLM (with style, motion, and negative context)."""
        if not self.llm_client:
            return self._basic_suggestion(briq)
    
        try:
            recent_suggestions = self.suggestion_history[-3:] if self.suggestion_history else []
            suggestions_text = (
                "\n".join([f"Cycle {s['cycle']}: {s['suggestion']}" for s in recent_suggestions])
                or "None yet (first cycle)"
            )
    
            user_prompt = INSPEQTOR_EVOLVE_PROMPT.format(
                current_prompt=briq.prompt,
                style_hints=getattr(briq, 'style_hints', '') or "(none provided)",
                motion_prompt=getattr(briq, 'motion_prompt', '') or "(none provided)",
                negative_prompt=getattr(briq, 'negative_prompt', '') or "(none provided)",
                cycle_index=briq.cycle_index,
                target_cycles=self.config.get("cycle", {}).get("max_cycles", 100),
                previous_suggestions=suggestions_text,
            )
    
            response = call_llm(
                self.llm_client,
                system_prompt=INSPEQTOR_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                config=self.config,
            )
    
            data = json.loads(response)
    
            suggestion = data.get("evolution_suggestion", "")
            if self.mode == "innovative":
                innovative = data.get("innovative_idea", "")
                if innovative:
                    suggestion = f"{suggestion}; {innovative}"
    
            return suggestion or self._basic_suggestion(briq)
    
        except Exception as e:
            logger.warning("InspeQtor LLM evolution failed, falling back to basic suggestion: %s", e)
            return self._basic_suggestion(briq)

    def get_cycle_summary(self, briq: VisualBriq) -> Dict[str, Any]:
        """Generate summary of completed cycle."""
        return {
            "cycle_index": briq.cycle_index,
            "briq_id": briq.briq_id,
            "prompt": briq.prompt[:100] + "..." if len(briq.prompt) > 100 else briq.prompt,
            "status": briq.status.value,
            "generation_time": briq.generation_time,
            "raw_video": str(briq.raw_video_path),
            "looped_video": str(briq.looped_video_path),
            "evolution_suggestion": briq.evolution_suggestion,
            "mode": briq.mode.value,
            "seed": briq.seed,
        }


__all__ = ["InspeQtor", "INSPEQTOR_SYSTEM_PROMPT"]
