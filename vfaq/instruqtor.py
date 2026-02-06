#!/usr/bin/env python3
"""
instruqtor.py - Visual Instruction Agent
═══════════════════════════════════════════════════════════════════════════════

InstruQtor is the first agent in the Visual FaQtory pipeline.

Responsibilities:
  1. Parse tasq.md and extract CREATIVE INTENT only
  2. Analyze input mode (text/image/video)
  3. Create structured VisualBriq with refined prompt and specs
  4. For cycle N>0: Use previous cycle's video as base input
  5. ENFORCE strict separation: tasq.md = creative, config.yaml = mechanical

STRICT SEPARATION (v0.0.5-alpha):
  tasq.md MAY contain: title, mode, backend, input_image, descriptive prompt text
  tasq.md MUST NOT control: fps, duration, frame counts, resolution, diffusion steps
  config.yaml controls ALL mechanical/technical parameters.

Part of QonQrete Visual FaQtory v0.0.5-alpha
"""
import os
import re
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

from .visual_briq import (
    VisualBriq, GenerationSpec, InputMode, BriqStatus,
    generate_briq_id, CycleState
)

logger = logging.getLogger(__name__)

# Forbidden keys in tasq.md frontmatter - these belong in config.yaml ONLY
FORBIDDEN_TASQ_KEYS = {
    'fps', 'duration', 'resolution', 'width', 'height',
    'video_frames', 'clip_seconds', 'steps', 'cfg_scale',
    'sampler', 'motion_bucket_id', 'noise_aug_strength',
    'video_fps', 'output_fps', 'output_codec', 'output_quality',
    'denoise_strength', 'crossfade_frames'
}


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES FOR LLM
# ═══════════════════════════════════════════════════════════════════════════════

INSTRUQTOR_SYSTEM_PROMPT = """You are InstruQtor, a visual generation instruction specialist for the QonQrete Visual FaQtory.

Your job is to transform raw creative prompts into optimized Stable Diffusion / AI video prompts.

You understand:
- SD/SDXL prompt syntax and weighting
- Quality tags that improve generation (masterpiece, highly detailed, etc.)
- Style keywords that work well for video generation
- How to balance creativity with technical effectiveness

When given a raw prompt, you:
1. Keep the core creative intent
2. Add appropriate quality boosters
3. Structure for best SD results
4. Suggest motion-friendly compositions for video

For cycle N>0 (evolution mode):
- Make SUBTLE variations, not dramatic changes
- Evolve colors, lighting, or atmosphere gradually
- Maintain visual continuity with previous cycles

Output format (JSON):
{
    "refined_prompt": "the optimized prompt",
    "quality_tags": ["masterpiece", "best quality", ...],
    "style_tags": ["cinematic", "moody", ...],
    "negative_prompt": "things to avoid",
    "motion_hint": "suggestion for video motion",
    "reasoning": "brief explanation of choices"
}"""

INSTRUQTOR_REFINE_PROMPT = """Refine this raw visual prompt for Stable Diffusion + Video generation:

RAW PROMPT:
{raw_prompt}

MODE: {mode}
CYCLE: {cycle_index}
{previous_context}

Create an optimized prompt that will generate stunning visuals.
Respond with JSON only."""

INSTRUQTOR_EVOLVE_PROMPT = """Subtly evolve this prompt for the next visual cycle:

CURRENT PROMPT:
{current_prompt}

EVOLUTION SUGGESTION FROM INSPECTOR:
{evolution_suggestion}

CYCLE: {cycle_index}

Make a SUBTLE variation - we want smooth visual evolution, not jarring changes.
Keep 80% similar, evolve 20%.

Respond with JSON only."""


class InstruQtor:
    """
    The instruction agent that prepares VisualBriqs for generation.

    Uses LLM to refine prompts and create optimal generation specs.
    Enforces strict config/tasq separation.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        worqspace_dir: Path,
        qodeyard_dir: Path,
        llm_provider: Optional[Any] = None
    ):
        self.config = config
        self.worqspace_dir = Path(worqspace_dir)
        self.qodeyard_dir = Path(qodeyard_dir)
        self.llm = None
        if llm_provider:
            from .llm_utils import create_llm_client
            # visual_faqtory passes a dict; turn it into a real client
            self.llm = create_llm_client(llm_provider) if isinstance(llm_provider, dict) else llm_provider

        # Load generation config from config.yaml ONLY
        gen_config = config.get('generation', {})
        self.default_spec = GenerationSpec(
            width=gen_config.get('width', 1024),
            height=gen_config.get('height', 576),
            cfg_scale=gen_config.get('cfg_scale', 7.0),
            steps=gen_config.get('steps', 30),
            sampler=gen_config.get('sampler', 'euler_ancestral'),
            video_frames=gen_config.get('video_frames', 25),
            video_fps=gen_config.get('video_fps', 8),
            clip_seconds=gen_config.get('clip_seconds', 8.0),
            motion_bucket_id=gen_config.get('motion_bucket_id', 127),
            noise_aug_strength=gen_config.get('noise_aug_strength', 0.02),
            denoise_strength=config.get('chaining', {}).get('denoise_strength', 0.4)
        )

        # Quality/style defaults from config
        drift_config = config.get('prompt_drift', {})
        self.default_quality_tags = drift_config.get('quality_tags', [
            'masterpiece', 'best quality', 'highly detailed'
        ])
        self.default_negative = drift_config.get('negative_prompt',
            'blurry, low quality, watermark, text, deformed'
        )

    def parse_tasq(self) -> Tuple[str, InputMode, Optional[Path], Dict[str, Any]]:
        """
        Parse tasq.md file and extract CREATIVE INTENT only.

        Enforces strict separation: mechanical parameters in tasq.md trigger warnings.

        Returns:
            (prompt, mode, input_image_path, creative_overrides)
        """
        tasq_file = self.config.get('input', {}).get('tasq_file', 'tasq.md')
        tasq_path = self.worqspace_dir / tasq_file

        if not tasq_path.exists():
            raise FileNotFoundError(f"tasq.md not found at: {tasq_path}")

        content = tasq_path.read_text(encoding='utf-8')

        # Parse YAML frontmatter if present
        frontmatter = {}
        body = content

        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                try:
                    frontmatter = yaml.safe_load(parts[1]) or {}
                except yaml.YAMLError as e:
                    logger.warning(f"Failed to parse frontmatter: {e}")
                body = parts[2]

        # ENFORCE strict separation: warn about forbidden keys
        found_forbidden = set(frontmatter.keys()) & FORBIDDEN_TASQ_KEYS
        if found_forbidden:
            logger.warning(
                f"[InstruQtor] tasq.md contains mechanical parameters that belong "
                f"in config.yaml: {found_forbidden}. These will be IGNORED. "
                f"Move them to worqspace/config.yaml under the appropriate section."
            )

        # Extract mode (allowed in tasq.md)
        mode_str = frontmatter.get('mode', 'text').lower()
        try:
            mode = InputMode(mode_str)
        except ValueError:
            logger.warning(f"Unknown mode '{mode_str}', defaulting to text")
            mode = InputMode.TEXT

        # Extract input_image (allowed in tasq.md for image mode)
        input_image = None
        if mode == InputMode.IMAGE:
            img_path = frontmatter.get('input_image') or frontmatter.get('base_image')
            if img_path:
                input_image = self._resolve_path(img_path)

        # Extract prompt from body
        prompt = self._extract_prompt(body)

        # Extract negative prompt if in body
        negative = self._extract_section(body, 'negative')

        # Collect ONLY creative overrides (filtered)
        creative_overrides = {
            'seed': frontmatter.get('seed'),
            'negative_prompt': negative or frontmatter.get('negative_prompt'),
            'title': frontmatter.get('title'),
            'backend': frontmatter.get('backend'),
            'drift_preset': frontmatter.get('drift_preset'),
        }
        creative_overrides = {k: v for k, v in creative_overrides.items() if v is not None}

        return prompt, mode, input_image, creative_overrides

    def _resolve_path(self, path_str: str) -> Path:
        """Resolve path relative to worqspace."""
        path = Path(path_str)
        if path.is_absolute():
            return path
        return (self.worqspace_dir / path).resolve()

    def _extract_prompt(self, body: str) -> str:
        """Extract main prompt from markdown body."""
        lines = []
        in_code = False
        skip_section = False

        for line in body.split('\n'):
            if line.strip().startswith('```'):
                in_code = not in_code
                continue
            if in_code:
                continue
            if re.match(r'^#{1,2}\s*[Nn]egative', line):
                skip_section = True
                continue
            if skip_section and re.match(r'^#{1,2}\s', line):
                skip_section = False
            if skip_section:
                continue
            if line.strip().startswith('#'):
                continue
            lines.append(line)

        return '\n'.join(lines).strip()

    def _extract_section(self, body: str, section_name: str) -> Optional[str]:
        """Extract content from a named section."""
        pattern = rf'^#{1,2}\s*{section_name}[^\n]*\n(.*?)(?=^#{1,2}\s|\Z)'
        match = re.search(pattern, body, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def create_briq(
        self,
        cycle_index: int,
        previous_briq: Optional[VisualBriq] = None,
        evolution_suggestion: Optional[str] = None
    ) -> VisualBriq:
        """
        Create a VisualBriq for the given cycle.

        Mode handling (v0.0.5-alpha):
          Cycle 0:
            - text mode: text → image → video
            - image mode: requires input_image, skips image gen, feeds into video
          Cycle N (N>0):
            - video mode: video → video (variation + evolution)
            - The pipeline NEVER hard-resets visual identity unless explicitly instructed.

        Args:
            cycle_index: Current cycle number (0-indexed)
            previous_briq: The briq from previous cycle (for evolution)
            evolution_suggestion: InspeQtor's suggestion for next cycle

        Returns:
            VisualBriq ready for ConstruQtor
        """
        logger.info(f"[InstruQtor] Creating briq for cycle {cycle_index}")

        # Parse tasq for creative intent
        raw_prompt, mode, input_image, creative_overrides = self.parse_tasq()

        # Get seed
        seed = creative_overrides.get('seed', 42) + cycle_index

        # Generate briq ID
        briq_id = generate_briq_id(cycle_index, seed)

        # Spec comes ENTIRELY from config.yaml - never from tasq.md
        spec = GenerationSpec(
            width=self.default_spec.width,
            height=self.default_spec.height,
            cfg_scale=self.default_spec.cfg_scale,
            steps=self.default_spec.steps,
            sampler=self.default_spec.sampler,
            video_frames=self.default_spec.video_frames,
            video_fps=self.default_spec.video_fps,
            clip_seconds=self.default_spec.clip_seconds,
            motion_bucket_id=self.default_spec.motion_bucket_id,
            noise_aug_strength=self.default_spec.noise_aug_strength,
            denoise_strength=self.default_spec.denoise_strength
        )

        # Base paths
        base_image = None
        base_video = None

        # Determine prompt and mode for this cycle
        if cycle_index == 0:
            # === CYCLE 0 ===
            if mode == InputMode.TEXT:
                # text → image → video
                prompt = self._refine_prompt(raw_prompt, mode, cycle_index)
            elif mode == InputMode.IMAGE:
                # image → video (skip image gen)
                if not input_image or not input_image.exists():
                    raise ValueError(
                        f"[InstruQtor] Image mode requires a valid input_image. "
                        f"Got: {input_image}. Set 'input_image' or 'base_image' in tasq.md."
                    )
                base_image = input_image
                prompt = self._refine_prompt(raw_prompt, mode, cycle_index)
            elif mode == InputMode.VIDEO:
                raise ValueError(
                    "[InstruQtor] Video mode is only valid for cycle N>0 "
                    "(requires previous cycle output). Use 'text' or 'image' for cycle 0."
                )
            else:
                raise ValueError(f"[InstruQtor] Unknown mode: {mode}")
        else:
            # === CYCLE N>0: video → video evolution ===
            if previous_briq and previous_briq.looped_video_path:
                mode = InputMode.VIDEO
                base_video = previous_briq.looped_video_path
                base_image = None

                # Evolve the prompt without hard-resetting visual identity
                prompt = self._evolve_prompt(
                    previous_briq.prompt,
                    evolution_suggestion or "",
                    cycle_index
                )
            else:
                # Fallback: no previous briq available, re-derive from tasq
                logger.warning(
                    f"[InstruQtor] Cycle {cycle_index} has no previous briq. "
                    f"Falling back to tasq.md prompt (this may reset visual identity)."
                )
                # IMAGE mode fallback: still use input_image from tasq.md
                if mode == InputMode.IMAGE:
                    if input_image and input_image.exists():
                        base_image = input_image
                        logger.info(
                            f"[InstruQtor] IMAGE mode fallback: using input_image "
                            f"from tasq.md: {input_image}"
                        )
                    else:
                        raise ValueError(
                            f"[InstruQtor] IMAGE mode requires a valid input_image "
                            f"for fallback on cycle {cycle_index}. "
                            f"Got: {input_image}. Set 'input_image' in tasq.md."
                        )
                prompt = self._refine_prompt(raw_prompt, mode, cycle_index)

        # Get negative prompt
        negative = creative_overrides.get('negative_prompt', self.default_negative)

        # Create the briq
        briq = VisualBriq(
            briq_id=briq_id,
            cycle_index=cycle_index,
            mode=mode,
            prompt=prompt,
            negative_prompt=negative,
            quality_tags=self.default_quality_tags.copy(),
            style_tags=[],
            seed=seed,
            base_image_path=base_image,
            base_video_path=base_video,
            spec=spec,
            status=BriqStatus.PENDING
        )

        logger.info(f"[InstruQtor] Created briq {briq_id} mode={mode.value}")
        return briq

    def _refine_prompt(self, raw_prompt: str, mode: InputMode, cycle_index: int) -> str:
        """Refine raw prompt using LLM (or basic processing if no LLM)."""
        if self.llm:
            return self._refine_with_llm(raw_prompt, mode, cycle_index)
        return self._basic_refine(raw_prompt)

    def _evolve_prompt(self, current_prompt: str, suggestion: str, cycle_index: int) -> str:
        """Evolve prompt for next cycle using LLM (or basic delta if no LLM)."""
        if self.llm:
            return self._evolve_with_llm(current_prompt, suggestion, cycle_index)
        return self._basic_evolve(current_prompt, cycle_index)

    def _basic_refine(self, prompt: str) -> str:
        """Basic prompt cleanup without LLM."""
        prompt = ' '.join(prompt.split())
        for tag in self.default_quality_tags:
            prompt = prompt.replace(f", {tag}", "").replace(f"{tag}, ", "")
        return prompt.strip()

    def _basic_evolve(self, prompt: str, cycle_index: int) -> str:
        """Basic prompt evolution without LLM - subtle variations."""
        color_shifts = [
            "deep blue tones", "purple haze", "neon pink accents",
            "electric cyan", "golden hour light", "crimson highlights",
            "emerald glow", "amber warmth", "silver moonlight"
        ]
        mood_shifts = [
            "slightly more intense", "calmer atmosphere",
            "increased energy", "ethereal feeling", "grittier texture"
        ]
        color = color_shifts[cycle_index % len(color_shifts)]
        mood = mood_shifts[cycle_index % len(mood_shifts)]
        return f"{prompt}, {color}, {mood}"

    def _refine_with_llm(self, raw_prompt: str, mode: InputMode, cycle_index: int) -> str:
        """Refine prompt using LLM."""
        try:
            from .llm_utils import call_llm as llm_call
            user_prompt = INSTRUQTOR_REFINE_PROMPT.format(
                raw_prompt=raw_prompt,
                mode=mode.value,
                cycle_index=cycle_index,
                previous_context=""
            )
            response = llm_call(
                self.llm,
                system_prompt=INSTRUQTOR_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                config=self.config
            )
            import json
            data = json.loads(response)
            return data.get('refined_prompt', raw_prompt)
        except Exception as e:
            logger.warning(f"LLM refinement failed: {e}, using basic")
            return self._basic_refine(raw_prompt)

    def _evolve_with_llm(self, current_prompt: str, suggestion: str, cycle_index: int) -> str:
        """Evolve prompt using LLM."""
        try:
            from .llm_utils import call_llm as llm_call
            user_prompt = INSTRUQTOR_EVOLVE_PROMPT.format(
                current_prompt=current_prompt,
                evolution_suggestion=suggestion or "Continue evolving naturally",
                cycle_index=cycle_index
            )
            response = llm_call(
                self.llm,
                system_prompt=INSTRUQTOR_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                config=self.config
            )
            import json
            data = json.loads(response)
            return data.get('refined_prompt', current_prompt)
        except Exception as e:
            logger.warning(f"LLM evolution failed: {e}, using basic")
            return self._basic_evolve(current_prompt, cycle_index)


__all__ = ['InstruQtor', 'INSTRUQTOR_SYSTEM_PROMPT']
