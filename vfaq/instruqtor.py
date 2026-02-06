#!/usr/bin/env python3
"""
instruqtor.py - Visual Instruction Agent
═══════════════════════════════════════════════════════════════════════════════

InstruQtor is the first agent in the Visual FaQtory pipeline.

Responsibilities:
  1. Load Prompt Bundle (tasq.md + negative_prompt.md + style_hints.md
     + motion_prompt.md) using the PromptBundle loader
  2. Analyze input mode (text/image/video)
  3. Create structured VisualBriq with refined prompt and specs
  4. For cycle N>0: Use previous cycle's video as base input
  5. ENFORCE strict separation: creative files = creative, config.yaml = mechanical

STRICT SEPARATION (v0.0.7-alpha):
  tasq.md / style_hints.md / motion_prompt.md MAY contain: creative intent only
  config.yaml controls ALL mechanical/technical parameters.

Prompt Bundle (v0.0.7-alpha):
  - tasq.md                base creative prompt (required)
  - negative_prompt.md     negative prompt source of truth (optional)
  - style_hints.md         style + evolution constraints (optional)
  - motion_prompt.md       video motion intent (optional)

LLM output fields (v0.0.7-alpha):
  - refined_prompt         optimized image/video prompt
  - negative_prompt        optional LLM-refined negative
  - style_tags             extracted style keywords
  - motion_hint            short motion guidance for video
  - video_prompt           optional dedicated video prompt

Part of QonQrete Visual FaQtory v0.0.7-alpha
"""
import os
import re
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

from .visual_briq import (
    VisualBriq, GenerationSpec, InputMode, BriqStatus,
    generate_briq_id, CycleState
)
from .prompt_bundle import PromptBundle, load_prompt_bundle

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
- Style constraints provided in STYLE_HINTS
- Motion direction provided in MOTION_PROMPT
- What should be excluded via NEGATIVE_PROMPT

When given a raw prompt and its supporting context, you:
1. Keep the core creative intent
2. Add appropriate quality boosters
3. Structure for best SD results
4. Respect STYLE_HINTS for visual style constraints and evolution direction
5. Incorporate MOTION_PROMPT intent into a short motion_hint
6. Suggest a video_prompt that combines visual + motion intent

For cycle N>0 (evolution mode):
- Make SUBTLE variations, not dramatic changes
- Evolve colors, lighting, or atmosphere gradually
- Maintain visual continuity with previous cycles

Output format (JSON):
{
    "refined_prompt": "the optimized image prompt",
    "quality_tags": ["masterpiece", "best quality", ...],
    "style_tags": ["cinematic", "moody", ...],
    "negative_prompt": "things to avoid (optional, leave empty to use user's)",
    "motion_hint": "short motion guidance for video stage",
    "video_prompt": "optional dedicated video prompt combining visual + motion",
    "reasoning": "brief explanation of choices"
}"""

INSTRUQTOR_REFINE_PROMPT = """Refine this raw visual prompt for Stable Diffusion + Video generation:

RAW PROMPT:
{raw_prompt}

STYLE_HINTS:
{style_hints}

MOTION_PROMPT:
{motion_prompt}

NEGATIVE_PROMPT_SOURCE:
{negative_prompt}

MODE: {mode}
CYCLE: {cycle_index}
{previous_context}

Create an optimized prompt that will generate stunning visuals.
Respect the STYLE_HINTS for visual direction and the MOTION_PROMPT for movement intent.
Respond with JSON only."""

INSTRUQTOR_EVOLVE_PROMPT = """Subtly evolve this prompt for the next visual cycle:

CURRENT PROMPT:
{current_prompt}

EVOLUTION SUGGESTION FROM INSPECTOR:
{evolution_suggestion}

STYLE_HINTS:
{style_hints}

MOTION_PROMPT:
{motion_prompt}

NEGATIVE_PROMPT_SOURCE:
{negative_prompt}

CYCLE: {cycle_index}

Make a SUBTLE variation - we want smooth visual evolution, not jarring changes.
Keep 80% similar, evolve 20%. Stay within the boundaries of STYLE_HINTS.

Respond with JSON only."""


class InstruQtor:
    """
    The instruction agent that prepares VisualBriqs for generation.

    Uses PromptBundle to load creative files, and LLM to refine prompts.
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

        # Cache the prompt bundle (loaded once, reused per cycle)
        self._bundle: Optional[PromptBundle] = None

    def _load_bundle(self) -> PromptBundle:
        """Load prompt bundle from worqspace files (cached after first load)."""
        if self._bundle is None:
            self._bundle = load_prompt_bundle(self.worqspace_dir, self.config)
            logger.info(
                f"[InstruQtor] PromptBundle loaded: "
                f"prompt={len(self._bundle.raw_prompt)}ch, "
                f"negative={self._bundle._negative_source}, "
                f"style={self._bundle._style_source}, "
                f"motion={self._bundle._motion_source}"
            )
        return self._bundle

    def parse_tasq(self) -> Tuple[str, InputMode, Optional[Path], Dict[str, Any]]:
        """
        Parse tasq.md file and extract CREATIVE INTENT only.
        Now delegates to PromptBundle loader for unified loading.

        Enforces strict separation: mechanical parameters in tasq.md trigger warnings.

        Returns:
            (prompt, mode, input_image_path, creative_overrides)
        """
        bundle = self._load_bundle()

        # Convert mode string to InputMode
        try:
            mode = InputMode(bundle.mode)
        except ValueError:
            logger.warning(f"Unknown mode '{bundle.mode}', defaulting to text")
            mode = InputMode.TEXT

        # Collect ONLY creative overrides (filtered)
        creative_overrides = {}
        if bundle.seed is not None:
            creative_overrides['seed'] = bundle.seed
        if bundle.title:
            creative_overrides['title'] = bundle.title
        if bundle.backend_override:
            creative_overrides['backend'] = bundle.backend_override
        if bundle.drift_preset:
            creative_overrides['drift_preset'] = bundle.drift_preset

        return bundle.raw_prompt, mode, bundle.input_image, creative_overrides

    def create_briq(
        self,
        cycle_index: int,
        previous_briq: Optional[VisualBriq] = None,
        evolution_suggestion: Optional[str] = None
    ) -> VisualBriq:
        """
        Create a VisualBriq for the given cycle.

        Mode handling (v0.0.7-alpha):
          Cycle 0:
            - text mode: text → image → video
            - image mode: requires input_image, skips image gen, feeds into video
          Cycle N (N>0):
            - video mode: video → video (variation + evolution)
            - The pipeline NEVER hard-resets visual identity unless explicitly instructed.

        Prompt Bundle integration (v0.0.7-alpha):
          - style_hints.md + motion_prompt.md + negative_prompt.md are loaded
          - LLM receives full bundle context
          - Briq stores all bundle fields for auditability

        Args:
            cycle_index: Current cycle number (0-indexed)
            previous_briq: The briq from previous cycle (for evolution)
            evolution_suggestion: InspeQtor's suggestion for next cycle

        Returns:
            VisualBriq ready for ConstruQtor
        """
        logger.info(f"[InstruQtor] Creating briq for cycle {cycle_index}")

        # Load prompt bundle
        bundle = self._load_bundle()

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
        llm_result = None

        if cycle_index == 0:
            # === CYCLE 0 ===
            if mode == InputMode.TEXT:
                llm_result = self._refine_prompt_bundle(raw_prompt, mode, cycle_index, bundle)
            elif mode == InputMode.IMAGE:
                if not input_image or not input_image.exists():
                    raise ValueError(
                        f"[InstruQtor] Image mode requires a valid input_image. "
                        f"Got: {input_image}. Set 'input_image' or 'base_image' in tasq.md."
                    )
                base_image = input_image
                llm_result = self._refine_prompt_bundle(raw_prompt, mode, cycle_index, bundle)
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

                llm_result = self._evolve_prompt_bundle(
                    previous_briq.prompt,
                    evolution_suggestion or "",
                    cycle_index,
                    bundle
                )
            else:
                # Fallback: no previous briq available, re-derive from tasq
                logger.warning(
                    f"[InstruQtor] Cycle {cycle_index} has no previous briq. "
                    f"Falling back to tasq.md prompt (this may reset visual identity)."
                )
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
                llm_result = self._refine_prompt_bundle(raw_prompt, mode, cycle_index, bundle)

        # Extract values from LLM result dict
        prompt = llm_result.get('refined_prompt', raw_prompt)
        style_tags = llm_result.get('style_tags', [])
        motion_hint = llm_result.get('motion_hint', '')
        llm_video_prompt = llm_result.get('video_prompt', '')
        llm_negative = llm_result.get('negative_prompt', '')

        # Negative prompt precedence:
        #   1) bundle.negative_prompt (already resolved from file/frontmatter/config)
        #   2) LLM-returned negative (only if non-empty AND source was default)
        #   3) config default
        negative = bundle.negative_prompt
        if llm_negative and bundle._negative_source == "config_default":
            negative = llm_negative

        # Create the briq
        briq = VisualBriq(
            briq_id=briq_id,
            cycle_index=cycle_index,
            mode=mode,
            prompt=prompt,
            negative_prompt=negative,
            quality_tags=self.default_quality_tags.copy(),
            style_tags=style_tags if style_tags else [],
            seed=seed,
            base_image_path=base_image,
            base_video_path=base_video,
            spec=spec,
            status=BriqStatus.PENDING,
            # Prompt bundle fields (v0.0.7)
            style_hints=bundle.style_hints,
            motion_prompt=bundle.motion_prompt,
            motion_hint=motion_hint,
        )

        # Finalize video_prompt after briq creation (so get_full_prompt() works)
        if llm_video_prompt:
            briq.video_prompt = llm_video_prompt
        elif motion_hint:
            briq.video_prompt = f"{briq.get_full_prompt()}, {motion_hint}"
        else:
            briq.video_prompt = briq.get_full_prompt()

        logger.info(f"[InstruQtor] Created briq {briq_id} mode={mode.value}")
        logger.info(f"  → prompt: {prompt[:80]}...")
        logger.info(f"  → style_hints: {len(bundle.style_hints)} chars loaded")
        logger.info(f"  → motion_prompt: {len(bundle.motion_prompt)} chars loaded")
        logger.info(f"  → motion_hint: {motion_hint[:60] if motion_hint else '(none)'}")
        logger.info(f"  → video_prompt: {briq.video_prompt[:80]}...")
        return briq

    # ═══════════════════════════════════════════════════════════════════════════
    # Prompt refinement (with bundle context)
    # ═══════════════════════════════════════════════════════════════════════════

    def _refine_prompt_bundle(
        self, raw_prompt: str, mode: InputMode,
        cycle_index: int, bundle: PromptBundle
    ) -> Dict[str, Any]:
        """Refine raw prompt using LLM (with full bundle context).
        Returns dict with refined_prompt, style_tags, motion_hint, video_prompt, etc."""
        if self.llm:
            return self._refine_with_llm_bundle(raw_prompt, mode, cycle_index, bundle)
        return self._basic_refine_bundle(raw_prompt, bundle)

    def _evolve_prompt_bundle(
        self, current_prompt: str, suggestion: str,
        cycle_index: int, bundle: PromptBundle
    ) -> Dict[str, Any]:
        """Evolve prompt for next cycle using LLM (with full bundle context).
        Returns dict with refined_prompt, style_tags, motion_hint, video_prompt, etc."""
        if self.llm:
            return self._evolve_with_llm_bundle(current_prompt, suggestion, cycle_index, bundle)
        return self._basic_evolve_bundle(current_prompt, cycle_index, bundle)

    def _basic_refine_bundle(self, prompt: str, bundle: PromptBundle) -> Dict[str, Any]:
        """Basic prompt cleanup without LLM — deterministic fallback."""
        prompt = ' '.join(prompt.split())
        for tag in self.default_quality_tags:
            prompt = prompt.replace(f", {tag}", "").replace(f"{tag}, ", "")
        return {
            'refined_prompt': prompt.strip(),
            'style_tags': [],
            'motion_hint': '',
            'video_prompt': '',
            'negative_prompt': '',
        }

    def _basic_evolve_bundle(self, prompt: str, cycle_index: int, bundle: PromptBundle) -> Dict[str, Any]:
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
        return {
            'refined_prompt': f"{prompt}, {color}, {mood}",
            'style_tags': [],
            'motion_hint': '',
            'video_prompt': '',
            'negative_prompt': '',
        }

    def _refine_with_llm_bundle(
        self, raw_prompt: str, mode: InputMode,
        cycle_index: int, bundle: PromptBundle
    ) -> Dict[str, Any]:
        """Refine prompt using LLM with full bundle context."""
        try:
            from .llm_utils import call_llm as llm_call
            user_prompt = INSTRUQTOR_REFINE_PROMPT.format(
                raw_prompt=raw_prompt,
                style_hints=bundle.style_hints or "(none provided)",
                motion_prompt=bundle.motion_prompt or "(none provided)",
                negative_prompt=bundle.negative_prompt or "(none provided)",
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
            data = json.loads(response)
            data.setdefault('refined_prompt', raw_prompt)
            data.setdefault('style_tags', [])
            data.setdefault('motion_hint', '')
            data.setdefault('video_prompt', '')
            data.setdefault('negative_prompt', '')
            return data
        except Exception as e:
            logger.warning(f"LLM refinement failed: {e}, using basic")
            return self._basic_refine_bundle(raw_prompt, bundle)

    def _evolve_with_llm_bundle(
        self, current_prompt: str, suggestion: str,
        cycle_index: int, bundle: PromptBundle
    ) -> Dict[str, Any]:
        """Evolve prompt using LLM with full bundle context."""
        try:
            from .llm_utils import call_llm as llm_call
            user_prompt = INSTRUQTOR_EVOLVE_PROMPT.format(
                current_prompt=current_prompt,
                evolution_suggestion=suggestion or "Continue evolving naturally",
                style_hints=bundle.style_hints or "(none provided)",
                motion_prompt=bundle.motion_prompt or "(none provided)",
                negative_prompt=bundle.negative_prompt or "(none provided)",
                cycle_index=cycle_index
            )
            response = llm_call(
                self.llm,
                system_prompt=INSTRUQTOR_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                config=self.config
            )
            data = json.loads(response)
            data.setdefault('refined_prompt', current_prompt)
            data.setdefault('style_tags', [])
            data.setdefault('motion_hint', '')
            data.setdefault('video_prompt', '')
            data.setdefault('negative_prompt', '')
            return data
        except Exception as e:
            logger.warning(f"LLM evolution failed: {e}, using basic")
            return self._basic_evolve_bundle(current_prompt, cycle_index, bundle)


__all__ = ['InstruQtor', 'INSTRUQTOR_SYSTEM_PROMPT']
