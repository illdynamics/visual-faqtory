#!/usr/bin/env python3
"""
prompt_bundle.py - Prompt Bundle Loader
═══════════════════════════════════════════════════════════════════════════════

Loads the full creative prompt bundle from worqspace files:
  - tasq.md              (base creative prompt — existing)
  - negative_prompt.md   (negative prompt source of truth — NEW)
  - style_hints.md       (style + evolution constraints — NEW)
  - motion_prompt.md     (video motion intent — NEW)

Backward-compatible: if new files are missing, falls back to existing
behavior (tasq.md body, config defaults, etc.).

Precedence for negative_prompt:
  1. tasq.md frontmatter `negative_prompt:` (power-user override)
  2. negative_prompt.md file content
  3. Negative section inside tasq.md body (## Negative)
  4. config.yaml prompt_drift.negative_prompt

Part of QonQrete Visual FaQtory v0.0.7-alpha
"""
import re
import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Forbidden mechanical keys in tasq/style/motion files
FORBIDDEN_CREATIVE_KEYS = {
    'fps', 'duration', 'resolution', 'width', 'height',
    'video_frames', 'clip_seconds', 'steps', 'cfg_scale',
    'sampler', 'motion_bucket_id', 'noise_aug_strength',
    'video_fps', 'output_fps', 'output_codec', 'output_quality',
    'denoise_strength', 'crossfade_frames'
}


@dataclass
class PromptBundle:
    """
    Complete creative prompt bundle loaded from worqspace files.
    Contains everything the InstruQtor needs to create a VisualBriq.
    """
    raw_prompt: str = ""
    negative_prompt: str = ""
    style_hints: str = ""
    motion_prompt: str = ""
    title: Optional[str] = None
    backend_override: Optional[str] = None
    mode: str = "text"
    input_image: Optional[Path] = None
    seed: Optional[int] = None
    drift_preset: Optional[str] = None
    # Track sources for debugging
    _negative_source: str = "default"
    _style_source: str = "none"
    _motion_source: str = "none"


def load_prompt_bundle(
    worqspace_dir: Path,
    config: Dict[str, Any]
) -> PromptBundle:
    """
    Load the full prompt bundle from worqspace files.

    Reads:
      - tasq.md (always required)
      - negative_prompt.md (optional)
      - style_hints.md (optional)
      - motion_prompt.md (optional)

    Args:
        worqspace_dir: Path to worqspace directory
        config: Full config dict

    Returns:
        PromptBundle with all creative data loaded
    """
    worqspace_dir = Path(worqspace_dir)
    input_config = config.get('input', {})
    drift_config = config.get('prompt_drift', {})

    bundle = PromptBundle()

    # ──────────────────────────────────────────────────────────────────────
    # 1. Load tasq.md (required — existing behavior)
    # ──────────────────────────────────────────────────────────────────────
    tasq_file = input_config.get('tasq_file', 'tasq.md')
    tasq_path = worqspace_dir / tasq_file

    if not tasq_path.exists():
        raise FileNotFoundError(f"[PromptBundle] tasq.md not found at: {tasq_path}")

    content = tasq_path.read_text(encoding='utf-8')

    # Parse YAML frontmatter
    frontmatter = {}
    body = content

    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            try:
                frontmatter = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError as e:
                logger.warning(f"[PromptBundle] Failed to parse tasq.md frontmatter: {e}")
            body = parts[2]

    # Enforce mechanical separation on frontmatter
    found_forbidden = set(frontmatter.keys()) & FORBIDDEN_CREATIVE_KEYS
    if found_forbidden:
        logger.warning(
            f"[PromptBundle] tasq.md contains mechanical parameters that belong "
            f"in config.yaml: {found_forbidden}. These will be IGNORED."
        )

    # Extract fields from frontmatter
    bundle.mode = frontmatter.get('mode', 'text').lower()
    bundle.title = frontmatter.get('title')
    bundle.backend_override = frontmatter.get('backend')
    bundle.seed = frontmatter.get('seed')
    bundle.drift_preset = frontmatter.get('drift_preset')

    # Input image (for image mode)
    img_path_str = frontmatter.get('input_image') or frontmatter.get('base_image')
    if img_path_str:
        img_path = Path(img_path_str)
        if not img_path.is_absolute():
            img_path = (worqspace_dir / img_path).resolve()
        bundle.input_image = img_path

    # Extract main prompt from body
    bundle.raw_prompt = _extract_prompt(body)

    # Extract negative from tasq body (## Negative section)
    tasq_body_negative = _extract_section(body, 'negative')

    logger.info(f"[PromptBundle] Loaded tasq.md: mode={bundle.mode}, "
                f"prompt={len(bundle.raw_prompt)} chars")

    # ──────────────────────────────────────────────────────────────────────
    # 2. Load negative_prompt.md (optional)
    # ──────────────────────────────────────────────────────────────────────
    neg_file = input_config.get('negative_prompt_file', 'negative_prompt.md')
    neg_path = worqspace_dir / neg_file
    neg_from_file = ""

    if neg_path.exists():
        neg_from_file = neg_path.read_text(encoding='utf-8').strip()
        if neg_from_file:
            logger.info(f"[PromptBundle] Loaded {neg_file}: {len(neg_from_file)} chars")
        else:
            logger.info(f"[PromptBundle] {neg_file} exists but is empty")

    # Negative prompt precedence:
    #   1) tasq frontmatter negative_prompt (power-user override)
    #   2) negative_prompt.md file
    #   3) tasq body ## Negative section
    #   4) config prompt_drift.negative_prompt
    fm_negative = frontmatter.get('negative_prompt')

    if fm_negative:
        bundle.negative_prompt = fm_negative
        bundle._negative_source = "tasq_frontmatter"
    elif neg_from_file:
        bundle.negative_prompt = neg_from_file
        bundle._negative_source = f"file:{neg_file}"
    elif tasq_body_negative:
        bundle.negative_prompt = tasq_body_negative
        bundle._negative_source = "tasq_body_section"
    else:
        bundle.negative_prompt = drift_config.get(
            'negative_prompt',
            'blurry, low quality, watermark, text, deformed'
        )
        bundle._negative_source = "config_default"

    logger.info(f"[PromptBundle] Negative prompt source: {bundle._negative_source}")

    # ──────────────────────────────────────────────────────────────────────
    # 3. Load style_hints.md (optional)
    # ──────────────────────────────────────────────────────────────────────
    style_file = input_config.get('style_hints_file', 'style_hints.md')
    style_path = worqspace_dir / style_file

    if style_path.exists():
        bundle.style_hints = style_path.read_text(encoding='utf-8').strip()
        bundle._style_source = f"file:{style_file}"
        if bundle.style_hints:
            logger.info(f"[PromptBundle] Loaded {style_file}: {len(bundle.style_hints)} chars")
        else:
            logger.info(f"[PromptBundle] {style_file} exists but is empty")
    else:
        logger.info(f"[PromptBundle] No {style_file} found (optional, skipping)")

    # ──────────────────────────────────────────────────────────────────────
    # 4. Load motion_prompt.md (optional)
    # ──────────────────────────────────────────────────────────────────────
    motion_file = input_config.get('motion_prompt_file', 'motion_prompt.md')
    motion_path = worqspace_dir / motion_file

    if motion_path.exists():
        bundle.motion_prompt = motion_path.read_text(encoding='utf-8').strip()
        bundle._motion_source = f"file:{motion_file}"
        if bundle.motion_prompt:
            logger.info(f"[PromptBundle] Loaded {motion_file}: {len(bundle.motion_prompt)} chars")
        else:
            logger.info(f"[PromptBundle] {motion_file} exists but is empty")
    else:
        logger.info(f"[PromptBundle] No {motion_file} found (optional, skipping)")

    return bundle


# ═══════════════════════════════════════════════════════════════════════════════
# Text extraction helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_prompt(body: str) -> str:
    """Extract main prompt from markdown body (skip code blocks and sections)."""
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


def _extract_section(body: str, section_name: str) -> Optional[str]:
    """Extract content from a named markdown section."""
    pattern = rf'^#{1,2}\s*{section_name}[^\n]*\n(.*?)(?=^#{1,2}\s|\Z)'
    match = re.search(pattern, body, re.IGNORECASE | re.MULTILINE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


__all__ = ['PromptBundle', 'load_prompt_bundle']
