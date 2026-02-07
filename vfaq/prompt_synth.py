#!/usr/bin/env python3
"""
prompt_synth.py - Deterministic Prompt Synthesis (NO LLM Required)
═══════════════════════════════════════════════════════════════════════════════

Synthesizes prompts from base prompt + style_hints + evolution_lines
WITHOUT any LLM dependency. Fully deterministic: same cycle index = same prompt.

Used when llm.enabled == false OR no LLM provider is configured.

Rules:
  1. base_prompt always comes first (tasq.md body)
  2. style_hints are ALWAYS appended (verbatim)
  3. evolution_lines are deterministically selected based on cycle_index
  4. negative_prompt.md is applied verbatim to generation requests
  5. motion_prompt.md is stored and injected into video workflows

Part of QonQrete Visual FaQtory v0.3.5-beta
"""
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT EVOLUTION LINES (used when evolution_lines.md is missing)
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_EVOLUTION_LINES = [
    # Material & Surface
    "Increase surface oxidation and verdigris on all brass components",
    "Introduce fresh weld seams and glowing heat scars",
    "Add micro-scratches, dents, and battle-worn paint chips",
    "Shift metal balance toward brushed steel and blackened iron",
    "Add oily reflections and wet specular highlights",
    "Introduce cracked enamel paint with hazard underlayers",
    # Biomech
    "Tentacle cables thicken and gain suction-cup rivets",
    "Add translucent bio-tubing pulsing with neon fluid",
    "Introduce subtle organic skin texture over machine joints",
    "Add ink-like leaks forming circuit glyphs",
    "Increase biomechanical asymmetry",
    "Add dormant squid-eye lenses embedded in walls",
    # Energy & Reactor
    "Reactor glow shifts from gold to cyan-white plasma",
    "Add unstable energy arcs jumping between rings",
    "Increase light pulse frequency and intensity",
    "Introduce heat distortion near the core",
    "Add emergency containment clamps",
    "Reactor rings rotate at mismatched speeds",
    # UI / Control Room
    "Add more floating diagnostic panels",
    "Increase CRT flicker and scanline density",
    "Introduce corrupted UI symbols and warning icons",
    "Add holographic schematics of squid anatomy",
    "Overlay matrix-style data rain",
    "UI color palette shifts toward toxic green",
    # Atmosphere & FX
    "Increase volumetric fog density",
    "Add drifting embers and sparks",
    "Introduce smoke jets from floor vents",
    "Add dust motes illuminated by god rays",
    "Increase chromatic aberration",
    "Add controlled datamosh blocks at frame edges",
    # Color & Style
    "Reduce saturation except for reactor core",
    "Push palette toward cold cyan and steel",
    "Introduce radioactive yellow accents",
    "Increase contrast and crush blacks",
    "Add posterized color transitions",
    "Enhance comic-etched linework",
    # Composition-Preserving
    "Maintain silhouette but alter internal details",
    "Add layered depth through parallax cues",
    "Increase foreground rim lighting",
    "Darken background to emphasize core",
    "Add subtle vignette",
    "Shift camera micro-angle slightly upward",
]


def load_evolution_lines(
    worqspace_dir: Path,
    config: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Load evolution lines from worqspace/evolution_lines.md.
    Falls back to built-in defaults if file is missing.

    Lines starting with '#' or '-' prefix are cleaned.
    Empty lines and pure-comment lines are skipped.

    Args:
        worqspace_dir: Path to worqspace directory
        config: Optional config dict (for custom filename)

    Returns:
        List of evolution line strings (non-empty, cleaned)
    """
    filename = "evolution_lines.md"
    if config:
        filename = config.get("input", {}).get("evolution_lines_file", filename)

    evo_path = Path(worqspace_dir) / filename

    if evo_path.exists():
        raw = evo_path.read_text(encoding="utf-8")
        lines = _parse_evolution_lines(raw)
        if lines:
            logger.info(
                f"[PromptSynth] Loaded {len(lines)} evolution lines from {filename}"
            )
            return lines
        logger.warning(
            f"[PromptSynth] {filename} exists but contains no valid lines, "
            f"using {len(DEFAULT_EVOLUTION_LINES)} built-in defaults"
        )
    else:
        logger.info(
            f"[PromptSynth] No {filename} found, "
            f"using {len(DEFAULT_EVOLUTION_LINES)} built-in defaults"
        )

    return DEFAULT_EVOLUTION_LINES.copy()


def _parse_evolution_lines(raw_text: str) -> List[str]:
    """Parse evolution lines from markdown text.

    Handles:
      - Lines starting with '- ' (bullet points)
      - Lines with [tag] prefixes (e.g. '[material] ...')
      - Plain text lines
      - Skips headers (#), empty lines, and horizontal rules (---)
    """
    lines = []
    for raw_line in raw_text.split("\n"):
        line = raw_line.strip()

        # Skip empty, headers, horizontal rules, and pure comments
        if not line:
            continue
        if line.startswith("#"):
            continue
        if line.startswith("---"):
            continue
        if line.startswith("```"):
            continue

        # Strip bullet prefix
        if line.startswith("- "):
            line = line[2:].strip()
        elif line.startswith("* "):
            line = line[2:].strip()

        # Keep [tag] prefixes — they're useful for thematic grouping
        if line:
            lines.append(line)

    return lines


def synthesize_prompt(
    base_prompt: str,
    style_hints: str,
    evolution_lines: List[str],
    cycle_index: int,
    max_mutations: int = 2,
) -> str:
    """
    Deterministic prompt synthesis — NO LLM, NO randomness.

    Same cycle_index + same inputs = identical output. Always.

    Args:
        base_prompt:     Raw creative prompt from tasq.md body
        style_hints:     Full content of style_hints.md (appended verbatim)
        evolution_lines: List of mutation strings
        cycle_index:     Current generation cycle (0-indexed)
        max_mutations:   Max evolution lines to select per cycle

    Returns:
        Synthesized prompt string ready for backend injection
    """
    parts = []

    # 1. Base prompt ALWAYS first
    if base_prompt.strip():
        parts.append(base_prompt.strip())

    # 2. Style hints ALWAYS appended (verbatim, trusted operator input)
    if style_hints and style_hints.strip():
        parts.append(f"STYLE_HINTS: {style_hints.strip()}")

    # 3. Evolution mutations (deterministic selection)
    if evolution_lines:
        mutations = select_evolution_mutations(
            evolution_lines, cycle_index, max_mutations
        )
        if mutations:
            mutation_text = "\n".join(f"- {m}" for m in mutations)
            parts.append(f"EVOLUTION_MUTATIONS:\n{mutation_text}")

    return "\n".join(parts)


def synthesize_video_prompt(
    base_prompt: str,
    style_hints: str,
    motion_prompt: str,
    evolution_lines: List[str],
    cycle_index: int,
    max_mutations: int = 2,
) -> str:
    """
    Deterministic video prompt synthesis — appends motion_prompt at the END.

    For backends that support text conditioning on video generation.

    Args:
        base_prompt:     Raw creative prompt
        style_hints:     Style hints content
        motion_prompt:   Motion intent from motion_prompt.md
        evolution_lines: Evolution mutation strings
        cycle_index:     Current cycle
        max_mutations:   Max mutations per cycle

    Returns:
        Video prompt string with motion_prompt appended
    """
    # Start with image prompt synthesis
    image_prompt = synthesize_prompt(
        base_prompt, style_hints, evolution_lines, cycle_index, max_mutations
    )

    # Append motion prompt at the END (for video-specific conditioning)
    if motion_prompt and motion_prompt.strip():
        return f"{image_prompt}\nMOTION: {motion_prompt.strip()}"

    return image_prompt


def select_evolution_mutations(
    evolution_lines: List[str],
    cycle_index: int,
    max_mutations: int = 2,
) -> List[str]:
    """
    Deterministically select N evolution lines based on cycle_index.

    Selection formula: index = (cycle_index * 7 + i * 13) % len(lines)
    This creates a pseudo-random but fully reproducible sequence.

    Args:
        evolution_lines: Available mutation strings
        cycle_index:     Current cycle
        max_mutations:   Maximum mutations to select

    Returns:
        List of selected mutation strings (deduplicated)
    """
    if not evolution_lines:
        return []

    n = min(max_mutations, len(evolution_lines))
    selected = []
    seen_indices = set()

    for i in range(n):
        idx = (cycle_index * 7 + i * 13) % len(evolution_lines)
        # Avoid duplicates within same cycle
        attempt = 0
        while idx in seen_indices and attempt < len(evolution_lines):
            idx = (idx + 1) % len(evolution_lines)
            attempt += 1
        if idx not in seen_indices:
            seen_indices.add(idx)
            selected.append(evolution_lines[idx])

    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# MOTION BUCKET ID MAPPING (for SVD backends that don't support text)
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_MOTION_KEYWORD_MAP = {
    "slow": 60,
    "slow cinematic": 70,
    "slow cinematic push": 80,
    "gentle": 80,
    "moderate": 110,
    "medium": 127,
    "dynamic": 150,
    "fast": 160,
    "chaotic": 175,
    "chaotic glitch": 180,
    "extreme": 200,
    "static": 40,
    "still": 30,
}


def map_motion_to_bucket_id(
    motion_prompt: str,
    keyword_map: Optional[Dict[str, int]] = None,
    default_bucket_id: int = 127,
) -> int:
    """
    Map motion_prompt keywords to motion_bucket_id for SVD backends.

    Scans motion_prompt for known keywords and returns the matching
    bucket ID. Longer keyword matches take priority (most specific).

    Args:
        motion_prompt:     Motion intent text
        keyword_map:       Custom keyword → bucket_id mapping (optional)
        default_bucket_id: Fallback if no keywords match

    Returns:
        Integer motion_bucket_id value
    """
    if not motion_prompt:
        return default_bucket_id

    mapping = keyword_map or DEFAULT_MOTION_KEYWORD_MAP
    prompt_lower = motion_prompt.lower()

    # Sort by key length descending (longest match wins)
    best_match = None
    best_len = 0

    for keyword, bucket_id in mapping.items():
        if keyword.lower() in prompt_lower and len(keyword) > best_len:
            best_match = bucket_id
            best_len = len(keyword)

    if best_match is not None:
        logger.info(
            f"[PromptSynth] Motion keyword mapped to motion_bucket_id={best_match}"
        )
        return best_match

    return default_bucket_id


__all__ = [
    "synthesize_prompt",
    "synthesize_video_prompt",
    "select_evolution_mutations",
    "load_evolution_lines",
    "map_motion_to_bucket_id",
    "DEFAULT_EVOLUTION_LINES",
    "DEFAULT_MOTION_KEYWORD_MAP",
]
