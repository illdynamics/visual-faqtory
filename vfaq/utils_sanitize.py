#!/usr/bin/env python3
"""
utils_sanitize.py - Prompt Sanitization + Moderation
═══════════════════════════════════════════════════════════════════════════════

Stage-safe input filtering for crowd prompts:
  - Length enforcement
  - Character restriction
  - Banned word/regex filtering
  - Whitespace normalization
  - URL stripping

Part of QonQrete Visual FaQtory v0.3.5-beta
"""
import re
import logging
from typing import Tuple, Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Default minimal banned words (stage safety, not moral policing)
DEFAULT_BANNED_WORDS = ["nazi", "hitler", "kill", "murder", "porn", "nsfw"]


def sanitize_prompt(
    text: str,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, bool, str]:
    """
    Sanitize and moderate a crowd prompt.

    Args:
        text: Raw prompt text
        config: crowd.moderation config section

    Returns:
        (sanitized_text, accepted, rejection_reason)
    """
    cfg = config or {}
    max_len = cfg.get('max_len', 120)
    min_len = cfg.get('min_len', 3)
    allow_unicode = cfg.get('allow_unicode', False)
    banned_words = cfg.get('banned_words', DEFAULT_BANNED_WORDS)
    banned_regex = cfg.get('banned_regex', [])
    enabled = cfg.get('enabled', True)

    if not text:
        return "", False, "empty prompt"

    # Strip and normalize whitespace
    sanitized = ' '.join(text.strip().split())

    # Strip URLs
    sanitized = re.sub(r'https?://\S+', '', sanitized).strip()
    sanitized = re.sub(r'www\.\S+', '', sanitized).strip()

    # Character restriction
    if not allow_unicode:
        # Allow ASCII printable + common punctuation
        sanitized = re.sub(r'[^\x20-\x7E]', '', sanitized).strip()

    # Collapse whitespace again after filtering
    sanitized = ' '.join(sanitized.split())

    # Length checks
    if len(sanitized) < min_len:
        return sanitized, False, f"too short (min {min_len} chars)"
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len].rsplit(' ', 1)[0]  # Trim at word boundary

    # Moderation checks
    if enabled:
        lower = sanitized.lower()

        # Banned words
        for word in banned_words:
            if word.lower() in lower:
                return sanitized, False, f"blocked: contains banned word"

        # Banned regex
        for pattern in banned_regex:
            try:
                if re.search(pattern, sanitized, re.IGNORECASE):
                    return sanitized, False, f"blocked: matches banned pattern"
            except re.error:
                logger.warning(f"[Sanitize] Invalid regex pattern: {pattern}")

    return sanitized, True, ""


def sanitize_name(name: str, max_len: int = 30) -> str:
    """Sanitize a display name."""
    if not name:
        return "anon"
    # ASCII only, strip weird chars
    clean = re.sub(r'[^\w\s-]', '', name.strip())
    clean = ' '.join(clean.split())
    if not clean:
        return "anon"
    return clean[:max_len]


__all__ = ['sanitize_prompt', 'sanitize_name', 'DEFAULT_BANNED_WORDS']
