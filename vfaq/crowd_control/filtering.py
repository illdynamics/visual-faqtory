#!/usr/bin/env python3
"""
filtering.py — Crowd Control Input Filtering & Sanitization
═══════════════════════════════════════════════════════════════════════════════

Handles:
  - Whitespace normalization (collapse, strip, single-line)
  - Max length enforcement
  - Empty rejection
  - Bad word / phrase filtering with extendable file

Part of Visual FaQtory v0.9.0-beta
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class PromptFilter:
    """Sanitizes and filters crowd-submitted prompts.

    Loads a badwords file on init and checks submissions against it.
    The badwords file is a plain text file with one word/phrase per line.
    Lines starting with # are treated as comments. Blank lines are ignored.
    """

    def __init__(self, badwords_path: Optional[str] = None, max_chars: int = 300):
        self.max_chars = max_chars
        self._badwords: List[str] = []
        self._badword_patterns: List[re.Pattern] = []
        if badwords_path:
            self._load_badwords(badwords_path)

    def _load_badwords(self, path: str) -> None:
        """Load bad words from file. One word/phrase per line."""
        p = Path(path)
        if not p.exists():
            logger.warning(f"[Filter] Badwords file not found: {path} — filtering disabled")
            return
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                self._badwords.append(stripped.lower())
                # Build regex pattern: word boundary match, case insensitive
                escaped = re.escape(stripped)
                self._badword_patterns.append(
                    re.compile(rf"\b{escaped}\b", re.IGNORECASE)
                )
            logger.info(f"[Filter] Loaded {len(self._badwords)} bad words/phrases from {path}")
        except Exception as e:
            logger.error(f"[Filter] Failed to load badwords file: {e}")

    def sanitize(self, raw_input: str) -> str:
        """Clean the raw input string.

        Steps:
          1. Strip leading/trailing whitespace
          2. Remove newlines and carriage returns (single line only)
          3. Collapse multiple spaces to single space
          4. Strip again
        """
        text = raw_input.strip()
        text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def validate(self, prompt: str) -> Tuple[bool, Optional[str]]:
        """Validate a sanitized prompt.

        Returns:
            (ok, rejection_reason)
            ok=True means the prompt passes all checks.
        """
        # Empty check
        if not prompt:
            return False, "empty_prompt"

        # Length check
        if len(prompt) > self.max_chars:
            return False, f"too_long:{len(prompt)}>{self.max_chars}"

        # Bad word check
        for pattern in self._badword_patterns:
            if pattern.search(prompt):
                return False, "bad_word_detected"

        return True, None
