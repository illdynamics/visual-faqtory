#!/usr/bin/env python3
"""Shared version helpers for Visual FaQtory."""
from __future__ import annotations
from pathlib import Path
_DEFAULT_VERSION = "v0.9.3-beta"

def read_version(default: str = _DEFAULT_VERSION) -> str:
    version_file = Path(__file__).resolve().parents[1] / "VERSION"
    try:
        return version_file.read_text(encoding="utf-8").strip()
    except OSError:
        return default

__version__ = read_version()
