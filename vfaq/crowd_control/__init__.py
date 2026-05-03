#!/usr/bin/env python3
"""
vfaq/crowd_control/ — Crowd Control for Visual FaQtory
═══════════════════════════════════════════════════════════════════════════════

Live audience prompt injection system. Viewers scan a QR code, submit prompts,
and the Visual FaQtory generator injects them into the next visual cycle.

Components:
  - server.py    : FastAPI app (prompt page, queue API, QR code)
  - db.py        : SQLite queue + rate limiting + audit
  - filtering.py : Input sanitization + bad word filtering
  - models.py    : Data models + config
  - client.py    : Generator-side HTTP client (fail-open)

Part of Visual FaQtory v0.9.3-beta
"""

from .models import CrowdControlConfig, Submission, SubmissionStatus
from .client import CrowdClient
from .db import CrowdDB
from .filtering import PromptFilter
from .server import create_crowd_app

__all__ = [
    "CrowdControlConfig",
    "CrowdClient",
    "CrowdDB",
    "PromptFilter",
    "Submission",
    "SubmissionStatus",
    "create_crowd_app",
]
