#!/usr/bin/env python3
"""
models.py — Crowd Control Data Models
═══════════════════════════════════════════════════════════════════════════════

Pydantic-free data containers for the Crowd Control MVP.
No external dependencies beyond stdlib.

Part of Visual FaQtory v0.9.0-beta
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class SubmissionStatus(str, Enum):
    """Status of a crowd prompt submission."""
    QUEUED = "queued"
    SERVED = "served"
    REJECTED = "rejected"


@dataclass
class Submission:
    """A single crowd-submitted prompt."""
    id: int = 0
    created_at: str = ""
    ip: str = ""
    prompt: str = ""
    status: str = SubmissionStatus.QUEUED.value
    rejection_reason: Optional[str] = None


@dataclass
class CrowdControlConfig:
    """Configuration for the crowd control system.

    Used by both the FastAPI server and the generator client.
    """
    enabled: bool = False
    base_url: str = "http://127.0.0.1:8808/visuals"
    pop_token: str = "CHANGE_ME_LONG_RANDOM"
    timeout_seconds: float = 1.0
    inject_label: str = "Audience mutation request"
    inject_mode: str = "append"           # append | replace
    max_chars: int = 300
    rate_limit_seconds: int = 600
    max_queue: int = 100
    badwords_path: str = "worqspace/badwords.txt"
    public_url: str = "https://wonq.tv/visuals"
    prefix: str = "/visuals"
    db_path: str = "worqspace/crowdcontrol.sqlite3"

    @classmethod
    def from_dict(cls, data: dict) -> CrowdControlConfig:
        """Build config from a dict (e.g. parsed YAML section)."""
        if not data:
            return cls()
        return cls(
            enabled=data.get("enabled", False),
            base_url=data.get("base_url", cls.base_url),
            pop_token=data.get("pop_token", cls.pop_token),
            timeout_seconds=float(data.get("timeout_seconds", cls.timeout_seconds)),
            inject_label=data.get("inject_label", cls.inject_label),
            inject_mode=data.get("inject_mode", cls.inject_mode),
            max_chars=int(data.get("max_chars", cls.max_chars)),
            rate_limit_seconds=int(data.get("rate_limit_seconds", cls.rate_limit_seconds)),
            max_queue=int(data.get("max_queue", cls.max_queue)),
            badwords_path=data.get("badwords_path", cls.badwords_path),
            public_url=data.get("public_url", cls.public_url),
            prefix=data.get("prefix", cls.prefix),
            db_path=data.get("db_path", cls.db_path),
        )
