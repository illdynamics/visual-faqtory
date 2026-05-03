#!/usr/bin/env python3
"""
models.py — Crowd Control Data Models
═══════════════════════════════════════════════════════════════════════════════

Pydantic-free data containers for the Crowd Control MVP.
No external dependencies beyond stdlib.

Part of Visual FaQtory v0.9.3-beta
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional


logger = logging.getLogger(__name__)


class SubmissionStatus(str, Enum):
    """Status of a crowd prompt submission."""
    QUEUED = "queued"
    CLAIMED = "claimed"
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
    # v0.9.1: controls HOW the audience prompt drives the next cycle.
    #
    #   "as_image_source" (default — preserves prior behaviour):
    #     The crowd-injected cycle is rendered as IMG2VID with the previous
    #     cycle's last frame supplied as the SOURCE (init) image. The audience
    #     prompt + (in append mode) the story paragraph drive the motion.
    #     This keeps tight visual continuity from cycle N-1 → cycle N.
    #
    #   "as_reference":
    #     The crowd-injected cycle is rendered as TEXT2VID with the previous
    #     cycle's last frame supplied as a REFERENCE image (when the model
    #     supports reference_image_urls). The audience prompt is the primary
    #     visual driver. This gives a softer style/identity tether rather
    #     than hard frame continuity. If the model does not support reference
    #     images, the field is auto-omitted and the cycle runs as a pure
    #     text2vid prompt.
    inject_source_mode: str = "as_image_source"
    bake_mode: str = "reinject_keyframe"  # off | direct_video | reinject_keyframe
    bake_use_morph: bool = True
    bake_denoise_min: float = 0.58
    bake_denoise_max: float = 0.82
    bake_prompt_prefix: str = (
        "AUDIENCE PROMPT IS THE PRIMARY VISUAL MUTATION. Strongly transform the scene to match this request "
        "while preserving enough continuity from the source image:"
    )
    discard_smart_prefetch_on_crowd: bool = True
    ack_after_success: bool = True
    requeue_on_failure: bool = True
    claim_timeout_seconds: int = 900
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

        def _bool(value, default: bool) -> bool:
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"1", "true", "yes", "on"}:
                    return True
                if lowered in {"0", "false", "no", "off"}:
                    return False
            logger.warning(
                "[CrowdControlConfig] Invalid bool value '%s' — using default=%s",
                value,
                default,
            )
            return default

        def _int(value, default: int, *, minimum: Optional[int] = None) -> int:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                logger.warning(
                    "[CrowdControlConfig] Invalid int value '%s' — using default=%s",
                    value,
                    default,
                )
                return default
            if minimum is not None and parsed < minimum:
                logger.warning(
                    "[CrowdControlConfig] Int value %s below minimum %s — using default=%s",
                    parsed,
                    minimum,
                    default,
                )
                return default
            return parsed

        def _float(value, default: float, *, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                logger.warning(
                    "[CrowdControlConfig] Invalid float value '%s' — using default=%s",
                    value,
                    default,
                )
                return default
            if minimum is not None and parsed < minimum:
                logger.warning(
                    "[CrowdControlConfig] Float value %s below minimum %s — using default=%s",
                    parsed,
                    minimum,
                    default,
                )
                return default
            if maximum is not None and parsed > maximum:
                logger.warning(
                    "[CrowdControlConfig] Float value %s above maximum %s — using default=%s",
                    parsed,
                    maximum,
                    default,
                )
                return default
            return parsed

        # Validate inject_source_mode against the allowed enum.
        ism_raw = data.get("inject_source_mode", cls.inject_source_mode)
        ism = str(ism_raw or "").strip().lower() or cls.inject_source_mode
        if ism not in {"as_image_source", "as_reference"}:
            logger.warning(
                "[CrowdControlConfig] Unsupported inject_source_mode '%s' — using default '%s'",
                ism_raw,
                cls.inject_source_mode,
            )
            ism = cls.inject_source_mode
        # Validate inject_mode similarly so a typo can't silently break flow.
        im_raw = data.get("inject_mode", cls.inject_mode)
        im = str(im_raw or "").strip().lower() or cls.inject_mode
        if im not in {"append", "replace"}:
            logger.warning(
                "[CrowdControlConfig] Unsupported inject_mode '%s' — using default '%s'",
                im_raw,
                cls.inject_mode,
            )
            im = cls.inject_mode

        bake_mode_raw = data.get("bake_mode", cls.bake_mode)
        bake_mode = str(bake_mode_raw or "").strip().lower() or cls.bake_mode
        if bake_mode not in {"off", "direct_video", "reinject_keyframe"}:
            logger.warning(
                "[CrowdControlConfig] Unsupported bake_mode '%s' — using default '%s'",
                bake_mode_raw,
                cls.bake_mode,
            )
            bake_mode = cls.bake_mode

        bake_denoise_min = _float(
            data.get("bake_denoise_min", cls.bake_denoise_min),
            cls.bake_denoise_min,
            minimum=0.0,
            maximum=1.0,
        )
        bake_denoise_max = _float(
            data.get("bake_denoise_max", cls.bake_denoise_max),
            cls.bake_denoise_max,
            minimum=0.0,
            maximum=1.0,
        )
        if bake_denoise_min > bake_denoise_max:
            logger.warning(
                "[CrowdControlConfig] bake_denoise_min > bake_denoise_max (%s > %s) — swapping values",
                bake_denoise_min,
                bake_denoise_max,
            )
            bake_denoise_min, bake_denoise_max = bake_denoise_max, bake_denoise_min

        return cls(
            enabled=_bool(data.get("enabled", cls.enabled), cls.enabled),
            base_url=data.get("base_url", cls.base_url),
            pop_token=data.get("pop_token", cls.pop_token),
            timeout_seconds=_float(data.get("timeout_seconds", cls.timeout_seconds), cls.timeout_seconds, minimum=0.1),
            inject_label=data.get("inject_label", cls.inject_label),
            inject_mode=im,
            inject_source_mode=ism,
            bake_mode=bake_mode,
            bake_use_morph=_bool(data.get("bake_use_morph", cls.bake_use_morph), cls.bake_use_morph),
            bake_denoise_min=bake_denoise_min,
            bake_denoise_max=bake_denoise_max,
            bake_prompt_prefix=str(data.get("bake_prompt_prefix", cls.bake_prompt_prefix) or cls.bake_prompt_prefix),
            discard_smart_prefetch_on_crowd=_bool(
                data.get("discard_smart_prefetch_on_crowd", cls.discard_smart_prefetch_on_crowd),
                cls.discard_smart_prefetch_on_crowd,
            ),
            ack_after_success=_bool(data.get("ack_after_success", cls.ack_after_success), cls.ack_after_success),
            requeue_on_failure=_bool(data.get("requeue_on_failure", cls.requeue_on_failure), cls.requeue_on_failure),
            claim_timeout_seconds=_int(
                data.get("claim_timeout_seconds", cls.claim_timeout_seconds),
                cls.claim_timeout_seconds,
                minimum=1,
            ),
            max_chars=_int(data.get("max_chars", cls.max_chars), cls.max_chars, minimum=1),
            rate_limit_seconds=_int(data.get("rate_limit_seconds", cls.rate_limit_seconds), cls.rate_limit_seconds, minimum=0),
            max_queue=_int(data.get("max_queue", cls.max_queue), cls.max_queue, minimum=1),
            badwords_path=data.get("badwords_path", cls.badwords_path),
            public_url=data.get("public_url", cls.public_url),
            prefix=data.get("prefix", cls.prefix),
            db_path=data.get("db_path", cls.db_path),
        )
