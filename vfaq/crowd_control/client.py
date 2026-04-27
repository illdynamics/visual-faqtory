#!/usr/bin/env python3
"""
client.py — Crowd Control Queue Client
═══════════════════════════════════════════════════════════════════════════════

Lightweight HTTP client used by the Visual FaQtory generator to pop
crowd prompts from the Crowd Control server.

Fail-open by design: all errors return None so the generator can
continue in story mode.

Part of Visual FaQtory v0.9.0-beta
"""
from __future__ import annotations

import logging
from typing import Optional

import requests

from .models import CrowdControlConfig

logger = logging.getLogger(__name__)


class CrowdClient:
    """Client for popping prompts from the Crowd Control queue.

    Fail-open: if anything goes wrong (timeout, connection refused,
    bad response), pop_next() returns None and the generator continues
    with story mode. No exceptions are raised.
    """

    def __init__(self, config: CrowdControlConfig):
        self._base_url = config.base_url.rstrip("/")
        self._token = config.pop_token
        self._timeout = config.timeout_seconds
        self._enabled = config.enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def pop_next(self) -> Optional[str]:
        """Pop the next queued prompt from the server.

        Returns:
            The prompt string, or None if the queue is empty or
            any error occurred (fail-open).
        """
        if not self._enabled:
            return None

        url = f"{self._base_url}/api/next"
        headers = {"Authorization": f"Bearer {self._token}"}

        try:
            resp = requests.get(url, headers=headers, timeout=self._timeout)
            if resp.status_code == 200:
                data = resp.json()
                prompt = data.get("prompt")
                if prompt:
                    logger.info(f"[CrowdClient] Got crowd prompt: {prompt[:80]}...")
                    return prompt
                logger.debug("[CrowdClient] Queue empty")
                return None
            elif resp.status_code == 401:
                logger.warning("[CrowdClient] Auth failed — check pop_token in config")
                return None
            else:
                logger.warning(
                    f"[CrowdClient] Unexpected status {resp.status_code}: {resp.text[:200]}"
                )
                return None
        except requests.ConnectionError:
            logger.warning("[CrowdClient] Connection refused — crowd server down? Continuing story mode.")
            return None
        except requests.Timeout:
            logger.warning("[CrowdClient] Timeout — continuing story mode.")
            return None
        except Exception as e:
            logger.warning(f"[CrowdClient] Unexpected error: {e} — continuing story mode.")
            return None

    def health(self) -> Optional[dict]:
        """Check server health. Returns health dict or None on failure."""
        if not self._enabled:
            return None
        url = f"{self._base_url}/api/health"
        try:
            resp = requests.get(url, timeout=self._timeout)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return None
