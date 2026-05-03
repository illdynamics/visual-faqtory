#!/usr/bin/env python3
"""
client.py — Crowd Control Queue Client
═══════════════════════════════════════════════════════════════════════════════

Lightweight HTTP client used by the Visual FaQtory generator to claim,
ack, and requeue crowd prompts from the Crowd Control server.

Fail-open by design: all errors return None so the generator can
continue in story mode.

Part of Visual FaQtory v0.9.3-beta
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

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
        self._claims: Dict[int, str] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def claim_next(self) -> Optional[dict]:
        """Claim the next queued prompt from the server.

        Returns:
            A dict with at least {'id', 'prompt'} (and optional claim_id),
            or None if the queue is empty or
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
                sub_id = data.get("id")
                if prompt and sub_id is not None:
                    claim = {
                        "id": int(sub_id),
                        "prompt": prompt,
                        "claim_id": data.get("claim_id"),
                    }
                    if claim.get("claim_id"):
                        self._claims[int(sub_id)] = str(claim["claim_id"])
                    logger.info(
                        f"[CrowdClient] Claimed crowd prompt #{claim['id']}: {prompt[:80]}..."
                    )
                    return claim
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

    def ack(self, submission_id: int, claim_id: Optional[str] = None) -> bool:
        """Ack a claimed prompt as served."""
        if not self._enabled:
            return False
        claim_token = claim_id or self._claims.get(int(submission_id))
        try:
            resp = requests.post(
                f"{self._base_url}/api/ack",
                headers={"Authorization": f"Bearer {self._token}"},
                json={"id": int(submission_id), "claim_id": claim_token},
                timeout=self._timeout,
            )
            if resp.status_code == 200 and (resp.json() or {}).get("ok", False):
                self._claims.pop(int(submission_id), None)
                logger.info(f"[CrowdClient] Acked crowd prompt #{submission_id}")
                return True
            logger.warning(f"[CrowdClient] Ack failed ({resp.status_code}): {resp.text[:200]}")
        except Exception as e:
            logger.warning(f"[CrowdClient] Ack error: {e}")
        return False

    def requeue(self, submission_id: int, reason: str = "", claim_id: Optional[str] = None) -> bool:
        """Requeue a claimed prompt after generation failure."""
        if not self._enabled:
            return False
        claim_token = claim_id or self._claims.get(int(submission_id))
        try:
            resp = requests.post(
                f"{self._base_url}/api/requeue",
                headers={"Authorization": f"Bearer {self._token}"},
                json={"id": int(submission_id), "claim_id": claim_token, "reason": reason},
                timeout=self._timeout,
            )
            if resp.status_code == 200 and (resp.json() or {}).get("ok", False):
                self._claims.pop(int(submission_id), None)
                logger.info(f"[CrowdClient] Requeued crowd prompt #{submission_id}")
                return True
            logger.warning(f"[CrowdClient] Requeue failed ({resp.status_code}): {resp.text[:200]}")
        except Exception as e:
            logger.warning(f"[CrowdClient] Requeue error: {e}")
        return False

    def pop_next(self) -> Optional[str]:
        """Backward-compatible API: claim and immediately ack the prompt.

        This preserves legacy semantics for callers that still expect
        pop-next to mark a prompt as served instantly.
        """
        claim = self.claim_next()
        if not claim:
            return None
        self.ack(int(claim["id"]), claim_id=claim.get("claim_id"))
        return str(claim["prompt"])

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
