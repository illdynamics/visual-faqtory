#!/usr/bin/env python3
"""
crowd_queue.py - Crowd Prompt Queue with Rate Limiting
═══════════════════════════════════════════════════════════════════════════════

In-memory prompt queue for live crowd interaction:
  - Thread-safe deque
  - Per-IP and per-name rate limiting (sliding window)
  - Prompt sanitization integration
  - Optional SQLite persistence (future)

Part of QonQrete Visual FaQtory v0.3.5-beta
"""
import uuid
import time
import hashlib
import logging
import threading
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple

from .utils_sanitize import sanitize_prompt, sanitize_name

logger = logging.getLogger(__name__)


@dataclass
class PromptItem:
    """A queued crowd prompt."""
    id: str = ""
    ts: float = 0.0
    name: str = "anon"
    prompt: str = ""
    ip_hash: str = ""
    votes: int = 0

    def __post_init__(self):
        if not self.id:
            self.id = uuid.uuid4().hex[:12]
        if not self.ts:
            self.ts = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RateLimiter:
    """Simple sliding-window rate limiter."""

    def __init__(self, window_seconds: int = 30, max_requests: int = 3):
        self.window = window_seconds
        self.max_requests = max_requests
        self._buckets: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> Tuple[bool, str]:
        """Check if key is within rate limit. Returns (allowed, reason)."""
        now = time.time()
        with self._lock:
            if key not in self._buckets:
                self._buckets[key] = []

            # Prune old entries
            self._buckets[key] = [
                t for t in self._buckets[key] if now - t < self.window
            ]

            if len(self._buckets[key]) >= self.max_requests:
                wait = self.window - (now - self._buckets[key][0])
                return False, f"rate limited (wait {wait:.0f}s)"

            self._buckets[key].append(now)
            return True, ""


class PromptQueue:
    """
    Thread-safe crowd prompt queue.

    Usage:
        queue = PromptQueue(config)
        ok, reason = queue.submit("neon skull", "Dave", "192.168.1.5")
        item = queue.pop_next()
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        queue_cfg = cfg.get('queue', {})
        mod_cfg = cfg.get('moderation', {})

        self.max_depth = queue_cfg.get('max_depth', 50)
        self._queue: deque = deque(maxlen=self.max_depth)
        self._lock = threading.Lock()
        self._mod_config = mod_cfg

        # Rate limiters
        ip_rl = queue_cfg.get('per_ip_rate_limit', {})
        name_rl = queue_cfg.get('per_name_rate_limit', {})
        self._ip_limiter = RateLimiter(
            window_seconds=ip_rl.get('window_seconds', 30),
            max_requests=ip_rl.get('max_requests', 3)
        )
        self._name_limiter = RateLimiter(
            window_seconds=name_rl.get('window_seconds', 60),
            max_requests=name_rl.get('max_requests', 2)
        )

        # Stats
        self.accepted = 0
        self.rejected = 0

    def submit(
        self, prompt_text: str, name: str = "", ip: str = ""
    ) -> Tuple[bool, str]:
        """
        Submit a prompt to the queue.

        Returns:
            (accepted: bool, reason: str)
        """
        # Sanitize name
        clean_name = sanitize_name(name)

        # Sanitize prompt
        sanitized, ok, reason = sanitize_prompt(prompt_text, self._mod_config)
        if not ok:
            self.rejected += 1
            logger.info(f"[CrowdQueue] Rejected from {clean_name}: {reason}")
            return False, reason

        # Rate limit by IP
        if ip:
            ip_hash = hashlib.sha256(ip.encode()).hexdigest()[:16]
            ip_ok, ip_reason = self._ip_limiter.check(ip_hash)
            if not ip_ok:
                self.rejected += 1
                return False, ip_reason
        else:
            ip_hash = ""

        # Rate limit by name
        if clean_name and clean_name != "anon":
            name_ok, name_reason = self._name_limiter.check(clean_name.lower())
            if not name_ok:
                self.rejected += 1
                return False, name_reason

        # Queue it
        item = PromptItem(
            prompt=sanitized,
            name=clean_name,
            ip_hash=ip_hash,
        )

        with self._lock:
            if len(self._queue) >= self.max_depth:
                self.rejected += 1
                return False, "queue full"
            self._queue.append(item)
            self.accepted += 1

        logger.info(f"[CrowdQueue] Accepted from {clean_name}: '{sanitized[:40]}...' (depth={self.depth()})")
        return True, ""

    def pop_next(self) -> Optional[PromptItem]:
        """Pop the next prompt from the queue."""
        with self._lock:
            if self._queue:
                return self._queue.popleft()
            return None

    def peek_next(self) -> Optional[PromptItem]:
        """Peek at the next prompt without removing."""
        with self._lock:
            if self._queue:
                return self._queue[0]
            return None

    def list_top(self, n: int = 5) -> List[PromptItem]:
        """List top N queued items."""
        with self._lock:
            return list(self._queue)[:n]

    def depth(self) -> int:
        """Current queue depth."""
        with self._lock:
            return len(self._queue)

    def stats(self) -> Dict[str, Any]:
        """Queue statistics."""
        return {
            'depth': self.depth(),
            'accepted': self.accepted,
            'rejected': self.rejected,
            'max_depth': self.max_depth,
        }


__all__ = ['PromptItem', 'PromptQueue', 'RateLimiter']
