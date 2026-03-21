#!/usr/bin/env python3
"""
db.py — Crowd Control SQLite Database Layer
═══════════════════════════════════════════════════════════════════════════════

WAL-mode SQLite with atomic pop, rate limiting, and audit trail.
No external dependencies — stdlib sqlite3 only.

Part of QonQrete Visual FaQtory v0.5.9-beta
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import Submission, SubmissionStatus

logger = logging.getLogger(__name__)

# ── SQL Schemas ──────────────────────────────────────────────────────────────

_SCHEMA_SUBMISSIONS = """
CREATE TABLE IF NOT EXISTS submissions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at       TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    ip               TEXT    NOT NULL,
    prompt           TEXT    NOT NULL,
    status           TEXT    NOT NULL DEFAULT 'queued',
    rejection_reason TEXT
);
"""

_SCHEMA_RATE_LIMIT = """
CREATE TABLE IF NOT EXISTS rate_limit (
    ip             TEXT PRIMARY KEY,
    last_submit_at TEXT NOT NULL
);
"""

_INDEX_SUBMISSIONS_STATUS = """
CREATE INDEX IF NOT EXISTS idx_submissions_status
ON submissions(status, id);
"""


class CrowdDB:
    """Thread-safe SQLite wrapper for the Crowd Control queue.

    Uses WAL mode for concurrent reads and IMMEDIATE transactions for
    safe atomic pops.
    """

    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        self._ensure_dir()
        self._init_db()

    def _ensure_dir(self) -> None:
        """Create parent directory for the DB file if needed."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        """Open a new connection with WAL mode and safe defaults."""
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        conn = self._connect()
        try:
            conn.executescript(
                _SCHEMA_SUBMISSIONS + _SCHEMA_RATE_LIMIT + _INDEX_SUBMISSIONS_STATUS
            )
            conn.commit()
            logger.info(f"[CrowdDB] Database ready: {self._db_path}")
        finally:
            conn.close()

    # ── Rate Limiting ────────────────────────────────────────────────────────

    def check_rate_limit(self, ip: str, window_seconds: int) -> Tuple[bool, Optional[int]]:
        """Check if an IP is rate-limited.

        Returns:
            (allowed, seconds_remaining)
            allowed=True means the submission can proceed.
            If not allowed, seconds_remaining is how long to wait.
        """
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT last_submit_at FROM rate_limit WHERE ip = ?", (ip,)
                ).fetchone()
                if row is None:
                    return True, None
                last_at = datetime.fromisoformat(row["last_submit_at"])
                now = datetime.now(timezone.utc)
                elapsed = (now - last_at).total_seconds()
                if elapsed >= window_seconds:
                    return True, None
                remaining = int(window_seconds - elapsed) + 1
                return False, remaining
            finally:
                conn.close()

    def update_rate_limit(self, ip: str) -> None:
        """Record a successful submission for rate-limit tracking."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT INTO rate_limit (ip, last_submit_at)
                       VALUES (?, ?)
                       ON CONFLICT(ip) DO UPDATE SET last_submit_at = excluded.last_submit_at""",
                    (ip, now_iso),
                )
                conn.commit()
            finally:
                conn.close()

    # ── Queue Operations ─────────────────────────────────────────────────────

    def enqueue(self, ip: str, prompt: str) -> int:
        """Add a prompt to the queue. Returns the new submission ID."""
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "INSERT INTO submissions (ip, prompt, status) VALUES (?, ?, ?)",
                    (ip, prompt, SubmissionStatus.QUEUED.value),
                )
                conn.commit()
                sub_id = cur.lastrowid
                logger.info(f"[CrowdDB] Enqueued submission #{sub_id} from {ip}")
                return sub_id
            finally:
                conn.close()

    def reject(self, ip: str, prompt: str, reason: str) -> int:
        """Record a rejected submission for the audit trail. Returns the submission ID."""
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "INSERT INTO submissions (ip, prompt, status, rejection_reason) VALUES (?, ?, ?, ?)",
                    (ip, prompt, SubmissionStatus.REJECTED.value, reason),
                )
                conn.commit()
                sub_id = cur.lastrowid
                logger.info(f"[CrowdDB] Rejected submission #{sub_id} from {ip}: {reason}")
                return sub_id
            finally:
                conn.close()

    def pop_next(self) -> Optional[str]:
        """Atomically pop the oldest queued prompt.

        Uses BEGIN IMMEDIATE for write-lock safety.

        Returns:
            The prompt string, or None if queue is empty.
        """
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT id, prompt FROM submissions WHERE status = ? ORDER BY id ASC LIMIT 1",
                    (SubmissionStatus.QUEUED.value,),
                ).fetchone()
                if row is None:
                    conn.commit()
                    return None
                conn.execute(
                    "UPDATE submissions SET status = ? WHERE id = ?",
                    (SubmissionStatus.SERVED.value, row["id"]),
                )
                conn.commit()
                logger.info(f"[CrowdDB] Popped submission #{row['id']}")
                return row["prompt"]
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def queue_length(self) -> int:
        """Return the number of queued (pending) submissions."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM submissions WHERE status = ?",
                (SubmissionStatus.QUEUED.value,),
            ).fetchone()
            return row["cnt"] if row else 0
        finally:
            conn.close()

    def queue_preview(self, limit: int = 3) -> List[Dict[str, str]]:
        """Return the next N queued prompts without consuming them.

        Returns a list of dicts with 'prompt' and 'created_at' keys.
        IP addresses are never exposed.
        """
        limit = max(1, min(limit, 10))  # clamp 1..10
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT prompt, created_at FROM submissions "
                "WHERE status = ? ORDER BY id ASC LIMIT ?",
                (SubmissionStatus.QUEUED.value, limit),
            ).fetchall()
            return [{"prompt": r["prompt"], "created_at": r["created_at"]} for r in rows]
        finally:
            conn.close()

    def status_counts(self) -> Dict[str, int]:
        """Return aggregate counts by submission status.

        Returns dict with keys: queued, served, rejected, total.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS cnt FROM submissions GROUP BY status"
            ).fetchall()
            counts = {r["status"]: r["cnt"] for r in rows}
            return {
                "queued": counts.get(SubmissionStatus.QUEUED.value, 0),
                "served": counts.get(SubmissionStatus.SERVED.value, 0),
                "rejected": counts.get(SubmissionStatus.REJECTED.value, 0),
                "total": sum(counts.values()),
            }
        finally:
            conn.close()
