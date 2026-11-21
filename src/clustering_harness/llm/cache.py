"""Persistent response cache (SQLite).

Keys are SHA-1 digests of the full request (model identity, messages, generation budget, seed, JSON mode,
temperature), so a cached answer is only reused for a byte-identical request to the same model. Cached
values are the *final* parsed outputs, including contract failures (``{"_error": ...}``), which makes
reruns and resumed runs reproduce earlier decisions exactly."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path

__all__ = ["ResponseCache", "request_key"]


def request_key(model_id: str, messages: list, max_tokens: int, seed: int, as_json: bool, temperature: float,
                settings: dict | None = None) -> str:
    """Cache key of one request. ``settings`` holds model options that change answers without changing the
    request (reasoning flag, thinking/effort, extra provider fields); empty settings give the original key."""
    parts = [model_id, messages, max_tokens, seed, as_json, temperature]
    if settings:
        parts.append(settings)
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


class ResponseCache:
    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=60, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, model TEXT, stage TEXT, tag TEXT, "
                           "out TEXT, n_calls INTEGER, secs REAL, t REAL)")

    def checkpoint(self) -> None:
        """Fold the write-ahead log into the main file (call before copying the cache file elsewhere)."""
        with self._lock:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def get(self, key: str) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT out FROM responses WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else None

    def put(self, key: str, out: dict, model: str = "", stage: str = "", tag: str = "", n_calls: int = 1, secs: float = 0.0) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?,?,?,?)",
                               (key, model, stage, tag, json.dumps(out), n_calls, secs, time.time()))

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return self._conn.execute("SELECT 1 FROM responses WHERE key=?", (key,)).fetchone() is not None

    def stage_calls(self, prefix: str) -> int:
        with self._lock:
            r = self._conn.execute("SELECT COALESCE(SUM(n_calls),0) FROM responses WHERE stage LIKE ?", (prefix + "%",)).fetchone()
        return int(r[0])

    def stage_counts(self) -> dict:
        with self._lock:
            rows = self._conn.execute("SELECT stage, COUNT(*) FROM responses GROUP BY stage").fetchall()
        return dict(rows)

    def error_count(self) -> int:
        with self._lock:
            r = self._conn.execute("SELECT COUNT(*) FROM responses WHERE out LIKE '%\"_error\"%'").fetchone()
        return int(r[0])
