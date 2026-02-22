"""
Seen-store for processed email attachments.

Stores **two** kinds of fingerprints so the same content is never processed twice:
  1. IMAP Message-ID header  (deduplicates at the email level)
  2. SHA-256 hash of the attachment bytes (deduplicates identical files
     arriving from different messages / senders)

Storage: a simple JSON file on disk — no external dependency required.
Thread-safety: protected by threading.Lock (safe for asyncio + run_in_threadpool).
"""

import hashlib
import json
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()


class SeenStore:
    """
    Persistent, thread-safe store of already-processed message IDs and file
    content hashes.

    Args:
        store_path: Path to the JSON file used for persistence.
    """

    def __init__(self, store_path: Path) -> None:
        self._path = store_path
        self._data: dict[str, list[str]] = {"message_ids": [], "file_hashes": []}
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def is_message_seen(self, message_id: str | None) -> bool:
        """Return True if this Message-ID was already processed."""
        if not message_id:
            return False
        with _LOCK:
            return message_id in self._data["message_ids"]

    def is_file_seen(self, file_bytes: bytes) -> bool:
        """Return True if a file with the same content was already processed."""
        h = _sha256(file_bytes)
        with _LOCK:
            return h in self._data["file_hashes"]

    def mark_message(self, message_id: str | None) -> None:
        """Mark a Message-ID as processed (persists immediately)."""
        if not message_id:
            return
        with _LOCK:
            if message_id not in self._data["message_ids"]:
                self._data["message_ids"].append(message_id)
                self._save()

    def mark_file(self, file_bytes: bytes) -> None:
        """Mark a file content hash as processed (persists immediately)."""
        h = _sha256(file_bytes)
        with _LOCK:
            if h not in self._data["file_hashes"]:
                self._data["file_hashes"].append(h)
                self._save()

    def stats(self) -> dict[str, int]:
        with _LOCK:
            return {
                "seen_message_ids": len(self._data["message_ids"]),
                "seen_file_hashes": len(self._data["file_hashes"]),
            }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                self._data["message_ids"] = list(raw.get("message_ids", []))
                self._data["file_hashes"] = list(raw.get("file_hashes", []))
                logger.debug(
                    "SeenStore loaded: %d message IDs, %d file hashes",
                    len(self._data["message_ids"]),
                    len(self._data["file_hashes"]),
                )
            except Exception as exc:
                logger.warning("SeenStore: could not load %s — starting fresh. %s", self._path, exc)

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.error("SeenStore: failed to persist to %s: %s", self._path, exc)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
