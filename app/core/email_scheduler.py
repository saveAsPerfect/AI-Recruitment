"""
Background email scheduler — polls the IMAP inbox every N minutes (default 60).

Architecture:
  • Runs as a single asyncio.Task started from FastAPI lifespan.
  • Delegates the blocking IMAP work to a thread pool via run_in_threadpool.
  • Uses SeenStore for idempotency (no duplicate processing).
  • Respects EMAIL_SCHEDULER_ENABLED to allow disabling when IMAP is not configured.

Idempotency guarantees:
  1. Message-ID level  — each IMAP Message-ID is stored after first processing.
  2. File-hash level   — SHA-256 of attachment bytes is stored; identical files
                         arriving in different messages are skipped.
"""

import asyncio
import logging
from pathlib import Path

from fastapi.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.core.email_seen_store import SeenStore

logger = logging.getLogger(__name__)
settings = get_settings()

# Module-level singleton — shared between scheduler and manual triggers
_seen_store: SeenStore | None = None


def get_seen_store() -> SeenStore:
    """Return (lazily initialised) global SeenStore."""
    global _seen_store
    if _seen_store is None:
        store_path = Path(settings.DATA_DIR) / "email_seen.json"
        _seen_store = SeenStore(store_path)
    return _seen_store


async def run_email_poll(es) -> dict:
    """
    Perform one email-poll cycle:
      1. Fetch attachments from IMAP (blocking — run in thread pool).
      2. Skip already-seen message IDs and duplicate file hashes.
      3. Parse new attachments into candidates and index into Elasticsearch.

    Returns a summary dict for logging / status endpoint.
    """
    # Import here to avoid circular imports
    from app.core.email_inbox import fetch_resume_attachments, EmailInboxError
    from app.utils.resume_parser import (
        extract_text, parse_resume_with_llm, build_candidate,
        TextExtractionError, ResumeParseError, NotResumeError,
    )
    from app.core.storage import upsert_candidate
    from app.core.embeddings import EmbeddingService
    from app.models.schemas import Candidate

    seen = get_seen_store()

    # ── Step 1: fetch from IMAP ────────────────────────────────────────────────
    try:
        fetched = await run_in_threadpool(
            fetch_resume_attachments,
            max_messages=settings.EMAIL_SCHEDULER_MAX_MESSAGES,
            include_seen=False,   # only UNSEEN, IMAP server pre-filters
            mark_seen=True,
        )
    except EmailInboxError as exc:
        logger.error("Email scheduler: IMAP fetch failed: %s", exc)
        return {"status": "error", "error": str(exc)}

    items = fetched.get("items", [])
    total_fetched = fetched.get("fetched_messages", 0)

    parsed_count = 0
    skipped_count = 0
    failed_count = 0

    embedder = EmbeddingService(settings.EMBEDDING_MODEL)

    for item in items:
        message_id: str | None = item.get("message_id")
        saved_path = Path(item["saved_path"])

        # ── Step 2a: idempotency by Message-ID ────────────────────────────────
        if seen.is_message_seen(message_id):
            logger.debug("Email scheduler: skip (seen message_id=%s)", message_id)
            skipped_count += 1
            try:
                saved_path.unlink(missing_ok=True)
            except Exception:
                pass
            continue

        # ── Step 2b: idempotency by file content hash ─────────────────────────
        try:
            file_bytes = saved_path.read_bytes()
        except OSError as exc:
            logger.warning("Email scheduler: cannot read %s: %s", saved_path, exc)
            failed_count += 1
            continue

        if seen.is_file_seen(file_bytes):
            logger.debug(
                "Email scheduler: skip duplicate file content (%s)", item["attachment_name"]
            )
            skipped_count += 1
            try:
                saved_path.unlink(missing_ok=True)
            except Exception:
                pass
            # Still mark the message so we don't re-check it next cycle
            seen.mark_message(message_id)
            continue

        # ── Step 3: parse and index ────────────────────────────────────────────
        try:
            raw_text = await run_in_threadpool(extract_text, str(saved_path))
            parsed = await run_in_threadpool(
                parse_resume_with_llm,
                raw_text,
                settings.OPENAI_API_KEY,
                settings.OPENAI_PARSE_MODEL,
            )
            c_dict = build_candidate(parsed, raw_text)
            candidate = Candidate(**c_dict)

            def _emb_text(c: dict) -> str:
                parts = [
                    c.get("role") or "",
                    " ".join(c.get("skills") or []),
                    c.get("education") or "",
                    c.get("experience") or "",
                ]
                return " ".join(p for p in parts if p)

            emb = await run_in_threadpool(embedder.encode_one, _emb_text(c_dict))
            await upsert_candidate(es, candidate, emb)

            # Mark as processed only after successful indexing
            seen.mark_message(message_id)
            seen.mark_file(file_bytes)
            parsed_count += 1
            logger.info(
                "Email scheduler: indexed candidate '%s' from '%s'",
                candidate.name,
                item["attachment_name"],
            )

        except (TextExtractionError, ResumeParseError) as exc:
            logger.warning(
                "Email scheduler: parse failed for %s: %s", item["attachment_name"], exc
            )
            # Mark as seen to not retry forever on a malformed file
            seen.mark_message(message_id)
            seen.mark_file(file_bytes)
            failed_count += 1

        except NotResumeError as exc:
            logger.warning(
                "Email scheduler: skipped non-resume attachment %s: %s",
                item["attachment_name"], exc,
            )
            seen.mark_message(message_id)
            seen.mark_file(file_bytes)
            skipped_count += 1

        except Exception:
            logger.exception(
                "Email scheduler: unexpected error for %s", item["attachment_name"]
            )
            failed_count += 1

    summary = {
        "status": "ok",
        "fetched_messages": total_fetched,
        "attachments_found": len(items),
        "parsed_candidates": parsed_count,
        "skipped_duplicates": skipped_count,
        "failed": failed_count,
    }
    logger.info("Email scheduler cycle done: %s", summary)
    return summary


async def email_scheduler_loop(app) -> None:
    """
    Infinite async loop that runs email polling every EMAIL_SCHEDULER_INTERVAL_MINUTES.

    Started as a background asyncio.Task by the FastAPI lifespan context manager.
    """
    interval_seconds = settings.EMAIL_SCHEDULER_INTERVAL_MINUTES * 60

    logger.info(
        "Email scheduler started — interval: %d min, IMAP: %s@%s",
        settings.EMAIL_SCHEDULER_INTERVAL_MINUTES,
        settings.IMAP_USER or "(not set)",
        settings.IMAP_HOST or "(not set)",
    )

    # Wait a short delay on startup so ES is fully ready before first cycle
    await asyncio.sleep(settings.EMAIL_SCHEDULER_STARTUP_DELAY_SECONDS)

    while True:
        try:
            es = app.state.es
            await run_email_poll(es)
        except asyncio.CancelledError:
            logger.info("Email scheduler: cancelled, shutting down.")
            break
        except Exception:
            logger.exception("Email scheduler: unhandled error in poll cycle")

        try:
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            logger.info("Email scheduler: cancelled during sleep, shutting down.")
            break
