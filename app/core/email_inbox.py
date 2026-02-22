"""
Inbound email integration (IMAP) for automatic resume collection.

Security measures implemented here:
  • Attachment extensions are whitelisted (IMAP_ALLOWED_EXTENSIONS).
  • A hard blacklist of executable / dangerous extensions is applied on top of
    the whitelist, so misconfigured environments cannot accidentally allow them.
  • Attachment **size** is capped at IMAP_MAX_ATTACHMENT_BYTES (default 10 MB).
    Oversized payloads are logged and discarded without being written to disk.
  • Filenames are sanitised to prevent path-traversal attacks — only
    alphanumeric chars, dots, underscores, and hyphens are allowed.
  • No attachment payload is ever executed; bytes are written to disk only.
"""
import imaplib
import logging
import re
from datetime import datetime
from email import message_from_bytes
from email.header import decode_header
from email.utils import parseaddr
from pathlib import Path
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Extensions that must NEVER be saved, regardless of whitelist configuration.
_BLOCKED_EXTENSIONS: frozenset[str] = frozenset({
    # Executables
    ".exe", ".com", ".bat", ".cmd", ".msi", ".ps1", ".psm1", ".psd1",
    # Scripts
    ".sh", ".bash", ".zsh", ".fish", ".ksh", ".csh",
    ".vbs", ".vba", ".js", ".jse", ".wsf", ".wsh",
    # Libraries / loaders
    ".dll", ".so", ".dylib",
    # Archives that may auto-extract and execute
    ".jar", ".scr",
    # Office macros
    ".xlsm", ".xltm", ".xlam", ".docm", ".dotm", ".pptm", ".potm",
    # Other
    ".hta", ".pif", ".lnk", ".reg", ".inf",
})


class EmailInboxError(RuntimeError):
    """Raised when IMAP fetch cannot be completed."""


def fetch_resume_attachments(
    *,
    max_messages: int = 20,
    include_seen: bool = False,
    mark_seen: bool = True,
) -> dict[str, Any]:
    """
    Fetch resume-like attachments from IMAP and save to local storage.

    Security:
      - Only whitelisted extensions are accepted.
      - Blocked/executable extensions are always rejected.
      - Attachments exceeding IMAP_MAX_ATTACHMENT_BYTES are skipped.
      - Filenames are sanitised before writing to disk.

    Returns:
      {
        "fetched_messages": int,
        "items": [
          {
            "message_id": str | None,
            "from_email": str | None,
            "subject": str | None,
            "attachment_name": str,
            "saved_path": str,
          },
          ...
        ]
      }
    """
    _validate_imap_settings()
    allowed_ext = _allowed_extensions(settings.IMAP_ALLOWED_EXTENSIONS)
    save_dir = Path(settings.RESUMES_DIR) / "email"
    save_dir.mkdir(parents=True, exist_ok=True)

    mailbox = None
    try:
        mailbox = _connect_imap()
        status, _ = mailbox.select(settings.IMAP_FOLDER)
        if status != "OK":
            raise EmailInboxError(f"Failed to select IMAP folder: {settings.IMAP_FOLDER}")

        criteria = "ALL" if include_seen else "UNSEEN"
        status, search_data = mailbox.search(None, criteria)
        if status != "OK":
            raise EmailInboxError(f"IMAP search failed with criteria '{criteria}'")

        msg_ids = (search_data[0] or b"").split()
        if not msg_ids:
            return {"fetched_messages": 0, "items": []}
        msg_ids = msg_ids[-max_messages:]

        items: list[dict[str, Any]] = []
        for msg_id in msg_ids:
            status, msg_data = mailbox.fetch(msg_id, "(RFC822)")
            if status != "OK" or not msg_data:
                logger.warning("IMAP fetch failed for message id=%s", msg_id)
                continue

            raw = _extract_raw_email(msg_data)
            if not raw:
                logger.warning("No raw RFC822 payload for message id=%s", msg_id)
                continue

            email_message = message_from_bytes(raw)
            subject = _decode_mime(email_message.get("Subject"))
            from_email = parseaddr(email_message.get("From", ""))[1] or None
            message_id = (email_message.get("Message-ID") or "").strip() or None

            attachments_saved = _save_resume_attachments(
                email_message=email_message,
                save_dir=save_dir,
                allowed_ext=allowed_ext,
                max_bytes=settings.IMAP_MAX_ATTACHMENT_BYTES,
            )

            for saved in attachments_saved:
                items.append(
                    {
                        "message_id": message_id,
                        "from_email": from_email,
                        "subject": subject,
                        "attachment_name": saved["attachment_name"],
                        "saved_path": saved["saved_path"],
                    }
                )

            if mark_seen:
                mailbox.store(msg_id, "+FLAGS", "\\Seen")

        return {"fetched_messages": len(msg_ids), "items": items}

    except EmailInboxError:
        raise
    except Exception as e:
        raise EmailInboxError(f"IMAP receive failed: {e}") from e
    finally:
        if mailbox is not None:
            try:
                mailbox.close()
            except Exception:
                pass
            try:
                mailbox.logout()
            except Exception:
                pass


def _connect_imap():
    if settings.IMAP_USE_SSL:
        mailbox = imaplib.IMAP4_SSL(settings.IMAP_HOST, settings.IMAP_PORT)
    else:
        mailbox = imaplib.IMAP4(settings.IMAP_HOST, settings.IMAP_PORT)
    mailbox.login(settings.IMAP_USER, settings.IMAP_PASSWORD)
    return mailbox


def _validate_imap_settings() -> None:
    required = {
        "IMAP_HOST": settings.IMAP_HOST,
        "IMAP_USER": settings.IMAP_USER,
        "IMAP_PASSWORD": settings.IMAP_PASSWORD,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise EmailInboxError(
            "IMAP is not configured. Missing env: " + ", ".join(missing)
        )


def _allowed_extensions(raw: str) -> set[str]:
    """
    Build the set of *permitted* extensions from the config string,
    then remove any that appear in the hard-blocked list.
    """
    exts = {e.strip().lower() for e in raw.split(",") if e.strip()}
    safe = (exts or {".pdf", ".doc", ".docx", ".txt"}) - _BLOCKED_EXTENSIONS
    blocked_overlap = exts & _BLOCKED_EXTENSIONS
    if blocked_overlap:
        logger.warning(
            "email_inbox: the following extensions are in IMAP_ALLOWED_EXTENSIONS "
            "but are blocked for security: %s", blocked_overlap
        )
    return safe


def _decode_mime(value: str | None) -> str | None:
    if not value:
        return None
    parts = []
    for part, enc in decode_header(value):
        if isinstance(part, bytes):
            parts.append(part.decode(enc or "utf-8", errors="replace"))
        else:
            parts.append(part)
    return "".join(parts).strip() or None


def _extract_raw_email(msg_data: list[Any]) -> bytes:
    for part in msg_data:
        if isinstance(part, tuple) and len(part) >= 2 and isinstance(part[1], (bytes, bytearray)):
            return bytes(part[1])
    return b""


def _safe_filename(name: str) -> str:
    """
    Sanitise a filename to prevent path-traversal and shell-injection.
    Only ASCII letters, digits, dots, hyphens, underscores are kept.
    Leading dots/underscores are stripped to avoid hidden files.
    """
    # Strip any directory component first
    name = Path(name).name
    sanitised = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("._")
    return sanitised or "attachment.bin"


def _save_resume_attachments(
    email_message,
    save_dir: Path,
    allowed_ext: set[str],
    max_bytes: int,
) -> list[dict[str, str]]:
    """
    Walk the email MIME tree and save whitelisted, size-limited attachments.

    Args:
        email_message: Parsed email.message.Message object.
        save_dir:      Directory to write files into.
        allowed_ext:   Set of lowercase extensions that are permitted.
        max_bytes:     Maximum attachment size in bytes; larger payloads are
                       discarded and a warning is logged.

    Returns:
        List of dicts with 'attachment_name' and 'saved_path'.
    """
    saved: list[dict[str, str]] = []

    for idx, part in enumerate(email_message.walk()):
        filename = _decode_mime(part.get_filename())
        if not filename:
            continue

        ext = Path(filename).suffix.lower()

        # Security check 1 — hard-blocked extensions (executables, scripts, etc.)
        if ext in _BLOCKED_EXTENSIONS:
            logger.warning(
                "email_inbox: rejected attachment with blocked extension: '%s'", filename
            )
            continue

        # Security check 2 — whitelist
        if ext not in allowed_ext:
            logger.debug(
                "email_inbox: skipped attachment with non-whitelisted extension: '%s'", filename
            )
            continue

        payload = part.get_payload(decode=True)
        if not payload:
            continue

        # Security check 3 — size limit
        payload_size = len(payload)
        if payload_size > max_bytes:
            logger.warning(
                "email_inbox: attachment '%s' is too large (%d bytes > %d limit), skipped.",
                filename, payload_size, max_bytes,
            )
            continue

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        safe_name = _safe_filename(filename)
        out_path = save_dir / f"{ts}_{idx}_{safe_name}"
        out_path.write_bytes(payload)

        logger.debug(
            "email_inbox: saved attachment '%s' → %s (%d bytes)", filename, out_path, payload_size
        )
        saved.append(
            {
                "attachment_name": filename,
                "saved_path": str(out_path),
            }
        )

    return saved
