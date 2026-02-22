"""
Prostoj skript dlja proverki email_inbox.
Podkljuchaetsja k real'nomu IMAP, ishhet pis'ma s rezjume i sohranjat vlozhenija.

Zapusk:
    python scripts/check_email.py
    python scripts/check_email.py --max 5 --include-seen
"""
import sys
import os
import argparse

# Root project -> sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from app.core.config import get_settings
from app.core.email_inbox import fetch_resume_attachments, EmailInboxError


def main():
    parser = argparse.ArgumentParser(description="Check email_inbox")
    parser.add_argument("--max", type=int, default=10, help="Max messages (default 10)")
    parser.add_argument("--include-seen", action="store_true", help="Include seen messages")
    parser.add_argument("--no-mark-seen", action="store_true", help="Do NOT mark messages as seen")
    args = parser.parse_args()

    settings = get_settings()

    # -- 1. Show settings --------------------------------------------------
    print("=" * 60)
    print("  EMAIL INBOX CHECK")
    print("=" * 60)
    print(f"  IMAP_HOST:       {settings.IMAP_HOST or '[NOT SET]'}")
    print(f"  IMAP_PORT:       {settings.IMAP_PORT}")
    print(f"  IMAP_USER:       {settings.IMAP_USER or '[NOT SET]'}")
    print(f"  IMAP_PASSWORD:   {'***' if settings.IMAP_PASSWORD else '[NOT SET]'}")
    print(f"  IMAP_FOLDER:     {settings.IMAP_FOLDER}")
    print(f"  IMAP_USE_SSL:    {settings.IMAP_USE_SSL}")
    print(f"  ALLOWED_EXT:     {settings.IMAP_ALLOWED_EXTENSIONS}")
    print(f"  RESUMES_DIR:     {settings.RESUMES_DIR}")
    print(f"  max_messages:    {args.max}")
    print(f"  include_seen:    {args.include_seen}")
    print(f"  mark_seen:       {not args.no_mark_seen}")
    print("=" * 60)

    if not settings.IMAP_HOST or not settings.IMAP_USER or not settings.IMAP_PASSWORD:
        print("\n[ERROR] IMAP not configured. Set IMAP_HOST, IMAP_USER, IMAP_PASSWORD in .env")
        sys.exit(1)

    # -- 2. Connect and fetch -----------------------------------------------
    print("\n[...] Connecting to IMAP...")

    try:
        result = fetch_resume_attachments(
            max_messages=args.max,
            include_seen=args.include_seen,
            mark_seen=not args.no_mark_seen,
        )
    except EmailInboxError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    # -- 3. Print results ----------------------------------------------------
    fetched = result["fetched_messages"]
    items = result["items"]

    print(f"\n[OK] Done!")
    print(f"   Messages processed:    {fetched}")
    print(f"   Attachments saved:     {len(items)}")

    if not items:
        print("\n[INFO] No resume attachments found.")
        if not args.include_seen:
            print("   Try --include-seen to include already read messages.")
    else:
        print("\n[ATTACHMENTS]")
        print("-" * 60)
        for i, item in enumerate(items, 1):
            print(f"\n  [{i}]")
            print(f"      From:       {item.get('from_email') or '-'}")
            print(f"      Subject:    {item.get('subject') or '-'}")
            print(f"      File:       {item['attachment_name']}")
            print(f"      Saved to:   {item['saved_path']}")
            print(f"      Message-ID: {item.get('message_id') or '-'}")
        print("-" * 60)

    print("\nDone.")


if __name__ == "__main__":
    main()
