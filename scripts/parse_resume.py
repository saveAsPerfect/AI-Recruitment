"""
Prostoj skript dlja proverki parsera rezjume.
Berdjot fajl, isvlekaet tekst, parsit cherez LLM i vyvodit JSON.

Zapusk:
    python scripts/parse_resume.py path/to/resume.pdf
"""
import sys
import os
import json
import argparse

# Root project -> sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from app.core.config import get_settings
from app.utils.resume_parser import extract_text, parse_resume_with_llm, build_candidate, TextExtractionError, ResumeParseError


def main():
    parser = argparse.ArgumentParser(description="Parse resume and output JSON")
    parser.add_argument("file", help="Path to the resume file (PDF/DOCX/TXT)")
    args = parser.parse_args()

    file_path = args.file
    settings = get_settings()

    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Extracting text from {file_path}...", file=sys.stderr)
    try:
        raw_text = extract_text(file_path)
    except TextExtractionError as e:
        print(f"[ERROR] Text extraction failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Text extracted ({len(raw_text)} chars). Parsing with LLM...", file=sys.stderr)
    try:
        parsed_data = parse_resume_with_llm(
            text=raw_text, 
            api_key=settings.OPENAI_API_KEY, 
            model=settings.OPENAI_PARSE_MODEL
        )
    except ResumeParseError as e:
        print(f"[ERROR] LLM parsing failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Build standard candidate dict
    candidate_dict = build_candidate(parsed_data, raw_text)
    
    # We dont need to print the huge raw_text in the output log to keep it clean
    del candidate_dict["raw_text"]

    print("\n[RESULT]")
    # Print JSON output to stdout
    print(json.dumps(candidate_dict, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
