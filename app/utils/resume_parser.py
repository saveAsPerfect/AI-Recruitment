"""
Resume parser: text extraction + LangChain/OpenAI structured parsing.

Text extraction pipeline for PDF (three attempts, in order):
  1. pdfminer.six  — best for text-layer PDFs
  2. PyPDF2        — alternative text-layer parser
  3. Tesseract OCR — for scanned/image-only PDFs
       pdf → pdf2image (poppler) → PIL images → pytesseract → text

DOCX:  python-docx
TXT:   direct read

Parsing:
  LangChain + OpenAI → structured JSON (name, email, role, skills, education, experience)
  If OPENAI_API_KEY is not set → raises ResumeParseError with a clear message.
  There is intentionally NO regex fallback: regex extraction produces low-quality
  structured data that silently degrades matching quality. Fail fast and visibly.
"""
import logging
import uuid
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Minimum characters considered "successfully extracted"
_MIN_TEXT_LENGTH = 50


# ── Custom exceptions ─────────────────────────────────────────────────────────

class TextExtractionError(RuntimeError):
    """Raised when no method could extract readable text from the file."""


class ResumeParseError(RuntimeError):
    """Raised when structured parsing is not possible (e.g. no API key)."""


# ── Text extraction ───────────────────────────────────────────────────────────

def extract_text(file_path: str) -> str:
    """
    Extract raw text from a resume file.

    PDF extraction order:
      1. pdfminer  (text-layer)
      2. PyPDF2    (text-layer fallback)
      3. Tesseract OCR (scanned / image PDF)

    Raises TextExtractionError if all methods fail or return empty text.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _from_pdf(str(path))
    elif suffix in (".docx", ".doc"):
        return _from_docx(str(path))
    elif suffix in (".txt", ".md", ".rtf"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            raise TextExtractionError(f"File is empty: {path.name}")
        return text
    else:
        raise TextExtractionError(f"Unsupported file type: {suffix}")


def _from_pdf(path: str) -> str:
    """Try pdfminer → PyPDF2 → Tesseract OCR, in that order."""

    # ── Attempt 1: pdfminer ───────────────────────────────────────────────────
    text = _pdf_pdfminer(path)
    if _has_enough_text(text):
        logger.debug(f"PDF extracted via pdfminer: {path}")
        return text

    # ── Attempt 2: PyPDF2 ─────────────────────────────────────────────────────
    text = _pdf_pypdf2(path)
    if _has_enough_text(text):
        logger.debug(f"PDF extracted via PyPDF2: {path}")
        return text

    # ── Attempt 3: Tesseract OCR ──────────────────────────────────────────────
    logger.info(
        f"Text-layer extraction returned <{_MIN_TEXT_LENGTH} chars for {path}. "
        "Falling back to Tesseract OCR."
    )
    text = _pdf_ocr(path)
    if _has_enough_text(text):
        logger.info(f"PDF extracted via OCR: {path}")
        return text

    raise TextExtractionError(
        f"Could not extract readable text from PDF '{Path(path).name}'. "
        "All three methods (pdfminer, PyPDF2, Tesseract OCR) returned empty or near-empty output. "
        "The file may be corrupted or in an unsupported format."
    )


def _pdf_pdfminer(path: str) -> str:
    try:
        from pdfminer.high_level import extract_text as _extract
        return _extract(path) or ""
    except Exception as e:
        logger.debug(f"pdfminer failed: {e}")
        return ""


def _pdf_pypdf2(path: str) -> str:
    try:
        import PyPDF2
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except Exception as e:
        logger.debug(f"PyPDF2 failed: {e}")
        return ""


def _pdf_ocr(path: str) -> str:
    """
    Convert each PDF page to an image with pdf2image (uses poppler),
    then run Tesseract OCR via pytesseract.

    Requires system packages: poppler-utils, tesseract-ocr, tesseract-ocr-rus
    (all installed in Docker via apt-get).
    """
    try:
        import pytesseract
        from pdf2image import convert_from_path
        from PIL import Image

        # Convert all pages to PIL images at 300 DPI for good OCR accuracy
        images: list[Image.Image] = convert_from_path(path, dpi=300)
        logger.info(f"OCR: converting {len(images)} page(s) for {Path(path).name}")

        page_texts = []
        for i, img in enumerate(images):
            # lang="rus+eng" handles both Russian and English text
            page_text = pytesseract.image_to_string(img, lang="rus+eng")
            page_texts.append(page_text)
            logger.debug(f"OCR page {i + 1}: {len(page_text)} chars extracted")

        return "\n\n".join(page_texts)

    except ImportError as e:
        missing = str(e).split("'")[1] if "'" in str(e) else str(e)
        logger.error(
            f"OCR dependency missing: {missing}. "
            "Install: pip install pytesseract pdf2image  "
            "and system packages: apt-get install tesseract-ocr tesseract-ocr-rus poppler-utils"
        )
        return ""
    except Exception as e:
        logger.error(f"Tesseract OCR failed: {e}")
        return ""


def _from_docx(path: str) -> str:
    try:
        from docx import Document
        doc = Document(path)
        text = "\n".join(p.text for p in doc.paragraphs)
        if not _has_enough_text(text):
            raise TextExtractionError(f"DOCX appears empty: {Path(path).name}")
        return text
    except TextExtractionError:
        raise
    except Exception as e:
        raise TextExtractionError(f"DOCX extraction failed for {Path(path).name}: {e}") from e


def _has_enough_text(text: str) -> bool:
    return bool(text) and len(text.strip()) >= _MIN_TEXT_LENGTH


# ── LangChain structured parsing ──────────────────────────────────────────────

class _ParsedCandidate(BaseModel):
    """Schema the LLM must return."""
    name: str = Field(description="Full name of the candidate")
    email: Optional[str] = Field(None, description="Email address, or null if absent")
    role: Optional[str] = Field(None, description="Current or most recent job title")
    skills: list[str] = Field(
        default_factory=list,
        description="Technical skills, frameworks and tools — all lowercase strings",
    )
    education: Optional[str] = Field(
        None,
        description="Degree specialty only, e.g. 'Computer Science'. No university name.",
    )
    experience: Optional[str] = Field(
        None,
        description="2-4 sentence plain-text summary of work history (companies, roles, achievements)",
    )


_SYSTEM_PROMPT = """You are an expert HR data extractor.
Given a resume text, extract the following fields and return ONLY valid JSON.
Set missing fields to null (or [] for lists). Do NOT invent data not present in the resume.

Fields:
- name        — full name
- email       — email address or null
- role        — current/most recent job title, e.g. "Senior Python Developer"
- skills      — flat list of technical skills/tools, all lowercase
- education   — degree specialty only, e.g. "Computer Science" (no university name)
- experience  — 2-4 sentence summary of work history"""


def parse_resume_with_llm(text: str, api_key: str, model: str = "gpt-4o-mini") -> dict:
    """
    Parse resume text into structured fields using LangChain + OpenAI.

    Raises ResumeParseError if OPENAI_API_KEY is missing or the LLM call fails.
    There is no silent regex fallback — parsing failures should be visible and fixable.
    """
    if not api_key:
        raise ResumeParseError(
            "OPENAI_API_KEY is not configured. "
            "Set it in your .env file to enable LLM-based resume parsing."
        )

    try:
        return _langchain_parse(text, api_key, model)
    except Exception as e:
        raise ResumeParseError(
            f"LLM resume parsing failed: {e}. "
            "Check your OPENAI_API_KEY and network connectivity."
        ) from e


def _langchain_parse(text: str, api_key: str, model: str) -> dict:
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from app.core.config import get_settings

    settings = get_settings()
    llm = ChatOpenAI(
        model=model,
        temperature=0,
        api_key=api_key,
        base_url=settings.OPENAI_BASE_URL
    )
    parser = JsonOutputParser(pydantic_object=_ParsedCandidate)

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("human", "RESUME TEXT:\n\n{resume}\n\n{format_instructions}"),
    ])

    chain = prompt | llm | parser
    result: dict = chain.invoke({
        "resume": text[:6000],   # stay well within context limits
        "format_instructions": parser.get_format_instructions(),
    })

    # Normalise skills to lowercase non-empty strings
    result["skills"] = [str(s).lower().strip() for s in result.get("skills", []) if s]
    return result


# ── Public interface ──────────────────────────────────────────────────────────

def build_candidate(parsed: dict, raw_text: str) -> dict:
    """Combine LLM-parsed fields with a fresh UUID into a Candidate dict."""
    return {
        "id": str(uuid.uuid4()),
        "name": parsed.get("name") or "Unknown",
        "email": parsed.get("email"),
        "role": parsed.get("role"),
        "skills": parsed.get("skills", []),
        "education": parsed.get("education"),
        "experience": parsed.get("experience"),
        "raw_text": raw_text,
    }
