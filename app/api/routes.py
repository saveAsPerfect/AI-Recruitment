"""
FastAPI routes for AI Recruiting Agent.
"""
import time
import uuid
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, Request
from fastapi.concurrency import run_in_threadpool
from elasticsearch import AsyncElasticsearch

from app.core.config import get_settings
from app.core.email_inbox import fetch_resume_attachments, EmailInboxError
from app.core.email_scheduler import run_email_poll, get_seen_store
from app.core.matching import run_bm25, run_semantic, run_llm, run_hybrid
from app.core.storage import (
    upsert_candidate, upsert_vacancy,
    get_vacancy, list_candidates, list_vacancies,
    delete_candidate,
)
from app.core.embeddings import EmbeddingService
from app.models.schemas import (
    Candidate, Vacancy, MatchResult,
    RecommendationResponse, MatchingMethod,
    EmailIngestRequest, EmailIngestResponse, EmailIngestItem,
)
from app.utils.resume_parser import extract_text, parse_resume_with_llm, build_candidate, TextExtractionError, ResumeParseError, NotResumeError

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()


def get_es(request: Request) -> AsyncElasticsearch:
    return request.app.state.es


def get_embedder() -> EmbeddingService:
    return EmbeddingService(settings.EMBEDDING_MODEL)


def _scored_to_match(r, rank: int) -> MatchResult:
    return MatchResult(
        candidate_id=r.candidate_id,
        candidate_name=r.candidate_name,
        score=r.score,
        bm25_score=r.bm25_score,
        semantic_score=r.semantic_score,
        llm_score=r.llm_score,
        explanation=r.explanation,
        matched_skills=r.matched_skills,
        missing_skills=r.missing_skills,
        rank=rank,
        from_cache=r.from_cache,
    )


# ── Recommendations ───────────────────────────────────────────────────────────

@router.get("/recommendations", response_model=RecommendationResponse, tags=["Matching"])
async def recommendations(
    job_id: str = Query(..., description="Vacancy ID"),
    method: MatchingMethod = Query(MatchingMethod.HYBRID),
    top_k: int = Query(5, ge=1, le=50),
    es: AsyncElasticsearch = Depends(get_es),
):
    """
    **Core endpoint** — returns top-K candidates for a vacancy.

    Methods:
    - `bm25` — Elasticsearch BM25 baseline
    - `semantic` — Dense vector cosine similarity
    - `llm` — BM25 pre-filter → GPT scoring with LLM cache
    - `hybrid` — BM25 + Dense + RRF + cosine rerank + LLM (best quality)
    """
    start = time.time()
    vacancy = await get_vacancy(es, job_id)
    if not vacancy:
        raise HTTPException(404, f"Vacancy '{job_id}' not found")

    embedder = get_embedder()
    vacancy_emb = embedder.encode_one(_vacancy_search_text(vacancy))

    prefilter = settings.BM25_CANDIDATES

    if method == MatchingMethod.BM25:
        scored = await run_bm25(es, vacancy, top_k)
    elif method == MatchingMethod.SEMANTIC:
        scored = await run_semantic(es, vacancy_emb, vacancy, top_k)
    elif method == MatchingMethod.LLM:
        scored = await run_llm(es, vacancy, vacancy_emb, settings.OPENAI_API_KEY, settings.OPENAI_MODEL, top_k, prefilter)
    else:  # HYBRID
        scored = await run_hybrid(es, vacancy, vacancy_emb, settings.OPENAI_API_KEY, settings.OPENAI_MODEL, top_k, prefilter)

    all_candidates = await list_candidates(es)

    return RecommendationResponse(
        job_id=job_id,
        job_title=vacancy["title"],
        method=method,
        top_candidates=[_scored_to_match(r, i + 1) for i, r in enumerate(scored)],
        total_candidates_evaluated=len(all_candidates),
        processing_time_seconds=round(time.time() - start, 3),
    )


# ── Candidates ────────────────────────────────────────────────────────────────

@router.post("/candidates/upload", response_model=Candidate, tags=["Candidates"])
async def upload_resume(
    file: UploadFile = File(...),
    es: AsyncElasticsearch = Depends(get_es),
):
    """
    Upload a resume (PDF / DOCX / TXT).
    LangChain + OpenAI parses it into structured Candidate fields.
    Stored in Elasticsearch with dense vector embedding.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".docx", ".doc", ".txt"}:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    tmp_path = Path(settings.RESUMES_DIR) / f"tmp_{uuid.uuid4().hex}{suffix}"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_bytes(await file.read())

    try:
        return await _ingest_candidate_from_file(tmp_path, es)
    except TextExtractionError as e:
        raise HTTPException(422, str(e))
    except NotResumeError as e:
        raise HTTPException(422, str(e))
    except ResumeParseError as e:
        raise HTTPException(503, str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/candidates", response_model=Candidate, tags=["Candidates"])
async def create_candidate(
    candidate: Candidate,
    es: AsyncElasticsearch = Depends(get_es),
):
    """Manually create / upsert a candidate."""
    if not candidate.id:
        candidate.id = str(uuid.uuid4())
    embedder = get_embedder()
    emb = embedder.encode_one(_candidate_search_text(candidate.dict()))
    await upsert_candidate(es, candidate, emb)
    return candidate


@router.get("/candidates", response_model=list[Candidate], tags=["Candidates"])
async def get_candidates(es: AsyncElasticsearch = Depends(get_es)):
    rows = await list_candidates(es)
    return [Candidate(**{k: v for k, v in r.items() if k != "embedding"}) for r in rows]


@router.delete("/candidates/{candidate_id}", tags=["Candidates"])
async def remove_candidate(candidate_id: str, es: AsyncElasticsearch = Depends(get_es)):
    ok = await delete_candidate(es, candidate_id)
    if not ok:
        raise HTTPException(404, "Candidate not found")
    return {"deleted": candidate_id}


@router.post("/emails/fetch", response_model=EmailIngestResponse, tags=["Email"])
async def fetch_from_email(
    request: EmailIngestRequest,
    es: AsyncElasticsearch = Depends(get_es),
):
    """
    Fetch new resumes from IMAP inbox, save attachments locally, optionally parse into candidates.
    """
    try:
        fetched = await run_in_threadpool(
            fetch_resume_attachments,
            max_messages=request.max_messages,
            include_seen=request.include_seen,
            mark_seen=request.mark_seen,
        )
    except EmailInboxError as e:
        raise HTTPException(503, str(e))

    items: list[EmailIngestItem] = []
    parsed_count = 0
    for item in fetched["items"]:
        base = {
            "message_id": item.get("message_id"),
            "from_email": item.get("from_email"),
            "subject": item.get("subject"),
            "attachment_name": item["attachment_name"],
            "saved_path": item["saved_path"],
        }
        if not request.parse_resumes:
            items.append(EmailIngestItem(**base, status="saved"))
            continue

        try:
            candidate = await _ingest_candidate_from_file(Path(item["saved_path"]), es)
            parsed_count += 1
            items.append(
                EmailIngestItem(**base, status="parsed", candidate_id=candidate.id)
            )
        except NotResumeError as e:
            items.append(EmailIngestItem(**base, status="not_resume", error=str(e)))
        except (TextExtractionError, ResumeParseError) as e:
            items.append(EmailIngestItem(**base, status="failed", error=str(e)))
        except Exception as e:
            logger.exception("Unexpected error while parsing resume from email")
            items.append(EmailIngestItem(**base, status="failed", error=str(e)))

    return EmailIngestResponse(
        fetched_messages=fetched["fetched_messages"],
        saved_attachments=len(fetched["items"]),
        parsed_candidates=parsed_count,
        items=items,
    )


# ── Vacancies ─────────────────────────────────────────────────────────────────

@router.post("/vacancies", response_model=Vacancy, tags=["Vacancies"])
async def create_vacancy(vacancy: Vacancy, es: AsyncElasticsearch = Depends(get_es)):
    """Create / upsert a vacancy. Embeds and stores in Elasticsearch."""
    if not vacancy.id:
        vacancy.id = str(uuid.uuid4())
    embedder = get_embedder()
    emb = embedder.encode_one(_vacancy_search_text(vacancy.dict()))
    await upsert_vacancy(es, vacancy, emb)
    return vacancy


@router.get("/vacancies", response_model=list[Vacancy], tags=["Vacancies"])
async def get_vacancies(es: AsyncElasticsearch = Depends(get_es)):
    rows = await list_vacancies(es)
    return [Vacancy(**{k: v for k, v in r.items() if k != "embedding"}) for r in rows]


@router.get("/vacancies/{vacancy_id}", response_model=Vacancy, tags=["Vacancies"])
async def get_one_vacancy(vacancy_id: str, es: AsyncElasticsearch = Depends(get_es)):
    row = await get_vacancy(es, vacancy_id)
    if not row:
        raise HTTPException(404, "Vacancy not found")
    return Vacancy(**{k: v for k, v in row.items() if k != "embedding"})


# ── Health ────────────────────────────────────────────────────────────────────


@router.get("/health", tags=["System"])
async def health(es: AsyncElasticsearch = Depends(get_es)):
    try:
        info = await es.info()
        es_ok = True
        es_version = info["version"]["number"]
    except Exception as e:
        es_ok = False
        es_version = str(e)

    seen = get_seen_store()
    return {
        "status": "ok" if es_ok else "degraded",
        "elasticsearch": es_ok,
        "es_version": es_version,
        "version": "2.0.0",
        "email_scheduler": {
            "enabled": settings.EMAIL_SCHEDULER_ENABLED,
            "interval_minutes": settings.EMAIL_SCHEDULER_INTERVAL_MINUTES,
            **seen.stats(),
        },
    }


@router.post("/emails/trigger", tags=["Email"])
async def trigger_email_poll(es: AsyncElasticsearch = Depends(get_es)):
    """
    Manually trigger one email-poll cycle immediately (same logic as scheduler).
    Useful for testing or ad-hoc inbox checks without waiting for the next
    scheduled run.
    """
    result = await run_email_poll(es)
    return result


@router.get("/emails/scheduler/status", tags=["Email"])
async def scheduler_status():
    """Return seen-store statistics and scheduler configuration."""
    seen = get_seen_store()
    return {
        "scheduler_enabled": settings.EMAIL_SCHEDULER_ENABLED,
        "interval_minutes": settings.EMAIL_SCHEDULER_INTERVAL_MINUTES,
        "imap_host": settings.IMAP_HOST or None,
        "imap_user": settings.IMAP_USER or None,
        **seen.stats(),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _candidate_search_text(c: dict) -> str:
    parts = [
        c.get("role") or "",
        " ".join(c.get("skills") or []),
        c.get("education") or "",
        c.get("experience") or "",
    ]
    return " ".join(p for p in parts if p)


def _vacancy_search_text(v: dict) -> str:
    parts = [
        v.get("title") or "",
        v.get("role") or "",
        " ".join(v.get("required_skills") or []),
        v.get("required_education") or "",
        v.get("description") or "",
    ]
    return " ".join(p for p in parts if p)


async def _ingest_candidate_from_file(path: Path, es: AsyncElasticsearch) -> Candidate:
    raw_text = extract_text(str(path))
    parsed = parse_resume_with_llm(raw_text, settings.OPENAI_API_KEY, settings.OPENAI_PARSE_MODEL)
    c_dict = build_candidate(parsed, raw_text)
    candidate = Candidate(**c_dict)

    embedder = get_embedder()
    emb = embedder.encode_one(_candidate_search_text(c_dict))
    await upsert_candidate(es, candidate, emb)
    return candidate
