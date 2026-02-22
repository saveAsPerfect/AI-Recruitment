"""
Storage layer — all persistence goes through Elasticsearch.

Stores:
  - Candidates with dense_vector embedding
  - Vacancies with dense_vector embedding
  - LLM cache: (vacancy_id, candidate_id) → score + explanation
"""
import logging
from datetime import datetime, timezone
from typing import Optional

from elasticsearch import AsyncElasticsearch, NotFoundError

from app.core.config import get_settings
from app.models.schemas import Candidate, Vacancy, LLMCacheEntry

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Candidate storage ─────────────────────────────────────────────────────────

async def upsert_candidate(es: AsyncElasticsearch, candidate: Candidate, embedding: list[float]) -> None:
    doc = candidate.dict()
    doc["embedding"] = embedding
    await es.index(
        index=settings.ES_INDEX_CANDIDATES,
        id=candidate.id,
        document=doc,
    )


async def get_candidate(es: AsyncElasticsearch, candidate_id: str) -> Optional[dict]:
    try:
        r = await es.get(index=settings.ES_INDEX_CANDIDATES, id=candidate_id)
        return r["_source"]
    except NotFoundError:
        return None


async def list_candidates(es: AsyncElasticsearch, size: int = 1000) -> list[dict]:
    r = await es.search(
        index=settings.ES_INDEX_CANDIDATES,
        body={"query": {"match_all": {}}, "size": size},
    )
    return [h["_source"] for h in r["hits"]["hits"]]


async def delete_candidate(es: AsyncElasticsearch, candidate_id: str) -> bool:
    try:
        await es.delete(index=settings.ES_INDEX_CANDIDATES, id=candidate_id)
        return True
    except NotFoundError:
        return False


# ── Vacancy storage ───────────────────────────────────────────────────────────

async def upsert_vacancy(es: AsyncElasticsearch, vacancy: Vacancy, embedding: list[float]) -> None:
    doc = vacancy.dict()
    doc["embedding"] = embedding
    await es.index(
        index=settings.ES_INDEX_VACANCIES,
        id=vacancy.id,
        document=doc,
    )


async def get_vacancy(es: AsyncElasticsearch, vacancy_id: str) -> Optional[dict]:
    try:
        r = await es.get(index=settings.ES_INDEX_VACANCIES, id=vacancy_id)
        return r["_source"]
    except NotFoundError:
        return None


async def list_vacancies(es: AsyncElasticsearch, size: int = 200) -> list[dict]:
    r = await es.search(
        index=settings.ES_INDEX_VACANCIES,
        body={"query": {"match_all": {}}, "size": size},
    )
    return [h["_source"] for h in r["hits"]["hits"]]


# ── LLM cache ─────────────────────────────────────────────────────────────────

def _cache_id(vacancy_id: str, candidate_id: str) -> str:
    return f"{vacancy_id}_{candidate_id}"


async def get_llm_cache(
    es: AsyncElasticsearch, vacancy_id: str, candidate_id: str
) -> Optional[LLMCacheEntry]:
    try:
        r = await es.get(
            index=settings.ES_INDEX_LLM_CACHE,
            id=_cache_id(vacancy_id, candidate_id),
        )
        src = r["_source"]
        return LLMCacheEntry(
            vacancy_id=src["vacancy_id"],
            candidate_id=src["candidate_id"],
            score=src["score"],
            explanation=src["explanation"],
            updated_at=src["updated_at"],
        )
    except NotFoundError:
        return None


async def set_llm_cache(
    es: AsyncElasticsearch,
    entry: LLMCacheEntry,
) -> None:
    doc = {
        "vacancy_id": entry.vacancy_id,
        "candidate_id": entry.candidate_id,
        "score": entry.score,
        "explanation": entry.explanation,
        "updated_at": entry.updated_at.isoformat(),
    }
    await es.index(
        index=settings.ES_INDEX_LLM_CACHE,
        id=_cache_id(entry.vacancy_id, entry.candidate_id),
        document=doc,
    )


async def get_all_cache_for_vacancy(
    es: AsyncElasticsearch, vacancy_id: str
) -> dict[str, LLMCacheEntry]:
    """Return all cached results for a vacancy as {candidate_id: entry}."""
    r = await es.search(
        index=settings.ES_INDEX_LLM_CACHE,
        body={
            "query": {"term": {"vacancy_id": vacancy_id}},
            "size": 2000,
        },
    )
    result = {}
    for h in r["hits"]["hits"]:
        src = h["_source"]
        entry = LLMCacheEntry(
            vacancy_id=src["vacancy_id"],
            candidate_id=src["candidate_id"],
            score=src["score"],
            explanation=src["explanation"],
            updated_at=src["updated_at"],
        )
        result[src["candidate_id"]] = entry
    return result
