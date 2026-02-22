"""
Elasticsearch client layer.

Indexes:
  candidates  — stores Candidate docs + dense vector (384-dim)
  vacancies   — stores Vacancy docs + dense vector (384-dim)
  llm_cache   — (vacancy_id, candidate_id) → score, explanation, updated_at
"""
import logging
from typing import Optional

from elasticsearch import AsyncElasticsearch, NotFoundError

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def get_es_client() -> AsyncElasticsearch:
    return AsyncElasticsearch(
        settings.ES_HOST,
        retry_on_timeout=True,
        max_retries=3,
    )


# ── Index mappings ────────────────────────────────────────────────────────────

CANDIDATE_MAPPING = {
    "mappings": {
        "properties": {
            "id":         {"type": "keyword"},
            "name":       {"type": "text", "analyzer": "standard"},
            "email":      {"type": "keyword"},
            "role":       {"type": "text", "analyzer": "standard"},
            "skills":     {"type": "keyword"},
            "education":  {"type": "text", "analyzer": "standard"},
            "experience": {"type": "text", "analyzer": "standard"},
            "raw_text":   {"type": "text", "analyzer": "standard"},
            "embedding":  {
                "type": "dense_vector",
                "dims": settings.EMBEDDING_DIM,
                "index": True,
                "similarity": "cosine",
            },
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
}

VACANCY_MAPPING = {
    "mappings": {
        "properties": {
            "id":                 {"type": "keyword"},
            "title":              {"type": "text", "analyzer": "standard"},
            "role":               {"type": "text", "analyzer": "standard"},
            "required_skills":    {"type": "keyword"},
            "required_education": {"type": "text", "analyzer": "standard"},
            "description":        {"type": "text", "analyzer": "standard"},
            "embedding":          {
                "type": "dense_vector",
                "dims": settings.EMBEDDING_DIM,
                "index": True,
                "similarity": "cosine",
            },
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
}

LLM_CACHE_MAPPING = {
    "mappings": {
        "properties": {
            "vacancy_id":   {"type": "keyword"},
            "candidate_id": {"type": "keyword"},
            "score":        {"type": "float"},
            "explanation":  {"type": "text"},
            "updated_at":   {"type": "date"},
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
}


async def init_indices(es: AsyncElasticsearch) -> None:
    """Create ES indices if they don't exist."""
    pairs = [
        (settings.ES_INDEX_CANDIDATES, CANDIDATE_MAPPING),
        (settings.ES_INDEX_VACANCIES,  VACANCY_MAPPING),
        (settings.ES_INDEX_LLM_CACHE,  LLM_CACHE_MAPPING),
    ]
    for index_name, mapping in pairs:
        exists = await es.indices.exists(index=index_name)
        if not exists:
            await es.indices.create(index=index_name, body=mapping)
            logger.info(f"Created ES index: {index_name}")
        else:
            logger.debug(f"ES index already exists: {index_name}")
