"""
AI Recruiting Agent v2 — FastAPI application.
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.elasticsearch import get_es_client, init_indices
from app.api.routes import router

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    Path(settings.RESUMES_DIR).mkdir(parents=True, exist_ok=True)
    es = get_es_client()
    app.state.es = es
    await init_indices(es)
    logger.info("AI Recruiting Agent v2 started ✓")

    # Start background email scheduler (if IMAP is configured and enabled)
    email_task: asyncio.Task | None = None
    if settings.EMAIL_SCHEDULER_ENABLED and settings.IMAP_HOST and settings.IMAP_USER:
        from app.core.email_scheduler import email_scheduler_loop
        email_task = asyncio.create_task(
            email_scheduler_loop(app),
            name="email_scheduler",
        )
        logger.info(
            "Email scheduler task started (interval=%d min).",
            settings.EMAIL_SCHEDULER_INTERVAL_MINUTES,
        )
    else:
        logger.info(
            "Email scheduler disabled "
            "(EMAIL_SCHEDULER_ENABLED=%s, IMAP_HOST='%s', IMAP_USER='%s').",
            settings.EMAIL_SCHEDULER_ENABLED,
            settings.IMAP_HOST,
            settings.IMAP_USER,
        )

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    if email_task is not None and not email_task.done():
        email_task.cancel()
        try:
            await email_task
        except asyncio.CancelledError:
            pass
        logger.info("Email scheduler task stopped.")

    await es.close()
    logger.info("Elasticsearch connection closed.")


app = FastAPI(
    title="AI Recruiting Agent",
    description="""
## AI Recruiting Agent — Elasticsearch + LangChain + LLM

Four matching strategies:

| Method | Description |
|--------|-------------|
| **BM25** | Elasticsearch full-text baseline |
| **Semantic** | Dense vector cosine similarity (all-MiniLM-L6-v2) |
| **LLM** | BM25 top-20 → GPT scoring with explanation |
| **Hybrid** | BM25 + Dense + RRF + cosine rerank + LLM (best) |

### Storage
- **Elasticsearch** — candidates + vacancies + embeddings + LLM cache
- **LangChain + OpenAI** — structured resume parsing

### Cache
LLM results cached per (vacancy_id, candidate_id) — no redundant API calls.
    """,
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.APP_HOST, port=settings.APP_PORT, reload=True)
