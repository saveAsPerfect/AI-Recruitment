"""
Application configuration — loaded from environment variables / .env file.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # OpenAI / LLM (Groq / Ollama compatible)
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_PARSE_MODEL: str = "gpt-4o-mini"
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"

    # Inbound email (IMAP)
    IMAP_HOST: str = ""
    IMAP_PORT: int = 993
    IMAP_USER: str = ""
    IMAP_PASSWORD: str = ""
    IMAP_FOLDER: str = "INBOX"
    IMAP_USE_SSL: bool = True
    IMAP_ALLOWED_EXTENSIONS: str = ".pdf,.doc,.docx,.txt"
    # Max size of a single email attachment (bytes). Default: 10 MB.
    IMAP_MAX_ATTACHMENT_BYTES: int = 10 * 1024 * 1024

    # Elasticsearch
    ES_HOST: str = "http://elasticsearch:9200"
    ES_INDEX_CANDIDATES: str = "candidates"
    ES_INDEX_VACANCIES: str = "vacancies"
    ES_INDEX_LLM_CACHE: str = "llm_cache"

    # Embedding model
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # Matching
    TOP_K_RESULTS: int = 5
    BM25_CANDIDATES: int = 20          # BM25 pre-filter before LLM
    BM25_WEIGHT: float = 0.5           # RRF / hybrid blend
    DENSE_WEIGHT: float = 0.5

    # Data
    DATA_DIR: str = "./data"
    RESUMES_DIR: str = "./data/resumes"

    # Email scheduler
    EMAIL_SCHEDULER_ENABLED: bool = True
    EMAIL_SCHEDULER_INTERVAL_MINUTES: int = 60
    EMAIL_SCHEDULER_STARTUP_DELAY_SECONDS: int = 15   # wait for ES on cold start
    EMAIL_SCHEDULER_MAX_MESSAGES: int = 50

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
