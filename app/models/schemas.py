"""
Pydantic schemas — new data model per requirements.
"""
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class MatchingMethod(str, Enum):
    BM25 = "bm25"
    SEMANTIC = "semantic"
    LLM = "llm"
    HYBRID = "hybrid"


# ── Domain models ─────────────────────────────────────────────────────────────

class Candidate(BaseModel):
    id: str
    name: str
    email: Optional[str] = None
    role: Optional[str] = None                   # e.g. "Senior Backend Developer"
    skills: list[str] = Field(default_factory=list)
    education: Optional[str] = None              # just the specialty: "Computer Science"
    experience: Optional[str] = None             # free-form text about work history
    raw_text: str = ""                           # original resume text (for re-indexing)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class Vacancy(BaseModel):
    id: str
    title: str
    role: Optional[str] = None                   # target role name
    required_skills: list[str] = Field(default_factory=list)
    required_education: Optional[str] = None     # e.g. "Computer Science"
    description: str = ""                        # full job description text

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ── Result models ─────────────────────────────────────────────────────────────

class MatchResult(BaseModel):
    candidate_id: str
    candidate_name: str
    score: float = Field(ge=0.0, le=1.0)
    bm25_score: Optional[float] = None
    semantic_score: Optional[float] = None
    llm_score: Optional[float] = None
    explanation: Optional[str] = None
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    rank: int = 1
    from_cache: bool = False


class RecommendationResponse(BaseModel):
    job_id: str
    job_title: str
    method: MatchingMethod
    top_candidates: list[MatchResult]
    total_candidates_evaluated: int
    processing_time_seconds: float
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ── LLM cache entry ───────────────────────────────────────────────────────────

class LLMCacheEntry(BaseModel):
    vacancy_id: str
    candidate_id: str
    score: float
    explanation: str
    updated_at: datetime = Field(default_factory=datetime.utcnow)
