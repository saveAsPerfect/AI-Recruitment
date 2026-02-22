# 🤖 AI Recruiting Agent v2

**Stack:** Elasticsearch · LangChain · OpenAI · sentence-transformers · FastAPI · Streamlit · Docker

---

## Data Model

```python
# Candidate
{
  id, name, email,
  role,              # "Senior Python Developer"
  skills,            # ["python", "fastapi", "docker"]
  education,         # "Computer Science"  (specialty only)
  experience,        # free-form text about work history
}

# Vacancy
{
  id, title, role,
  required_skills,       # ["python", "fastapi", "docker"]
  required_education,    # "Computer Science"
  description,           # full job description
}
```

---

## Matching Methods

| Method | Description | When to use |
|--------|-------------|-------------|
| **BM25** | ES `multi_match` with field boosts | Fast baseline, large pools |
| **Semantic** | KNN dense vector cosine similarity | Concept/synonym matching |
| **LLM** | BM25(top-20) → GPT-4o-mini scoring | High accuracy, small pool |
| **Hybrid** | BM25 + Dense → RRF → cosine rerank → LLM | Best overall quality |

### Hybrid Pipeline

```
Vacancy
  ├─► BM25 (top-20) ──────────────────┐
  │                                    ├─► RRF Fusion
  ├─► Dense KNN (top-20) ─────────────┘
  │                          │
  │               Cosine Rerank (stored embeddings)
  │                          │
  └─► LLM (GPT-4o-mini) ◄───┘  cached in ES
```

### LLM Cache

`llm_cache` index stores `(vacancy_id, candidate_id) → score, explanation, updated_at`.  
Cache is checked before every LLM call — no redundant API costs.

---

## Quick Start

```bash
# 1. Configure
cp .env.example .env
# Edit .env → add OPENAI_API_KEY

# 2. Start
docker compose up --build

# 3. Seed sample data
docker compose exec api python scripts/seed_data.py

# 4. Access
# Swagger:   http://localhost:8000/docs
# Streamlit: http://localhost:8501
# ES:        http://localhost:9200
```

---

## API Endpoints

```http
GET  /api/v1/recommendations?job_id=<id>&method=hybrid&top_k=5
POST /api/v1/candidates/upload         — PDF/DOCX/TXT → LangChain parse → ES
POST /api/v1/candidates                — manual candidate creation
GET  /api/v1/candidates
POST /api/v1/vacancies
GET  /api/v1/vacancies
GET  /api/v1/health
```

---

## Project Structure

```
ai-recruiting-agent/
├── app/
│   ├── main.py                  # FastAPI app + ES lifespan
│   ├── api/routes.py            # REST endpoints
│   ├── core/
│   │   ├── config.py            # Settings via env vars
│   │   ├── elasticsearch.py     # ES client + index mappings
│   │   ├── storage.py           # ES CRUD (candidates, vacancies, cache)
│   │   ├── embeddings.py        # Singleton embedding service
│   │   └── matching.py          # BM25 · Semantic · LLM · Hybrid + RRF
│   ├── models/schemas.py        # Pydantic schemas
│   └── utils/resume_parser.py   # LangChain + OpenAI structured parsing
├── streamlit_app.py             # Frontend
├── scripts/seed_data.py         # Sample data loader
├── notebooks/                   # Architecture + demo notebook
├── Dockerfile
├── Dockerfile.streamlit
├── docker-compose.yml           # ES + API + Frontend
└── .env.example
```
