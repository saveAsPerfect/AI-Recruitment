"""
Seed script: loads sample vacancies and candidates into Elasticsearch.

Usage:
    ES_HOST=http://localhost:9200 python scripts/seed_data.py
"""
import asyncio
import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.config import get_settings
from app.core.elasticsearch import get_es_client, init_indices
from app.core.storage import upsert_candidate, upsert_vacancy
from app.core.embeddings import EmbeddingService
from app.models.schemas import Candidate, Vacancy

settings = get_settings()

VACANCIES = [
    {
        "id": str(uuid.uuid4()),
        "title": "Senior Python Developer",
        "role": "Backend Developer",
        "required_skills": ["python", "fastapi", "postgresql", "docker"],
        "required_education": "Computer Science",
        "description": (
            "We are looking for a Senior Python Developer to build scalable backend services. "
            "You'll design REST APIs in FastAPI, work with PostgreSQL and Redis, and deploy to AWS "
            "using Docker and Kubernetes."
        ),
    },
    {
        "id": str(uuid.uuid4()),
        "title": "Machine Learning Engineer",
        "role": "ML Engineer",
        "required_skills": ["python", "pytorch", "scikit-learn", "nlp"],
        "required_education": "Computer Science",
        "description": (
            "Ищем ML инженера для работы над рекомендательными системами. "
            "Нужен опыт с PyTorch/TensorFlow, NLP, трансформерами (BERT, GPT). "
            "Будет плюсом знание MLflow, Spark, Docker."
        ),
    },
]

CANDIDATES = [
    {
        "id": str(uuid.uuid4()),
        "name": "Иван Петров",
        "email": "ivan.petrov@example.com",
        "role": "Senior Backend Developer",
        "skills": ["python", "fastapi", "django", "postgresql", "redis", "docker", "kubernetes", "aws", "git"],
        "education": "Computer Science",
        "experience": (
            "5 years backend experience. TechCorp 2020-2024: built microservices in FastAPI, "
            "optimized PostgreSQL queries (3x speedup), introduced Redis caching. "
            "StartupXYZ 2019-2020: REST API development in Flask."
        ),
        "raw_text": "Senior Python Developer 5 years fastapi django postgresql redis docker kubernetes aws",
    },
    {
        "id": str(uuid.uuid4()),
        "name": "Maria Smirnova",
        "email": "maria.smirnova@example.com",
        "role": "ML Engineer",
        "skills": ["python", "pytorch", "tensorflow", "scikit-learn", "nlp", "bert", "mlflow", "pandas", "numpy"],
        "education": "Applied Mathematics",
        "experience": (
            "4 years in ML. DataDriven Inc 2021-2024: recommendation system with BERT embeddings (CTR +25%), "
            "NLP pipeline for RU/EN text classification. "
            "Analytics Corp 2020-2021: feature engineering, A/B testing."
        ),
        "raw_text": "ML engineer pytorch tensorflow scikit-learn NLP BERT MLflow pandas numpy",
    },
    {
        "id": str(uuid.uuid4()),
        "name": "Alex Kozlov",
        "email": "alex.kozlov@example.com",
        "role": "Frontend Developer",
        "skills": ["javascript", "typescript", "react", "vue", "node.js", "mongodb", "docker"],
        "education": "Software Engineering",
        "experience": (
            "3 years frontend. WebAgency 2022-2024: SPA in React/TypeScript, REST API integration. "
            "Junior Developer 2021: HTML/CSS/jQuery."
        ),
        "raw_text": "Frontend developer JavaScript TypeScript React Vue Node.js MongoDB",
    },
    {
        "id": str(uuid.uuid4()),
        "name": "Olga Volkova",
        "email": "olga.volkova@example.com",
        "role": "Data Engineer",
        "skills": ["python", "spark", "kafka", "postgresql", "airflow", "docker", "aws", "sql"],
        "education": "Computer Science",
        "experience": (
            "4 years data engineering. BigData Corp 2021-2024: ETL pipelines in Apache Spark, "
            "Kafka streaming, Airflow orchestration. "
            "StartFintech 2020-2021: PostgreSQL DWH, data modeling."
        ),
        "raw_text": "Data engineer python spark kafka postgresql airflow docker aws",
    },
]


async def main():
    print("Connecting to Elasticsearch...")
    es = get_es_client()
    await init_indices(es)

    embedder = EmbeddingService(settings.EMBEDDING_MODEL)
    print(f"Embedding model: {settings.EMBEDDING_MODEL}")

    print("\nSeeding vacancies...")
    for vdata in VACANCIES:
        v = Vacancy(**vdata)
        text = f"{v.title} {v.role} {' '.join(v.required_skills)} {v.description}"
        emb = embedder.encode_one(text)
        await upsert_vacancy(es, v, emb)
        print(f"  ✓ {v.title}  (id: {v.id[:8]}...)")

    print("\nSeeding candidates...")
    for cdata in CANDIDATES:
        c = Candidate(**cdata)
        text = f"{c.role} {' '.join(c.skills)} {c.education} {c.experience}"
        emb = embedder.encode_one(text)
        await upsert_candidate(es, c, emb)
        print(f"  ✓ {c.name}  ({c.role})")

    await es.close()
    print("\n✅ Seed complete!")
    print("   API docs: http://localhost:8000/docs")
    print("   UI:       http://localhost:8501")


if __name__ == "__main__":
    asyncio.run(main())
