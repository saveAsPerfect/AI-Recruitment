"""
Seed script: loads candidates and vacancies from CSV dataset into Elasticsearch.
Dataset: data/synthetic_resume_job_dataset.csv
"""
import asyncio
import os
import sys
import pandas as pd
import re
import uuid
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.config import get_settings
from app.core.elasticsearch import get_es_client, init_indices
from app.core.storage import upsert_candidate, upsert_vacancy
from app.core.embeddings import EmbeddingService
from app.models.schemas import Candidate, Vacancy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_resume_job_dataset.csv")

def parse_resume(text):
    # Example: "Machine Learning Engineer. Skills: PyTorch... Education: Computer Science. Experience..."
    role = "Unknown Role"
    skills = []
    edu = ""
    exp = ""
    
    # Try to extract role (first sentence)
    parts = text.split(". ", 1)
    if len(parts) > 1:
        role = parts[0]
        rest = parts[1]
    else:
        rest = text

    # Extract skills
    skills_match = re.search(r"Skills: (.*?)\. (?:Education|Experience|$)", text)
    if skills_match:
        skills = [s.strip().lower() for s in skills_match.group(1).split(",")]
    
    # Extract education
    edu_match = re.search(r"Education: (.*?)\. (?:Experience|$|Опыт)", text)
    if edu_match:
        edu = edu_match.group(1).strip()
    
    # The rest is experience
    exp_match = re.search(r"(?:Experience|Опыт работы): (.*)$", text, re.IGNORECASE)
    if exp_match:
        exp = exp_match.group(1).strip()
    else:
        exp = text

    return role, skills, edu, exp

def parse_job(text):
    # Example: "Data Scientist (IT Company). Required skills: ... Education: ..."
    title = "Unknown Job"
    role = ""
    skills = []
    edu = ""
    desc = text
    
    # Try to extract title
    parts = text.split(". ", 1)
    if len(parts) > 1:
        title = parts[0]
    
    # Extract skills
    skills_match = re.search(r"Required skills: (.*?)\. (?:Education|$)", text)
    if skills_match:
        skills = [s.strip().lower() for s in skills_match.group(1).split(",")]
        
    # Extract education
    edu_match = re.search(r"Education: (.*?)\. (?:Description|$)", text)
    if edu_match:
        edu = edu_match.group(1).strip()
        
    return title, skills, edu, desc

async def main():
    if not os.path.exists(CSV_PATH):
        logger.error(f"CSV file not found at {CSV_PATH}")
        return

    logger.info(f"Reading CSV from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)

    logger.info("Connecting to Elasticsearch...")
    es = get_es_client()
    await init_indices(es)
    
    embedder = EmbeddingService(settings.EMBEDDING_MODEL)
    
    # ── Seed Vacancies ────────────────────────────────────────────────────────
    logger.info("Seeding Vacancies...")
    unique_jobs = df.drop_duplicates(subset=["vacancy_id"])
    for _, row in unique_jobs.iterrows():
        title, skills, edu, desc = parse_job(row["job_text"])
        v = Vacancy(
            id=str(row["vacancy_id"]),
            title=title,
            role=title,
            required_skills=skills,
            required_education=edu,
            description=desc
        )
        text_for_emb = f"{v.title} {v.role} {' '.join(v.required_skills)} {v.description}"
        emb = embedder.encode_one(text_for_emb)
        await upsert_vacancy(es, v, emb)
        logger.info(f"  ✓ Vacancy: {title}")

    # ── Seed Candidates ───────────────────────────────────────────────────────
    logger.info("Seeding Candidates...")
    unique_candidates = df.drop_duplicates(subset=["candidate_id"])
    
    # Generic names for synthetic candidates
    names = ["Alexander", "Maria", "Dmitry", "Elena", "Ivan", "Olga", "Sergey", "Anna", "Petr", "Svetlana"]
    
    for i, row in unique_candidates.iterrows():
        role, skills, edu, exp = parse_resume(row["resume_text"])
        name = names[i % len(names)] + f" {i}"
        
        c = Candidate(
            id=str(row["candidate_id"]),
            name=name,
            email=f"candidate_{row['candidate_id']}@example.com",
            role=role,
            skills=skills,
            education=edu,
            experience=exp,
            raw_text=row["resume_text"]
        )
        
        text_for_emb = f"{c.role} {' '.join(c.skills)} {c.education} {c.experience}"
        emb = embedder.encode_one(text_for_emb)
        await upsert_candidate(es, c, emb)
        logger.info(f"  ✓ Candidate: {name} ({role})")

    await es.close()
    logger.info("Seed completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
