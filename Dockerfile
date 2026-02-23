# ── Stage 1: build deps ───────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model (baked into image — no network needed at runtime)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System dependencies:
#   poppler-utils  — required by pdf2image to convert PDF pages to images
#   tesseract-ocr  — OCR engine
#   tesseract-ocr-rus — Russian language pack for Tesseract
#   tesseract-ocr-eng — English language pack (usually bundled, listed explicitly)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn
COPY --from=builder /root/.cache /root/.cache

COPY app/ ./app/
COPY scripts/ ./scripts/

RUN mkdir -p /app/data/resumes /app/data/vacancies

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
