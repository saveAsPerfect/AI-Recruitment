"""
Embedding service — wraps sentence-transformers all-MiniLM-L6-v2.
Singleton so the model loads once per process.
"""
import logging
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Singleton wrapper around sentence-transformers."""

    _instance: "EmbeddingService | None" = None

    def __new__(cls, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._model_name = model_name
        return cls._instance

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
            logger.info("Embedding model loaded.")

    def encode(self, texts: Union[str, list[str]]) -> np.ndarray:
        self._load()
        if isinstance(texts, str):
            texts = [texts]
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )

    def encode_one(self, text: str) -> list[float]:
        return self.encode(text)[0].tolist()
