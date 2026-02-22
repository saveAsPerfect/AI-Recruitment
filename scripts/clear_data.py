"""
Clean script: deletes all indices (candidates, vacancies, llm_cache) from Elasticsearch.
"""
import asyncio
import os
import sys
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.config import get_settings
from app.core.elasticsearch import get_es_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

async def main():
    logger.info("Connecting to Elasticsearch...")
    es = get_es_client()
    
    indices = [
        settings.ES_INDEX_CANDIDATES,
        settings.ES_INDEX_VACANCIES,
        settings.ES_INDEX_LLM_CACHE
    ]
    
    logger.info(f"Indices to delete: {indices}")
    
    for index in indices:
        try:
            exists = await es.indices.exists(index=index)
            if exists:
                await es.indices.delete(index=index)
                logger.info(f"  \u2713 Deleted index: {index}")
            else:
                logger.info(f"  \u26a0 Index does not exist: {index}")
        except Exception as e:
            logger.error(f"  \u2717 Failed to delete index {index}: {e}")

    await es.close()
    logger.info("\n\u2705 Database clean complete!")

if __name__ == "__main__":
    asyncio.run(main())
