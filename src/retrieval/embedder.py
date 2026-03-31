"""
src/retrieval/embedder.py
─────────────────────────
Wraps the Ollama embedding model (nomic-embed-text by default).
Returns a LangChain-compatible Embeddings object used by the
vector store and query pipeline.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_ollama import OllamaEmbeddings

from config.settings import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embedder() -> OllamaEmbeddings:
    """
    Return a cached Ollama embeddings instance.

    Cached so the same object is reused across ingestion and queries,
    avoiding repeated model warm-up overhead.
    """
    log.info(
        "embedder_initialised",
        model=settings.ollama_embed_model,
        base_url=settings.ollama_base_url,
    )
    return OllamaEmbeddings(
        model=settings.ollama_embed_model,
        base_url=settings.ollama_base_url,
    )
