"""
src/retrieval/retriever.py
──────────────────────────
Retrieves relevant document chunks for a user query.

Features:
- Semantic similarity search (top-k)
- Similarity score filtering (min_similarity_score)
- Optional metadata filtering (e.g. filter by file_type or source)
- Returns chunks sorted by relevance with scores attached
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from config.settings import settings
from src.retrieval.vectorstore import get_vectorstore
from src.utils.logger import get_logger

log = get_logger(__name__)


def retrieve(
    query: str,
    top_k: Optional[int] = None,
    metadata_filter: Optional[Dict] = None,
    min_score: Optional[float] = None,
) -> List[Tuple[Document, float]]:
    """
    Retrieve the most relevant chunks for *query*.

    Parameters
    ----------
    query : str
        User's natural-language question.
    top_k : int, optional
        Number of chunks to retrieve.  Defaults to settings.top_k.
    metadata_filter : dict, optional
        ChromaDB / pgvector where-clause, e.g. {"file_type": "pdf"}.
    min_score : float, optional
        Minimum cosine similarity to include a chunk.
        Defaults to settings.min_similarity_score.

    Returns
    -------
    List[Tuple[Document, float]]
        List of (document, similarity_score) pairs, highest score first.
        Returns an empty list when nothing passes the threshold.
    """
    k = top_k or settings.top_k
    threshold = min_score if min_score is not None else settings.min_similarity_score

    store = get_vectorstore()

    try:
        results: List[Tuple[Document, float]] = (
            store.similarity_search_with_relevance_scores(
                query=query,
                k=k,
                filter=metadata_filter,
            )
        )
    except Exception as exc:  # noqa: BLE001
        log.error("retrieval_error", query=query[:80], error=str(exc))
        return []

    # Filter by minimum similarity threshold
    filtered = [(doc, score) for doc, score in results if score >= threshold]

    log.info(
        "retrieval_complete",
        query_preview=query[:60],
        retrieved=len(results),
        after_filter=len(filtered),
        threshold=threshold,
    )

    return filtered


def retrieve_documents(
    query: str,
    top_k: Optional[int] = None,
    metadata_filter: Optional[Dict] = None,
) -> List[Document]:
    """Convenience wrapper — returns only Document objects (no scores)."""
    return [doc for doc, _ in retrieve(query, top_k=top_k, metadata_filter=metadata_filter)]
