"""
src/retrieval/vectorstore.py
─────────────────────────────
pgvector-backed vector store using PostgreSQL.

The table is created automatically on first use via LangChain's
PGVector integration. Each chunk is stored with:
  - its embedding vector
  - the full page_content
  - all metadata (source, file_name, chunk_id, page, etc.)

Public API
----------
get_vectorstore()   → VectorStore
add_chunks(chunks)  → None  (idempotent upsert by chunk_id)
delete_by_source()  → None  (remove all chunks for a file)
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document

from config.settings import settings
from src.retrieval.embedder import get_embedder
from src.utils.logger import get_logger

log = get_logger(__name__)

# Name of the pgvector collection (maps to a table in Postgres)
_COLLECTION_NAME = "rag_documents"


@lru_cache(maxsize=1)
def get_vectorstore() -> PGVector:
    """
    Return a cached PGVector instance connected to PostgreSQL.

    The pgvector extension and the collection table are created
    automatically by LangChain if they do not already exist.
    """
    log.info(
        "vectorstore_init",
        backend="pgvector",
        collection=_COLLECTION_NAME,
        url=settings.postgres_url.split("@")[-1],  # log host only, not credentials
    )

    store = PGVector(
        collection_name=_COLLECTION_NAME,
        connection_string=settings.postgres_url,
        embedding_function=get_embedder(),
        use_jsonb=True,          # store metadata as JSONB — faster filtering
        pre_delete_collection=False,   # keep existing data on restart
    )
    return store


def add_chunks(chunks: List[Document]) -> None:
    """
    Upsert chunks into pgvector.

    chunk_id is used as the document ID — so re-ingesting the same
    file produces the same IDs and overwrites existing rows cleanly.
    """
    if not chunks:
        log.warning("add_chunks_called_with_empty_list")
        return

    store = get_vectorstore()
    ids = [c.metadata["chunk_id"] for c in chunks]
    store.add_documents(chunks, ids=ids)

    log.info("chunks_upserted_to_pgvector", count=len(chunks))


def delete_by_source(source_path: str) -> None:
    """
    Remove all chunks belonging to a specific source file.
    Useful for incremental updates when a document changes.
    """
    store = get_vectorstore()
    store.delete(filter={"source": source_path})
    log.info("chunks_deleted", source=source_path)


def collection_stats() -> dict:
    """Return basic stats about the current collection."""
    store = get_vectorstore()
    try:
        # PGVector exposes the underlying SQLAlchemy session
        with store._make_sync_session() as session:
            from sqlalchemy import text
            result = session.execute(
                text(
                    "SELECT COUNT(*) FROM langchain_pg_embedding "
                    "WHERE collection_id = ("
                    "  SELECT uuid FROM langchain_pg_collection WHERE name = :name"
                    ")"
                ),
                {"name": _COLLECTION_NAME},
            )
            count = result.scalar() or 0
        return {"collection": _COLLECTION_NAME, "total_chunks": count}
    except Exception as exc:   # noqa: BLE001
        log.warning("stats_unavailable", error=str(exc))
        return {"collection": _COLLECTION_NAME, "total_chunks": "unknown"}
