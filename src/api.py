"""
src/api.py
──────────
FastAPI application exposing the RAG assistant over HTTP.

Endpoints:
  POST /ask           — Ask a question
  POST /ingest        — Trigger document ingestion
  DELETE /documents   — Remove all chunks for a source file
  GET  /stats         — pgvector collection stats
  GET  /health        — Health check
  GET  /docs          — Swagger UI (built-in)

Run:
    uvicorn src.api:app --reload --port 8000
"""

from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config.settings import settings
from src.generation.chain import ask
from src.ingestion.chunker import chunk_documents
from src.ingestion.cleaner import clean_documents
from src.ingestion.loader import load_documents
from src.retrieval.vectorstore import add_chunks, delete_by_source, collection_stats
from src.utils.logger import get_logger

log = get_logger(__name__)

app = FastAPI(
    title="RAG Assistant API",
    description="Enterprise document Q&A — LangChain + Ollama + pgvector",
    version="2.0.0",
)


# ── Request / Response models ─────────────────────────────────────────────


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    metadata_filter: Optional[Dict] = None


class SourceItem(BaseModel):
    file_name: str
    source: str
    page: str | int
    file_type: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    retrieved: int
    grounded: bool


class IngestResponse(BaseModel):
    status: str
    documents_loaded: int
    chunks_stored: int


class DeleteRequest(BaseModel):
    source_path: str = Field(..., description="Absolute path of the source file to remove")


class HealthResponse(BaseModel):
    status: str
    llm_model: str
    embed_model: str
    vector_store: str
    postgres_host: str


# ── Endpoints ─────────────────────────────────────────────────────────────


@app.post("/ask", response_model=AskResponse)
async def ask_question(req: AskRequest) -> AskResponse:
    """Ask a question against the indexed documents."""
    try:
        result = ask(
            question=req.question,
            metadata_filter=req.metadata_filter,
            top_k=req.top_k,
        )
        return AskResponse(**result)
    except Exception as exc:
        log.error("api_ask_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents() -> IngestResponse:
    """Trigger a full re-ingestion from the raw docs directory."""
    try:
        raw_docs = load_documents()
        cleaned = clean_documents(raw_docs)
        chunks = chunk_documents(cleaned)
        add_chunks(chunks)
        return IngestResponse(
            status="success",
            documents_loaded=len(raw_docs),
            chunks_stored=len(chunks),
        )
    except Exception as exc:
        log.error("api_ingest_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/documents")
async def delete_document(req: DeleteRequest) -> dict:
    """Remove all chunks for a specific source file from pgvector."""
    try:
        delete_by_source(req.source_path)
        return {"status": "deleted", "source": req.source_path}
    except Exception as exc:
        log.error("api_delete_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/stats")
async def stats() -> dict:
    """Return pgvector collection statistics."""
    return collection_stats()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    host = settings.postgres_url.split("@")[-1] if "@" in settings.postgres_url else "unknown"
    return HealthResponse(
        status="ok",
        llm_model=settings.ollama_llm_model,
        embed_model=settings.ollama_embed_model,
        vector_store="pgvector",
        postgres_host=host,
    )
