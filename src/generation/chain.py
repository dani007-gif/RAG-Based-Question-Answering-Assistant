"""
src/generation/chain.py
────────────────────────
Assembles and runs the full RAG query pipeline:

    User question
        → Retrieval (semantic search)
        → Guardrail check (enough context?)
        → Prompt assembly
        → Ollama LLM
        → Answer + sources

Returns a structured result dict so callers can display
both the answer and its provenance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_ollama import ChatOllama

from config.settings import settings
from src.generation.prompt import QA_PROMPT, format_context
from src.guardrails.guardrails import check_retrieval, GuardrailException
from src.retrieval.retriever import retrieve
from src.utils.logger import get_logger

log = get_logger(__name__)


def _build_llm() -> ChatOllama:
    return ChatOllama(
        model=settings.ollama_llm_model,
        base_url=settings.ollama_base_url,
        temperature=0.1,      # low temperature → factual, deterministic answers
    )


def ask(
    question: str,
    metadata_filter: Optional[Dict] = None,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run the full RAG pipeline for *question*.

    Parameters
    ----------
    question : str
        User's question.
    metadata_filter : dict, optional
        Pass-through to retriever (e.g. restrict to a department).
    top_k : int, optional
        Override retrieval count.

    Returns
    -------
    dict with keys:
        answer   : str   — generated answer (or fallback message)
        sources  : list  — list of source metadata dicts
        retrieved: int   — number of chunks retrieved
        grounded : bool  — True when guardrail passed
    """
    log.info("query_received", question=question[:80])

    # 1. Retrieve relevant chunks
    results = retrieve(question, top_k=top_k, metadata_filter=metadata_filter)
    docs = [doc for doc, _ in results]
    scores = [score for _, score in results]

    # 2. Guardrail: refuse if retrieval quality is too low
    try:
        check_retrieval(docs, scores)
    except GuardrailException as exc:
        log.warning("guardrail_triggered", reason=str(exc))
        return {
            "answer": str(exc),
            "sources": [],
            "retrieved": len(docs),
            "grounded": False,
        }

    # 3. Build context and prompt
    context = format_context(docs)
    prompt = QA_PROMPT.format_messages(context=context, question=question)

    # 4. Generate answer
    llm = _build_llm()
    response = llm.invoke(prompt)
    answer = response.content

    # 5. Collect source citations
    sources = _deduplicate_sources(docs)

    log.info(
        "query_answered",
        question=question[:60],
        chunks_used=len(docs),
        answer_length=len(answer),
    )

    return {
        "answer": answer,
        "sources": sources,
        "retrieved": len(docs),
        "grounded": True,
    }


def _deduplicate_sources(docs: List) -> List[Dict[str, Any]]:
    """Return unique source entries from the retrieved chunks."""
    seen = set()
    sources = []
    for doc in docs:
        key = (
            doc.metadata.get("source", ""),
            doc.metadata.get("page", doc.metadata.get("chunk_index", "")),
        )
        if key not in seen:
            seen.add(key)
            sources.append(
                {
                    "file_name": doc.metadata.get("file_name", "unknown"),
                    "source": doc.metadata.get("source", ""),
                    "page": doc.metadata.get("page", doc.metadata.get("chunk_index", "?")),
                    "file_type": doc.metadata.get("file_type", ""),
                }
            )
    return sources
