"""
src/guardrails/guardrails.py
─────────────────────────────
Guardrails applied before and after generation to ensure
the assistant only answers from retrieved evidence.

Checks implemented:
1. Minimum retrieved chunks  — refuse if nothing useful found
2. Minimum similarity score  — refuse if best match is too weak
3. Prompt injection detection — warn if suspicious patterns in chunks
4. Answer length sanity      — warn if answer is suspiciously short
"""

from __future__ import annotations

import re
from typing import List

from langchain_core.documents import Document

from config.settings import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

# Fallback message shown to the user
FALLBACK_MESSAGE = (
    "I could not find a reliable answer in the indexed documents. "
    "Please consult the source document directly or rephrase your question."
)

# Patterns that may indicate prompt injection inside a document
_INJECTION_PATTERNS = [
    re.compile(r"ignore (all )?(previous|prior|above) instructions", re.IGNORECASE),
    re.compile(r"you are now", re.IGNORECASE),
    re.compile(r"(system|assistant):\s*", re.IGNORECASE),
    re.compile(r"<\|im_start\|>", re.IGNORECASE),
]


class GuardrailException(Exception):
    """Raised when a guardrail blocks the generation step."""


def check_retrieval(
    docs: List[Document],
    scores: List[float],
    min_chunks: int = 1,
) -> None:
    """
    Validate that retrieval produced usable context.

    Raises
    ------
    GuardrailException
        When no chunks were retrieved, or all scores are below threshold.
    """
    if not docs:
        raise GuardrailException(FALLBACK_MESSAGE)

    best_score = max(scores) if scores else 0.0
    if best_score < settings.min_similarity_score:
        log.warning(
            "low_similarity_retrieval",
            best_score=best_score,
            threshold=settings.min_similarity_score,
        )
        raise GuardrailException(FALLBACK_MESSAGE)

    # Prompt injection scan
    for doc in docs:
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(doc.page_content):
                log.warning(
                    "prompt_injection_detected",
                    source=doc.metadata.get("source", "unknown"),
                    pattern=pattern.pattern,
                )
                # Sanitise: replace the suspicious content
                doc.page_content = pattern.sub("[REDACTED]", doc.page_content)


def check_answer(answer: str) -> str:
    """
    Post-generation check on the LLM output.

    Currently:
    - Strips leading/trailing whitespace
    - Warns if the answer is suspiciously short (< 10 chars)

    Returns the (possibly modified) answer string.
    """
    answer = answer.strip()

    if len(answer) < 10:
        log.warning("suspiciously_short_answer", answer=answer)

    return answer
