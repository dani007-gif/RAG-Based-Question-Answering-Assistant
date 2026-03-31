"""
src/ingestion/cleaner.py
────────────────────────
Cleans raw text extracted from documents.

Problems addressed:
- Repeated whitespace / newlines
- PDF header/footer artifacts (page numbers, running titles)
- Broken hyphenation across lines
- Null bytes and non-printable characters
- Excessive punctuation runs
"""

from __future__ import annotations

import re
from typing import List

from langchain_core.documents import Document

from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Regex patterns ──────────────────────────────────────────────────────────

# e.g. "Page 3 of 17", "- 3 -", "3"  on a line by itself
_PAGE_NUMBER_RE = re.compile(
    r"^\s*[-–]?\s*\d{1,4}\s*[-–]?\s*$|"
    r"^\s*[Pp]age\s+\d+\s*(of\s+\d+)?\s*$",
    re.MULTILINE,
)

# Broken hyphenation: "informa-\ntion" → "information"
_HYPHEN_BREAK_RE = re.compile(r"-\n(\S)")

# Three or more consecutive newlines → two newlines (paragraph break)
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

# Multiple spaces / tabs → single space
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")

# Non-printable / control characters (keep newline \n = 0x0A)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x09\x0b-\x1f\x7f]")


def clean_text(text: str) -> str:
    """Apply all cleaning steps to a raw text string."""
    text = _CONTROL_CHAR_RE.sub(" ", text)
    text = _PAGE_NUMBER_RE.sub("", text)
    text = _HYPHEN_BREAK_RE.sub(r"\1", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def clean_documents(documents: List[Document]) -> List[Document]:
    """
    Clean the page_content of every Document in *documents*.

    Empty documents (after cleaning) are dropped and logged.
    """
    cleaned: List[Document] = []
    dropped = 0

    for doc in documents:
        new_content = clean_text(doc.page_content)
        if not new_content:
            log.warning(
                "empty_document_after_cleaning",
                source=doc.metadata.get("source", "unknown"),
            )
            dropped += 1
            continue
        doc.page_content = new_content
        cleaned.append(doc)

    log.info(
        "cleaning_complete",
        kept=len(cleaned),
        dropped=dropped,
    )
    return cleaned
