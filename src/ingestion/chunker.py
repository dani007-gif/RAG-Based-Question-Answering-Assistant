"""
src/ingestion/chunker.py
────────────────────────
Splits cleaned documents into retrieval-ready chunks.

Strategy:
1. Split first on markdown headings / double newlines (structural split)
2. Then enforce a hard token-size limit with overlap

Each chunk receives enriched metadata so the retrieval layer can
filter and trace answers back to their source.
"""

from __future__ import annotations

import uuid
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

# Structural separators tried in order before character-level splitting
_SEPARATORS = [
    "\n## ",   # H2 markdown heading
    "\n### ",  # H3
    "\n\n",    # Paragraph break
    "\n",      # Line break
    " ",       # Word boundary
    "",        # Character boundary (last resort)
]


def _make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        separators=_SEPARATORS,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        add_start_index=True,   # adds 'start_index' metadata
    )


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split *documents* into chunks and enrich each chunk with metadata.

    Added metadata per chunk:
    - chunk_id        : deterministic UUID (source + start_index)
    - chunk_index     : sequential index within the source document
    - total_chunks    : total chunks from the same source page
    """
    splitter = _make_splitter()
    chunks: List[Document] = []

    for doc in documents:
        splits = splitter.split_documents([doc])

        for idx, split in enumerate(splits):
            # Deterministic chunk_id based on source + position
            raw_id = f"{split.metadata.get('source', '')}-{split.metadata.get('start_index', idx)}"
            split.metadata["chunk_id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, raw_id))
            split.metadata["chunk_index"] = idx
            split.metadata["total_chunks"] = len(splits)
            chunks.append(split)

    log.info(
        "chunking_complete",
        input_docs=len(documents),
        output_chunks=len(chunks),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return chunks
