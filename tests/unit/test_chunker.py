"""tests/unit/test_chunker.py"""

import pytest
from langchain_core.documents import Document

from src.ingestion.chunker import chunk_documents


def _make_doc(text: str, source: str = "test.pdf") -> Document:
    return Document(
        page_content=text,
        metadata={"source": source, "file_name": source, "file_type": "pdf", "file_hash": "abc"},
    )


class TestChunkDocuments:
    def test_empty_input_returns_empty(self):
        assert chunk_documents([]) == []

    def test_short_doc_produces_one_chunk(self):
        doc = _make_doc("This is a short document.")
        chunks = chunk_documents([doc])
        assert len(chunks) >= 1

    def test_long_doc_produces_multiple_chunks(self):
        # ~2000 chars — well above any reasonable chunk_size
        long_text = "Word " * 400
        doc = _make_doc(long_text)
        chunks = chunk_documents([doc])
        assert len(chunks) > 1

    def test_chunk_has_required_metadata(self):
        doc = _make_doc("Some content for metadata check.")
        chunks = chunk_documents([doc])
        assert chunks, "Expected at least one chunk"
        meta = chunks[0].metadata
        assert "chunk_id" in meta
        assert "chunk_index" in meta
        assert "total_chunks" in meta

    def test_chunk_id_is_deterministic(self):
        doc = _make_doc("Stable content that should produce stable IDs.")
        chunks_a = chunk_documents([doc])
        chunks_b = chunk_documents([doc])
        ids_a = [c.metadata["chunk_id"] for c in chunks_a]
        ids_b = [c.metadata["chunk_id"] for c in chunks_b]
        assert ids_a == ids_b

    def test_source_metadata_preserved(self):
        doc = _make_doc("Content.", source="my_manual.pdf")
        chunks = chunk_documents([doc])
        for chunk in chunks:
            assert chunk.metadata.get("file_name") == "my_manual.pdf"

    def test_multiple_docs_chunked_independently(self):
        docs = [_make_doc(f"Document {i}. " * 50, f"doc_{i}.pdf") for i in range(3)]
        chunks = chunk_documents(docs)
        sources = {c.metadata["file_name"] for c in chunks}
        assert sources == {"doc_0.pdf", "doc_1.pdf", "doc_2.pdf"}
