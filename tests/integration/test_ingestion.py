"""
tests/integration/test_ingestion.py
─────────────────────────────────────
Integration tests for the full ingestion pipeline.
Uses temporary directories and mock documents — no Ollama required.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.ingestion.chunker import chunk_documents
from src.ingestion.cleaner import clean_documents
from src.ingestion.loader import load_documents


@pytest.fixture()
def sample_text_dir(tmp_path: Path) -> Path:
    """Create a temp directory with sample .txt files."""
    (tmp_path / "doc1.txt").write_text(
        "This is the first document about our vacation policy. "
        "Employees are entitled to 25 days of paid time off per year. "
        "Unused days can be carried over up to a maximum of 5 days.",
        encoding="utf-8",
    )
    (tmp_path / "doc2.txt").write_text(
        "Security policy: All employees must use two-factor authentication. "
        "Passwords must be at least 16 characters long and rotated every 90 days.",
        encoding="utf-8",
    )
    (tmp_path / "ignored.xyz").write_text("Should be ignored.")
    return tmp_path


class TestLoadDocuments:
    def test_loads_txt_files(self, sample_text_dir: Path):
        docs = load_documents(str(sample_text_dir))
        file_names = [d.metadata["file_name"] for d in docs]
        assert "doc1.txt" in file_names
        assert "doc2.txt" in file_names

    def test_ignores_unsupported_extensions(self, sample_text_dir: Path):
        docs = load_documents(str(sample_text_dir))
        file_names = [d.metadata["file_name"] for d in docs]
        assert "ignored.xyz" not in file_names

    def test_metadata_fields_present(self, sample_text_dir: Path):
        docs = load_documents(str(sample_text_dir))
        for doc in docs:
            assert "source" in doc.metadata
            assert "file_name" in doc.metadata
            assert "file_type" in doc.metadata
            assert "file_hash" in doc.metadata

    def test_raises_on_missing_directory(self):
        with pytest.raises(FileNotFoundError):
            load_documents("/nonexistent/path/abc123")


class TestFullIngestionPipeline:
    def test_pipeline_produces_chunks(self, sample_text_dir: Path):
        docs = load_documents(str(sample_text_dir))
        cleaned = clean_documents(docs)
        chunks = chunk_documents(cleaned)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.page_content.strip()
            assert "chunk_id" in chunk.metadata

    def test_chunk_ids_are_unique(self, sample_text_dir: Path):
        docs = load_documents(str(sample_text_dir))
        cleaned = clean_documents(docs)
        chunks = chunk_documents(cleaned)

        ids = [c.metadata["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs detected"

    def test_pipeline_is_idempotent(self, sample_text_dir: Path):
        """Running the pipeline twice should produce the same chunk IDs."""
        def run():
            docs = load_documents(str(sample_text_dir))
            cleaned = clean_documents(docs)
            return {c.metadata["chunk_id"] for c in chunk_documents(cleaned)}

        ids_first = run()
        ids_second = run()
        assert ids_first == ids_second
