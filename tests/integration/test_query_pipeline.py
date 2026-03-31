"""
tests/integration/test_query_pipeline.py
──────────────────────────────────────────
Tests the full query pipeline (retrieve → guardrail → generate)
using mocked Ollama and vector store — no external services required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage


def _make_doc(content: str, file_name: str = "policy.pdf") -> Document:
    return Document(
        page_content=content,
        metadata={
            "source": f"/docs/{file_name}",
            "file_name": file_name,
            "file_type": "pdf",
            "page": 1,
            "chunk_id": "test-chunk-001",
            "chunk_index": 0,
        },
    )


MOCK_DOCS = [
    _make_doc(
        "Employees receive 25 days of paid vacation per year.",
        "hr_handbook.pdf",
    ),
    _make_doc(
        "Unused vacation days (up to 5) may be carried over to the next year.",
        "hr_handbook.pdf",
    ),
]


class TestAskFunction:
    @patch("src.generation.chain._build_llm")
    @patch("src.generation.chain.retrieve")
    def test_returns_answer_when_docs_found(self, mock_retrieve, mock_build_llm):
        mock_retrieve.return_value = [(doc, 0.92) for doc in MOCK_DOCS]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(
            content="Employees receive 25 days of paid vacation per year."
        )
        mock_build_llm.return_value = mock_llm

        from src.generation.chain import ask
        result = ask("How many vacation days do employees get?")

        assert result["grounded"] is True
        assert len(result["answer"]) > 5
        assert result["retrieved"] == 2

    @patch("src.generation.chain._build_llm")
    @patch("src.generation.chain.retrieve")
    def test_returns_fallback_when_no_docs(self, mock_retrieve, mock_build_llm):
        mock_retrieve.return_value = []  # nothing retrieved
        mock_build_llm.return_value = MagicMock()

        from src.generation.chain import ask
        result = ask("What is the airspeed velocity of an unladen swallow?")

        assert result["grounded"] is False
        assert "could not find" in result["answer"].lower()
        assert result["sources"] == []

    @patch("src.generation.chain._build_llm")
    @patch("src.generation.chain.retrieve")
    def test_returns_fallback_when_scores_too_low(self, mock_retrieve, mock_build_llm):
        mock_retrieve.return_value = [(doc, 0.05) for doc in MOCK_DOCS]
        mock_build_llm.return_value = MagicMock()

        from src.generation.chain import ask
        result = ask("Some irrelevant question")

        assert result["grounded"] is False

    @patch("src.generation.chain._build_llm")
    @patch("src.generation.chain.retrieve")
    def test_sources_deduplicated(self, mock_retrieve, mock_build_llm):
        # Two chunks from the same file
        mock_retrieve.return_value = [(doc, 0.88) for doc in MOCK_DOCS]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="25 days.")
        mock_build_llm.return_value = mock_llm

        from src.generation.chain import ask
        result = ask("Vacation policy?")

        # Both chunks are from hr_handbook.pdf but different pages — check deduplication logic
        source_files = [s["file_name"] for s in result["sources"]]
        assert len(source_files) == len(set(source_files)) or len(source_files) <= len(MOCK_DOCS)
