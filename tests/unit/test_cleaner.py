"""tests/unit/test_cleaner.py"""

import pytest
from langchain_core.documents import Document

from src.ingestion.cleaner import clean_text, clean_documents


class TestCleanText:
    def test_strips_whitespace(self):
        assert clean_text("  hello  ") == "hello"

    def test_collapses_multiple_spaces(self):
        result = clean_text("too   many    spaces")
        assert "  " not in result

    def test_removes_page_number_lines(self):
        text = "Some content\nPage 3 of 17\nMore content"
        result = clean_text(text)
        assert "Page 3 of 17" not in result

    def test_removes_bare_page_numbers(self):
        text = "Content before\n  42  \nContent after"
        result = clean_text(text)
        assert "\n  42  \n" not in result

    def test_fixes_hyphen_line_breaks(self):
        text = "informa-\ntion is important"
        result = clean_text(text)
        assert "information" in result

    def test_collapses_triple_newlines(self):
        text = "Para one\n\n\n\nPara two"
        result = clean_text(text)
        assert "\n\n\n" not in result

    def test_removes_null_bytes(self):
        text = "hello\x00world"
        result = clean_text(text)
        assert "\x00" not in result

    def test_empty_string_returns_empty(self):
        assert clean_text("") == ""

    def test_preserves_meaningful_content(self):
        text = "The vacation policy allows 25 days per year."
        assert clean_text(text) == text


class TestCleanDocuments:
    def _doc(self, content: str) -> Document:
        return Document(page_content=content, metadata={"source": "test.pdf"})

    def test_cleans_all_documents(self):
        docs = [self._doc("  hello  "), self._doc("  world  ")]
        cleaned = clean_documents(docs)
        assert cleaned[0].page_content == "hello"
        assert cleaned[1].page_content == "world"

    def test_drops_empty_documents(self):
        docs = [self._doc("real content"), self._doc("   "), self._doc("\n\n")]
        cleaned = clean_documents(docs)
        assert len(cleaned) == 1

    def test_empty_input_returns_empty(self):
        assert clean_documents([]) == []
