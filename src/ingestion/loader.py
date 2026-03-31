"""
src/ingestion/loader.py
───────────────────────
Loads documents from the raw directory.
Supported formats: PDF, Markdown, plain text, DOCX.

Returns a list of LangChain Document objects enriched with metadata.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
)
from langchain_core.documents import Document

from config.settings import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

# Map file extension → loader class
_LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".md": UnstructuredMarkdownLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
}


def _file_hash(path: Path) -> str:
    """SHA-256 of file content — used for change detection."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def load_documents(directory: str | None = None) -> List[Document]:
    """
    Recursively load all supported documents from *directory*.
    Falls back to settings.raw_docs_dir when omitted.

    Returns
    -------
    List[Document]
        Each document carries metadata:
          - source      : absolute file path
          - file_name   : filename only
          - file_type   : extension (pdf / md / txt / docx)
          - file_hash   : SHA-256 for change detection
          - page        : page number (PDFs only)
    """
    raw_dir = Path(directory or settings.raw_docs_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw docs directory not found: {raw_dir}")

    docs: List[Document] = []
    skipped = 0

    for file_path in sorted(raw_dir.rglob("*")):
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()
        loader_cls = _LOADER_MAP.get(ext)

        if loader_cls is None:
            log.debug("skipping_unsupported_file", path=str(file_path))
            skipped += 1
            continue

        try:
            loader = loader_cls(str(file_path))
            pages = loader.load()

            file_hash = _file_hash(file_path)

            for doc in pages:
                doc.metadata.update(
                    {
                        "source": str(file_path.resolve()),
                        "file_name": file_path.name,
                        "file_type": ext.lstrip("."),
                        "file_hash": file_hash,
                    }
                )
            docs.extend(pages)
            log.info("loaded_file", path=file_path.name, pages=len(pages))

        except Exception as exc:  # noqa: BLE001
            log.error("load_failed", path=str(file_path), error=str(exc))

    log.info(
        "loading_complete",
        total_docs=len(docs),
        skipped_files=skipped,
    )
    return docs
