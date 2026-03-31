#!/usr/bin/env python
"""
scripts/ingest.py
──────────────────
CLI entry point for the ingestion pipeline.

Usage:
    python scripts/ingest.py                          # ingest from default dir
    python scripts/ingest.py --dir /path/to/docs      # custom directory
    python scripts/ingest.py --confluence             # also pull Confluence pages
    python scripts/ingest.py --dir ./docs --confluence
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.ingestion.chunker import chunk_documents
from src.ingestion.cleaner import clean_documents
from src.ingestion.loader import load_documents
from src.retrieval.vectorstore import add_chunks
from src.utils.logger import get_logger

log = get_logger(__name__)
console = Console()
app = typer.Typer()


@app.command()
def main(
    directory: Optional[Path] = typer.Option(
        None,
        "--dir",
        "-d",
        help="Directory containing source documents (defaults to RAW_DOCS_DIR in .env)",
    ),
    confluence: bool = typer.Option(
        False,
        "--confluence",
        "-c",
        help="Also ingest pages from Confluence (requires credentials in .env)",
    ),
) -> None:
    console.print(
        Panel.fit(
            "[bold cyan]RAG Assistant — Ingestion Pipeline[/bold cyan]",
            border_style="cyan",
        )
    )

    all_docs = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:

        # ── Step 1: Load local documents ──────────────────────────────
        task = progress.add_task("Loading documents…", total=None)
        docs = load_documents(str(directory) if directory else None)
        progress.update(task, description=f"[green]Loaded {len(docs)} document pages")
        all_docs.extend(docs)

        # ── Step 2: Load Confluence pages (optional) ──────────────────
        if confluence:
            task2 = progress.add_task("Fetching Confluence pages…", total=None)
            from src.ingestion.confluence import load_confluence_pages
            conf_docs = load_confluence_pages()
            progress.update(task2, description=f"[green]Fetched {len(conf_docs)} Confluence pages")
            all_docs.extend(conf_docs)

        # ── Step 3: Clean ─────────────────────────────────────────────
        task3 = progress.add_task("Cleaning text…", total=None)
        cleaned = clean_documents(all_docs)
        progress.update(task3, description=f"[green]Cleaned → {len(cleaned)} documents kept")

        # ── Step 4: Chunk ─────────────────────────────────────────────
        task4 = progress.add_task("Chunking documents…", total=None)
        chunks = chunk_documents(cleaned)
        progress.update(task4, description=f"[green]Created {len(chunks)} chunks")

        # ── Step 5: Embed + store ─────────────────────────────────────
        task5 = progress.add_task("Embedding and storing chunks…", total=None)
        add_chunks(chunks)
        progress.update(task5, description=f"[green]Stored {len(chunks)} chunks in vector index")

    console.print("\n[bold green]✓ Ingestion complete![/bold green]")
    console.print(
        f"  Documents loaded : {len(all_docs)}\n"
        f"  Chunks stored    : {len(chunks)}"
    )


if __name__ == "__main__":
    app()
