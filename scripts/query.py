#!/usr/bin/env python
"""
scripts/query.py
─────────────────
CLI entry point for querying the RAG assistant.

Usage:
    python scripts/query.py                                          # interactive mode
    python scripts/query.py --question "What is the vacation policy?"
    python scripts/query.py --question "..." --top-k 8
    python scripts/query.py --question "..." --filter '{"file_type":"pdf"}'
"""

from __future__ import annotations

import json
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from src.generation.chain import ask
from src.utils.logger import get_logger

log = get_logger(__name__)
console = Console()
app = typer.Typer()


def _display_result(result: dict) -> None:
    """Pretty-print the answer and sources to the terminal."""
    grounded = result.get("grounded", False)
    answer = result["answer"]

    # Answer panel
    panel_style = "green" if grounded else "yellow"
    panel_title = "Answer" if grounded else "Answer (low confidence)"
    console.print(
        Panel(
            Markdown(answer),
            title=panel_title,
            border_style=panel_style,
        )
    )

    # Sources table
    sources = result.get("sources", [])
    if sources:
        table = Table(title="Sources", show_header=True, header_style="bold cyan")
        table.add_column("File", style="dim")
        table.add_column("Page / Chunk")
        table.add_column("Type")
        for src in sources:
            table.add_row(
                src.get("file_name", "—"),
                str(src.get("page", "—")),
                src.get("file_type", "—"),
            )
        console.print(table)
    else:
        console.print("[dim]No sources returned.[/dim]")

    console.print(
        f"\n[dim]Chunks retrieved: {result.get('retrieved', 0)} | "
        f"Grounded: {grounded}[/dim]\n"
    )


@app.command()
def main(
    question: Optional[str] = typer.Option(
        None, "--question", "-q", help="Question to ask"
    ),
    top_k: Optional[int] = typer.Option(
        None, "--top-k", "-k", help="Number of chunks to retrieve"
    ),
    metadata_filter: Optional[str] = typer.Option(
        None,
        "--filter",
        "-f",
        help='JSON metadata filter, e.g. \'{"file_type":"pdf"}\'',
    ),
) -> None:
    console.print(
        Panel.fit(
            "[bold cyan]RAG Assistant[/bold cyan]  —  Ask questions over your documents",
            border_style="cyan",
        )
    )

    parsed_filter = None
    if metadata_filter:
        try:
            parsed_filter = json.loads(metadata_filter)
        except json.JSONDecodeError:
            console.print("[red]Invalid JSON for --filter. Ignoring.[/red]")

    if question:
        # Single-shot mode
        console.print(f"\n[bold]Q:[/bold] {question}\n")
        result = ask(question, metadata_filter=parsed_filter, top_k=top_k)
        _display_result(result)
    else:
        # Interactive REPL mode
        console.print(
            "[dim]Interactive mode. Type your question and press Enter. "
            "Type 'exit' or Ctrl-C to quit.[/dim]\n"
        )
        while True:
            try:
                user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Goodbye.[/dim]")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q"}:
                console.print("[dim]Goodbye.[/dim]")
                break

            result = ask(user_input, metadata_filter=parsed_filter, top_k=top_k)
            _display_result(result)


if __name__ == "__main__":
    app()
