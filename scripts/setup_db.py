#!/usr/bin/env python
"""
scripts/setup_db.py
────────────────────
One-time database setup script.
Run this ONCE after starting PostgreSQL to enable pgvector
and verify the connection before ingesting documents.

Usage:
    python scripts/setup_db.py
"""

import sys

import psycopg2
from rich.console import Console
from rich.panel import Panel

# Load settings before importing anything else
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.settings import settings

console = Console()


def setup():
    console.print(
        Panel.fit(
            "[bold cyan]RAG Assistant — Database Setup[/bold cyan]",
            border_style="cyan",
        )
    )

    # Parse connection string for psycopg2
    url = settings.postgres_url.replace("postgresql+psycopg2://", "postgresql://")

    console.print(f"\n[dim]Connecting to: {url.split('@')[-1]}[/dim]")

    try:
        conn = psycopg2.connect(url)
        conn.autocommit = True
        cur = conn.cursor()

        # 1. Enable pgvector extension
        console.print("\n[cyan]Step 1:[/cyan] Enabling pgvector extension…")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        console.print("[green]  ✓ pgvector extension enabled[/green]")

        # 2. Verify
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
        row = cur.fetchone()
        if row:
            console.print(f"[green]  ✓ pgvector version: {row[0]}[/green]")
        else:
            console.print("[red]  ✗ pgvector not found — check your PostgreSQL image[/red]")
            sys.exit(1)

        # 3. Show existing tables
        cur.execute("""
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public'
            ORDER BY tablename;
        """)
        tables = [r[0] for r in cur.fetchall()]
        if tables:
            console.print(f"\n[dim]Existing tables: {', '.join(tables)}[/dim]")
        else:
            console.print("\n[dim]No tables yet — they will be created on first ingest.[/dim]")

        cur.close()
        conn.close()

        console.print(
            "\n[bold green]✓ Database ready![/bold green] "
            "You can now run: [cyan]python scripts/ingest.py[/cyan]"
        )

    except psycopg2.OperationalError as e:
        console.print(f"\n[bold red]✗ Connection failed:[/bold red] {e}")
        console.print(
            "\n[yellow]Is PostgreSQL running?[/yellow]\n"
            "  • Local:  make sure postgres is started\n"
            "  • Docker: run [cyan]docker-compose up -d postgres[/cyan] first"
        )
        sys.exit(1)


if __name__ == "__main__":
    setup()
