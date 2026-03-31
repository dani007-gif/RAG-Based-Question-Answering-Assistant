#!/usr/bin/env python
"""
scripts/stats.py
─────────────────
Show how many chunks are stored in the pgvector collection.

Usage:
    python scripts/stats.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rich.console import Console
from rich.table import Table
from src.retrieval.vectorstore import collection_stats

console = Console()

stats = collection_stats()

table = Table(title="pgvector Collection Stats", show_header=True, header_style="bold cyan")
table.add_column("Key", style="bold")
table.add_column("Value", justify="right")
for k, v in stats.items():
    table.add_row(k, str(v))

console.print(table)
