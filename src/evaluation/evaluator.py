"""
src/evaluation/evaluator.py
────────────────────────────
Evaluates retrieval quality against a benchmark of labelled questions.

Metrics computed:
- Recall@k    : fraction of relevant docs found in top-k
- Precision@k : fraction of top-k that are relevant
- Hit Rate    : fraction of questions with ≥1 relevant doc in top-k
- MRR         : Mean Reciprocal Rank

Benchmark format (JSON):
[
  {
    "question": "What is the vacation policy?",
    "relevant_sources": ["hr_handbook.pdf", "vacation_policy.pdf"]
  },
  ...
]

Usage:
    python -m src.evaluation.evaluator --benchmark data/eval_questions.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import typer
from rich.console import Console
from rich.table import Table

from config.settings import settings
from src.retrieval.retriever import retrieve
from src.utils.logger import get_logger

log = get_logger(__name__)
console = Console()
app = typer.Typer()


def _recall_at_k(retrieved_sources: List[str], relevant_sources: List[str]) -> float:
    if not relevant_sources:
        return 1.0
    hits = sum(
        1
        for rel in relevant_sources
        if any(rel in src for src in retrieved_sources)
    )
    return hits / len(relevant_sources)


def _precision_at_k(retrieved_sources: List[str], relevant_sources: List[str]) -> float:
    if not retrieved_sources:
        return 0.0
    hits = sum(
        1
        for src in retrieved_sources
        if any(rel in src for rel in relevant_sources)
    )
    return hits / len(retrieved_sources)


def _reciprocal_rank(retrieved_sources: List[str], relevant_sources: List[str]) -> float:
    for rank, src in enumerate(retrieved_sources, start=1):
        if any(rel in src for rel in relevant_sources):
            return 1.0 / rank
    return 0.0


def evaluate(benchmark: List[Dict[str, Any]], top_k: int | None = None) -> Dict[str, float]:
    k = top_k or settings.top_k

    recalls, precisions, rr_list, hits = [], [], [], []

    for item in benchmark:
        question = item["question"]
        relevant = item.get("relevant_sources", [])

        results = retrieve(question, top_k=k)
        retrieved_sources = [doc.metadata.get("file_name", "") for doc, _ in results]

        recalls.append(_recall_at_k(retrieved_sources, relevant))
        precisions.append(_precision_at_k(retrieved_sources, relevant))
        rr_list.append(_reciprocal_rank(retrieved_sources, relevant))
        hits.append(1 if any(rel in src for src in retrieved_sources for rel in relevant) else 0)

    n = len(benchmark) or 1
    metrics = {
        f"Recall@{k}": sum(recalls) / n,
        f"Precision@{k}": sum(precisions) / n,
        "Hit Rate": sum(hits) / n,
        "MRR": sum(rr_list) / n,
    }
    return metrics


@app.command()
def main(
    benchmark: Path = typer.Option(
        Path("data/eval_questions.json"),
        "--benchmark",
        help="Path to benchmark JSON file",
    ),
    top_k: int = typer.Option(settings.top_k, "--top-k", help="Retrieval k"),
) -> None:
    if not benchmark.exists():
        console.print(f"[red]Benchmark file not found: {benchmark}[/red]")
        raise typer.Exit(1)

    data = json.loads(benchmark.read_text())
    console.print(f"[cyan]Running evaluation on {len(data)} questions (k={top_k})…[/cyan]")

    metrics = evaluate(data, top_k=top_k)

    table = Table(title="Retrieval Evaluation Results", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Score", justify="right")
    for metric, value in metrics.items():
        table.add_row(metric, f"{value:.4f}")

    console.print(table)


if __name__ == "__main__":
    app()
