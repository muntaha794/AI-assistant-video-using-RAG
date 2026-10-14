"""Evaluation harness tests: retrieval metrics + grounded-answer checks."""

from __future__ import annotations

from pathlib import Path

from app.evaluation.retrieval import (
    evaluate_groundedness,
    load_examples,
    run_evaluation,
)
from app.services.rag.embeddings import HashingEmbeddings
from app.services.rag.store import InMemoryVectorStore

DATASET = Path(__file__).resolve().parents[1] / "eval" / "dataset.jsonl"


def test_dataset_loads() -> None:
    examples = load_examples(DATASET)
    assert len(examples) >= 8
    assert examples[0].question and examples[0].context


def test_groundedness_discriminates_supported_vs_hallucinated() -> None:
    examples = load_examples(DATASET)
    metrics = evaluate_groundedness(examples, threshold=0.7)
    assert metrics.evaluated == len(examples)
    # Every supported answer passes and every hallucinated answer is flagged.
    assert metrics.grounded_pass_rate == 1.0
    assert metrics.hallucination_flag_rate == 1.0


async def test_retrieval_metrics_on_fixture_corpus() -> None:
    examples = load_examples(DATASET)
    report = await run_evaluation(examples, HashingEmbeddings(384), InMemoryVectorStore(), k=4)
    # The hashing embeddings are lexical, and questions share vocabulary with
    # their gold contexts, so retrieval should be strong on this corpus.
    assert report.retrieval.evaluated == len(examples)
    assert report.retrieval.hit_rate >= 0.75, report.retrieval
    assert report.retrieval.mrr >= 0.5, report.retrieval
