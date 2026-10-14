"""Evaluation harness: retrieval metrics and grounded-answer checks.

Metrics
-------
* **Hit rate@k** — fraction of questions whose gold context appears in the
  top-k retrieved chunks.
* **MRR@k** — mean reciprocal rank of the gold context.
* **Grounded-answer check** — a canned *supported* answer must score ≥ the
  groundedness threshold, and a canned *hallucinated* answer must score below
  it. This validates that the groundedness scorer actually discriminates.

The harness runs on a JSONL dataset of (question, context) pairs using the
same chunker/embeddings/vector store as production, so it works offline
(hashing embeddings) or against the real backends.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.services.rag.embeddings import Embeddings
from app.services.rag.store import VectorItem, VectorStore
from app.services.translation import TranslatedSegment


@dataclass
class EvalExample:
    id: str
    question: str
    context: str
    grounded_answer: str
    hallucinated_answer: str
    meeting_id: str = "eval-meeting"
    user_id: str = "eval-user"


@dataclass
class RetrievalMetrics:
    hit_rate: float
    mrr: float
    evaluated: int
    k: int


@dataclass
class GroundednessMetrics:
    grounded_pass_rate: float
    hallucination_flag_rate: float
    grounded_threshold: float
    evaluated: int


@dataclass
class EvalReport:
    retrieval: RetrievalMetrics
    groundedness: GroundednessMetrics
    per_example: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "retrieval": self.retrieval.__dict__,
            "groundedness": self.groundedness.__dict__,
            "per_example": self.per_example,
        }


def load_examples(path: str | Path) -> list[EvalExample]:
    examples: list[EvalExample] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append(EvalExample(**row))
    return examples


def _as_chunk(context: str, idx: int) -> TranslatedSegment:
    """Wrap a dataset context as a one-segment transcript."""
    return TranslatedSegment(
        start_ms=idx * 60_000, end_ms=(idx + 1) * 60_000, text=context, seq=idx
    )


async def build_index(
    examples: list[EvalExample],
    embeddings: Embeddings,
    store: VectorStore,
    collection: str = "eval",
) -> None:
    """Index each example's context as its own retrievable chunk."""
    texts = [example.context for example in examples]
    vectors = await embeddings.embed_documents(texts)
    items = [
        VectorItem(
            id=f"{example.meeting_id}:{idx:04d}",
            document=example.context,
            vector=vector,
            metadata={
                "meeting_id": example.meeting_id,
                "user_id": example.user_id,
                "chunk_id": f"c{idx:04d}",
                "start_ms": idx * 60_000,
                "end_ms": (idx + 1) * 60_000,
                "speakers": "SPEAKER_00",
            },
        )
        for idx, (example, vector) in enumerate(zip(examples, vectors, strict=True))
    ]
    await store.upsert(collection, items)


async def evaluate_retrieval(
    examples: list[EvalExample],
    embeddings: Embeddings,
    store: VectorStore,
    *,
    k: int = 4,
    collection: str = "eval",
) -> RetrievalMetrics:
    """Hit rate@k and MRR@k of the gold context per question."""
    hits = 0
    reciprocal_rank_sum = 0.0
    for example in examples:
        query = await embeddings.embed_query(example.question)
        retrieved = await store.query(
            collection, query, k=k, where={"meeting_id": example.meeting_id}
        )
        rank = None
        for position, hit in enumerate(retrieved, start=1):
            if hit.document.strip() == example.context.strip():
                rank = position
                break
        if rank is not None:
            hits += 1
            reciprocal_rank_sum += 1.0 / rank
    count = max(len(examples), 1)
    return RetrievalMetrics(
        hit_rate=hits / count,
        mrr=reciprocal_rank_sum / count,
        evaluated=len(examples),
        k=k,
    )


def evaluate_groundedness(
    examples: list[EvalExample], *, threshold: float = 0.7
) -> GroundednessMetrics:
    """A grounded answer must pass; a hallucinated one must not."""
    from app.services.rag.engine import groundedness_score

    grounded_passes = 0
    hallucination_flagged = 0
    for example in examples:
        grounded_score = groundedness_score(example.grounded_answer, example.context)
        hallucinated_score = groundedness_score(example.hallucinated_answer, example.context)
        if grounded_score >= threshold:
            grounded_passes += 1
        if hallucinated_score < threshold:
            hallucination_flagged += 1
    count = max(len(examples), 1)
    return GroundednessMetrics(
        grounded_pass_rate=grounded_passes / count,
        hallucination_flag_rate=hallucination_flagged / count,
        grounded_threshold=threshold,
        evaluated=len(examples),
    )


async def run_evaluation(
    examples: list[EvalExample],
    embeddings: Embeddings,
    store: VectorStore,
    *,
    k: int = 4,
    groundedness_threshold: float = 0.7,
) -> EvalReport:
    await build_index(examples, embeddings, store)
    retrieval = await evaluate_retrieval(examples, embeddings, store, k=k)
    groundedness = evaluate_groundedness(examples, threshold=groundedness_threshold)
    return EvalReport(retrieval=retrieval, groundedness=groundedness)
