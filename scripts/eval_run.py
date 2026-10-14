#!/usr/bin/env python
"""Run the retrieval + groundedness evaluation and print a JSON report.

Usage:
    python scripts/eval_run.py                       # offline defaults (hashing + memory)
    python scripts/eval_run.py --dataset eval/dataset.jsonl --top-k 6

Exit code is non-zero when metrics fall below the thresholds, so this can gate
CI or a pre-release checklist.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.retrieval import load_examples, run_evaluation  # noqa: E402
from app.services.rag.embeddings import HashingEmbeddings  # noqa: E402
from app.services.rag.store import InMemoryVectorStore  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="eval/dataset.jsonl")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--min-hit-rate", type=float, default=0.75)
    parser.add_argument("--min-mrr", type=float, default=0.5)
    parser.add_argument("--min-grounded-pass-rate", type=float, default=1.0)
    args = parser.parse_args()

    examples = load_examples(args.dataset)
    if not examples:
        print(f"No examples loaded from {args.dataset}", file=sys.stderr)
        return 2

    report = await run_evaluation(
        examples, HashingEmbeddings(384), InMemoryVectorStore(), k=args.top_k
    )
    payload = report.as_dict()
    print(json.dumps(payload, indent=2))

    retrieval = payload["retrieval"]
    groundedness = payload["groundedness"]
    failures: list[str] = []
    if retrieval["hit_rate"] < args.min_hit_rate:
        failures.append(f"hit_rate {retrieval['hit_rate']:.2f} < {args.min_hit_rate}")
    if retrieval["mrr"] < args.min_mrr:
        failures.append(f"mrr {retrieval['mrr']:.2f} < {args.min_mrr}")
    if groundedness["grounded_pass_rate"] < args.min_grounded_pass_rate:
        rate = groundedness["grounded_pass_rate"]
        failures.append(f"grounded_pass_rate {rate:.2f} < {args.min_grounded_pass_rate}")
    if groundedness["hallucination_flag_rate"] < 1.0:
        failures.append("hallucinated answers were not all flagged as ungrounded")

    if failures:
        print("\nEVAL FAILED:\n- " + "\n- ".join(failures), file=sys.stderr)
        return 1
    print("\nEVAL PASSED", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(asyncio.run(main()))
