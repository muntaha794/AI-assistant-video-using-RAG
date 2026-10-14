"""Prometheus metrics.

Exposed on ``GET /metrics``. Every metric below feeds the observability
goals of the project: processing duration, LLM token usage and error rates.
"""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

REGISTRY = CollectorRegistry()

# ── HTTP ──────────────────────────────────────────────────────────────────────
HTTP_REQUESTS = Counter(
    "http_requests_total",
    "HTTP requests",
    ["method", "path", "status"],
    registry=REGISTRY,
)
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "path"],
    registry=REGISTRY,
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

# ── Jobs / pipeline ───────────────────────────────────────────────────────────
JOBS_TOTAL = Counter(
    "meeting_jobs_total",
    "Meeting processing jobs by terminal status",
    ["status"],
    registry=REGISTRY,
)
JOB_DURATION = Histogram(
    "meeting_job_duration_seconds",
    "End-to-end meeting processing duration",
    registry=REGISTRY,
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1200, 3600),
)
STAGE_DURATION = Histogram(
    "pipeline_stage_duration_seconds",
    "Duration of each pipeline stage",
    ["stage"],
    registry=REGISTRY,
    buckets=(0.1, 0.5, 1.0, 5.0, 15.0, 60.0, 180.0, 600.0, 1800.0),
)
ACTIVE_JOBS = Gauge(
    "active_meeting_jobs",
    "Jobs currently being processed by this process",
    registry=REGISTRY,
)

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_CALLS = Counter(
    "llm_calls_total",
    "LLM calls by model and outcome",
    ["model", "outcome"],
    registry=REGISTRY,
)
LLM_LATENCY = Histogram(
    "llm_call_duration_seconds",
    "LLM call latency",
    ["model"],
    registry=REGISTRY,
    buckets=(0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)
LLM_TOKENS = Counter(
    "llm_tokens_total",
    "Tokens consumed from the LLM API",
    ["model", "direction"],  # direction: prompt | completion
    registry=REGISTRY,
)

# ── RAG ───────────────────────────────────────────────────────────────────────
RAG_QUERIES = Counter(
    "rag_queries_total",
    "RAG chat queries by outcome (ok | ungrounded)",
    ["outcome"],
    registry=REGISTRY,
)
RAG_GROUNDEDNESS = Histogram(
    "rag_groundedness_score",
    "Fraction of answer sentences supported by retrieved context",
    registry=REGISTRY,
    buckets=(0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0),
)
RETRIEVAL_HITS = Counter(
    "rag_retrieved_chunks_total",
    "Chunks returned by retrieval",
    registry=REGISTRY,
)


def render_metrics() -> bytes:
    """Serialize the registry in Prometheus text exposition format."""
    return generate_latest(REGISTRY)
