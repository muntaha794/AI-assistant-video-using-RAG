"""RAG engine tests: retrieval scoping, citations, groundedness."""

from __future__ import annotations

from app.config import Settings
from app.services.rag.embeddings import HashingEmbeddings
from app.services.rag.engine import (
    RAGEngine,
    groundedness_score,
    parse_citation_ids,
    render_context,
    sentence_support,
)
from app.services.rag.store import InMemoryVectorStore
from app.services.translation import TranslatedSegment
from tests.fakes import FakeLLM

MEETING_ID = "11111111-1111-1111-1111-111111111111"
USER_ID = "22222222-2222-2222-2222-222222222222"


def _segments() -> list[TranslatedSegment]:
    return [
        TranslatedSegment(
            start_ms=0,
            end_ms=12_000,
            text="[0:00] SPEAKER_00: We agreed to prioritize mobile support this quarter.",
            seq=0,
        ),
        TranslatedSegment(
            start_ms=12_000,
            end_ms=30_000,
            text="[0:12] SPEAKER_01: Carol will audit the vendor costs before renewal.",
            seq=1,
        ),
    ]


# ── Pure functions ────────────────────────────────────────────────────────────


def test_parse_citation_ids_dedupes_and_orders() -> None:
    assert parse_citation_ids("First [S2] then [S1] and again [S2].") == [2, 1]


def test_sentence_support_requires_overlap() -> None:
    context_tokens = {"mobile", "support", "prioritized", "quarter"}
    assert sentence_support("We prioritized mobile support this quarter.", context_tokens)
    assert not sentence_support("The CEO bought a yacht in Monaco.", context_tokens)
    # Explicit refusals count as grounded.
    assert sentence_support("The transcript does not contain that information.", set())


def test_groundedness_score_is_fraction_supported() -> None:
    context = "We prioritized mobile support this quarter. Carol will audit vendor costs."
    supported = "We prioritized mobile support this quarter. Carol will audit vendor costs [S1]."
    hallucinated = "We prioritized mobile support. Also, the CEO resigned in disgrace [S1]."
    assert groundedness_score(supported, context) == 1.0
    assert groundedness_score(hallucinated, context) < 0.7
    assert groundedness_score("", context) == 1.0


def test_render_context_includes_ids_and_ranges() -> None:
    from app.services.rag.engine import RetrievedChunk

    chunk = RetrievedChunk(
        chunk_id="c0000",
        text="hello",
        score=0.9,
        start_ms=0,
        end_ms=60_000,
        speakers=["SPEAKER_00"],
    )
    rendered = render_context([chunk])
    assert rendered.startswith("[S1] (0:00-1:00; speakers: SPEAKER_00)")


# ── Engine behaviour ──────────────────────────────────────────────────────────


def _engine(settings: Settings, llm: FakeLLM | None = None) -> RAGEngine:
    return RAGEngine(
        llm=llm or FakeLLM(),
        embeddings=HashingEmbeddings(settings.embedding_dim),
        store=InMemoryVectorStore(),
        settings=settings,
    )


async def test_index_and_answer_produce_timestamp_citations(settings: Settings) -> None:
    engine = _engine(
        settings,
        llm=FakeLLM(
            chat_response=(
                "We agreed to prioritize mobile support this quarter [S1]. "
                "Carol will audit vendor costs before renewal [S1]."
            )
        ),
    )
    chunks = await engine.index_segments(MEETING_ID, USER_ID, _segments())
    assert chunks == 1  # short transcript → single chunk

    result = await engine.answer(
        MEETING_ID, USER_ID, "What did we decide about mobile support and vendor costs?"
    )
    assert result.citations, "expected at least one citation"
    citation = result.citations[0]
    assert citation.start_ms == 0
    assert citation.end_ms == 30_000
    assert citation.start_label == "0:00"
    assert citation.quote
    assert result.grounded_score >= 0.7


async def test_retrieval_is_scoped_to_meeting_and_user(settings: Settings) -> None:
    engine = _engine(settings)
    await engine.index_segments(MEETING_ID, USER_ID, _segments())

    # Same meeting id, different tenant → nothing retrieved.
    hits = await engine.retrieve(
        MEETING_ID, "99999999-9999-9999-9999-999999999999", "mobile support"
    )
    assert hits == []
    # Unknown meeting → nothing either.
    hits = await engine.retrieve("00000000-0000-0000-0000-000000000000", USER_ID, "mobile support")
    assert hits == []


async def test_drop_meeting_removes_vectors(settings: Settings) -> None:
    engine = _engine(settings)
    await engine.index_segments(MEETING_ID, USER_ID, _segments())
    await engine.drop_meeting(MEETING_ID)
    hits = await engine.retrieve(MEETING_ID, USER_ID, "mobile support")
    assert hits == []


async def test_answer_without_retrievable_context_still_responds(settings: Settings) -> None:
    engine = _engine(
        settings, llm=FakeLLM(chat_response="I could not find that in the transcript.")
    )
    await engine.index_segments(MEETING_ID, USER_ID, _segments())
    result = await engine.answer(
        MEETING_ID,
        USER_ID,
        "zzz unrelated gibberish qqq",
    )
    assert result.citations == [] or isinstance(result.citations, list)
    assert 0.0 <= result.grounded_score <= 1.0
