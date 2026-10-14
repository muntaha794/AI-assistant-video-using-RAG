"""LLM wrapper tests: JSON extraction, usage tracking, error normalization."""

from __future__ import annotations

import json

import pytest

from app.errors import ExternalServiceError, TransientProcessingError
from app.services.action_items import extract_action_items
from app.services.llm import LLMResult, MistralLLM, UsageTracker, extract_json
from app.services.summarizer import render_transcript, summarize_meeting
from app.services.translation import TranslatedSegment
from tests.fakes import FakeLLM

# ── extract_json ──────────────────────────────────────────────────────────────


def test_extract_json_plain() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_in_code_fence() -> None:
    assert extract_json('Here you go:\n```json\n{"a": [1, 2]}\n```\nDone.') == {"a": [1, 2]}


def test_extract_json_embedded_in_prose() -> None:
    text = 'Sure! The result is {"decision": "ship it", "count": 3} as requested.'
    assert extract_json(text) == {"decision": "ship it", "count": 3}


def test_extract_json_bare_array() -> None:
    assert extract_json('[{"x": 1}]') == [{"x": 1}]


def test_extract_json_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        extract_json("no json here at all")


# ── UsageTracker ─────────────────────────────────────────────────────────────


def test_usage_tracker_accumulates_and_snapshots() -> None:
    tracker = UsageTracker()
    tracker.add(100, 20)
    tracker.add(50, 10)
    assert tracker.snapshot() == {
        "prompt_tokens": 150,
        "completion_tokens": 30,
        "calls": 2,
    }
    tracker.reset()
    assert tracker.snapshot() == {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}


# ── Error normalization (MistralLLM without a network) ──────────────────────


def test_build_llm_requires_api_key() -> None:
    with pytest.raises(ExternalServiceError):
        MistralLLM(api_key=None, model="mistral-small-latest")


@pytest.mark.asyncio
async def test_mistral_llm_retries_transient_then_succeeds() -> None:
    llm = MistralLLM(api_key="k", model="mistral-small-latest", max_retries=3, timeout_seconds=5)

    calls = {"n": 0}

    def flaky(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientProcessingError("still down")
        return LLMResult(
            text="ok", model="mistral-small-latest", prompt_tokens=3, completion_tokens=2
        )

    # complete() retries transient failures with exponential backoff.
    llm._call_blocking = flaky  # type: ignore[method-assign]
    result = await llm.complete([{"role": "user", "content": "hi"}])
    assert calls["n"] == 3
    assert result.text == "ok"
    # Usage was recorded from the successful attempt.
    assert llm.usage.snapshot()["prompt_tokens"] == 3


@pytest.mark.asyncio
async def test_mistral_llm_does_not_retry_auth_errors() -> None:
    llm = MistralLLM(api_key="k", model="mistral-small-latest", max_retries=3)

    def auth_failure(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise ExternalServiceError("401 invalid api key")

    llm._call_blocking = auth_failure  # type: ignore[method-assign]
    with pytest.raises(ExternalServiceError):
        await llm.complete([{"role": "user", "content": "hi"}])


# ── Prompt-level flows with the fake LLM ─────────────────────────────────────


def _segments() -> list[TranslatedSegment]:
    return [
        TranslatedSegment(
            start_ms=0,
            end_ms=8000,
            text="Alice will ship the beta by 2026-11-01.",
            speaker="SPEAKER_00",
            seq=0,
        ),
        TranslatedSegment(
            start_ms=8000,
            end_ms=16000,
            text="We also decided to prioritize mobile support over the dashboard.",
            speaker="SPEAKER_01",
            seq=1,
        ),
    ]


@pytest.mark.asyncio
async def test_summarize_meeting_parses_decisions_with_timestamps() -> None:
    llm = FakeLLM()
    result = await summarize_meeting(llm, _segments())
    assert "Alice" in result.summary
    assert len(result.key_decisions) == 2
    # "0:16" and "0:08" both resolve to the segment that contains/starts at 8000 ms.
    assert result.key_decisions[0].start_ms == 8_000
    assert result.key_decisions[1].start_ms == 8_000
    # The system prompt instructed JSON-only output.
    assert "key_decisions" in llm.calls[0][0]["content"]


@pytest.mark.asyncio
async def test_extract_action_items_normalizes_payload() -> None:
    llm = FakeLLM()
    items = await extract_action_items(llm, _segments())
    assert len(items) == 3
    assert items[0]["owner"] == "Alice"
    assert items[0]["due_date"].isoformat() == "2026-11-01"
    # Evidence anchored to the utterance at 0:08 (the segment starting at 8000 ms).
    assert items[0]["source_start_ms"] == 8_000
    assert items[0]["source_quote"]
    assert items[1]["due_date"] is None or items[1]["due_date"].isoformat() >= "2026-10-13"


@pytest.mark.asyncio
async def test_summarize_meeting_survives_invalid_json() -> None:
    llm = FakeLLM(summary_response="Sorry, I cannot produce JSON right now.")
    result = await summarize_meeting(llm, _segments())
    assert result.summary == "Sorry, I cannot produce JSON right now."
    assert result.key_decisions == []


def test_render_transcript_includes_time_and_speaker() -> None:
    text = render_transcript(_segments())
    assert text == (
        "[0:00] SPEAKER_00: Alice will ship the beta by 2026-11-01.\n"
        "[0:08] SPEAKER_01: We also decided to prioritize mobile support over the dashboard."
    )


def test_usage_snapshot_is_json_serializable() -> None:
    llm = FakeLLM()
    snapshot = llm.usage.snapshot()
    assert json.dumps(snapshot)
