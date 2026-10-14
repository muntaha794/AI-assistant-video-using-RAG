"""Deterministic test doubles.

The fakes keep the real service protocols so the production wiring (and the
whole pipeline) is exercised with zero network, GPU or model downloads.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.config import Settings
from app.services.diarization import SpeakerSpan
from app.services.llm import LLMResult, Message, UsageTracker
from app.services.rag.embeddings import HashingEmbeddings
from app.services.rag.store import InMemoryVectorStore
from app.services.registry import ServiceRegistry
from app.services.storage import AudioStorage
from app.services.transcription import TranscriptResult, TranscriptSegmentData


class FakeTranscriber:
    """Returns a canned transcript; optionally raises to simulate failures."""

    def __init__(self, result: TranscriptResult | None = None, *, fail_times: int = 0) -> None:
        self.result = result or TranscriptResult(
            language="en",
            duration_seconds=40.0,
            segments=[
                TranscriptSegmentData(
                    0, 8000, "Hi everyone, thanks for joining the quarterly planning meeting."
                ),
                TranscriptSegmentData(
                    8000, 16000, "Alice will own the beta launch and ship it by 2026-11-01."
                ),
                TranscriptSegmentData(
                    16000,
                    26000,
                    "We decided to prioritize mobile support over the reporting dashboard.",
                ),
                TranscriptSegmentData(
                    26000,
                    40000,
                    "Bob will send the vendor contract to Carol by Friday for review.",
                ),
            ],
        )
        self.fail_times = fail_times
        self.calls = 0

    async def transcribe(self, audio_path: Path) -> TranscriptResult:
        self.calls += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("simulated whisper failure")
        return self.result


class FakeTranslator:
    """Deterministic 'translation': prefixes segments with a marker."""

    async def translate_batch(self, texts: list[str]) -> list[str]:
        return [f"[EN] {text}" for text in texts]


class FakeDiarizer:
    """Two speakers: SPEAKER_00 for the first 12s, SPEAKER_01 afterwards."""

    def __init__(self, spans: list[SpeakerSpan] | None = None) -> None:
        self.spans = spans or [
            SpeakerSpan(0, 12_000, "SPEAKER_00"),
            SpeakerSpan(12_000, 60_000, "SPEAKER_01"),
        ]

    async def diarize(self, audio_path: Path, duration_seconds: float) -> list[SpeakerSpan]:
        return self.spans


SUMMARY_RESPONSE = json.dumps(
    {
        "summary": "The team planned the Q4 beta launch. Alice owns shipping the beta "
        "by November 1st, and mobile support was prioritized over reporting.",
        "key_decisions": [
            {"decision": "Prioritize mobile support over the reporting dashboard", "time": "0:16"},
            {"decision": "Beta launches by 2026-11-01, owned by Alice", "time": "0:08"},
        ],
    }
)

ACTION_ITEMS_RESPONSE = json.dumps(
    {
        "action_items": [
            {
                "title": "Ship the beta launch",
                "description": "Own the beta release end to end.",
                "owner": "Alice",
                "due_date": "2026-11-01",
                "priority": "high",
                "time": "0:08",
            },
            {
                "title": "Send the vendor contract to Carol",
                "description": "Contract review before signing.",
                "owner": "Bob",
                "due_date": "Friday",
                "priority": "medium",
                "time": "0:26",
            },
            {
                "title": "Update the roadmap",
                "description": "Reflect mobile-first priority.",
                "owner": None,
                "due_date": None,
                "priority": "low",
                "time": "0:16",
            },
        ]
    }
)


class FakeLLM:
    """Scripted LLM: dispatches on the system prompt, records usage."""

    def __init__(
        self,
        *,
        summary_response: str = SUMMARY_RESPONSE,
        action_items_response: str = ACTION_ITEMS_RESPONSE,
        chat_response: str | None = None,
    ) -> None:
        self.summary_response = summary_response
        self.action_items_response = action_items_response
        # Default: cite the first retrieved context block.
        self.chat_response = chat_response or (
            "The team decided to prioritize mobile support over the reporting dashboard [S1]."
        )
        self.usage = UsageTracker()
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResult:
        self.calls.append(messages)
        system = messages[0]["content"] if messages else ""
        if "key_decisions" in system:
            text = self.summary_response
        elif "action_items" in system:
            text = self.action_items_response
        else:
            text = self.chat_response
        self.usage.add(prompt_tokens=120, completion_tokens=40)
        return LLMResult(text=text, model="fake-model", prompt_tokens=120, completion_tokens=40)


def build_fake_registry(settings: Settings, llm: FakeLLM | None = None) -> ServiceRegistry:
    return ServiceRegistry(
        settings=settings,
        storage=AudioStorage(settings),
        transcriber=FakeTranscriber(),
        diarizer=FakeDiarizer(),
        translator=FakeTranslator(),
        llm=llm or FakeLLM(),
        embeddings=HashingEmbeddings(settings.embedding_dim),
        vector_store=InMemoryVectorStore(),
    )


# Canonical bytes that pass the mp3 magic-byte check.
FAKE_MP3_BYTES = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\x00" * 256
