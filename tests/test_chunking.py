"""Chunker tests: windows, overlap, timestamp ranges."""

from __future__ import annotations

from app.services.rag.chunking import chunk_segments
from app.services.translation import TranslatedSegment


def _segments(count: int = 12, words: int = 12) -> list[TranslatedSegment]:
    return [
        TranslatedSegment(
            start_ms=i * 5_000,
            end_ms=(i + 1) * 5_000,
            text=" ".join(["word"] * words),
            speaker="SPEAKER_00" if i % 2 == 0 else "SPEAKER_01",
            seq=i,
        )
        for i in range(count)
    ]


def test_empty_transcript_has_no_chunks() -> None:
    assert chunk_segments([]) == []


def test_small_transcript_is_single_chunk() -> None:
    chunks = chunk_segments(_segments(count=2, words=5), max_chars=1200, overlap_chars=100)
    assert len(chunks) == 1
    assert chunks[0].start_ms == 0
    assert chunks[0].end_ms == 10_000
    assert set(chunks[0].speakers) == {"SPEAKER_00", "SPEAKER_01"}


def test_large_transcript_is_windowed_with_overlap() -> None:
    segments = _segments(count=20, words=10)
    chunks = chunk_segments(segments, max_chars=400, overlap_chars=120)
    assert len(chunks) > 1
    # Consecutive chunks overlap in time (carry-over window).
    for prev, cur in zip(chunks, chunks[1:], strict=False):
        assert cur.start_ms <= prev.end_ms
    # Chunk ids are unique and ordered; ranges are monotonic.
    assert [c.chunk_id for c in chunks] == sorted(c.chunk_id for c in chunks)
    assert chunks[0].start_ms == 0
    assert chunks[-1].end_ms == segments[-1].end_ms


def test_chunk_text_contains_time_and_speaker_markers() -> None:
    chunks = chunk_segments(_segments(count=1, words=3), max_chars=1200, overlap_chars=0)
    assert "[0:00] SPEAKER_00:" in chunks[0].text
