"""Unit tests for the pure helpers: timing, normalization, diarization merge."""

from __future__ import annotations

from datetime import date

from app.services.action_items import (
    normalize_action_items,
    normalize_owner,
    normalize_priority,
    parse_due_date,
)
from app.services.diarization import SpeakerSpan, best_effort_speaker, label_segments
from app.services.timing import format_timestamp, nearest_segment_index, parse_timecode
from app.services.translation import (
    TranscriptSegmentData,
    TranslatedSegment,
    apply_translation,
    is_hindi_text,
    needs_translation,
)

# ── Timing ────────────────────────────────────────────────────────────────────


def test_format_timestamp() -> None:
    assert format_timestamp(0) == "0:00"
    assert format_timestamp(65_000) == "1:05"
    assert format_timestamp(3_725_000) == "1:02:05"


def test_parse_timecode() -> None:
    assert parse_timecode("1:05") == 65_000
    assert parse_timecode("1:02:05") == 3_725_000
    assert parse_timecode("garbage") is None
    assert parse_timecode(None) is None
    assert parse_timecode("10:99") is None


def test_nearest_segment_index_prefers_containment() -> None:
    ranges = [(0, 1000), (1000, 5000), (5000, 9000)]
    assert nearest_segment_index(2500, ranges) == 1
    assert nearest_segment_index(9500, ranges) == 2  # closest start
    assert nearest_segment_index(0, []) is None


# ── Action items ──────────────────────────────────────────────────────────────


def test_normalize_owner_placeholders() -> None:
    assert normalize_owner("TBD") is None
    assert normalize_owner("  --  ") is None
    assert normalize_owner(None) is None
    assert normalize_owner("@Alice Smith.") == "Alice Smith"


def test_normalize_priority_defaults_and_aliases() -> None:
    assert normalize_priority(None) == "medium"
    assert normalize_priority("URGENT") == "high"
    assert normalize_priority("p3") == "low"
    assert normalize_priority("whatever") == "medium"


def test_parse_due_date_formats_and_relatives() -> None:
    today = date(2026, 10, 13)  # a Tuesday
    assert parse_due_date("2026-11-01") == date(2026, 11, 1)
    assert parse_due_date("01/11/2026") == date(2026, 11, 1)
    assert parse_due_date("tomorrow", today=today) == date(2026, 10, 14)
    assert parse_due_date("friday", today=today) == date(2026, 10, 16)
    assert parse_due_date("next monday", today=today) == date(2026, 10, 19)
    assert parse_due_date("someday maybe") is None
    assert parse_due_date(None) is None


def test_normalize_action_items_attaches_evidence() -> None:
    segments = [
        TranslatedSegment(start_ms=0, end_ms=8000, text="Please ship the beta by November.", seq=0),
        TranslatedSegment(start_ms=8000, end_ms=16000, text="Unrelated chatter.", seq=1),
    ]
    raw = [
        {
            "title": "  Ship the beta ",
            "owner": "TBD",
            "due_date": "2026-11-01",
            "priority": "urgent",
            "time": "0:03",
        },
        {"title": ""},
    ]
    items = normalize_action_items(raw, segments)
    assert len(items) == 1  # empty title dropped
    item = items[0]
    assert item["title"] == "Ship the beta"
    assert item["owner"] is None
    assert item["priority"] == "high"
    assert item["source_start_ms"] == 0
    assert item["source_end_ms"] == 8000
    assert "ship the beta" in item["source_quote"].lower()


# ── Translation ───────────────────────────────────────────────────────────────


def test_hindi_detection_by_script() -> None:
    assert is_hindi_text("क्या हम लॉन्च की तारीख तय कर सकते हैं")
    assert not is_hindi_text("Let's decide the launch date today")
    assert not is_hindi_text("   ")


def test_needs_translation_uses_language_hint() -> None:
    segments = [TranscriptSegmentData(0, 1000, "नमस्ते दोस्तों")]
    assert needs_translation("hi", segments)
    assert needs_translation(None, segments)  # script fallback
    assert not needs_translation("en", [TranscriptSegmentData(0, 1000, "hello there")])


def test_apply_translation_preserves_timestamps_and_original() -> None:
    segments = [
        TranscriptSegmentData(0, 2000, "नमस्ते"),
        TranscriptSegmentData(2000, 4000, "धन्यवाद"),
    ]
    translated = apply_translation(segments, ["[EN] hello"])
    assert [t.start_ms for t in translated] == [0, 2000]
    assert translated[0].is_translated and translated[0].original_text == "नमस्ते"
    # Missing translation line falls back to the original text.
    assert translated[1].is_translated is False
    assert translated[1].text == "धन्यवाद"


# ── Diarization merge ─────────────────────────────────────────────────────────


def test_best_effort_speaker_majority_overlap() -> None:
    spans = [SpeakerSpan(0, 10_000, "SPEAKER_00"), SpeakerSpan(10_000, 30_000, "SPEAKER_01")]
    assert best_effort_speaker(2_000, 5_000, spans) == "SPEAKER_00"
    assert best_effort_speaker(12_000, 20_000, spans) == "SPEAKER_01"
    # Boundary-straddling segment → majority overlap wins.
    assert best_effort_speaker(9_500, 12_000, spans) == "SPEAKER_01"
    # Outside every span → below the 30% threshold, keep the default label.
    assert best_effort_speaker(50_000, 60_000, spans) == "SPEAKER_00"


def test_label_segments_without_spans_defaults() -> None:
    segments = [TranscriptSegmentData(0, 1000, "a"), TranscriptSegmentData(1000, 2000, "b")]
    assert label_segments(segments, []) == ["SPEAKER_00", "SPEAKER_00"]
