"""Meeting, transcript and summary schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from app.services.timing import format_timestamp


class SourceType(str, Enum):
    upload = "upload"
    youtube = "youtube"


class YouTubeIn(BaseModel):
    url: str = Field(max_length=1024, examples=["https://www.youtube.com/watch?v=dQw4w9WgXcQ"])
    title: str | None = Field(default=None, max_length=300)


class MeetingOut(BaseModel):
    id: uuid.UUID
    title: str
    source_type: SourceType
    source_uri: str | None = None
    original_filename: str | None = None
    status: str
    stage: str | None = None
    progress: int
    language: str | None = None
    duration_seconds: float | None = None
    segment_count: int
    speaker_count: int
    error: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None

    model_config = {"from_attributes": True}


class KeyDecisionOut(BaseModel):
    decision: str
    start_ms: int | None = None
    start_label: str | None = None


class MeetingDetailOut(MeetingOut):
    summary: str | None = None
    key_decisions: list[KeyDecisionOut] = Field(default_factory=list)
    action_item_count: int = 0


class MeetingListOut(BaseModel):
    items: list[MeetingOut]
    total: int
    limit: int
    offset: int


class SegmentOut(BaseModel):
    seq: int
    start_ms: int
    end_ms: int
    start_label: str
    end_label: str
    speaker: str
    text: str
    is_translated: bool

    @classmethod
    def from_segment(cls, segment) -> SegmentOut:  # type: ignore[no-untyped-def]
        return cls(
            seq=segment.seq,
            start_ms=segment.start_ms,
            end_ms=segment.end_ms,
            start_label=format_timestamp(segment.start_ms),
            end_label=format_timestamp(segment.end_ms),
            speaker=segment.speaker,
            text=segment.text,
            is_translated=segment.is_translated,
        )


class TranscriptOut(BaseModel):
    meeting_id: uuid.UUID
    language: str | None
    speakers: list[str]
    segments: list[SegmentOut]


class SummaryOut(BaseModel):
    meeting_id: uuid.UUID
    summary: str
    key_decisions: list[KeyDecisionOut]
    usage: dict[str, int] | None = None


class MeetingAccepted(BaseModel):
    """202 response: the meeting queued for asynchronous processing."""

    meeting_id: uuid.UUID
    job_id: uuid.UUID
    status: str = "queued"
    status_url: str
    poll_hint_seconds: int = 5
