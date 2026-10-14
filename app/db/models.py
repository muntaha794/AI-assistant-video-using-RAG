"""ORM models.

Design notes
------------
* UUID primary keys (portable across PostgreSQL and SQLite) so ids can be
  exposed publicly without sequential-enumeration leaks.
* Enums are stored as plain strings with Python-side ``str`` enums: keeps
  Alembic migrations simple and avoids PostgreSQL enum-type churn.
* ``user_id`` is denormalized onto ``ActionItem`` and ``ChatMessage`` so every
  per-user query is a single indexed lookup (defence in depth for tenant
  isolation on top of the join-based checks in the repositories).
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(256))
    full_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    meetings: Mapped[list[Meeting]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Meeting(Base):
    __tablename__ = "meetings"
    __table_args__ = (Index("ix_meetings_user_created", "user_id", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )

    title: Mapped[str] = mapped_column(String(300))
    source_type: Mapped[str] = mapped_column(String(20))  # upload | youtube
    source_uri: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    original_filename: Mapped[str | None] = mapped_column(String(300), nullable=True)
    audio_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    # Processing state (mirrors the Job row, denormalized for fast UI polling)
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    stage: Mapped[str | None] = mapped_column(String(40), nullable=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Transcript facts
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    segment_count: Mapped[int] = mapped_column(Integer, default=0)
    speaker_count: Mapped[int] = mapped_column(Integer, default=0)

    # Summary (one summary per meeting; regenerated on reprocess)
    summary_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    key_decisions: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    summary_usage: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped[User] = relationship(back_populates="meetings")
    segments: Mapped[list[TranscriptSegment]] = relationship(
        back_populates="meeting", cascade="all, delete-orphan", order_by="TranscriptSegment.seq"
    )
    action_items: Mapped[list[ActionItem]] = relationship(
        back_populates="meeting", cascade="all, delete-orphan"
    )
    chat_messages: Mapped[list[ChatMessage]] = relationship(
        back_populates="meeting", cascade="all, delete-orphan"
    )
    jobs: Mapped[list[Job]] = relationship(back_populates="meeting", cascade="all, delete-orphan")


class TranscriptSegment(Base):
    """One timestamped utterance of the meeting."""

    __tablename__ = "transcript_segments"
    __table_args__ = (Index("ix_segments_meeting_seq", "meeting_id", "seq"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    meeting_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), index=True
    )
    seq: Mapped[int] = mapped_column(Integer)  # 0-based order in the transcript
    start_ms: Mapped[int] = mapped_column(Integer)
    end_ms: Mapped[int] = mapped_column(Integer)
    speaker: Mapped[str] = mapped_column(String(64), default="SPEAKER_00")
    text: Mapped[str] = mapped_column(Text)  # language of `speaker_language`
    original_text: Mapped[str | None] = mapped_column(Text, nullable=True)  # pre-translation
    is_translated: Mapped[bool] = mapped_column(Boolean, default=False)

    meeting: Mapped[Meeting] = relationship(back_populates="segments")


class ActionItem(Base):
    __tablename__ = "action_items"
    __table_args__ = (
        Index("ix_action_items_user_status", "user_id", "status"),
        Index("ix_action_items_user_due", "user_id", "due_date"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    meeting_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))

    title: Mapped[str] = mapped_column(String(500))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    owner: Mapped[str | None] = mapped_column(String(200), nullable=True)
    due_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    priority: Mapped[str] = mapped_column(String(16), default="medium")  # low|medium|high
    # todo | in_progress | done | cancelled
    status: Mapped[str] = mapped_column(String(20), default="todo")

    # Evidence: where in the meeting this item was stated
    source_start_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_end_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_quote: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    meeting: Mapped[Meeting] = relationship(back_populates="action_items")


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    __table_args__ = (Index("ix_chat_meeting_created", "meeting_id", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    meeting_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))

    role: Mapped[str] = mapped_column(String(20))  # user | assistant
    content: Mapped[str] = mapped_column(Text)
    citations: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    grounded_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    meeting: Mapped[Meeting] = relationship(back_populates="chat_messages")


class Job(Base):
    """Durable async-processing queue row (claimed with SKIP LOCKED on PG)."""

    __tablename__ = "jobs"
    __table_args__ = (Index("ix_jobs_status_next_run", "status", "next_run_at"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    meeting_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(40), default="process_meeting")

    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    stage: Mapped[str | None] = mapped_column(String(40), nullable=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    next_run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    token_usage: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    meeting: Mapped[Meeting] = relationship(back_populates="jobs")
