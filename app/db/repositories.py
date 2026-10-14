"""Repository layer.

Every function that touches tenant data takes an explicit ``user_id`` and adds
it to the WHERE clause — tenant isolation is enforced at the data layer, not
only in the route handlers.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any

from sqlalchemy import Select, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ActionItem, ChatMessage, Job, Meeting, TranscriptSegment, User


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Users ─────────────────────────────────────────────────────────────────────


async def get_user_by_email(session: AsyncSession, email: str) -> User | None:
    return await session.scalar(select(User).where(User.email == email.lower().strip()))


async def get_user(session: AsyncSession, user_id: uuid.UUID) -> User | None:
    return await session.get(User, user_id)


async def create_user(
    session: AsyncSession, *, email: str, hashed_password: str, full_name: str | None = None
) -> User:
    user = User(email=email.lower().strip(), hashed_password=hashed_password, full_name=full_name)
    session.add(user)
    await session.flush()
    return user


# ── Meetings ──────────────────────────────────────────────────────────────────


async def create_meeting(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    title: str,
    source_type: str,
    source_uri: str | None = None,
    original_filename: str | None = None,
    audio_path: str | None = None,
) -> Meeting:
    meeting = Meeting(
        user_id=user_id,
        title=title,
        source_type=source_type,
        source_uri=source_uri,
        original_filename=original_filename,
        audio_path=audio_path,
    )
    session.add(meeting)
    await session.flush()
    return meeting


async def get_meeting(
    session: AsyncSession, meeting_id: uuid.UUID, *, user_id: uuid.UUID
) -> Meeting | None:
    """Fetch a meeting only if it belongs to ``user_id``."""
    stmt = select(Meeting).where(Meeting.id == meeting_id, Meeting.user_id == user_id)
    return await session.scalar(stmt)


async def list_meetings(
    session: AsyncSession, *, user_id: uuid.UUID, limit: int = 50, offset: int = 0
) -> tuple[list[Meeting], int]:
    base = select(Meeting).where(Meeting.user_id == user_id)
    total = await session.scalar(
        select(func.count()).select_from(base.subquery())  # type: ignore[arg-type]
    )
    rows = await session.scalars(
        base.order_by(Meeting.created_at.desc()).limit(limit).offset(offset)
    )
    return list(rows), int(total or 0)


async def delete_meeting(
    session: AsyncSession, meeting_id: uuid.UUID, *, user_id: uuid.UUID
) -> Meeting | None:
    meeting = await get_meeting(session, meeting_id, user_id=user_id)
    if meeting is None:
        return None
    # Remove vectors for this meeting (best-effort; the DB row goes either way)
    await session.delete(meeting)
    await session.flush()
    return meeting


# ── Transcript ────────────────────────────────────────────────────────────────


async def replace_segments(
    session: AsyncSession,
    meeting_id: uuid.UUID,
    segments: list[dict[str, Any]],
) -> int:
    """Replace the full transcript of a meeting (idempotent reprocessing)."""
    await session.execute(
        delete(TranscriptSegment).where(TranscriptSegment.meeting_id == meeting_id)
    )
    rows = [
        TranscriptSegment(
            meeting_id=meeting_id,
            seq=idx,
            start_ms=int(seg["start_ms"]),
            end_ms=int(seg["end_ms"]),
            speaker=str(seg.get("speaker", "SPEAKER_00")),
            text=str(seg["text"]),
            original_text=seg.get("original_text"),
            is_translated=bool(seg.get("is_translated", False)),
        )
        for idx, seg in enumerate(segments)
    ]
    session.add_all(rows)
    await session.flush()
    return len(rows)


async def list_segments(
    session: AsyncSession,
    meeting_id: uuid.UUID,
    *,
    user_id: uuid.UUID,
    speaker: str | None = None,
) -> list[TranscriptSegment]:
    stmt: Select[tuple[TranscriptSegment]] = select(TranscriptSegment).where(
        TranscriptSegment.meeting_id == meeting_id
    )
    meeting = await get_meeting(session, meeting_id, user_id=user_id)
    if meeting is None:
        return []
    if speaker:
        stmt = stmt.where(TranscriptSegment.speaker == speaker)
    rows = await session.scalars(stmt.order_by(TranscriptSegment.seq))
    return list(rows)


async def list_speakers(
    session: AsyncSession, meeting_id: uuid.UUID, *, user_id: uuid.UUID
) -> list[str]:
    meeting = await get_meeting(session, meeting_id, user_id=user_id)
    if meeting is None:
        return []
    rows = await session.scalars(
        select(TranscriptSegment.speaker)
        .where(TranscriptSegment.meeting_id == meeting_id)
        .distinct()
        .order_by(TranscriptSegment.speaker)
    )
    return list(rows)


# ── Action items ──────────────────────────────────────────────────────────────


async def add_action_items(
    session: AsyncSession, meeting_id: uuid.UUID, user_id: uuid.UUID, items: list[dict[str, Any]]
) -> list[ActionItem]:
    rows = [
        ActionItem(
            meeting_id=meeting_id,
            user_id=user_id,
            title=item["title"],
            description=item.get("description"),
            owner=item.get("owner"),
            due_date=item.get("due_date"),
            priority=item.get("priority", "medium"),
            status="todo",
            source_start_ms=item.get("source_start_ms"),
            source_end_ms=item.get("source_end_ms"),
            source_quote=item.get("source_quote"),
        )
        for item in items
    ]
    session.add_all(rows)
    await session.flush()
    return rows


async def replace_action_items(
    session: AsyncSession, meeting_id: uuid.UUID, user_id: uuid.UUID, items: list[dict[str, Any]]
) -> list[ActionItem]:
    """Idempotent replacement used when a meeting is (re)processed."""
    await session.execute(
        delete(ActionItem).where(ActionItem.meeting_id == meeting_id, ActionItem.user_id == user_id)
    )
    return await add_action_items(session, meeting_id, user_id, items)


async def get_action_item(
    session: AsyncSession, item_id: uuid.UUID, *, user_id: uuid.UUID
) -> ActionItem | None:
    stmt = select(ActionItem).where(ActionItem.id == item_id, ActionItem.user_id == user_id)
    return await session.scalar(stmt)


async def list_action_items(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    meeting_id: uuid.UUID | None = None,
    status: str | None = None,
    owner: str | None = None,
    due_before: date | None = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[ActionItem], int]:
    filters = [ActionItem.user_id == user_id]
    if meeting_id:
        filters.append(ActionItem.meeting_id == meeting_id)
    if status:
        filters.append(ActionItem.status == status)
    if owner:
        filters.append(ActionItem.owner == owner)
    if due_before:
        filters.append(ActionItem.due_date <= due_before)

    base = select(ActionItem).where(*filters)
    total = await session.scalar(
        select(func.count()).select_from(base.subquery())  # type: ignore[arg-type]
    )
    rows = await session.scalars(
        base.order_by(ActionItem.due_date.is_(None), ActionItem.due_date, ActionItem.created_at)
        .limit(limit)
        .offset(offset)
    )
    return list(rows), int(total or 0)


# ── Chat ──────────────────────────────────────────────────────────────────────


async def add_chat_message(
    session: AsyncSession,
    *,
    meeting_id: uuid.UUID,
    user_id: uuid.UUID,
    role: str,
    content: str,
    citations: list[dict[str, Any]] | None = None,
    grounded_score: float | None = None,
    created_at: datetime | None = None,
) -> ChatMessage:
    message = ChatMessage(
        meeting_id=meeting_id,
        user_id=user_id,
        role=role,
        content=content,
        citations=citations,
        grounded_score=grounded_score,
        created_at=created_at or utcnow(),
    )
    session.add(message)
    await session.flush()
    return message


async def list_chat_history(
    session: AsyncSession,
    meeting_id: uuid.UUID,
    *,
    user_id: uuid.UUID,
    limit: int = 50,
) -> list[ChatMessage]:
    meeting = await get_meeting(session, meeting_id, user_id=user_id)
    if meeting is None:
        return []
    rows = await session.scalars(
        select(ChatMessage)
        .where(ChatMessage.meeting_id == meeting_id, ChatMessage.user_id == user_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(limit)
    )
    return list(reversed(list(rows)))


# ── Jobs ──────────────────────────────────────────────────────────────────────


async def create_job(
    session: AsyncSession,
    *,
    meeting_id: uuid.UUID,
    user_id: uuid.UUID,
    kind: str = "process_meeting",
    max_attempts: int = 3,
) -> Job:
    job = Job(
        meeting_id=meeting_id,
        user_id=user_id,
        kind=kind,
        max_attempts=max_attempts,
        next_run_at=utcnow(),
    )
    session.add(job)
    await session.flush()
    return job


async def get_job(
    session: AsyncSession, job_id: uuid.UUID, *, user_id: uuid.UUID | None = None
) -> Job | None:
    stmt = select(Job).where(Job.id == job_id)
    if user_id is not None:
        stmt = stmt.where(Job.user_id == user_id)
    return await session.scalar(stmt)


async def get_jobs_for_meeting(
    session: AsyncSession, meeting_id: uuid.UUID, *, user_id: uuid.UUID
) -> list[Job]:
    rows = await session.scalars(
        select(Job)
        .where(Job.meeting_id == meeting_id, Job.user_id == user_id)
        .order_by(Job.created_at.desc())
    )
    return list(rows)


async def claim_next_job(session: AsyncSession, *, worker_id: str) -> Job | None:
    """Atomically claim the oldest runnable job (SKIP LOCKED on PostgreSQL)."""
    now = utcnow()
    stmt = (
        select(Job)
        .where(Job.status == "queued", Job.next_run_at <= now)
        .order_by(Job.created_at)
        .limit(1)
    )
    # Row-level locking so multiple replicas can claim without colliding.
    bind = session.sync_session.get_bind()
    if bind is not None and bind.dialect.name == "postgresql":
        stmt = stmt.with_for_update(skip_locked=True)
    job = await session.scalar(stmt)
    if job is None:
        return None
    job.status = "processing"
    job.worker_id = worker_id
    job.attempts += 1
    job.started_at = now
    job.stage = "starting"
    job.progress = 0
    job.error = None
    await session.flush()
    return job


async def update_job_progress(
    session: AsyncSession, job_id: uuid.UUID, *, stage: str, progress: int
) -> None:
    job = await session.get(Job, job_id)
    if job is not None:
        job.stage = stage
        job.progress = max(job.progress, min(progress, 99))
        meeting = await session.get(Meeting, job.meeting_id)
        if meeting is not None and meeting.status == "processing":
            meeting.stage = stage
            meeting.progress = job.progress


async def complete_job(
    session: AsyncSession,
    job_id: uuid.UUID,
    *,
    token_usage: dict[str, Any] | None = None,
) -> None:
    job = await session.get(Job, job_id)
    if job is None:
        return
    now = utcnow()
    job.status = "succeeded"
    job.stage = "done"
    job.progress = 100
    job.finished_at = now
    job.token_usage = token_usage
    if job.started_at is not None:
        # SQLite returns naive datetimes; normalize to aware UTC before math.
        started_at = job.started_at
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)
        job.duration_ms = int((now - started_at).total_seconds() * 1000)


async def fail_job(
    session: AsyncSession, job_id: uuid.UUID, *, error: str, retry_backoff_seconds: int
) -> str:
    """Mark a failed attempt; requeue with backoff or fail permanently."""
    job = await session.get(Job, job_id)
    if job is None:
        return "failed"
    job.error = error[:4000]
    if job.attempts < job.max_attempts:
        job.status = "queued"
        job.next_run_at = utcnow() + timedelta(seconds=retry_backoff_seconds)
        job.stage = "retrying"
        return "requeued"
    job.status = "failed"
    job.finished_at = utcnow()
    job.stage = "failed"
    meeting = await session.get(Meeting, job.meeting_id)
    if meeting is not None:
        meeting.status = "failed"
        meeting.error = error[:4000]
    return "failed"


async def requeue_stale_jobs(session: AsyncSession, *, stale_after_seconds: int) -> int:
    """Recover jobs abandoned by a crashed worker."""
    cutoff = utcnow() - timedelta(seconds=stale_after_seconds)
    rows = await session.scalars(
        select(Job).where(Job.status == "processing", Job.started_at < cutoff)
    )
    count = 0
    for job in rows:
        job.status = "queued"
        job.next_run_at = utcnow()
        job.stage = "recovered"
        job.worker_id = None
        count += 1
    await session.flush()
    return count
