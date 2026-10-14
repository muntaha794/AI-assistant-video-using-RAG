"""Meeting routes.

Uploads and YouTube URLs are both processed **asynchronously**: these handlers
validate input, persist the meeting, enqueue a job and return 202 with a
status URL. Clients poll ``GET /jobs/{id}`` (or the meeting) until completed.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, File, Form, Query, Request, Response, UploadFile, status
from sqlalchemy import func, select

from app.api.deps import CurrentUser, RegistryDep, RunnerDep, SessionDep
from app.db import repositories
from app.db.models import ActionItem
from app.errors import ConflictError, NotFoundError, ValidationError
from app.schemas.meetings import (
    KeyDecisionOut,
    MeetingAccepted,
    MeetingDetailOut,
    MeetingListOut,
    MeetingOut,
    SegmentOut,
    SummaryOut,
    TranscriptOut,
    YouTubeIn,
)
from app.services.audio import validate_youtube_url
from app.services.jobs import submit_processing_job
from app.services.timing import format_timestamp

router = APIRouter(prefix="/meetings", tags=["meetings"])


# ── Creation ──────────────────────────────────────────────────────────────────


@router.post(
    "/upload",
    response_model=MeetingAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload an audio file (processed asynchronously)",
)
async def upload_meeting(
    request: Request,
    user: CurrentUser,
    session: SessionDep,
    registry: RegistryDep,
    runner: RunnerDep,
    file: Annotated[UploadFile, File(description="Audio file (mp3, wav, m4a, flac, ogg, webm…)")],
    title: Annotated[str | None, Form(max_length=300)] = None,
) -> MeetingAccepted:
    # Create the row first so the file lands in the meeting's own directory;
    # on validation failure the request session rolls back, leaving no row.
    meeting = await repositories.create_meeting(
        session,
        user_id=user.id,
        title=title or (file.filename or "Uploaded meeting").rsplit(".", 1)[0][:300],
        source_type="upload",
        original_filename=file.filename,
    )
    stored = await registry.storage.save_upload(
        meeting.id, file.filename or "audio", file.file, content_type=file.content_type
    )
    meeting.source_uri = stored.path.name
    meeting.audio_path = stored.path.name
    await session.commit()  # meeting row must exist before the job FK (PostgreSQL)
    job_id = await submit_processing_job(
        request.app.state.db,
        meeting_id=meeting.id,
        user_id=user.id,
        settings=request.app.state.settings,
    )
    return _accepted(request, meeting.id, job_id)


@router.post(
    "/youtube",
    response_model=MeetingAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Process a YouTube URL (downloaded + transcribed asynchronously)",
)
async def youtube_meeting(
    request: Request,
    payload: YouTubeIn,
    user: CurrentUser,
    session: SessionDep,
    registry: RegistryDep,
    runner: RunnerDep,
) -> MeetingAccepted:
    try:
        url = validate_youtube_url(payload.url)
    except ValueError as exc:
        raise ValidationError(str(exc), details={"field": "url"}) from exc

    meeting = await repositories.create_meeting(
        session,
        user_id=user.id,
        title=payload.title or "YouTube meeting",
        source_type="youtube",
        source_uri=url,
    )
    await session.commit()  # meeting row must exist before the job FK (PostgreSQL)
    job_id = await submit_processing_job(
        request.app.state.db,
        meeting_id=meeting.id,
        user_id=user.id,
        settings=request.app.state.settings,
    )
    return _accepted(request, meeting.id, job_id)


def _accepted(request: Request, meeting_id: uuid.UUID, job_id: uuid.UUID) -> MeetingAccepted:
    prefix = request.app.state.settings.api_v1_prefix
    return MeetingAccepted(
        meeting_id=meeting_id,
        job_id=job_id,
        status_url=f"{prefix}/jobs/{job_id}",
    )


# ── Reads ─────────────────────────────────────────────────────────────────────


@router.get("", response_model=MeetingListOut, summary="List my meetings")
async def list_meetings(
    user: CurrentUser,
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> MeetingListOut:
    meetings, total = await repositories.list_meetings(
        session, user_id=user.id, limit=limit, offset=offset
    )
    return MeetingListOut(
        items=[MeetingOut.model_validate(m) for m in meetings],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{meeting_id}", response_model=MeetingDetailOut, summary="Meeting detail + summary")
async def get_meeting(
    meeting_id: uuid.UUID, user: CurrentUser, session: SessionDep
) -> MeetingDetailOut:
    meeting = await repositories.get_meeting(session, meeting_id, user_id=user.id)
    if meeting is None:
        raise NotFoundError("Meeting not found")
    detail = MeetingDetailOut.model_validate(meeting)
    detail.summary = meeting.summary_text
    detail.key_decisions = [
        KeyDecisionOut(
            decision=str(d.get("decision", "")),
            start_ms=d.get("start_ms"),
            start_label=(
                format_timestamp(d["start_ms"]) if d.get("start_ms") is not None else None
            ),
        )
        for d in (meeting.key_decisions or [])
    ]
    count = await session.scalar(
        select(func.count())
        .select_from(ActionItem)
        .where(ActionItem.meeting_id == meeting.id, ActionItem.user_id == user.id)
    )
    detail.action_item_count = int(count or 0)
    return detail


@router.get(
    "/{meeting_id}/transcript", response_model=TranscriptOut, summary="Timestamped transcript"
)
async def get_transcript(
    meeting_id: uuid.UUID,
    user: CurrentUser,
    session: SessionDep,
    speaker: Annotated[str | None, Query(max_length=64)] = None,
    limit: Annotated[int, Query(ge=1, le=5000)] = 2000,
) -> TranscriptOut:
    meeting = await repositories.get_meeting(session, meeting_id, user_id=user.id)
    if meeting is None:
        raise NotFoundError("Meeting not found")
    segments = await repositories.list_segments(
        session, meeting_id, user_id=user.id, speaker=speaker
    )
    speakers = sorted({seg.speaker for seg in segments})
    return TranscriptOut(
        meeting_id=meeting_id,
        language=meeting.language,
        speakers=speakers,
        segments=[SegmentOut.from_segment(seg) for seg in segments[:limit]],
    )


@router.get("/{meeting_id}/summary", response_model=SummaryOut, summary="Summary + key decisions")
async def get_summary(meeting_id: uuid.UUID, user: CurrentUser, session: SessionDep) -> SummaryOut:
    meeting = await repositories.get_meeting(session, meeting_id, user_id=user.id)
    if meeting is None:
        raise NotFoundError("Meeting not found")
    if meeting.status != "completed" or not meeting.summary_text:
        raise ConflictError(
            "Meeting has not finished processing yet", details={"status": meeting.status}
        )
    return SummaryOut(
        meeting_id=meeting_id,
        summary=meeting.summary_text,
        key_decisions=[
            KeyDecisionOut(
                decision=str(d.get("decision", "")),
                start_ms=d.get("start_ms"),
                start_label=(
                    format_timestamp(d["start_ms"]) if d.get("start_ms") is not None else None
                ),
            )
            for d in (meeting.key_decisions or [])
        ],
        usage=meeting.summary_usage if isinstance(meeting.summary_usage, dict) else None,
    )


# ── Reprocessing / deletion ───────────────────────────────────────────────────


@router.post(
    "/{meeting_id}/reprocess",
    response_model=MeetingAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Re-run the pipeline for a meeting",
)
async def reprocess_meeting(
    request: Request,
    meeting_id: uuid.UUID,
    user: CurrentUser,
    session: SessionDep,
    runner: RunnerDep,
) -> MeetingAccepted:
    meeting = await repositories.get_meeting(session, meeting_id, user_id=user.id)
    if meeting is None:
        raise NotFoundError("Meeting not found")
    if meeting.status == "processing":
        raise ConflictError("Meeting is already being processed")
    meeting.status = "queued"
    meeting.stage = "queued"
    meeting.progress = 0
    meeting.error = None
    await session.commit()
    job_id = await submit_processing_job(
        request.app.state.db,
        meeting_id=meeting.id,
        user_id=user.id,
        settings=request.app.state.settings,
    )
    return _accepted(request, meeting.id, job_id)


@router.delete(
    "/{meeting_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a meeting and its data"
)
async def delete_meeting(
    meeting_id: uuid.UUID,
    user: CurrentUser,
    session: SessionDep,
    registry: RegistryDep,
    runner: RunnerDep,
) -> Response:
    meeting = await repositories.get_meeting(session, meeting_id, user_id=user.id)
    if meeting is None:
        raise NotFoundError("Meeting not found")
    if meeting.status == "processing":
        raise ConflictError("Meeting is being processed; retry after it finishes")
    # Vectors first (best-effort), then DB rows, then files on disk.
    await registry.rag.drop_meeting(str(meeting_id))
    await repositories.delete_meeting(session, meeting_id, user_id=user.id)
    registry.storage.delete_meeting_dir(meeting_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
