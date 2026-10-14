"""Chat routes: evidence-based Q&A over one meeting's transcript."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Query

from app.api.deps import CurrentUser, RegistryDep, SessionDep
from app.db import repositories
from app.errors import ConflictError, NotFoundError
from app.schemas.chat import ChatHistoryOut, ChatIn, ChatMessageOut, ChatOut, CitationOut
from app.services.rag.engine import GroundedAnswer

router = APIRouter(prefix="/meetings/{meeting_id}/chat", tags=["chat"])


@router.post("", response_model=ChatOut, summary="Ask a question about the meeting")
async def chat(
    meeting_id: uuid.UUID,
    payload: ChatIn,
    user: CurrentUser,
    session: SessionDep,
    registry: RegistryDep,
) -> ChatOut:
    meeting = await repositories.get_meeting(session, meeting_id, user_id=user.id)
    if meeting is None:
        raise NotFoundError("Meeting not found")
    if meeting.status != "completed":
        raise ConflictError("Meeting is not ready for chat yet", details={"status": meeting.status})

    history_rows = await repositories.list_chat_history(
        session, meeting_id, user_id=user.id, limit=registry.settings.chat_history_max_messages
    )
    history = [(row.role, row.content) for row in history_rows]

    result: GroundedAnswer = await registry.rag.answer(
        str(meeting_id), str(user.id), payload.question, history=history
    )

    citation_dicts = [citation.as_dict() for citation in result.citations]
    # Explicit, strictly increasing timestamps keep history ordering deterministic.
    stamped_at = datetime.now(timezone.utc)
    await repositories.add_chat_message(
        session,
        meeting_id=meeting_id,
        user_id=user.id,
        role="user",
        content=payload.question,
        created_at=stamped_at,
    )
    await repositories.add_chat_message(
        session,
        meeting_id=meeting_id,
        user_id=user.id,
        role="assistant",
        content=result.answer,
        citations=citation_dicts,
        grounded_score=result.grounded_score,
        created_at=stamped_at + timedelta(microseconds=1),
    )
    return ChatOut(
        meeting_id=meeting_id,
        answer=result.answer,
        citations=[CitationOut.model_validate(citation.as_dict()) for citation in result.citations],
        grounded_score=result.grounded_score,
        is_grounded=result.is_grounded,
        retrieved_chunks=len(result.retrieved),
    )


@router.get("/history", response_model=ChatHistoryOut, summary="Chat history for a meeting")
async def chat_history(
    meeting_id: uuid.UUID,
    user: CurrentUser,
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> ChatHistoryOut:
    meeting = await repositories.get_meeting(session, meeting_id, user_id=user.id)
    if meeting is None:
        raise NotFoundError("Meeting not found")
    rows = await repositories.list_chat_history(session, meeting_id, user_id=user.id, limit=limit)
    return ChatHistoryOut(
        meeting_id=meeting_id,
        messages=[ChatMessageOut.model_validate(row) for row in rows],
    )
