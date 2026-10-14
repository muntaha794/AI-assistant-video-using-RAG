"""Action-item tracker routes: cross-meeting tracker + per-meeting listing."""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Query

from app.api.deps import CurrentUser, SessionDep
from app.db import repositories
from app.errors import NotFoundError
from app.schemas.action_items import ActionItemListOut, ActionItemOut, ActionItemUpdate
from app.services.timing import format_timestamp

router = APIRouter(tags=["action-items"])


def _to_out(item) -> ActionItemOut:  # type: ignore[no-untyped-def]
    out = ActionItemOut.model_validate(item)
    out.source_start_label = (
        format_timestamp(item.source_start_ms) if item.source_start_ms is not None else None
    )
    return out


@router.get(
    "/action-items", response_model=ActionItemListOut, summary="Tracker: all my action items"
)
async def list_my_action_items(
    user: CurrentUser,
    session: SessionDep,
    status_filter: Annotated[
        str | None, Query(alias="status", pattern="^(todo|in_progress|done|cancelled)$")
    ] = None,
    owner: Annotated[str | None, Query(max_length=200)] = None,
    due_before: date | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ActionItemListOut:
    items, total = await repositories.list_action_items(
        session,
        user_id=user.id,
        status=status_filter,
        owner=owner,
        due_before=due_before,
        limit=limit,
        offset=offset,
    )
    return ActionItemListOut(
        items=[_to_out(item) for item in items], total=total, limit=limit, offset=offset
    )


@router.patch(
    "/action-items/{item_id}",
    response_model=ActionItemOut,
    summary="Update ownership, deadline, priority or status",
)
async def update_action_item(
    item_id: uuid.UUID, payload: ActionItemUpdate, user: CurrentUser, session: SessionDep
) -> ActionItemOut:
    item = await repositories.get_action_item(session, item_id, user_id=user.id)
    if item is None:
        raise NotFoundError("Action item not found")
    data = payload.model_dump(exclude_unset=True)
    for field_name, value in data.items():
        setattr(item, field_name, value)
    if "status" in data:
        item.completed_at = datetime.now(timezone.utc) if data["status"] == "done" else None
    await session.commit()
    return _to_out(item)


@router.get(
    "/meetings/{meeting_id}/action-items",
    response_model=ActionItemListOut,
    summary="Action items extracted from one meeting",
)
async def list_meeting_action_items(
    meeting_id: uuid.UUID, user: CurrentUser, session: SessionDep
) -> ActionItemListOut:
    meeting = await repositories.get_meeting(session, meeting_id, user_id=user.id)
    if meeting is None:
        raise NotFoundError("Meeting not found")
    items, total = await repositories.list_action_items(
        session, user_id=user.id, meeting_id=meeting_id
    )
    return ActionItemListOut(
        items=[_to_out(item) for item in items], total=total, limit=len(items), offset=0
    )
