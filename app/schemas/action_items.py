"""Action-item tracker schemas."""

from __future__ import annotations

import uuid
from datetime import date, datetime

from pydantic import BaseModel, Field

from app.services.action_items import ITEM_STATUSES, PRIORITIES


class ActionItemOut(BaseModel):
    id: uuid.UUID
    meeting_id: uuid.UUID
    title: str
    description: str | None = None
    owner: str | None = None
    due_date: date | None = None
    priority: str
    status: str
    source_start_ms: int | None = None
    source_start_label: str | None = None
    source_end_ms: int | None = None
    source_quote: str | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None

    model_config = {"from_attributes": True}


class ActionItemUpdate(BaseModel):
    """Editable tracker fields (ownership, deadline, progress)."""

    title: str | None = Field(default=None, max_length=500)
    description: str | None = Field(default=None, max_length=4000)
    owner: str | None = Field(default=None, max_length=200)
    due_date: date | None = None
    priority: str | None = Field(default=None, pattern=f"^({'|'.join(PRIORITIES)})$")
    status: str | None = Field(default=None, pattern=f"^({'|'.join(ITEM_STATUSES)})$")


class ActionItemListOut(BaseModel):
    items: list[ActionItemOut]
    total: int
    limit: int
    offset: int
