"""Job status schemas."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel


class JobOut(BaseModel):
    id: uuid.UUID
    meeting_id: uuid.UUID
    kind: str
    status: str  # queued | processing | succeeded | failed
    stage: str | None = None
    progress: int
    attempts: int
    max_attempts: int
    error: str | None = None
    token_usage: dict[str, int] | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: int | None = None

    model_config = {"from_attributes": True}
