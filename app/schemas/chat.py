"""Chat (RAG) request/response schemas."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class ChatIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class CitationOut(BaseModel):
    """Evidence for an answer: the transcript window it came from."""

    chunk_id: str
    start_ms: int
    end_ms: int
    start_label: str
    end_label: str
    speaker: str
    quote: str

    model_config = {"from_attributes": True}


class ChatOut(BaseModel):
    meeting_id: uuid.UUID
    answer: str
    citations: list[CitationOut]
    grounded_score: float
    is_grounded: bool
    retrieved_chunks: int


class ChatMessageOut(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    citations: list[CitationOut] | None = None
    grounded_score: float | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatHistoryOut(BaseModel):
    meeting_id: uuid.UUID
    messages: list[ChatMessageOut]
