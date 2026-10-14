"""Job status routes (poll after upload / YouTube submission)."""

from __future__ import annotations

import uuid

from fastapi import APIRouter

from app.api.deps import CurrentUser, SessionDep
from app.db import repositories
from app.errors import NotFoundError
from app.schemas.jobs import JobOut

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get(
    "/{job_id}",
    response_model=JobOut,
    summary="Job status (queued/processing/succeeded/failed)",
)
async def get_job_status(job_id: uuid.UUID, user: CurrentUser, session: SessionDep) -> JobOut:
    job = await repositories.get_job(session, job_id, user_id=user.id)
    if job is None:
        raise NotFoundError("Job not found")
    return JobOut.model_validate(job)
