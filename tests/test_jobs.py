"""Job runner tests: retry/backoff semantics and crash recovery."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest_asyncio

from app.config import Settings
from app.db import repositories
from app.db.base import Base
from app.db.models import Meeting, User
from app.db.session import Database, build_database
from app.services.jobs import JobContext, JobRunner, submit_processing_job
from app.services.registry import ServiceRegistry


@pytest_asyncio.fixture
async def db(settings: Settings) -> Database:
    database = build_database(settings)
    async with database.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield database
    await database.dispose()


async def _seed(db: Database) -> tuple[uuid.UUID, uuid.UUID]:
    async with db.session() as session:
        user = User(email=f"{uuid.uuid4().hex[:8]}@example.com", hashed_password="x")
        session.add(user)
        await session.flush()
        meeting = Meeting(user_id=user.id, title="t", source_type="upload")
        session.add(meeting)
        await session.flush()
        return meeting.id, user.id


async def test_job_succeeds_and_records_token_usage(
    db: Database, settings: Settings, registry: ServiceRegistry
) -> None:
    seen: list[uuid.UUID] = []

    async def pipeline(ctx: JobContext) -> None:
        await ctx.stage("working", 50)
        seen.append(ctx.meeting_id)
        ctx.registry.usage.add(10, 5)

    runner = JobRunner(db, registry, pipeline, settings)
    meeting_id, user_id = await _seed(db)
    job_id = await submit_processing_job(
        db, meeting_id=meeting_id, user_id=user_id, settings=settings
    )

    await runner.drain()

    async with db.session() as session:
        job = await repositories.get_job(session, job_id)
        meeting = await repositories.get_meeting(session, meeting_id, user_id=user_id)
    assert seen == [meeting_id]
    assert job is not None and job.status == "succeeded" and job.progress == 100
    assert job.duration_ms is not None
    assert job.token_usage == {"prompt_tokens": 10, "completion_tokens": 5, "calls": 1}
    assert meeting is not None and meeting.status == "completed" and meeting.progress == 100


async def test_failing_pipeline_retries_then_fails(
    db: Database, settings: Settings, registry: ServiceRegistry
) -> None:
    attempts = 0

    async def pipeline(ctx: JobContext) -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("boom")

    runner = JobRunner(db, registry, pipeline, settings)
    meeting_id, user_id = await _seed(db)
    job_id = await submit_processing_job(
        db, meeting_id=meeting_id, user_id=user_id, settings=settings
    )

    for _ in range(settings.job_max_attempts):
        await runner.drain()

    async with db.session() as session:
        job = await repositories.get_job(session, job_id)
        meeting = await repositories.get_meeting(session, meeting_id, user_id=user_id)
    assert attempts == settings.job_max_attempts
    assert job is not None and job.status == "failed" and job.attempts == settings.job_max_attempts
    assert "boom" in (job.error or "")
    assert meeting is not None and meeting.status == "failed"


async def test_transient_failure_recovers_on_retry(
    db: Database, settings: Settings, registry: ServiceRegistry
) -> None:
    calls = 0

    async def pipeline(ctx: JobContext) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient glitch")

    runner = JobRunner(db, registry, pipeline, settings)
    meeting_id, user_id = await _seed(db)
    job_id = await submit_processing_job(
        db, meeting_id=meeting_id, user_id=user_id, settings=settings
    )

    await runner.drain(max_jobs=1)
    async with db.session() as session:
        job = await repositories.get_job(session, job_id)
    assert job is not None and job.status == "queued" and job.attempts == 1

    await runner.drain(max_jobs=1)
    async with db.session() as session:
        job = await repositories.get_job(session, job_id)
    assert job is not None and job.status == "succeeded"


async def _noop_pipeline(ctx: JobContext) -> None:
    await ctx.stage("noop", 1)


async def test_future_jobs_are_not_claimed(
    db: Database, settings: Settings, registry: ServiceRegistry
) -> None:
    runner = JobRunner(db, registry, _noop_pipeline, settings)
    meeting_id, user_id = await _seed(db)
    async with db.session() as session:
        job = await repositories.create_job(session, meeting_id=meeting_id, user_id=user_id)
        job.next_run_at = datetime.now(timezone.utc) + timedelta(hours=1)
        job_id = job.id
    await runner.drain()
    async with db.session() as session:
        claimed = await repositories.get_job(session, job_id)
    assert claimed is not None and claimed.status == "queued" and claimed.attempts == 0


async def test_stale_jobs_are_requeued(db: Database, settings: Settings) -> None:
    meeting_id, user_id = await _seed(db)
    async with db.session() as session:
        job = await repositories.create_job(session, meeting_id=meeting_id, user_id=user_id)
        job.status = "processing"
        job.started_at = datetime.now(timezone.utc) - timedelta(hours=3)
        job_id = job.id
    async with db.session() as session:
        recovered = await repositories.requeue_stale_jobs(session, stale_after_seconds=3600)
    assert recovered == 1
    async with db.session() as session:
        job = await repositories.get_job(session, job_id)
    assert job is not None and job.status == "queued" and job.worker_id is None
