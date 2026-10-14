"""Asynchronous meeting processing.

Design
------
The queue is the ``jobs`` table itself (no Redis/broker needed):
* ``POST /meetings`` inserts a ``queued`` job row and returns 202 immediately.
* N in-process workers poll for the oldest runnable row and claim it
  atomically (``FOR UPDATE SKIP LOCKED`` on PostgreSQL, so multiple API/worker
  replicas scale horizontally).
* Failures are retried with exponential backoff up to ``max_attempts``;
  jobs abandoned by a crashed worker are requeued by the reaper.

Everything the pipeline needs (db, services) is injected via ``JobContext``,
which keeps the runner unit-testable with a fake pipeline.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.config import Settings
from app.db import repositories
from app.db.models import Meeting, utcnow
from app.db.session import Database
from app.errors import NotFoundError
from app.logging_config import get_logger
from app.metrics import ACTIVE_JOBS, JOB_DURATION, JOBS_TOTAL

if TYPE_CHECKING:  # pragma: no cover
    from app.services.registry import ServiceRegistry

logger = get_logger(__name__)

PipelineFn = Callable[["JobContext"], Awaitable[None]]


@dataclass
class JobContext:
    """Handle passed to the pipeline: identity, services, progress reporting."""

    job_id: uuid.UUID
    meeting_id: uuid.UUID
    user_id: uuid.UUID
    db: Database
    registry: ServiceRegistry

    async def stage(self, name: str, progress: int) -> None:
        """Report pipeline progress (job + meeting rows, for UI polling)."""
        async with self.db.session() as session:
            await repositories.update_job_progress(
                session, self.job_id, stage=name, progress=progress
            )
        logger.info("pipeline_stage", job_id=str(self.job_id), stage=name, progress=progress)


class JobRunner:
    """Polls the jobs table and executes meeting pipelines concurrently."""

    def __init__(
        self,
        db: Database,
        registry: ServiceRegistry,
        pipeline: PipelineFn,
        settings: Settings,
        *,
        poll_interval: float | None = None,
    ) -> None:
        self.db = db
        self.registry = registry
        self.pipeline = pipeline
        self.settings = settings
        self.poll_interval = (
            poll_interval if poll_interval is not None else settings.job_poll_interval_seconds
        )
        self.worker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task[None]] = []

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._tasks:
            return
        self._stop.clear()
        async with self.db.session() as session:
            recovered = await repositories.requeue_stale_jobs(
                session, stale_after_seconds=self.settings.job_stale_after_seconds
            )
        if recovered:
            logger.warning("requeued_stale_jobs", count=recovered)
        for index in range(self.settings.job_workers):
            self._tasks.append(asyncio.create_task(self._worker_loop(f"{self.worker_id}#{index}")))
        self._tasks.append(asyncio.create_task(self._reaper_loop()))
        logger.info(
            "job_runner_started",
            worker_id=self.worker_id,
            concurrency=self.settings.job_workers,
        )

    async def stop(self, *, timeout: float = 10.0) -> None:
        """Graceful shutdown: workers finish their current job, then exit."""
        self._stop.set()
        if not self._tasks:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._tasks, return_exceptions=True), timeout=timeout
            )
        except asyncio.TimeoutError:  # pragma: no cover - defensive
            for task in self._tasks:
                task.cancel()
        self._tasks.clear()
        logger.info("job_runner_stopped", worker_id=self.worker_id)

    # ── Worker loop ──────────────────────────────────────────────────────────

    async def _worker_loop(self, slot: str) -> None:
        while not self._stop.is_set():
            claimed = await self._claim(slot)
            if claimed is None:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), timeout=self.poll_interval)
                continue
            await self._execute(claimed)

    async def _claim(self, slot: str) -> repositories.Job | None:
        async with self.db.session() as session:
            job = await repositories.claim_next_job(session, worker_id=f"{slot}")
            if job is None:
                return None
            # Detach the fields we need; the session closes right after.
            claimed = repositories.Job(
                id=job.id, meeting_id=job.meeting_id, user_id=job.user_id, attempts=job.attempts
            )
            # Reflect the claim on the meeting row immediately (UI polls it).
            meeting = await session.get(Meeting, job.meeting_id)
            if meeting is not None:
                meeting.status = "processing"
                meeting.stage = "starting"
                meeting.progress = 0
                meeting.error = None
                meeting.started_at = utcnow()
            await session.commit()
        return claimed

    async def _execute(self, job: repositories.Job) -> None:
        started = time.perf_counter()
        usage_start = self.registry.usage.snapshot()
        ACTIVE_JOBS.inc()
        logger.info(
            "job_started",
            job_id=str(job.id),
            meeting_id=str(job.meeting_id),
            attempt=job.attempts,
        )
        async with self.db.session() as session:
            await repositories.update_job_progress(session, job.id, stage="starting", progress=0)
        try:
            context = JobContext(
                job_id=job.id,
                meeting_id=job.meeting_id,
                user_id=job.user_id,
                db=self.db,
                registry=self.registry,
            )
            await self.pipeline(context)
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            raise
        except Exception as exc:  # noqa: BLE001 - the runner is the error boundary
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("job_failed", job_id=str(job.id), attempt=job.attempts, error=error)
            async with self.db.session() as session:
                outcome = await repositories.fail_job(
                    session,
                    job.id,
                    error=error,
                    retry_backoff_seconds=self.settings.job_retry_backoff_seconds,
                )
            JOBS_TOTAL.labels(status=outcome).inc()
        else:
            usage_end = self.registry.usage.snapshot()
            token_usage = {key: usage_end[key] - usage_start.get(key, 0) for key in usage_end}
            async with self.db.session() as session:
                await repositories.complete_job(session, job.id, token_usage=token_usage)
                await self._mark_meeting_completed(session, job.meeting_id)
            JOBS_TOTAL.labels(status="succeeded").inc()
            logger.info(
                "job_succeeded",
                job_id=str(job.id),
                meeting_id=str(job.meeting_id),
                duration_ms=round((time.perf_counter() - started) * 1000),
            )
        finally:
            ACTIVE_JOBS.dec()
            JOB_DURATION.observe(time.perf_counter() - started)

    async def _mark_meeting_completed(self, session, meeting_id: uuid.UUID) -> None:  # type: ignore[no-untyped-def]
        meeting = await session.get(Meeting, meeting_id)
        if meeting is not None and meeting.status != "failed":
            meeting.status = "completed"
            meeting.stage = "done"
            meeting.progress = 100
            meeting.completed_at = utcnow()

    # ── Reaper ───────────────────────────────────────────────────────────────

    async def _reaper_loop(self) -> None:
        interval = max(self.settings.job_stale_after_seconds / 4, 5.0)
        while not self._stop.is_set():
            try:
                async with self.db.session() as session:
                    count = await repositories.requeue_stale_jobs(
                        session, stale_after_seconds=self.settings.job_stale_after_seconds
                    )
                if count:
                    logger.warning("reaper_requeued_jobs", count=count)
            except Exception:  # noqa: BLE001 - the reaper must never die
                logger.exception("reaper_failed")
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=interval)

    # ── Test helper ──────────────────────────────────────────────────────────

    async def drain(self, *, timeout: float = 30.0, max_jobs: int | None = None) -> None:
        """Process runnable jobs (used by tests and scripts).

        ``max_jobs`` bounds how many jobs are claimed in one call, which lets
        tests step through retry attempts deterministically.
        """
        deadline = time.monotonic() + timeout
        processed = 0
        while time.monotonic() < deadline:
            if max_jobs is not None and processed >= max_jobs:
                return
            claimed = await self._claim(self.worker_id)
            if claimed is None:
                return
            await self._execute(claimed)
            processed += 1


async def submit_processing_job(
    db: Database,
    *,
    meeting_id: uuid.UUID,
    user_id: uuid.UUID,
    settings: Settings,
) -> uuid.UUID:
    """Insert the queued job row (called from request handlers)."""
    async with db.session() as session:
        job = await repositories.create_job(
            session,
            meeting_id=meeting_id,
            user_id=user_id,
            max_attempts=settings.job_max_attempts,
        )
        return job.id


async def get_job_or_404(
    db: Database, job_id: uuid.UUID, *, user_id: uuid.UUID
) -> repositories.Job:
    async with db.session() as session:
        job = await repositories.get_job(session, job_id, user_id=user_id)
        if job is None:
            raise NotFoundError("Job not found")
        return job
