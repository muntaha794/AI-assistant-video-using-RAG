"""Standalone worker entrypoint.

Run with ``python -m app.worker`` to process meeting jobs without an HTTP
server (e.g. the optional ``worker`` service in docker-compose.yml). Workers
poll the same ``jobs`` table as the API, claiming work with SKIP LOCKED, so
they scale horizontally and survive API restarts.
"""

from __future__ import annotations

import asyncio
import contextlib

from app.config import get_settings
from app.db.base import Base
from app.db.session import build_database
from app.logging_config import configure_logging, get_logger
from app.pipeline import process_meeting
from app.services.jobs import JobRunner
from app.services.registry import build_service_registry

logger = get_logger(__name__)


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    db = build_database(settings)
    if settings.effective_auto_create_tables:
        async with db.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    registry = build_service_registry(settings)
    runner = JobRunner(db, registry, process_meeting, settings)
    await runner.start()
    logger.info("worker_started", worker_id=runner.worker_id)
    try:
        # Run until cancelled (Ctrl-C / SIGTERM).
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:  # pragma: no cover - signal path
        pass
    finally:
        await runner.stop()
        await db.dispose()


if __name__ == "__main__":  # pragma: no cover
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
