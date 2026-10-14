"""FastAPI application entrypoint.

``uvicorn app.main:app`` — wires settings, logging, database, services and the
job runner in the lifespan handler, plus request-id middleware, CORS and
consistent error bodies.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.error_handlers import register_exception_handlers
from app.api.routes import action_items, auth, chat, health, jobs, meetings
from app.config import Settings, get_settings
from app.db.base import Base
from app.db.session import build_database
from app.logging_config import configure_logging, get_logger
from app.metrics import HTTP_LATENCY, HTTP_REQUESTS
from app.pipeline import process_meeting
from app.services.jobs import JobRunner
from app.services.registry import ServiceRegistry, build_service_registry

logger = get_logger(__name__)

API_DESCRIPTION = """\
AI Meeting Assistant — production-grade meeting intelligence.

* **Ingest**: upload audio or paste a YouTube URL; processing happens asynchronously.
* **Transcript**: local Whisper with per-segment timestamps and speaker diarization;
  Hindi transcripts are translated to English with timestamps preserved.
* **Summary**: Mistral-powered summary, key decisions anchored to timestamps.
* **Action items**: extracted with owners, deadlines and evidence quotes; trackable
  across meetings.
* **Chat with evidence**: RAG over the transcript; every answer carries timestamp
  citations back into the recording.
"""


def create_app(
    settings: Settings | None = None, *, registry: ServiceRegistry | None = None
) -> FastAPI:
    """Application factory (tests inject a fake registry here)."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        app.state.settings = settings
        app.state.db = build_database(settings)
        app.state.registry = registry or build_service_registry(settings)
        if settings.effective_auto_create_tables:
            async with app.state.db.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        runner = JobRunner(app.state.db, app.state.registry, process_meeting, settings)
        app.state.runner = runner
        await runner.start()
        logger.info(
            "application_started",
            env=settings.app_env,
            db=settings.database_url.split("://")[0],
        )
        try:
            yield
        finally:
            await runner.stop()
            await app.state.db.dispose()
            logger.info("application_stopped")

    app = FastAPI(
        title=settings.app_name,
        description=API_DESCRIPTION,
        version="1.0.0",
        lifespan=lifespan,
        contact={"name": "AI Meeting Assistant"},
        license_info={"name": "MIT"},
    )

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        import structlog

        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:16])
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            elapsed = perf_counter() - started
            route = request.scope.get("route")
            path_label = getattr(route, "path", request.url.path)
            HTTP_REQUESTS.labels(request.method, path_label, str(status_code)).inc()
            HTTP_LATENCY.labels(request.method, path_label).observe(elapsed)
            structlog.contextvars.clear_contextvars()

    register_exception_handlers(app)

    app.include_router(health.router)
    prefix = settings.api_v1_prefix
    app.include_router(auth.router, prefix=prefix)
    app.include_router(meetings.router, prefix=prefix)
    app.include_router(chat.router, prefix=prefix)
    app.include_router(action_items.router, prefix=prefix)
    app.include_router(jobs.router, prefix=prefix)
    return app


app = create_app()
