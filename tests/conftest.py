"""Shared fixtures: isolated SQLite DB per test, app with fake services."""

from __future__ import annotations

import uuid
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.config import Settings, reset_settings_cache
from app.main import create_app
from tests.fakes import FakeLLM, build_fake_registry


@pytest.fixture
def settings(tmp_path) -> Settings:
    reset_settings_cache()
    return Settings(
        _env_file=None,  # ignore any real .env in the workspace
        app_env="test",
        log_level="WARNING",
        log_json=False,
        database_url=f"sqlite+aiosqlite:///{tmp_path}/test.db",
        upload_dir=tmp_path / "uploads",
        jwt_secret="test-secret-key-must-be-32-bytes-long",
        mistral_api_key="test-key",
        vector_store="memory",
        embedding_backend="hashing",
        auto_create_tables=True,
        # Tests drive the queue deterministically with runner.drain() instead
        # of background worker loops, avoiding claim/process races.
        job_workers=0,
        job_poll_interval_seconds=0.05,
        job_retry_backoff_seconds=0,
        rag_min_score=0.0,
        rag_top_k=4,
    )


@pytest_asyncio.fixture
async def app_obj(settings: Settings):
    registry = build_fake_registry(settings)
    app = create_app(settings, registry=registry)
    async with app.router.lifespan_context(app):
        yield app


@pytest.fixture
def registry(settings: Settings):
    """Standalone fake registry (used by job-runner tests without the app)."""
    return build_fake_registry(settings)


@pytest_asyncio.fixture
async def client(app_obj) -> AsyncClient:
    transport = ASGITransport(app=app_obj)
    async with AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        yield http_client


@pytest_asyncio.fixture
async def runner(app_obj):  # type: ignore[no-untyped-def]
    """The in-app job runner (for drain())."""
    return app_obj.state.runner


@pytest.fixture
def fake_llm(app_obj) -> FakeLLM:
    return app_obj.state.registry.llm


async def register_user(
    client: AsyncClient, email: str = "ana@example.com", password: str = "supersecret1"
) -> dict[str, str]:
    response = await client.post(
        "/api/v1/auth/register", json={"email": email, "password": password, "full_name": "Ana"}
    )
    assert response.status_code == 201, response.text
    return await login(client, email, password)


async def login(
    client: AsyncClient, email: str = "ana@example.com", password: str = "supersecret1"
) -> dict[str, str]:
    response = await client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200, response.text
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


async def upload_meeting(
    client: AsyncClient, headers: dict[str, str], *, filename: str = "standup.mp3"
) -> dict[str, Any]:
    from tests.fakes import FAKE_MP3_BYTES

    response = await client.post(
        "/api/v1/meetings/upload",
        headers=headers,
        files={"file": (filename, FAKE_MP3_BYTES, "audio/mpeg")},
        data={"title": "Weekly standup"},
    )
    assert response.status_code == 202, response.text
    return response.json()


@pytest.fixture
def anyio_backend() -> str:  # pragma: no cover - compat with trio-free installs
    return "asyncio"


def new_uuid() -> uuid.UUID:
    return uuid.uuid4()
