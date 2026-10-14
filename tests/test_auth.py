"""Authentication and per-user isolation tests."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from tests.conftest import register_user


async def test_register_returns_201_and_hides_password(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": "ana@example.com", "password": "supersecret1", "full_name": "Ana"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["email"] == "ana@example.com"
    assert "hashed_password" not in body and "password" not in body


async def test_duplicate_email_conflicts(client: AsyncClient) -> None:
    await register_user(client)
    response = await client.post(
        "/api/v1/auth/register", json={"email": "ana@example.com", "password": "anotherpass1"}
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "conflict"


async def test_login_issues_token_and_me_works(client: AsyncClient) -> None:
    headers = await register_user(client)
    response = await client.get("/api/v1/auth/me", headers=headers)
    assert response.status_code == 200
    assert response.json()["email"] == "ana@example.com"


async def test_wrong_password_is_401(client: AsyncClient) -> None:
    await register_user(client)
    response = await client.post(
        "/api/v1/auth/login", json={"email": "ana@example.com", "password": "wrong-password"}
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthenticated"


async def test_missing_token_is_401(client: AsyncClient) -> None:
    response = await client.get("/api/v1/auth/me")
    assert response.status_code == 401


@pytest.mark.parametrize("password", ["short", "1234567"])
async def test_weak_password_rejected(client: AsyncClient, password: str) -> None:
    response = await client.post(
        "/api/v1/auth/register", json={"email": "weak@example.com", "password": password}
    )
    assert response.status_code == 422


async def test_token_form_login(client: AsyncClient) -> None:
    await register_user(client)
    response = await client.post(
        "/api/v1/auth/token", data={"username": "ana@example.com", "password": "supersecret1"}
    )
    assert response.status_code == 200
    assert response.json()["token_type"] == "bearer"
