"""Shared FastAPI dependencies: database sessions, auth, service access."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.db.models import User
from app.db.repositories import get_user
from app.errors import AuthenticationError
from app.services.jobs import JobRunner
from app.services.registry import ServiceRegistry

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/token")


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_database(request: Request):  # type: ignore[no-untyped-def]
    return request.app.state.db


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Per-request session: commit on success, rollback on error."""
    async with request.app.state.db.session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_registry(request: Request) -> ServiceRegistry:
    return request.app.state.registry


def get_runner(request: Request) -> JobRunner:
    return request.app.state.runner


async def get_current_user(
    token: Annotated[str | None, Depends(oauth2_scheme)],
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> User:
    """Resolve the JWT bearer token to an active user (401 otherwise)."""
    from app.core.security import decode_access_token

    if not token:
        raise AuthenticationError("Missing bearer token")
    payload = decode_access_token(token, settings)
    try:
        user_id = uuid.UUID(payload["sub"])
    except (KeyError, ValueError) as exc:
        raise AuthenticationError("Token subject is not a valid user id") from exc
    user = await get_user(session, user_id)
    if user is None or not user.is_active:
        raise AuthenticationError("User no longer active")
    return user


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]
CurrentUser = Annotated[User, Depends(get_current_user)]
RegistryDep = Annotated[ServiceRegistry, Depends(get_registry)]
RunnerDep = Annotated[JobRunner, Depends(get_runner)]
