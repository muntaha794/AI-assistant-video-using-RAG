"""Authentication routes: register, login (JSON + OAuth2 form), current user."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, SessionDep, SettingsDep
from app.core.security import create_access_token, hash_password, verify_password
from app.db.models import User
from app.errors import AuthenticationError, ConflictError
from app.schemas.auth import LoginRequest, TokenOut, UserCreate, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create an account",
)
async def register(payload: UserCreate, session: SessionDep) -> User:
    from app.db.repositories import get_user_by_email

    existing = await get_user_by_email(session, payload.email)
    if existing is not None:
        raise ConflictError("An account with this email already exists")
    user = User(
        email=payload.email.lower().strip(),
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
    )
    session.add(user)
    try:
        await session.flush()
    except IntegrityError as exc:  # concurrent duplicate registration
        raise ConflictError("An account with this email already exists") from exc
    await session.commit()
    return user


@router.post(
    "/token", response_model=TokenOut, summary="Login (OAuth2 form, for Swagger Authorize)"
)
async def token_form(
    session: SessionDep,
    settings: SettingsDep,
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> TokenOut:
    return await _issue_token(session, settings, form_data.username, form_data.password)


@router.post("/login", response_model=TokenOut, summary="Login (JSON)")
async def login(payload: LoginRequest, session: SessionDep, settings: SettingsDep) -> TokenOut:
    return await _issue_token(session, settings, payload.email, payload.password)


@router.get("/me", response_model=UserOut, summary="Current user")
async def me(user: CurrentUser) -> User:
    return user


async def _issue_token(  # type: ignore[no-untyped-def]
    session: AsyncSession, settings, email: str, password: str
) -> TokenOut:
    user = await session.scalar(select(User).where(User.email == email.lower().strip()))
    if user is None or not verify_password(password, user.hashed_password):
        # Same message for unknown user and bad password: no account enumeration.
        raise AuthenticationError("Invalid email or password")
    token = create_access_token(str(user.id), settings)
    return TokenOut(access_token=token, expires_in=settings.access_token_expire_minutes * 60)
