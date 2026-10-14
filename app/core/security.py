"""Password hashing and JWT token helpers.

Password hashing uses PBKDF2-HMAC-SHA256 from the standard library: no native
build step, memory-hard enough with 310k iterations, and the format is
self-describing (``pbkdf2_sha256$iterations$salt$hash``) so the algorithm can
be upgraded later without invalidating existing passwords.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

from app.config import Settings
from app.errors import AuthenticationError

_ALGORITHM = "HS256"
_SCHEME = "pbkdf2_sha256"
_SALT_BYTES = 16
_KEY_BYTES = 32


def hash_password(password: str, *, iterations: int = 310_000) -> str:
    """Hash a password with PBKDF2-HMAC-SHA256."""
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{_SCHEME}${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time verification of a password against a stored hash."""
    try:
        scheme, iterations_s, salt_hex, digest_hex = stored.split("$")
        if scheme != _SCHEME:
            return False
        iterations = int(iterations_s)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except (ValueError, AttributeError):
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(candidate, expected)


def create_access_token(
    subject: str, settings: Settings, extra_claims: dict[str, Any] | None = None
) -> str:
    """Create a signed JWT for a user id (``sub``)."""
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": subject,
        "iat": now,
        "exp": now + timedelta(minutes=settings.access_token_expire_minutes),
        "iss": settings.app_name,
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, settings.jwt_secret, algorithm=_ALGORITHM)


def decode_access_token(token: str, settings: Settings) -> dict[str, Any]:
    """Decode and validate a JWT, raising ``AuthenticationError`` when invalid."""
    try:
        return jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[_ALGORITHM],
            issuer=settings.app_name,
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("Token has expired") from exc
    except jwt.PyJWTError as exc:
        raise AuthenticationError("Invalid token") from exc
