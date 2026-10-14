"""Typed application errors.

Services raise the exceptions below (never bare ``Exception``); the API layer
maps them to HTTP responses in one place (``app.api.error_handlers``), so error
handling stays consistent and clients always get a machine-readable code.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for all expected, user-facing errors."""

    status_code = 400
    code = "app_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class AuthorizationError(AppError):
    status_code = 403
    code = "forbidden"


class AuthenticationError(AppError):
    status_code = 401
    code = "unauthenticated"


class ConflictError(AppError):
    status_code = 409
    code = "conflict"


class ValidationError(AppError):
    status_code = 422
    code = "validation_error"


class PayloadTooLargeError(AppError):
    status_code = 413
    code = "payload_too_large"


class UnsupportedMediaError(AppError):
    status_code = 415
    code = "unsupported_media_type"


class RateLimitedError(AppError):
    status_code = 429
    code = "rate_limited"


class ExternalServiceError(AppError):
    """A dependency (Mistral, Whisper, yt-dlp, translator) failed."""

    status_code = 502
    code = "external_service_error"


class TransientProcessingError(ExternalServiceError):
    """A failure worth retrying (timeouts, 5xx, connection resets)."""

    status_code = 503
    code = "transient_processing_error"


class ProcessingError(AppError):
    """Permanent failure while processing a meeting."""

    status_code = 500
    code = "processing_error"
