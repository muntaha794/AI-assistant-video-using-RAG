"""Reliability primitives: retries with exponential backoff + timeouts.

Used around every external dependency (Mistral API, Whisper, yt-dlp,
translation). Retries are reserved for transient failures; permanent
validation errors are never retried.
"""

from __future__ import annotations

import asyncio
import collections.abc as cabc
import functools
import logging
from typing import Any, TypeVar

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.errors import TransientProcessingError

logger = logging.getLogger(__name__)

T = TypeVar("T")

_TRANSIENT_ERROR_NAMES = {
    "APITimeoutError",
    "APIConnectionError",
    "RateLimitError",
    "RemoteError",
    "ConnectError",
    "ReadTimeout",
    "TransportError",
}


def is_transient_exception(exc: BaseException) -> bool:
    """Whether a failure is worth retrying (timeouts, connection issues, 5xx)."""
    if isinstance(exc, TransientProcessingError):
        return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    if isinstance(exc, OSError):
        return True
    return type(exc).__name__ in _TRANSIENT_ERROR_NAMES


def retry_transient(
    *,
    attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 30.0,
) -> cabc.Callable[[cabc.Callable[..., Any]], cabc.Callable[..., Any]]:
    """Async decorator: retry transient failures with jittered exponential backoff."""

    def decorator(func: cabc.Callable[..., Any]) -> cabc.Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            retrying = AsyncRetrying(
                stop=stop_after_attempt(attempts),
                wait=wait_exponential_jitter(initial=min_wait, max=max_wait),
                retry=retry_if_exception(is_transient_exception),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True,
            )
            async for attempt in retrying:
                with attempt:
                    return await func(*args, **kwargs)
            raise RuntimeError("retry loop exhausted")  # pragma: no cover

        return wrapper

    return decorator


async def run_with_timeout(func: cabc.Callable[[], T], timeout: float, *, label: str) -> T:
    """Run a (possibly blocking) callable in a worker thread with a hard timeout."""

    async def _call() -> T:
        result = func()
        if asyncio.iscoroutine(result):
            return await result  # type: ignore[misc]
        return result

    try:
        return await asyncio.wait_for(_call(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise TransientProcessingError(f"{label} timed out after {timeout:.0f}s") from exc


def log_retry(state: RetryCallState) -> None:
    """Tenacity ``before_sleep`` hook with structured context."""
    exc = state.outcome.exception() if state.outcome else None
    logger.warning(
        "retrying_after_failure",
        extra={"attempt": state.attempt_number, "error": str(exc)},
    )
