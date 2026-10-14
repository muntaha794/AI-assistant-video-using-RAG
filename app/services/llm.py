"""LLM access layer.

A thin, typed wrapper around the official Mistral SDK with:
* retry with exponential backoff for transient failures (429/5xx/timeouts),
* hard per-call timeout,
* structured logging + Prometheus metrics for latency and token usage,
* a protocol so tests can inject a deterministic fake.

The Mistral free tier is used via ``mistralai`` with any model id from
``settings.mistral_model`` (default: ``mistral-small-latest``).
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.config import Settings
from app.core.reliability import retry_transient
from app.errors import ExternalServiceError, TransientProcessingError
from app.logging_config import get_logger
from app.metrics import LLM_CALLS, LLM_LATENCY, LLM_TOKENS

logger = get_logger(__name__)

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}


@dataclass
class LLMResult:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLM(Protocol):
    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResult: ...


@dataclass
class UsageTracker:
    """In-process token accounting, snapshotted into the Job row per run."""

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    def add(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.calls += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "calls": self.calls,
            }

    def reset(self) -> None:
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.calls = 0


class MistralLLM:
    """Async wrapper over the official ``mistralai`` client."""

    def __init__(
        self,
        api_key: str | None,
        model: str,
        *,
        timeout_seconds: float = 120.0,
        max_retries: int = 3,
        default_temperature: float = 0.2,
        default_max_tokens: int = 4096,
        usage: UsageTracker | None = None,
    ) -> None:
        if not api_key:
            raise ExternalServiceError(
                "MISTRAL_API_KEY is not configured (set it in the environment / .env)"
            )
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.usage = usage or UsageTracker()
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from mistralai import Mistral  # current SDK (>=1.0)
            except ImportError:  # pragma: no cover - legacy SDK fallback
                from mistralai.client import MistralClient  # type: ignore[import-not-found]

                self._client = MistralClient(api_key=self.api_key)
                return self._client
            self._client = Mistral(api_key=self.api_key)
        return self._client

    def _call_blocking(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResult:
        client = self._get_client()
        started = time.perf_counter()
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            request["response_format"] = {"type": "json_object"}

        try:
            response = client.chat.complete(**request)
        except Exception as exc:  # noqa: BLE001 - normalize SDK error surface
            message = str(exc)
            if json_mode and "response_format" in message.lower():
                request.pop("response_format", None)
                response = client.chat.complete(**request)
            elif "timeout" in message.lower() or "connection" in message.lower():
                raise TransientProcessingError(f"Mistral API unreachable: {message[:300]}") from exc
            elif "429" in message or "rate" in message.lower():
                raise TransientProcessingError(f"Mistral rate limit: {message[:300]}") from exc
            elif "401" in message or "403" in message or "api_key" in message.lower():
                raise ExternalServiceError(f"Mistral auth failed: {message[:300]}") from exc
            else:
                raise ExternalServiceError(f"Mistral call failed: {message[:300]}") from exc

        latency = time.perf_counter() - started
        choices = getattr(response, "choices", None) or []
        text = ""
        if choices:
            message_obj = choices[0].message
            text = (getattr(message_obj, "content", "") or "").strip()
        usage_obj = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage_obj, "completion_tokens", 0) or 0)
        return LLMResult(
            text=text,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_seconds=latency,
        )

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResult:
        temp = self.default_temperature if temperature is None else temperature
        tokens = self.default_max_tokens if max_tokens is None else max_tokens

        @retry_transient(attempts=self.max_retries, min_wait=1.0, max_wait=30.0)
        async def _invoke() -> LLMResult:
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._call_blocking,
                        messages,
                        temperature=temp,
                        max_tokens=tokens,
                        json_mode=json_mode,
                    ),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                raise TransientProcessingError(
                    f"Mistral call timed out after {self.timeout_seconds:.0f}s"
                ) from exc
            self.usage.add(result.prompt_tokens, result.completion_tokens)
            LLM_TOKENS.labels(model=self.model, direction="prompt").inc(result.prompt_tokens)
            LLM_TOKENS.labels(model=self.model, direction="completion").inc(
                result.completion_tokens
            )
            LLM_LATENCY.labels(model=self.model).observe(result.latency_seconds)
            LLM_CALLS.labels(model=self.model, outcome="ok").inc()
            return result

        try:
            result = await _invoke()
        except (ExternalServiceError, TransientProcessingError):
            LLM_CALLS.labels(model=self.model, outcome="error").inc()
            raise
        logger.info(
            "llm_call_finished",
            model=self.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_ms=round(result.latency_seconds * 1000, 1),
            json_mode=json_mode,
        )
        return result


# ── JSON extraction helper ────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Parse JSON from an LLM reply, tolerating code fences and prose."""
    text = text.strip()
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[idx:])
            if isinstance(value, (dict, list)):
                return value
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Model reply is not valid JSON: {text[:200]}")


def build_llm(settings: Settings, usage: UsageTracker | None = None) -> MistralLLM:
    return MistralLLM(
        api_key=settings.mistral_api_key,
        model=settings.mistral_model,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        default_temperature=settings.llm_temperature,
        default_max_tokens=settings.llm_max_tokens,
        usage=usage,
    )
