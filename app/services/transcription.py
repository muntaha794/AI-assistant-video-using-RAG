"""Speech-to-text with local Whisper.

The model is loaded lazily once per (model-name, device) and reused; all
blocking work runs in a worker thread so the event loop stays responsive.
A transcriber protocol allows the test-suite to swap in a deterministic fake.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from app.errors import ExternalServiceError
from app.logging_config import get_logger
from app.metrics import STAGE_DURATION

logger = get_logger(__name__)


@dataclass(frozen=True)
class TranscriptSegmentData:
    """One timestamped utterance, in milliseconds."""

    start_ms: int
    end_ms: int
    text: str


@dataclass
class TranscriptResult:
    language: str
    segments: list[TranscriptSegmentData] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def plain_text(self) -> str:
        return " ".join(segment.text.strip() for segment in self.segments if segment.text.strip())


class Transcriber(Protocol):
    async def transcribe(self, audio_path: Path) -> TranscriptResult: ...


class WhisperTranscriber:
    """Local Whisper implementation (openai-whisper package)."""

    def __init__(self, model_name: str = "base", device: str = "auto") -> None:
        self.model_name = model_name
        self.device = None if device == "auto" else device
        self._model = None
        self._lock = threading.Lock()

    def _load_model(self):  # type: ignore[no-untyped-def]
        with self._lock:
            if self._model is None:
                try:
                    import whisper  # noqa: PLC0415 - lazy optional dependency
                except ImportError as exc:  # pragma: no cover - needs requirements-ml
                    raise ExternalServiceError(
                        "openai-whisper is not installed (pip install -r requirements-ml.txt)"
                    ) from exc
                logger.info("loading_whisper_model", model=self.model_name, device=self.device)
                self._model = whisper.load_model(self.model_name, device=self.device)
        return self._model

    def _transcribe_blocking(self, audio_path: Path) -> TranscriptResult:
        model = self._load_model()
        # fp16 only makes sense on CUDA; whisper warns loudly on CPU otherwise.
        options: dict = {"fp16": self.device != "cpu"}
        result = model.transcribe(str(audio_path), **options)
        segments = [
            TranscriptSegmentData(
                start_ms=int(float(seg["start"]) * 1000),
                end_ms=int(float(seg["end"]) * 1000),
                text=str(seg["text"]).strip(),
            )
            for seg in result.get("segments", [])
            if str(seg.get("text", "")).strip()
        ]
        duration = segments[-1].end_ms / 1000.0 if segments else 0.0
        return TranscriptResult(
            language=str(result.get("language", "unknown")),
            segments=segments,
            duration_seconds=duration,
        )

    async def transcribe(self, audio_path: Path) -> TranscriptResult:
        started = time.perf_counter()
        try:
            result = await asyncio.to_thread(self._transcribe_blocking, audio_path)
        except ExternalServiceError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced as a processing error
            logger.exception("whisper_transcription_failed", error=str(exc))
            raise ExternalServiceError(f"Transcription failed: {exc}") from exc
        STAGE_DURATION.labels(stage="transcribe").observe(time.perf_counter() - started)
        logger.info(
            "transcription_finished",
            model=self.model_name,
            language=result.language,
            segments=len(result.segments),
            duration_seconds=round(result.duration_seconds, 2),
        )
        return result
