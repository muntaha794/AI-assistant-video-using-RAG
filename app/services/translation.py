"""Hindi → English translation with timestamp preservation.

Whisper reports the detected language; when it is Hindi the transcript is
translated segment-by-segment so every timestamp keeps pointing at the same
utterance. The original Hindi text is preserved in ``original_text``.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Protocol

from app.config import Settings
from app.errors import TransientProcessingError
from app.logging_config import get_logger
from app.metrics import STAGE_DURATION
from app.services.transcription import TranscriptSegmentData

logger = get_logger(__name__)

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


@dataclass
class TranslatedSegment:
    """A segment after optional translation (speaker attached post-diarization)."""

    start_ms: int
    end_ms: int
    text: str  # translated (or original when no translation happened)
    speaker: str = "SPEAKER_00"
    seq: int = 0  # 0-based position in the transcript
    original_text: str | None = None
    is_translated: bool = False


class Translator(Protocol):
    async def translate_batch(self, texts: list[str]) -> list[str]: ...


# ── Pure helpers (unit-tested) ────────────────────────────────────────────────


def is_hindi_text(text: str, *, min_ratio: float = 0.25) -> bool:
    """True when a meaningful share of the text is Devanagari script."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    devanagari = sum(1 for ch in letters if _DEVANAGARI_RE.match(ch))
    return devanagari / len(letters) >= min_ratio


def needs_translation(language: str | None, segments: list[TranscriptSegmentData]) -> bool:
    """Decide from Whisper's language hint, with a Devanagari fallback check."""
    if language and language.lower().startswith("hi"):
        return True
    if language and not language.lower().startswith(("hi", "unknown")):
        return False
    sample = " ".join(seg.text for seg in segments[:20])
    return is_hindi_text(sample)


# ── Translator implementations ────────────────────────────────────────────────


class GoogleTranslatorAdapter:
    """deep-translator wrapper: free Google Translate, batched and rate-limited."""

    def __init__(self, source: str = "hi", target: str = "en", batch_size: int = 20) -> None:
        self.source = source
        self.target = target
        self.batch_size = max(1, batch_size)

    async def translate_batch(self, texts: list[str]) -> list[str]:
        if not texts:
            return []

        def _translate(chunk: list[str]) -> list[str]:
            try:
                from deep_translator import GoogleTranslator  # noqa: PLC0415 - lazy dep
            except ImportError as exc:  # pragma: no cover - needs requirements-ml
                raise TransientProcessingError(
                    "deep-translator is not installed (pip install -r requirements-ml.txt)"
                ) from exc
            translator = GoogleTranslator(source=self.source, target=self.target)
            joined = "\n".join(chunk)
            translated = translator.translate(joined)
            lines = (translated or "").split("\n")
            # The API may collapse empty lines; pad back to the original count.
            while len(lines) < len(chunk):
                lines.append("")
            return lines[: len(chunk)]

        out: list[str] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            try:
                out.extend(await asyncio.to_thread(_translate, chunk))
            except TransientProcessingError:
                raise
            except Exception as exc:  # noqa: BLE001 - network failures are transient
                raise TransientProcessingError(f"Translation failed: {exc}") from exc
            # Be polite to the free endpoint between batches.
            await asyncio.sleep(0.4)
        return out


def apply_translation(
    segments: list[TranscriptSegmentData], translated_texts: list[str]
) -> list[TranslatedSegment]:
    """Pair each segment with its translation (timestamps untouched)."""
    result: list[TranslatedSegment] = []
    for idx, seg in enumerate(segments):
        translated = translated_texts[idx].strip() if idx < len(translated_texts) else ""
        if translated:
            result.append(
                TranslatedSegment(
                    start_ms=seg.start_ms,
                    end_ms=seg.end_ms,
                    text=translated,
                    seq=idx,
                    original_text=seg.text,
                    is_translated=True,
                )
            )
        else:  # translation missing for this line — keep the original
            result.append(
                TranslatedSegment(
                    start_ms=seg.start_ms,
                    end_ms=seg.end_ms,
                    text=seg.text,
                    seq=idx,
                    is_translated=False,
                )
            )
    return result


async def translate_transcript(
    segments: list[TranscriptSegmentData],
    language: str | None,
    translator: Translator,
    *,
    enabled: bool = True,
) -> list[TranslatedSegment]:
    """Translate Hindi segments to English, preserving timestamps."""
    passthrough = [
        TranslatedSegment(start_ms=s.start_ms, end_ms=s.end_ms, text=s.text, seq=idx)
        for idx, s in enumerate(segments)
    ]
    if not enabled or not needs_translation(language, segments):
        return passthrough

    started = time.perf_counter()
    texts = [seg.text for seg in segments]
    translated_texts = await translator.translate_batch(texts)
    STAGE_DURATION.labels(stage="translate").observe(time.perf_counter() - started)
    logger.info("transcript_translated", segments=len(segments), source="hi", target="en")
    return apply_translation(segments, translated_texts)


def build_translator(settings: Settings) -> Translator:
    return GoogleTranslatorAdapter(
        source=settings.translation_source_language,
        target=settings.translation_target_language,
        batch_size=settings.translation_batch_size,
    )
