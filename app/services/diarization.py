"""Speaker diarization.

Primary path: ``pyannote.audio`` speaker-diarization pipeline (requires an
``HF_TOKEN`` with access to the model). Fallback path: a deterministic
single-speaker labeller so the pipeline still produces a complete, correctly
timestamped transcript when diarization is unavailable — the README documents
how to enable the real thing.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.config import Settings
from app.logging_config import get_logger
from app.metrics import STAGE_DURATION
from app.services.transcription import TranscriptSegmentData

logger = get_logger(__name__)


@dataclass(frozen=True)
class SpeakerSpan:
    start_ms: int
    end_ms: int
    speaker: str


class Diarizer(Protocol):
    async def diarize(self, audio_path: Path, duration_seconds: float) -> list[SpeakerSpan]: ...


# ── Pure helpers (unit-tested) ────────────────────────────────────────────────


def best_effort_speaker(seg_start_ms: int, seg_end_ms: int, spans: list[SpeakerSpan]) -> str:
    """Assign the speaker whose spans overlap the segment the most."""
    best_speaker = "SPEAKER_00"
    best_overlap = 0
    duration = max(seg_end_ms - seg_start_ms, 1)
    for span in spans:
        overlap = min(span.end_ms, seg_end_ms) - max(span.start_ms, seg_start_ms)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = span.speaker
    # Require at least ~30% overlap before overriding the default label.
    if best_overlap / duration >= 0.3:
        return best_speaker
    return "SPEAKER_00"


def label_segments(segments: list[TranscriptSegmentData], spans: list[SpeakerSpan]) -> list[str]:
    if not spans:
        return ["SPEAKER_00"] * len(segments)
    return [best_effort_speaker(seg.start_ms, seg.end_ms, spans) for seg in segments]


def merge_speaker_runs(labels: list[str], segments: list[TranscriptSegmentData]) -> list[str]:
    """Collapse consecutive same-speaker labels so downstream text can be grouped."""
    if not labels:
        return []
    merged = [labels[0]]
    for label in labels[1:]:
        if label != merged[-1]:
            merged.append(label)
    return merged


class FallbackDiarizer:
    """No-op diarization: everything is attributed to SPEAKER_00."""

    async def diarize(self, audio_path: Path, duration_seconds: float) -> list[SpeakerSpan]:
        if duration_seconds > 0:
            return [
                SpeakerSpan(start_ms=0, end_ms=int(duration_seconds * 1000), speaker="SPEAKER_00")
            ]
        return []


class PyannoteDiarizer:
    """Real diarization via the pyannote speaker-diarization pipeline."""

    def __init__(
        self,
        *,
        hf_token: str,
        pipeline_name: str = "pyannote/speaker-diarization-3.1",
        max_speakers: int | None = None,
    ) -> None:
        self.hf_token = hf_token
        self.pipeline_name = pipeline_name
        self.max_speakers = max_speakers
        self._pipeline = None

    def _load(self):  # type: ignore[no-untyped-def]
        if self._pipeline is None:
            from pyannote.audio import Pipeline  # noqa: PLC0415 - lazy optional dep

            pipeline = Pipeline.from_pretrained(self.pipeline_name, use_auth_token=self.hf_token)
            self._pipeline = pipeline
        return self._pipeline

    def _diarize_blocking(self, audio_path: Path) -> list[SpeakerSpan]:
        pipeline = self._load()
        kwargs = {}
        if self.max_speakers:
            kwargs["min_speakers"] = 2
            kwargs["max_speakers"] = self.max_speakers
        annotation = pipeline(str(audio_path), **kwargs)
        spans = [
            SpeakerSpan(
                start_ms=int(turn.start * 1000),
                end_ms=int(turn.end * 1000),
                speaker=str(turn.label),
            )
            for turn in annotation.itertracks(yield_label=True)
        ]
        # Canonical, stable speaker names: SPEAKER_00, SPEAKER_01, ...
        renaming: dict[str, str] = {}
        for span in spans:
            renaming.setdefault(span.speaker, f"SPEAKER_{len(renaming):02d}")
        return [SpeakerSpan(span.start_ms, span.end_ms, renaming[span.speaker]) for span in spans]

    async def diarize(self, audio_path: Path, duration_seconds: float) -> list[SpeakerSpan]:
        started = time.perf_counter()
        spans = await asyncio.to_thread(self._diarize_blocking, audio_path)
        STAGE_DURATION.labels(stage="diarize").observe(time.perf_counter() - started)
        return spans


def build_diarizer(settings: Settings) -> Diarizer:
    """Prefer pyannote when configured and a token is present; else fall back."""
    if settings.diarization_enabled and settings.hf_token:
        return PyannoteDiarizer(
            hf_token=settings.hf_token,
            pipeline_name=settings.pyannote_pipeline,
            max_speakers=settings.diarization_max_speakers,
        )
    if settings.diarization_enabled:
        logger.warning(
            "diarization_disabled_missing_token",
            hint="Set HF_TOKEN (HuggingFace read token with pyannote access) to enable "
            "speaker diarization; falling back to single-speaker transcripts.",
        )
    return FallbackDiarizer()
