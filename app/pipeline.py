"""End-to-end meeting processing pipeline.

Runs inside a worker slot (see ``app.services.jobs``). Stages:

1. acquire   — download YouTube audio, or use the uploaded file
2. transcribe — local Whisper → timestamped segments + detected language
3. translate — Hindi → English with timestamps preserved (no-op otherwise)
4. diarize   — speaker labels per segment (pyannote or fallback)
5. persist   — transcript segments + meeting metadata in PostgreSQL
6. summarize — Mistral summary + timestamped key decisions
7. actions   — Mistral action items, normalized with owners/deadlines/evidence
8. index     — timestamped chunks → embeddings → vector store

All heavy blocking work happens in worker threads; the event loop stays free.
Each stage reports progress and is individually timed in Prometheus.
"""

from __future__ import annotations

import time
from pathlib import Path

from app.config import Settings
from app.db import repositories
from app.db.models import Meeting
from app.db.session import Database
from app.errors import ProcessingError
from app.logging_config import get_logger
from app.metrics import STAGE_DURATION
from app.services.audio import download_youtube_audio, normalize_to_wav16k_mono
from app.services.diarization import label_segments
from app.services.jobs import JobContext
from app.services.transcription import TranscriptSegmentData
from app.services.translation import translate_transcript

logger = get_logger(__name__)


async def process_meeting(ctx: JobContext) -> None:
    """Full pipeline for one meeting; raises on permanent failure."""
    registry = ctx.registry
    settings: Settings = registry.settings
    db: Database = ctx.db

    # ── Load the meeting ─────────────────────────────────────────────────────
    async with db.session() as session:
        meeting = await repositories.get_meeting(session, ctx.meeting_id, user_id=ctx.user_id)
        if meeting is None:
            raise ProcessingError(f"Meeting {ctx.meeting_id} not found for user")
        source_type = meeting.source_type
        source_uri = meeting.source_uri
        meeting_dir = registry.storage.meeting_dir(meeting.id)

    # ── 1. Acquire audio ─────────────────────────────────────────────────────
    await ctx.stage("acquiring", 5)
    started = time.perf_counter()
    if source_type == "youtube":
        downloaded = await download_youtube_audio(source_uri or "", meeting_dir)
        audio_path: Path = downloaded
        async with db.session() as session:
            meeting = await session.get(Meeting, ctx.meeting_id)
            if meeting is not None:
                meeting.audio_path = downloaded.name
    else:
        if not source_uri:
            raise ProcessingError("Meeting has no stored audio file")
        audio_path = meeting_dir / Path(source_uri).name
        if not audio_path.exists():
            raise ProcessingError("Uploaded audio file is missing (expired?)")
    # Whisper prefers 16 kHz mono wav; normalize when the media stack exists.
    try:
        audio_path = normalize_to_wav16k_mono(audio_path)
    except Exception as exc:  # noqa: BLE001 - non-fatal: whisper can resample
        logger.info("wav_normalization_skipped", reason=str(exc)[:200])
    STAGE_DURATION.labels(stage="acquire").observe(time.perf_counter() - started)

    # ── 2. Transcribe ────────────────────────────────────────────────────────
    await ctx.stage("transcribing", 15)
    transcript = await registry.transcriber.transcribe(audio_path)
    if not transcript.segments:
        raise ProcessingError("Transcription produced no segments (empty or unreadable audio?)")

    # ── 3. Translate (Hindi → English, timestamps preserved) ────────────────
    await ctx.stage("translating", 35)
    translated = await translate_transcript(
        transcript.segments,
        transcript.language,
        registry.translator,
        enabled=settings.translate_enabled,
    )

    # ── 4. Diarize ───────────────────────────────────────────────────────────
    await ctx.stage("diarizing", 45)
    duration_seconds = transcript.duration_seconds or (
        translated[-1].end_ms / 1000.0 if translated else 0.0
    )
    spans = await registry.diarizer.diarize(audio_path, duration_seconds)
    plain_segments = [
        TranscriptSegmentData(start_ms=seg.start_ms, end_ms=seg.end_ms, text=seg.text)
        for seg in translated
    ]
    labels = label_segments(plain_segments, spans)
    for segment, label in zip(translated, labels, strict=True):
        segment.speaker = label
    speaker_count = len(set(labels))

    # ── 5. Persist transcript ────────────────────────────────────────────────
    await ctx.stage("saving_transcript", 55)
    async with db.session() as session:
        await repositories.replace_segments(
            session,
            ctx.meeting_id,
            [
                {
                    "start_ms": seg.start_ms,
                    "end_ms": seg.end_ms,
                    "speaker": seg.speaker,
                    "text": seg.text,
                    "original_text": seg.original_text,
                    "is_translated": seg.is_translated,
                }
                for seg in translated
            ],
        )
        meeting = await session.get(Meeting, ctx.meeting_id)
        if meeting is not None:
            meeting.language = transcript.language
            meeting.duration_seconds = round(duration_seconds, 2)
            meeting.segment_count = len(translated)
            meeting.speaker_count = speaker_count

    # ── 6. Summarize ─────────────────────────────────────────────────────────
    await ctx.stage("summarizing", 65)
    usage_before = registry.usage.snapshot()
    summary = await registry.summarize(translated)
    usage_after = registry.usage.snapshot()
    summary_usage = {key: usage_after[key] - usage_before.get(key, 0) for key in usage_after}

    # ── 7. Action items ──────────────────────────────────────────────────────
    await ctx.stage("extracting_action_items", 80)
    action_items = await registry.extract_action_items(translated)

    async with db.session() as session:
        await repositories.replace_action_items(session, ctx.meeting_id, ctx.user_id, action_items)
        meeting = await session.get(Meeting, ctx.meeting_id)
        if meeting is not None:
            meeting.summary_text = summary.summary
            meeting.summary_usage = summary_usage
            meeting.key_decisions = [
                {"decision": d.decision, "start_ms": d.start_ms} for d in summary.key_decisions
            ]

    # ── 8. Index for RAG ─────────────────────────────────────────────────────
    await ctx.stage("indexing", 90)
    chunk_count = await registry.rag.index_segments(
        str(ctx.meeting_id), str(ctx.user_id), translated
    )

    logger.info(
        "meeting_processed",
        meeting_id=str(ctx.meeting_id),
        language=transcript.language,
        segments=len(translated),
        speakers=speaker_count,
        action_items=len(action_items),
        chunks=chunk_count,
    )
