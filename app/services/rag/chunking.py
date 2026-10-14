"""Transcript chunking that preserves timestamps for citations.

Each chunk is a window of consecutive transcript segments rendered as
``[M:SS] SPEAKER: text`` lines. The chunk remembers the millisecond range and
the speakers inside it, which is what makes timestamp citations possible:
retrieval hits map 1:1 back to moments in the recording.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.services.timing import format_timestamp
from app.services.translation import TranslatedSegment


@dataclass
class Chunk:
    chunk_id: str  # per-meeting id, e.g. "c0003"
    text: str
    start_ms: int
    end_ms: int
    speakers: list[str] = field(default_factory=list)
    first_seq: int = 0
    last_seq: int = 0

    @property
    def start_label(self) -> str:
        return format_timestamp(self.start_ms)

    @property
    def end_label(self) -> str:
        return format_timestamp(self.end_ms)


def render_segment(seg: TranslatedSegment) -> str:
    return f"[{format_timestamp(seg.start_ms)}] {seg.speaker}: {seg.text.strip()}"


def chunk_segments(
    segments: list[TranslatedSegment],
    *,
    max_chars: int = 1200,
    overlap_chars: int = 200,
) -> list[Chunk]:
    """Window segments into overlapping chunks bounded by ``max_chars``."""
    if not segments:
        return []

    chunks: list[Chunk] = []
    window: list[TranslatedSegment] = []
    window_len = 0

    def flush() -> None:
        if not window:
            return
        speakers: list[str] = []
        for seg in window:
            if seg.speaker not in speakers:
                speakers.append(seg.speaker)
        text = "\n".join(render_segment(seg) for seg in window)
        chunks.append(
            Chunk(
                chunk_id=f"c{len(chunks):04d}",
                text=text,
                start_ms=window[0].start_ms,
                end_ms=window[-1].end_ms,
                speakers=speakers,
                first_seq=window[0].seq,
                last_seq=window[-1].seq,
            )
        )

    for seg in segments:
        seg_len = len(render_segment(seg)) + 1
        if window and window_len + seg_len > max_chars:
            flush()
            # Carry an overlap of the trailing segments into the next window.
            overlap: list[TranslatedSegment] = []
            overlap_len = 0
            for prev in reversed(window):
                prev_len = len(render_segment(prev)) + 1
                if overlap_len + prev_len > overlap_chars:
                    break
                overlap.insert(0, prev)
                overlap_len += prev_len
            window = overlap
            window_len = overlap_len
        window.append(seg)
        window_len += seg_len

    flush()
    return chunks
