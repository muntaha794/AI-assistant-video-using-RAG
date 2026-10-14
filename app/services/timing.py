"""Timestamp helpers shared by prompts, citations and the UI."""

from __future__ import annotations

import re

_TIME_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{2})$")


def format_timestamp(ms: int | float) -> str:
    """Milliseconds → ``H:MM:SS`` (or ``M:SS`` under an hour)."""
    total_seconds = max(0, int(ms // 1000))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def parse_timecode(value: str | None) -> int | None:
    """``H:MM:SS`` / ``MM:SS`` → milliseconds; None when unparseable."""
    if not value:
        return None
    match = _TIME_RE.match(value.strip())
    if not match:
        return None
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    if minutes > 59 or seconds > 59:
        return None
    return ((hours * 60 + minutes) * 60 + seconds) * 1000


def nearest_segment_index(start_ms: int, segment_ranges: list[tuple[int, int]]) -> int | None:
    """Index of the segment containing ``start_ms``, else the closest following/preceding one.

    Ranges are treated as half-open ``[start, end)`` so a boundary timestamp
    belongs to the segment that *starts* there.
    """
    if not segment_ranges:
        return None
    for idx, (start, end) in enumerate(segment_ranges):
        if start <= start_ms < end:
            return idx
    # Not contained: pick the segment with the smallest distance to its start.
    return min(
        range(len(segment_ranges)),
        key=lambda idx: abs(segment_ranges[idx][0] - start_ms),
    )
