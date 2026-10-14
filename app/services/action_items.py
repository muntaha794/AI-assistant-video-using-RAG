"""Action-item extraction, normalization and tracker rules.

LLM output is untrusted input: this module normalizes owners, parses due
dates conservatively (no invented deadlines), clamps priorities, and anchors
every item to the transcript moment that requested it (timestamp citation).
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

from app.logging_config import get_logger
from app.services.llm import LLM, Message, extract_json
from app.services.summarizer import ACTION_ITEMS_SYSTEM_PROMPT, render_transcript
from app.services.timing import nearest_segment_index, parse_timecode
from app.services.translation import TranslatedSegment

logger = get_logger(__name__)

PRIORITIES = ("low", "medium", "high")
ITEM_STATUSES = ("todo", "in_progress", "done", "cancelled")

_NULL_OWNERS = {"", "none", "null", "n/a", "na", "tbd", "unknown", "unassigned", "-", "—", "?"}
_PRIORITY_ALIASES = {
    "low": "low",
    "minor": "low",
    "p3": "low",
    "medium": "medium",
    "normal": "medium",
    "med": "medium",
    "p2": "medium",
    "high": "high",
    "critical": "high",
    "urgent": "high",
    "p1": "high",
}

_DATE_FORMATS = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%b-%Y",
    "%d %B %Y",
    "%B %d, %Y",
    "%b %d, %Y",
)
_RELATIVE_DAYS = {"today": 0, "tomorrow": 1, "day after tomorrow": 2}


def normalize_owner(value: Any) -> str | None:
    """Clean an owner name; ``None`` for placeholder values."""
    if value is None:
        return None
    owner = re.sub(r"\s+", " ", str(value)).strip().strip("@–—").rstrip(".").strip("-–— ")
    if owner.lower() in _NULL_OWNERS:
        return None
    return owner[:200] or None


def normalize_priority(value: Any) -> str:
    if value is None:
        return "medium"
    return _PRIORITY_ALIASES.get(str(value).strip().lower(), "medium")


def parse_due_date(value: Any, *, today: date | None = None) -> date | None:
    """Conservative due-date parsing: ISO/common formats and simple relatives."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text in _NULL_OWNERS:
        return None
    today = today or datetime.now(timezone.utc).date()

    if text in _RELATIVE_DAYS:
        return today + timedelta(days=_RELATIVE_DAYS[text])
    week_match = re.match(
        r"(?:next\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", text
    )
    if week_match:
        weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        target = weekdays.index(week_match.group(1))
        delta = (target - today.weekday()) % 7
        if delta == 0 or text.startswith("next"):
            delta = delta or 7
        return today + timedelta(days=delta)

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _clean_text(value: Any, max_len: int) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text[:max_len] or None


def _attach_evidence(
    item: dict[str, Any], segments: list[TranslatedSegment], raw_time: Any
) -> dict[str, Any]:
    """Anchor an item to the transcript segment that requested it."""
    ms = parse_timecode(str(raw_time)) if raw_time else None
    if ms is None or not segments:
        return item
    ranges = [(seg.start_ms, seg.end_ms) for seg in segments]
    idx = nearest_segment_index(ms, ranges)
    if idx is None:
        return item
    segment = segments[idx]
    item["source_start_ms"] = segment.start_ms
    item["source_end_ms"] = segment.end_ms
    item["source_quote"] = segment.text.strip()[:1000]
    return item


def normalize_action_items(
    raw_items: list[Any], segments: list[TranslatedSegment], *, today: date | None = None
) -> list[dict[str, Any]]:
    """Turn untrusted LLM output into validated action-item payloads."""
    normalized: list[dict[str, Any]] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        title = _clean_text(raw.get("title"), 500)
        if not title:
            continue
        item: dict[str, Any] = {
            "title": title,
            "description": _clean_text(raw.get("description"), 4000),
            "owner": normalize_owner(raw.get("owner")),
            "due_date": parse_due_date(raw.get("due_date"), today=today),
            "priority": normalize_priority(raw.get("priority")),
            "source_start_ms": None,
            "source_end_ms": None,
            "source_quote": None,
        }
        normalized.append(_attach_evidence(item, segments, raw.get("time")))
    return normalized


async def extract_action_items(
    llm: LLM, segments: list[TranslatedSegment], *, max_items: int = 15
) -> list[dict[str, Any]]:
    """Extract and normalize action items from the transcript."""
    transcript = render_transcript(segments)
    messages: list[Message] = [
        {"role": "system", "content": ACTION_ITEMS_SYSTEM_PROMPT},
        {"role": "user", "content": f"Transcript:\n{transcript}"},
    ]
    result = await llm.complete(messages, json_mode=True)
    try:
        payload = extract_json(result.text)
    except ValueError:
        logger.warning("action_items_json_unparseable", snippet=result.text[:200])
        return []
    raw_items = payload.get("action_items") if isinstance(payload, dict) else payload
    if not isinstance(raw_items, list):
        raw_items = []
    return normalize_action_items(raw_items, segments)[:max_items]
