"""Meeting summarization and key-decision extraction.

The transcript is rendered with ``[M:SS] SPEAKER: text`` markers so the model
can anchor decisions to moments in the meeting; extracted decision timestamps
are resolved back to milliseconds for the API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.logging_config import get_logger
from app.services.llm import LLM, Message, extract_json
from app.services.timing import format_timestamp, nearest_segment_index, parse_timecode
from app.services.translation import TranslatedSegment

logger = get_logger(__name__)

SUMMARY_SYSTEM_PROMPT = """\
You are a precise meeting analyst. You will receive a meeting transcript where \
each line is formatted as `[M:SS] SPEAKER: utterance`.

Respond ONLY with JSON matching this schema:
{
  "summary": "A concise 2-4 paragraph summary of the meeting: context, discussion, outcomes.",
  "key_decisions": [
    {"decision": "What was decided", "time": "M:SS or H:MM:SS or null"}
  ]
}
Rules:
- key_decisions must contain the 3-8 most important decisions; use the transcript's \
nearest timestamp when you can, else null.
- Never invent facts that are not supported by the transcript.
"""

ACTION_ITEMS_SYSTEM_PROMPT = """\
You are a precise meeting analyst extracting action items from a transcript where \
each line is formatted as `[M:SS] SPEAKER: utterance`.

Respond ONLY with JSON matching this schema:
{
  "action_items": [
    {
      "title": "Short imperative task (max 120 chars)",
      "description": "1-2 sentences of detail, optional",
      "owner": "Person name if clearly stated, else null",
      "due_date": "YYYY-MM-DD if a deadline was stated or can be safely inferred from \
context like 'by Friday', else null",
      "priority": "low" | "medium" | "high",
      "time": "M:SS or H:MM:SS of the utterance that requested it, or null"
    }
  ]
}
Rules:
- Only include genuine commitments/tasks; do not list discussion points.
- At most 15 action items; prefer the clearly-owned ones.
- Use exactly the speaker names from the transcript for owners.
- Never invent names or dates that are not supported by the transcript.
"""


@dataclass
class KeyDecision:
    decision: str
    start_ms: int | None = None


@dataclass
class SummaryResult:
    summary: str
    key_decisions: list[KeyDecision] = field(default_factory=list)


def render_transcript(segments: list[TranslatedSegment]) -> str:
    """Render segments with time and speaker markers for the prompt."""
    lines = []
    for seg in segments:
        timecode = format_timestamp(seg.start_ms)
        # Keep the prompt compact but complete.
        lines.append(f"[{timecode}] {seg.speaker}: {seg.text.strip()}")
    return "\n".join(lines)


def _resolve_time(value: Any, segments: list[TranslatedSegment]) -> int | None:
    ms = parse_timecode(str(value)) if value else None
    if ms is None:
        return None
    ranges = [(seg.start_ms, seg.end_ms) for seg in segments]
    idx = nearest_segment_index(ms, ranges)
    return segments[idx].start_ms if idx is not None else ms


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


async def summarize_meeting(
    llm: LLM, segments: list[TranslatedSegment], *, max_transcript_chars: int = 60_000
) -> SummaryResult:
    """Summarize the meeting and extract timestamped key decisions."""
    transcript = render_transcript(segments)
    if len(transcript) > max_transcript_chars:
        half = max_transcript_chars // 2
        transcript = (
            transcript[:half] + "\n...[transcript elided for length]...\n" + transcript[-half:]
        )

    messages: list[Message] = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": f"Transcript:\n{transcript}"},
    ]
    result = await llm.complete(messages, json_mode=True)
    try:
        payload = extract_json(result.text)
    except ValueError:
        logger.warning("summary_json_unparseable", snippet=result.text[:200])
        return SummaryResult(summary=result.text.strip() or "Summary unavailable.")

    if isinstance(payload, list):  # tolerate a bare decisions array
        payload = {"summary": "", "key_decisions": payload}
    summary_text = str(payload.get("summary", "") or "").strip()
    decisions: list[KeyDecision] = []
    for raw in _as_list(payload.get("key_decisions")):
        if isinstance(raw, str):
            raw = {"decision": raw}
        if not isinstance(raw, dict):
            continue
        decision = str(raw.get("decision", "")).strip()
        if not decision:
            continue
        decisions.append(
            KeyDecision(decision=decision, start_ms=_resolve_time(raw.get("time"), segments))
        )
    return SummaryResult(summary=summary_text, key_decisions=decisions)
