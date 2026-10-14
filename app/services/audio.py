"""Audio acquisition: YouTube downloads and format normalization.

``yt-dlp`` and ``pydub``/ffmpeg are imported lazily so the API (and the
test-suite) runs without the heavy media stack installed.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from app.errors import ExternalServiceError, TransientProcessingError
from app.logging_config import get_logger

logger = get_logger(__name__)

_YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "music.youtube.com",
}
_YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def validate_youtube_url(url: str) -> str:
    """Validate that ``url`` is a canonical YouTube watch/shorts/embed URL."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("YouTube URL must use http(s)")
    host = (parsed.hostname or "").lower()
    if host not in _YOUTUBE_HOSTS:
        raise ValueError("URL host must be youtube.com or youtu.be")

    video_id: str | None = None
    if host == "youtu.be":
        video_id = parsed.path.lstrip("/").split("/")[0]
    elif parsed.path == "/watch":
        video_id = parse_qs(parsed.query).get("v", [None])[0]
    elif parsed.path.startswith(("/shorts/", "/embed/", "/live/")):
        video_id = parsed.path.split("/")[2] if len(parsed.path.split("/")) > 2 else None

    if not video_id or not _YOUTUBE_ID_RE.match(video_id):
        raise ValueError("URL does not contain a valid YouTube video id")
    return url.strip()


async def download_youtube_audio(url: str, dest_dir: Path, *, timeout: float = 900.0) -> Path:
    """Download the bestaudio track for a YouTube URL into ``dest_dir``."""
    validate_youtube_url(url)
    dest_dir.mkdir(parents=True, exist_ok=True)

    def _download() -> Path:
        try:
            import yt_dlp  # noqa: PLC0415 - lazy optional dependency
        except ImportError as exc:  # pragma: no cover - needs requirements-ml
            raise ExternalServiceError(
                "yt-dlp is not installed (pip install -r requirements-ml.txt)"
            ) from exc

        options = {
            "format": "bestaudio/best",
            "outtmpl": str(dest_dir / "source.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "retries": 3,
            "socket_timeout": 30,
            # Keep the pipeline deterministic: no post-processing beyond audio.
            "postprocessors": [],
        }
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([url])
        except yt_dlp.utils.DownloadError as exc:  # type: ignore[attr-defined]
            message = str(exc)
            if "unavailable" in message.lower() or "private" in message.lower():
                raise ExternalServiceError(f"YouTube video unavailable: {message[:300]}") from exc
            raise TransientProcessingError(f"YouTube download failed: {message[:300]}") from exc

        candidates = sorted(dest_dir.glob("source.*"))
        if not candidates:
            raise ExternalServiceError("yt-dlp finished but produced no audio file")
        return candidates[0]

    try:
        return await asyncio.wait_for(asyncio.to_thread(_download), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise TransientProcessingError(f"YouTube download timed out after {timeout:.0f}s") from exc


def normalize_to_wav16k_mono(source: Path, *, sample_rate: int = 16000) -> Path:
    """Convert any input to the 16 kHz mono WAV Whisper expects.

    Returns the original path when ffmpeg/pydub are unavailable and the file
    is already a wav (Whisper can resample itself).
    """
    if source.suffix.lower() == ".wav":
        return source
    try:
        from pydub import AudioSegment  # noqa: PLC0415 - lazy optional dependency
    except ImportError as exc:  # pragma: no cover - needs requirements-ml
        raise ExternalServiceError(
            "pydub/ffmpeg are required to convert this audio format "
            "(pip install -r requirements-ml.txt)"
        ) from exc

    audio = AudioSegment.from_file(str(source))
    audio = audio.set_frame_rate(sample_rate).set_channels(1).set_sample_width(2)
    target = source.with_suffix(".wav")
    audio.export(str(target), format="wav")
    return target


def audio_duration_seconds(path: Path) -> float | None:
    """Best-effort duration probe (None when the media stack is absent)."""
    try:
        from pydub import AudioSegment  # noqa: PLC0415 - lazy optional dependency

        return len(AudioSegment.from_file(str(path))) / 1000.0
    except Exception:  # noqa: BLE001 - duration is informational only
        return None
