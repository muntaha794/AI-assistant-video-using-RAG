"""Upload handling: size limits, type validation, safe paths, cleanup.

The audio file is streamed to disk in chunks with a hard byte cap (never
loads the whole file in memory), validated by extension *and* magic bytes,
and stored in a per-meeting directory that is deleted with the meeting.
"""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import BinaryIO, Protocol

from app.config import Settings
from app.errors import PayloadTooLargeError, UnsupportedMediaError
from app.logging_config import get_logger

logger = get_logger(__name__)

# (offset, bytes) signatures that identify real audio/video containers.
_MAGIC_SIGNATURES: list[tuple[int, bytes]] = [
    (0, b"ID3"),  # mp3
    (0, b"OggS"),  # ogg / opus
    (0, b"fLaC"),  # flac
    (0, b"RIFF"),  # wav / avi (wav checked via WAVE below)
    (8, b"WAVE"),  # wav
    (4, b"ftyp"),  # mp4 / m4a
    (0, b"\x1a\x45\xdf\xa3"),  # webm / mkv
    (0, b"\xff\xfb"),  # mp3 frame sync
    (0, b"\xff\xf3"),  # mp3 frame sync
    (0, b"\xff\xf2"),  # mp3 frame sync
]


@dataclass(frozen=True)
class StoredFile:
    path: Path
    size_bytes: int
    content_type: str | None


class UploadStream(Protocol):
    """Minimal file-like protocol satisfied by Starlette's UploadFile .file."""

    def read(self, size: int = -1) -> bytes: ...

    def seek(self, offset: int, whence: int = 0) -> int: ...


class AudioStorage:
    """Manages the on-disk layout under ``settings.upload_dir``."""

    def __init__(self, settings: Settings) -> None:
        self.root: Path = settings.upload_dir.resolve()
        self.max_bytes = settings.max_upload_bytes
        self.chunk_bytes = settings.upload_chunk_bytes
        self.allowed_extensions = {ext.lower() for ext in settings.allowed_audio_extensions}
        self.root.mkdir(parents=True, exist_ok=True)

    # ── Paths ────────────────────────────────────────────────────────────────

    def meeting_dir(self, meeting_id: uuid.UUID) -> Path:
        """Per-meeting directory; the uuid format itself blocks traversal."""
        safe_id = uuid.UUID(str(meeting_id))
        path = (self.root / str(safe_id)).resolve()
        if not path.is_relative_to(self.root):  # defence in depth
            raise UnsupportedMediaError("Invalid storage path")
        return path

    # ── Validation ───────────────────────────────────────────────────────────

    def validate_extension(self, filename: str | None) -> str:
        ext = Path(filename or "").suffix.lower()
        if not ext or ext not in self.allowed_extensions:
            raise UnsupportedMediaError(
                f"Unsupported file type '{ext or 'unknown'}'. "
                f"Allowed: {', '.join(sorted(self.allowed_extensions))}",
                details={"allowed_extensions": sorted(self.allowed_extensions)},
            )
        return ext

    @staticmethod
    def _looks_like_media(head: bytes) -> bool:
        return any(head[offset : offset + len(sig)] == sig for offset, sig in _MAGIC_SIGNATURES)

    # ── Write ────────────────────────────────────────────────────────────────

    async def save_upload(
        self,
        meeting_id: uuid.UUID,
        filename: str,
        stream: UploadStream,
        *,
        content_type: str | None = None,
    ) -> StoredFile:
        """Stream-validate and persist an uploaded audio file."""
        ext = self.validate_extension(filename)

        head = stream.read(64)
        if not head or not self._looks_like_media(head):
            raise UnsupportedMediaError(
                "File content does not look like an audio/video container "
                "(magic-byte check failed)."
            )
        stream.seek(0)

        target_dir = self.meeting_dir(meeting_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        # The user-supplied name is never trusted for the on-disk path.
        target = target_dir / f"source{ext}"

        total = 0
        try:
            with target.open("wb") as out:
                while True:
                    chunk = stream.read(self.chunk_bytes)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > self.max_bytes:
                        raise PayloadTooLargeError(
                            f"File exceeds the {self.max_bytes // (1024 * 1024)} MB limit",
                            details={"max_bytes": self.max_bytes},
                        )
                    out.write(chunk)
        except BaseException:
            self.delete_meeting_dir(meeting_id)
            raise

        if total == 0:
            self.delete_meeting_dir(meeting_id)
            raise UnsupportedMediaError("Uploaded file is empty")

        return StoredFile(path=target, size_bytes=total, content_type=content_type)

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def delete_meeting_dir(self, meeting_id: uuid.UUID) -> None:
        path = self.meeting_dir(meeting_id)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)

    def sweep_orphans(self, *, ttl_hours: int) -> int:
        """Remove per-meeting dirs older than the TTL (crash-safe cleanup)."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=ttl_hours)
        removed = 0
        for child in self.root.iterdir():
            if not child.is_dir():
                continue
            try:
                uuid.UUID(child.name)
            except ValueError:
                continue
            mtime = datetime.fromtimestamp(child.stat().st_mtime, tz=timezone.utc)
            if mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
        if removed:
            logger.info("swept_orphan_upload_dirs", count=removed)
        return removed


def resolve_audio_path(storage: AudioStorage, meeting_id: uuid.UUID, relative: str) -> Path:
    """Re-resolve a stored relative path under the meeting dir (no traversal)."""
    base = storage.meeting_dir(meeting_id)
    candidate = (base / relative).resolve()
    if not candidate.is_relative_to(base):
        raise UnsupportedMediaError("Invalid audio path")
    return candidate


def stream_from_bytes(data: BinaryIO) -> UploadStream:
    """Adapter so tests can reuse ``save_upload`` with plain file objects."""
    return data  # protocol-compatible
