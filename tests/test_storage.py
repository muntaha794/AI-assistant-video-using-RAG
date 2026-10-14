"""File-handling tests: size limits, magic-byte validation, cleanup."""

from __future__ import annotations

import io
import uuid
from pathlib import Path

import pytest

from app.config import Settings
from app.errors import PayloadTooLargeError, UnsupportedMediaError
from app.services.storage import AudioStorage
from tests.fakes import FAKE_MP3_BYTES


def _storage(tmp_path: Path, **overrides: object) -> AudioStorage:
    settings = Settings(_env_file=None, upload_dir=tmp_path / "uploads", **overrides)  # type: ignore[arg-type]
    return AudioStorage(settings)


async def test_valid_upload_is_streamed_to_disk(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    meeting_id = uuid.uuid4()
    stored = await storage.save_upload(
        meeting_id, "kickoff.mp3", io.BytesIO(FAKE_MP3_BYTES), content_type="audio/mpeg"
    )
    assert stored.path.exists()
    assert stored.path.name == "source.mp3"  # user filename never trusted for paths
    assert stored.size_bytes == len(FAKE_MP3_BYTES)
    assert stored.path.parent.name == str(meeting_id)


@pytest.mark.parametrize("filename", ["notes.txt", "image.png", "audio", "archive.zip"])
async def test_disallowed_extensions_rejected(tmp_path: Path, filename: str) -> None:
    storage = _storage(tmp_path)
    with pytest.raises(UnsupportedMediaError):
        await storage.save_upload(uuid.uuid4(), filename, io.BytesIO(FAKE_MP3_BYTES))


async def test_magic_byte_mismatch_rejected(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    text_bytes = b"definitely not audio, just some plain text" * 10
    with pytest.raises(UnsupportedMediaError):
        await storage.save_upload(uuid.uuid4(), "fake.mp3", io.BytesIO(text_bytes))
    # Nothing left behind on disk.
    assert not any((tmp_path / "uploads").iterdir())


async def test_oversized_upload_rejected_and_cleaned(tmp_path: Path) -> None:
    storage = _storage(tmp_path, max_upload_bytes=1024)
    big = b"ID3" + b"\x00" * 4096
    with pytest.raises(PayloadTooLargeError):
        await storage.save_upload(uuid.uuid4(), "huge.mp3", io.BytesIO(big))
    assert not any((tmp_path / "uploads").iterdir())


async def test_empty_upload_rejected(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    with pytest.raises(UnsupportedMediaError):
        await storage.save_upload(uuid.uuid4(), "empty.mp3", io.BytesIO(b""))


async def test_delete_meeting_dir(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    meeting_id = uuid.uuid4()
    await storage.save_upload(meeting_id, "a.mp3", io.BytesIO(FAKE_MP3_BYTES))
    storage.delete_meeting_dir(meeting_id)
    assert not storage.meeting_dir(meeting_id).exists()


async def test_orphan_sweep_removes_old_dirs(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    old_id = uuid.uuid4()
    await storage.save_upload(old_id, "a.mp3", io.BytesIO(FAKE_MP3_BYTES))
    # Age the directory beyond the TTL.
    import os
    import time

    old = time.time() - 48 * 3600
    os.utime(storage.meeting_dir(old_id), (old, old))
    removed = storage.sweep_orphans(ttl_hours=24)
    assert removed == 1
    assert not storage.meeting_dir(old_id).exists()
