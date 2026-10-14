"""End-to-end: upload → async pipeline → transcript, summary, actions, chat."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import AsyncClient

from app.services.transcription import TranscriptResult, TranscriptSegmentData
from tests.conftest import register_user, upload_meeting
from tests.fakes import FAKE_MP3_BYTES, FakeTranscriber


async def _process(client: AsyncClient, app_obj, headers: dict[str, str]) -> dict:  # type: ignore[no-untyped-def]
    accepted = await upload_meeting(client, headers)
    await app_obj.state.runner.drain()
    response = await client.get(f"/api/v1/meetings/{accepted['meeting_id']}", headers=headers)
    assert response.status_code == 200
    return {"accepted": accepted, "meeting": response.json()}


async def test_full_pipeline_english_meeting(client: AsyncClient, app_obj, fake_llm) -> None:  # type: ignore[no-untyped-def]
    headers = await register_user(client)
    result = await _process(client, app_obj, headers)
    accepted, meeting = result["accepted"], result["meeting"]

    # Meeting completed with transcript facts + summary
    assert meeting["status"] == "completed"
    assert meeting["segment_count"] == 4
    assert meeting["speaker_count"] == 2
    assert meeting["summary"].startswith("The team planned")
    assert len(meeting["key_decisions"]) == 2
    assert meeting["key_decisions"][0]["start_label"] == "0:16"
    assert meeting["action_item_count"] == 3

    # Job row reports success, stage, progress and token usage
    job = await client.get(accepted["status_url"], headers=headers)
    assert job.status_code == 200
    job_body = job.json()
    assert job_body["status"] == "succeeded"
    assert job_body["progress"] == 100
    assert job_body["duration_ms"] is not None
    assert job_body["token_usage"]["calls"] >= 2  # summary + action items

    # Timestamped, speaker-attributed transcript
    transcript = await client.get(
        f"/api/v1/meetings/{accepted['meeting_id']}/transcript", headers=headers
    )
    assert transcript.status_code == 200
    body = transcript.json()
    assert body["speakers"] == ["SPEAKER_00", "SPEAKER_01"]
    first = body["segments"][0]
    assert first["start_label"] == "0:00" and first["end_label"] == "0:08"
    assert first["speaker"] == "SPEAKER_00"
    # Segments 3+ (>= 12s) belong to the second speaker.
    assert body["segments"][3]["speaker"] == "SPEAKER_01"

    # Summary endpoint with usage stats
    summary = await client.get(
        f"/api/v1/meetings/{accepted['meeting_id']}/summary", headers=headers
    )
    assert summary.status_code == 200
    assert summary.json()["usage"]["prompt_tokens"] > 0

    # Action items: owners, deadlines, priorities + evidence anchors.
    # Ordering: due-date ascending (Friday < 2026-11-01), undated items last.
    items = await client.get(
        f"/api/v1/meetings/{accepted['meeting_id']}/action-items", headers=headers
    )
    assert items.status_code == 200
    item_list = items.json()["items"]
    by_title = {item["title"]: item for item in item_list}
    assert set(by_title) == {
        "Ship the beta launch",
        "Send the vendor contract to Carol",
        "Update the roadmap",
    }
    alice_item = by_title["Ship the beta launch"]
    assert alice_item["owner"] == "Alice"
    assert alice_item["due_date"] == "2026-11-01"
    assert alice_item["priority"] == "high"
    assert alice_item["source_start_label"] == "0:08"
    assert "ship it by 2026-11-01" in alice_item["source_quote"].lower()
    assert by_title["Send the vendor contract to Carol"]["owner"] == "Bob"
    assert by_title["Update the roadmap"]["owner"] is None  # unassigned stays unassigned

    # Tracker across meetings + filters
    tracker = await client.get("/api/v1/action-items?status=todo", headers=headers)
    assert tracker.json()["total"] == 3
    alice = await client.get("/api/v1/action-items?owner=Alice", headers=headers)
    assert alice.json()["total"] == 1

    # Ownership/deadline/status updates
    item_id = alice_item["id"]
    patch = await client.patch(
        f"/api/v1/action-items/{item_id}",
        headers=headers,
        json={"status": "done", "owner": "Alice Cooper"},
    )
    assert patch.status_code == 200
    assert patch.json()["status"] == "done"
    assert patch.json()["completed_at"] is not None
    assert patch.json()["owner"] == "Alice Cooper"

    # Chat with evidence
    chat = await client.post(
        f"/api/v1/meetings/{accepted['meeting_id']}/chat",
        headers=headers,
        json={"question": "What did we decide about the reporting dashboard and mobile support?"},
    )
    assert chat.status_code == 200
    chat_body = chat.json()
    assert "mobile support" in chat_body["answer"]
    assert chat_body["retrieved_chunks"] >= 1
    assert chat_body["is_grounded"] is True
    citation = chat_body["citations"][0]
    # The short standup transcript is a single chunk spanning the whole meeting.
    assert citation["start_label"] == "0:00"
    assert citation["speaker"] in {"SPEAKER_00", "SPEAKER_01"}

    # Chat history persisted (user + assistant turns)
    history = await client.get(
        f"/api/v1/meetings/{accepted['meeting_id']}/chat/history", headers=headers
    )
    assert [m["role"] for m in history.json()["messages"]] == ["user", "assistant"]

    # LLM was never called with real credentials
    assert fake_llm.calls  # the fake handled every LLM interaction


async def test_hindi_meeting_is_translated_with_timestamps(client: AsyncClient, app_obj) -> None:  # type: ignore[no-untyped-def]
    registry = app_obj.state.registry
    registry.transcriber = FakeTranscriber(
        result=TranscriptResult(
            language="hi",
            duration_seconds=10.0,
            segments=[
                TranscriptSegmentData(0, 5000, "क्या हम लॉन्च की तारीख तय कर सकते हैं"),
                TranscriptSegmentData(5000, 10_000, "हाँ, अगले महीने शुरू करते हैं"),
            ],
        )
    )
    headers = await register_user(client, "hindi@example.com")
    result = await _process(client, app_obj, headers)
    meeting_id = result["accepted"]["meeting_id"]

    transcript = await client.get(f"/api/v1/meetings/{meeting_id}/transcript", headers=headers)
    segments = transcript.json()["segments"]
    assert segments[0]["is_translated"] is True
    assert segments[0]["text"].startswith("[EN]")
    assert "लॉन्च" in segments[0]["text"]  # original preserved alongside
    assert [s["start_ms"] for s in segments] == [0, 5000]


async def test_user_isolation_everywhere(client: AsyncClient, app_obj) -> None:  # type: ignore[no-untyped-def]
    alice = await register_user(client, "alice@example.com")
    bob = await register_user(client, "bob@example.com", "password12345")
    result = await _process(client, app_obj, alice)
    meeting_id = result["accepted"]["meeting_id"]

    # Bob sees an empty list, 404s on direct access, and cannot chat.
    bob_list = await client.get("/api/v1/meetings", headers=bob)
    assert bob_list.json()["total"] == 0
    assert (await client.get(f"/api/v1/meetings/{meeting_id}", headers=bob)).status_code == 404
    assert (
        await client.get(f"/api/v1/meetings/{meeting_id}/transcript", headers=bob)
    ).status_code == 404
    assert (
        await client.post(
            f"/api/v1/meetings/{meeting_id}/chat", headers=bob, json={"question": "hi?"}
        )
    ).status_code == 404
    # Bob cannot see Alice's job or action items.
    job_id = result["accepted"]["job_id"]
    assert (await client.get(f"/api/v1/jobs/{job_id}", headers=bob)).status_code == 404
    bob_items = await client.get("/api/v1/action-items", headers=bob)
    assert bob_items.json()["total"] == 0


async def test_upload_rejects_non_audio_content(client: AsyncClient) -> None:
    headers = await register_user(client)
    response = await client.post(
        "/api/v1/meetings/upload",
        headers=headers,
        files={"file": ("notes.mp3", b"just text, definitely not audio", "audio/mpeg")},
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/watch?v=dQw4w9WgXcQ",
        "ftp://youtube.com/watch?v=dQw4w9WgXcQ",
        "not-a-url",
    ],
)
async def test_youtube_invalid_urls_rejected(client: AsyncClient, url: str) -> None:
    headers = await register_user(client)
    response = await client.post("/api/v1/meetings/youtube", headers=headers, json={"url": url})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


async def test_youtube_submission_is_queued(client: AsyncClient) -> None:
    headers = await register_user(client)
    response = await client.post(
        "/api/v1/meetings/youtube",
        headers=headers,
        json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "title": "Retro"},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["status_url"].endswith(body["job_id"])


async def test_youtube_pipeline_flow_with_download_stub(
    client: AsyncClient, app_obj, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def fake_download(url: str, dest_dir: Path, **kwargs: object) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / "source.webm"
        path.write_bytes(b"\x1a\x45\xdf\xa3" + b"\x00" * 64)
        return path

    monkeypatch.setattr("app.pipeline.download_youtube_audio", fake_download)
    headers = await register_user(client, "yt@example.com")
    response = await client.post(
        "/api/v1/meetings/youtube",
        headers=headers,
        json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
    )
    assert response.status_code == 202
    await app_obj.state.runner.drain()
    meeting = await client.get(f"/api/v1/meetings/{response.json()['meeting_id']}", headers=headers)
    assert meeting.json()["status"] == "completed"


async def test_delete_meeting_removes_data_and_vectors(client: AsyncClient, app_obj) -> None:  # type: ignore[no-untyped-def]
    headers = await register_user(client)
    result = await _process(client, app_obj, headers)
    meeting_id = result["accepted"]["meeting_id"]

    delete = await client.delete(f"/api/v1/meetings/{meeting_id}", headers=headers)
    assert delete.status_code == 204
    assert (await client.get(f"/api/v1/meetings/{meeting_id}", headers=headers)).status_code == 404
    stored = app_obj.state.registry.vector_store
    hits = await stored.query("meeting_chunks", [0.0] * 384, k=10, where={"meeting_id": meeting_id})
    assert hits == []


async def test_reprocess_meeting(client: AsyncClient, app_obj) -> None:  # type: ignore[no-untyped-def]
    headers = await register_user(client)
    result = await _process(client, app_obj, headers)
    meeting_id = result["accepted"]["meeting_id"]

    response = await client.post(f"/api/v1/meetings/{meeting_id}/reprocess", headers=headers)
    assert response.status_code == 202
    await app_obj.state.runner.drain()
    meeting = await client.get(f"/api/v1/meetings/{meeting_id}", headers=headers)
    assert meeting.json()["status"] == "completed"
    assert meeting.json()["segment_count"] == 4


async def test_chat_before_processing_conflicts(client: AsyncClient) -> None:
    headers = await register_user(client)
    accepted = await upload_meeting(client, headers)
    response = await client.post(
        f"/api/v1/meetings/{accepted['meeting_id']}/chat",
        headers=headers,
        json={"question": "What was decided?"},
    )
    assert response.status_code == 409


async def test_metrics_and_health_endpoints(client: AsyncClient) -> None:
    assert (await client.get("/healthz")).status_code == 200
    assert (await client.get("/readyz")).status_code == 200
    metrics = await client.get("/metrics")
    assert metrics.status_code == 200
    assert "http_requests_total" in metrics.text


async def test_openapi_document_exposes_endpoints(client: AsyncClient) -> None:
    response = await client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/api/v1/meetings/upload" in paths
    assert "/api/v1/meetings/{meeting_id}/chat" in paths
    assert "/api/v1/action-items" in paths
    assert "/api/v1/jobs/{job_id}" in paths


async def test_upload_requires_auth(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/meetings/upload", files={"file": ("a.mp3", FAKE_MP3_BYTES, "audio/mpeg")}
    )
    assert response.status_code == 401
