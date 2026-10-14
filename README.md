# AI Meeting Assistant

A production-style backend for meeting recordings: upload an audio file (or a
YouTube URL) and get back a timestamped transcript, an English translation, a
summary with key decisions, tracked action items, and a chat interface that
answers questions **only** from the transcript with `[S#]` citations mapped to
exact time ranges.

Built with **FastAPI + SQLAlchemy (async) + Alembic + PostgreSQL**, an
in-process job queue (`FOR UPDATE SKIP LOCKED` retries/reaping, no broker
required), a RAG engine with pluggable embeddings/vector stores, Mistral as
the LLM, Prometheus metrics, structured logs, and a full pytest suite that
runs with no network and no ML dependencies.

## Repository layout

| Path | Purpose |
| --- | --- |
| `app/` | FastAPI application: API routes, services (transcription, translation, summarizer, action items, RAG), job runner, config, metrics |
| `migrations/` | Alembic migrations |
| `tests/` | Pytest suite (79 tests, uses fakes — no network/ML needed) |
| `eval/`, `scripts/eval_run.py` | Retrieval + groundedness evaluation harness |
| `env.example` | Every environment variable the app reads, with defaults/comments |
| `Dockerfile`, `docker-compose.yml` | Production image and the `db` + `api` + `worker` stack |
| `.github/workflows/ci.yml` | CI: ruff check, ruff format, mypy, pytest matrix |
| `app.py`, `main.py`, `core/`, `utils/` | The original prototype, preserved unchanged (see below) |

## Quickstart

Requires Python ≥ 3.10 (3.11 recommended) and, for real runs, PostgreSQL —
tests default to SQLite and fakes.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-base.txt -r requirements-dev.txt

cp env.example .env          # at minimum set JWT_SECRET and MISTRAL_API_KEY
alembic upgrade head         # create tables
uvicorn app.main:app --reload --port 8000
```

Interactive API docs: <http://localhost:8000/docs> — health checks at
`/healthz`, `/readyz`, and Prometheus metrics at `/metrics`.

Run the background worker in a second shell so queued uploads get processed:

```bash
python -m app.worker
```

A **full install** (Whisper, torch, pyannote, ChromaDB, sentence-transformers)
is `pip install -r Requirements.txt`, which layers `requirements-ml.txt` on
top of the base. The API and test suite run without the ML extras thanks to
lightweight fallbacks (see *Optional heavy dependencies*).

## Configuration

All settings come from environment variables (`.env` is picked up
automatically) and are validated in `app/config.py`. See **`env.example`** for
the full annotated list. The ones you normally must set:

- `JWT_SECRET` — token signing key (required for auth)
- `MISTRAL_API_KEY` — powers summarization, action items, and chat
- `DATABASE_URL` — defaults to SQLite for local dev; Postgres in Docker/CI

Optional: `HF_TOKEN` (pyannote diarization), `WHISPER_MODEL`,
`EMBEDDING_BACKEND`, `RAG_TOP_K`, `JOB_WORKERS`, …

## API surface

All application routes live under `/api/v1`.

### Auth
| Method & path | Description |
| --- | --- |
| `POST /auth/register` | Create an account (201, password never echoed) |
| `POST /auth/login` | JSON login → bearer token |
| `POST /auth/token` | OAuth2 form login (for the Swagger Authorize button) |
| `GET /auth/me` | Current user |

### Meetings
| Method & path | Description |
| --- | --- |
| `POST /meetings/upload` | Multipart audio upload → **202** `{meeting_id, job_id}` |
| `POST /meetings/youtube` | Ingest a YouTube URL → **202** `{meeting_id, job_id}` |
| `GET /meetings` | List my meetings |
| `GET /meetings/{id}` | Meeting detail + status |
| `GET /meetings/{id}/transcript` | Timestamped, speaker-labelled transcript |
| `GET /meetings/{id}/summary` | Summary + key decisions with timestamps |
| `POST /meetings/{id}/reprocess` | Re-run the pipeline |
| `DELETE /meetings/{id}` | Delete meeting and all of its data (204) |

### Chat (RAG)
| Method & path | Description |
| --- | --- |
| `POST /meetings/{id}/chat` | Answer from the transcript only, with `[S#]` citations resolved to time ranges + a groundedness score |
| `GET /meetings/{id}/chat/history` | Prior turns for the meeting |

### Action items
| Method & path | Description |
| --- | --- |
| `GET /action-items` | Cross-meeting tracker for the current user |
| `GET /meetings/{id}/action-items` | Items extracted from one meeting |
| `PATCH /action-items/{item_id}` | Update status/owner/due date |

### Jobs & ops
| Method & path | Description |
| --- | --- |
| `GET /jobs/{job_id}` | Queue status: `queued` / `processing` / `succeeded` / `failed` |
| `GET /healthz`, `GET /readyz`, `GET /metrics` | Liveness, readiness, Prometheus |

**Async flow:** upload → 202 with a `job_id` → an in-process worker claims the
row (`SKIP LOCKED`, so API and worker replicas scale horizontally) → poll
`GET /jobs/{job_id}` until `succeeded`, then fetch summary/transcript/chat.
Failures retry with exponential backoff; jobs abandoned by a crashed worker
are requeued by the reaper.

## Testing & CI

```bash
pytest -q                      # 79 tests, offline, SQLite + fakes
ruff check app tests scripts
ruff format --check app tests scripts
mypy                           # type-checks the app/ package
```

CI (`.github/workflows/ci.yml`) runs exactly these gates plus a Python-version
matrix for the test suite. Coverage: `pytest --cov=app`.

## Evaluation harness

A fixture corpus (`eval/dataset.jsonl`) of supported and hallucinated
questions scores the retrieval and grounding pipeline:

```bash
python scripts/eval_run.py --dataset eval/dataset.jsonl
```

Reports hit rate, MRR, grounded pass rate, and hallucination-flag rate, and
exits non-zero if any threshold (`--min-hit-rate`, `--min-mrr`,
`--min-grounded-pass-rate`) is missed — usable as a regression gate.

## Docker

```bash
docker compose up --build
```

Starts PostgreSQL (`db`), the API (`api`, runs `alembic upgrade head` then
serves on 8000), and the standalone worker (`worker` — `python -m
app.worker`). The image installs the full dependency set and includes ffmpeg
for audio decoding.

## Optional heavy dependencies

`requirements-ml.txt` adds Whisper/torch for local transcription, pyannote for
diarization (needs `HF_TOKEN`), ChromaDB for persistent vectors,
sentence-transformers for real embeddings, and the Mistral SDK. Without them
the app falls back to the stub transcriber, in-memory vector store, and
hashing embeddings — which is also what the test-suite and eval harness use.

## Original prototype

`app.py` (Streamlit demo UI), `main.py`, `core/`, and `utils/` are the
original LangChain/Whisper prototype this API grew out of. They are kept
exactly as they were committed, for reference; the maintained application is
`app/`.
