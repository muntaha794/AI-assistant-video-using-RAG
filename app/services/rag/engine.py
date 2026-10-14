"""Evidence-based RAG engine.

Flow: transcript segments → timestamped chunks → embeddings → vector store.
At question time: retrieve top-k chunks (scoped to one meeting *and* one
user), have Mistral answer **only** from those blocks with ``[S#]`` markers,
map the cited markers back to millisecond ranges (timestamp citations), and
score groundedness — the fraction of answer sentences supported by the
retrieved context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.config import Settings
from app.logging_config import get_logger
from app.metrics import RAG_GROUNDEDNESS, RAG_QUERIES, RETRIEVAL_HITS
from app.services.llm import LLM, Message
from app.services.rag.chunking import Chunk, chunk_segments
from app.services.rag.embeddings import Embeddings, tokenize
from app.services.rag.store import VectorItem, VectorStore
from app.services.timing import format_timestamp
from app.services.translation import TranslatedSegment

logger = get_logger(__name__)

COLLECTION = "meeting_chunks"

ANSWER_SYSTEM_PROMPT = """\
You answer questions about ONE meeting transcript. You will receive numbered \
context blocks, each introduced by an id like [S1] with its time range and speakers.

Rules:
- Answer ONLY using facts from the context blocks. Never use outside knowledge.
- After each factual claim, cite the supporting block id(s) inline like [S1] or [S2][S5].
- If the context does not contain the answer, say so explicitly and do not cite anything.
- Quote exact phrases when it helps; keep the answer concise and in the user's language.
"""

_CITATION_RE = re.compile(r"\[S(\d+)\]")
_REFUSAL_RE = re.compile(
    r"(does not (?:contain|mention|include|state)|doesn't (?:contain|mention|include|state)|"
    r"do not (?:contain|mention|include)|don't (?:contain|mention)|"
    r"cannot answer|can't answer|"
    r"not (?:mentioned|stated|included|available)|no (?:information|record))",
    re.IGNORECASE,
)


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    start_ms: int
    end_ms: int
    speakers: list[str] = field(default_factory=list)

    @property
    def start_label(self) -> str:
        return format_timestamp(self.start_ms)

    @property
    def end_label(self) -> str:
        return format_timestamp(self.end_ms)


@dataclass
class Citation:
    """Evidence for a claim: the transcript window it came from."""

    chunk_id: str
    start_ms: int
    end_ms: int
    start_label: str
    end_label: str
    speaker: str
    quote: str

    def as_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "start_label": self.start_label,
            "end_label": self.end_label,
            "speaker": self.speaker,
            "quote": self.quote,
        }


@dataclass
class GroundedAnswer:
    answer: str
    citations: list[Citation]
    grounded_score: float
    retrieved: list[RetrievedChunk]
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def is_grounded(self) -> bool:
        return self.grounded_score >= 0.7


# ── Groundedness (pure, unit-tested) ──────────────────────────────────────────


def sentence_support(sentence: str, context_tokens: set[str]) -> bool:
    """A sentence is 'supported' when most content tokens appear in context."""
    if _REFUSAL_RE.search(sentence):
        return True  # explicit non-answer is grounded by definition
    tokens = tokenize(sentence)
    if not tokens:
        return True
    content = [t for t in tokens if len(t) > 2] or tokens
    overlap = sum(1 for token in content if token in context_tokens)
    return overlap / len(content) >= 0.6


def groundedness_score(answer: str, context: str) -> float:
    """Fraction of the answer's sentences supported by the retrieved context."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s.strip()]
    if not sentences:
        return 1.0
    context_tokens = set(tokenize(context))
    supported = sum(1 for sentence in sentences if sentence_support(sentence, context_tokens))
    return supported / len(sentences)


def parse_citation_ids(answer: str) -> list[int]:
    """Extract cited block numbers in order of first appearance."""
    seen: list[int] = []
    for match in _CITATION_RE.finditer(answer):
        idx = int(match.group(1))
        if idx not in seen:
            seen.append(idx)
    return seen


def render_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for pos, chunk in enumerate(chunks, start=1):
        speakers = ", ".join(chunk.speakers) if chunk.speakers else "unknown"
        blocks.append(
            f"[S{pos}] ({chunk.start_label}-{chunk.end_label}; speakers: {speakers})\n{chunk.text}"
        )
    return "\n\n".join(blocks)


class RAGEngine:
    """Indexes transcripts and answers questions with timestamp citations."""

    def __init__(
        self, llm: LLM, embeddings: Embeddings, store: VectorStore, settings: Settings
    ) -> None:
        self.llm = llm
        self.embeddings = embeddings
        self.store = store
        self.settings = settings

    # ── Indexing ─────────────────────────────────────────────────────────────

    async def index_segments(
        self, meeting_id: str, user_id: str, segments: list[TranslatedSegment]
    ) -> int:
        """Chunk, embed and upsert a transcript. Returns the chunk count."""
        chunks = chunk_segments(
            segments,
            max_chars=self.settings.rag_chunk_max_chars,
            overlap_chars=self.settings.rag_chunk_overlap_chars,
        )
        if not chunks:
            return 0
        vectors = await self.embeddings.embed_documents([chunk.text for chunk in chunks])
        items = [
            VectorItem(
                id=f"{meeting_id}:{chunk.chunk_id}",
                document=chunk.text,
                vector=vector,
                metadata={
                    "meeting_id": meeting_id,
                    "user_id": user_id,
                    "chunk_id": chunk.chunk_id,
                    "start_ms": chunk.start_ms,
                    "end_ms": chunk.end_ms,
                    "speakers": ", ".join(chunk.speakers),
                    "first_seq": chunk.first_seq,
                    "last_seq": chunk.last_seq,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        await self.store.upsert(COLLECTION, items)
        logger.info("meeting_indexed", meeting_id=meeting_id, chunks=len(items))
        return len(items)

    async def drop_meeting(self, meeting_id: str) -> None:
        await self.store.delete_where(COLLECTION, {"meeting_id": meeting_id})

    # ── Retrieval ────────────────────────────────────────────────────────────

    async def retrieve(
        self, meeting_id: str, user_id: str, question: str, *, k: int | None = None
    ) -> list[RetrievedChunk]:
        """Top-k chunks for a question, filtered to one meeting and one user."""
        top_k = k or self.settings.rag_top_k
        query_vector = await self.embeddings.embed_query(question)
        hits = await self.store.query(
            COLLECTION,
            query_vector,
            k=top_k,
            where={"meeting_id": meeting_id, "user_id": user_id},
        )
        RETRIEVAL_HITS.inc(len(hits))
        return [
            RetrievedChunk(
                chunk_id=str(hit.metadata.get("chunk_id", hit.id)),
                text=hit.document,
                score=hit.score,
                start_ms=int(hit.metadata.get("start_ms", 0)),
                end_ms=int(hit.metadata.get("end_ms", 0)),
                speakers=[
                    s.strip() for s in str(hit.metadata.get("speakers", "")).split(",") if s.strip()
                ],
            )
            for hit in hits
            if hit.score >= self.settings.rag_min_score
        ]

    # ── Answering ────────────────────────────────────────────────────────────

    async def answer(
        self,
        meeting_id: str,
        user_id: str,
        question: str,
        *,
        history: list[tuple[str, str]] | None = None,
    ) -> GroundedAnswer:
        retrieved = await self.retrieve(meeting_id, user_id, question)
        context = render_context(retrieved)

        messages: list[Message] = [{"role": "system", "content": ANSWER_SYSTEM_PROMPT}]
        for role, content in (history or [])[-self.settings.chat_history_max_messages :]:
            messages.append({"role": role, "content": content})

        if not retrieved:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"No context blocks were retrieved for this meeting.\nQuestion: {question}"
                    ),
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": (f"Context blocks:\n{context}\n\nQuestion: {question}"),
                }
            )

        result = await self.llm.complete(messages, temperature=0.1)
        answer_text = result.text.strip()

        citations = self._build_citations(answer_text, retrieved)
        score = groundedness_score(answer_text, context) if retrieved else 1.0
        RAG_GROUNDEDNESS.observe(score)
        RAG_QUERIES.labels(outcome="ok" if score >= 0.7 else "ungrounded").inc()
        logger.info(
            "rag_answer_produced",
            meeting_id=meeting_id,
            retrieved=len(retrieved),
            citations=len(citations),
            grounded_score=round(score, 3),
        )
        return GroundedAnswer(
            answer=answer_text,
            citations=citations,
            grounded_score=round(score, 4),
            retrieved=retrieved,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )

    def _build_citations(self, answer: str, retrieved: list[RetrievedChunk]) -> list[Citation]:
        citations: list[Citation] = []
        for idx in parse_citation_ids(answer):
            if 1 <= idx <= len(retrieved):
                chunk = retrieved[idx - 1]
                quote = re.sub(r"^\[[^\]]+\]\s*", "", chunk.text.split("\n")[0])
                citations.append(
                    Citation(
                        chunk_id=chunk.chunk_id,
                        start_ms=chunk.start_ms,
                        end_ms=chunk.end_ms,
                        start_label=chunk.start_label,
                        end_label=chunk.end_label,
                        speaker=chunk.speakers[0] if chunk.speakers else "SPEAKER_00",
                        quote=quote[:300],
                    )
                )
        return citations


def chunk_count_for(segments: list[TranslatedSegment], settings: Settings) -> int:
    """Utility for tests/eval: how many chunks a transcript would produce."""
    return len(
        chunk_segments(
            segments,
            max_chars=settings.rag_chunk_max_chars,
            overlap_chars=settings.rag_chunk_overlap_chars,
        )
    )


def chunk_to_retrieved(chunk: Chunk) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        score=1.0,
        start_ms=chunk.start_ms,
        end_ms=chunk.end_ms,
        speakers=list(chunk.speakers),
    )
