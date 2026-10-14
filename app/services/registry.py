"""Service registry: one place that builds and holds the heavy services.

The registry is created at app startup (and by the test-suite with fakes
injected), so routes and the job pipeline share the same LLM client,
vector store and usage tracker.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import Settings
from app.services.action_items import extract_action_items
from app.services.diarization import Diarizer, build_diarizer
from app.services.llm import LLM, UsageTracker, build_llm
from app.services.rag.embeddings import Embeddings, build_embeddings
from app.services.rag.engine import RAGEngine
from app.services.rag.store import InMemoryVectorStore, VectorStore
from app.services.storage import AudioStorage
from app.services.summarizer import summarize_meeting
from app.services.transcription import Transcriber, WhisperTranscriber
from app.services.translation import Translator, build_translator


@dataclass
class ServiceRegistry:
    """Wires concrete service implementations (swappable in tests)."""

    settings: Settings
    storage: AudioStorage
    transcriber: Transcriber
    diarizer: Diarizer
    translator: Translator
    llm: LLM
    embeddings: Embeddings
    vector_store: VectorStore
    rag: RAGEngine = field(init=False)
    usage: UsageTracker = field(init=False)

    def __post_init__(self) -> None:
        self.usage = getattr(self.llm, "usage", None) or UsageTracker()
        self.rag = RAGEngine(self.llm, self.embeddings, self.vector_store, self.settings)

    async def summarize(self, segments):  # type: ignore[no-untyped-def]
        return await summarize_meeting(self.llm, segments)

    async def extract_action_items(self, segments):  # type: ignore[no-untyped-def]
        return await extract_action_items(self.llm, segments)


def build_service_registry(settings: Settings) -> ServiceRegistry:
    """Production wiring: Whisper, pyannote/fallback, Chroma, Mistral, MiniLM."""
    if settings.vector_store == "memory":
        vector_store: VectorStore = InMemoryVectorStore()
    else:
        from app.services.rag.store import ChromaVectorStore

        vector_store = ChromaVectorStore(settings.chroma_persist_dir)

    return ServiceRegistry(
        settings=settings,
        storage=AudioStorage(settings),
        transcriber=WhisperTranscriber(settings.whisper_model, settings.whisper_device),
        diarizer=build_diarizer(settings),
        translator=build_translator(settings),
        llm=build_llm(settings),
        embeddings=build_embeddings(
            settings.embedding_backend, settings.embedding_model, settings.embedding_dim
        ),
        vector_store=vector_store,
    )
