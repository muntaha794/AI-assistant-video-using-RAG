"""Embedding backends behind one protocol.

* ``SentenceTransformerEmbeddings`` — production default (HuggingFace
  ``all-MiniLM-L6-v2``, 384-dim), loaded lazily and reused.
* ``HashingEmbeddings`` — deterministic, dependency-free feature-hashing
  embeddings used by the test-suite and eval harness so retrieval can be
  exercised without downloading models.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from collections import Counter
from typing import Protocol

from app.logging_config import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9']+")

# Small English stopword list: enough to make lexical overlap meaningful
# without shipping an NLP dependency into the lightweight install.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "no",
        "not",
        "of",
        "on",
        "or",
        "our",
        "she",
        "so",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "who",
        "will",
        "with",
        "you",
        "your",
    }
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class Embeddings(Protocol):
    dimension: int

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class HashingEmbeddings:
    """Signed feature hashing (the "hashing trick"), L2-normalized."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token, count in Counter(tokenize(text)).items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


class SentenceTransformerEmbeddings:
    """Production embeddings via sentence-transformers (lazy model load)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.dimension = 384
        self._model = None

    def _load(self):  # type: ignore[no-untyped-def]
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - needs requirements-ml
                raise RuntimeError(
                    "sentence-transformers is not installed (pip install -r requirements-ml.txt)"
                ) from exc
            logger.info("loading_embedding_model", model=self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        def _encode() -> list[list[float]]:
            model = self._load()
            vectors = model.encode(texts, normalize_embeddings=True)
            return [list(map(float, vector)) for vector in vectors]

        return await asyncio.to_thread(_encode)

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed_documents([text])
        return vectors[0]


def build_embeddings(backend: str, model_name: str, dimension: int = 384) -> Embeddings:
    if backend == "sentence-transformers":
        try:
            return SentenceTransformerEmbeddings(model_name)
        except Exception:  # noqa: BLE001 - fall back to hashing when unavailable
            logger.warning("embedding_backend_unavailable", backend=backend)
    return HashingEmbeddings(dimension)
