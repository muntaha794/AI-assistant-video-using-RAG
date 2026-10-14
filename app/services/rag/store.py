"""Vector stores behind one protocol.

* ``ChromaVectorStore`` — persistent local ChromaDB (production default).
* ``InMemoryVectorStore`` — pure-Python cosine store used by the test-suite
  and the evaluation harness; no external services or downloads.

Both use cosine similarity and a metadata subset filter, so retrieval
behaviour (scores, ``where`` filters) is consistent across backends.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from app.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class VectorItem:
    id: str
    document: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorHit:
    id: str
    document: str
    score: float  # cosine similarity in [-1, 1]
    metadata: dict[str, Any] = field(default_factory=dict)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _matches(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
    if not where:
        return True
    return all(metadata.get(key) == value for key, value in where.items())


class VectorStore(Protocol):
    async def upsert(self, collection: str, items: list[VectorItem]) -> None: ...

    async def query(
        self,
        collection: str,
        vector: list[float],
        *,
        k: int = 6,
        where: dict[str, Any] | None = None,
    ) -> list[VectorHit]: ...

    async def delete_where(self, collection: str, where: dict[str, Any]) -> int: ...


class InMemoryVectorStore:
    """Thread-safe-enough dict store (single-process tests)."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, VectorItem]] = {}

    async def upsert(self, collection: str, items: list[VectorItem]) -> None:
        bucket = self._data.setdefault(collection, {})
        for item in items:
            bucket[item.id] = item

    async def query(
        self,
        collection: str,
        vector: list[float],
        *,
        k: int = 6,
        where: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        bucket = self._data.get(collection, {})
        scored = [
            VectorHit(
                id=item.id,
                document=item.document,
                score=cosine(vector, item.vector),
                metadata=dict(item.metadata),
            )
            for item in bucket.values()
            if _matches(item.metadata, where)
        ]
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:k]

    async def delete_where(self, collection: str, where: dict[str, Any]) -> int:
        bucket = self._data.get(collection, {})
        doomed = [item_id for item_id, item in bucket.items() if _matches(item.metadata, where)]
        for item_id in doomed:
            del bucket[item_id]
        return len(doomed)


class ChromaVectorStore:
    """Persistent ChromaDB client (cosine space, metadata filters)."""

    def __init__(self, persist_dir: Path) -> None:
        self.persist_dir = persist_dir
        self._client: Any = None
        self._collections: dict[str, Any] = {}

    def _get_collection(self, name: str) -> Any:
        if self._client is None:
            import chromadb  # noqa: PLC0415 - lazy optional dependency

            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(
                name=name, metadata={"hnsw:space": "cosine"}
            )
        return self._collections[name]

    async def upsert(self, collection: str, items: list[VectorItem]) -> None:
        if not items:
            return

        def _upsert() -> None:
            col = self._get_collection(collection)
            col.upsert(
                ids=[item.id for item in items],
                documents=[item.document for item in items],
                embeddings=[item.vector for item in items],
                metadatas=[_json_safe(item.metadata) for item in items],
            )

        await asyncio.to_thread(_upsert)

    async def query(
        self,
        collection: str,
        vector: list[float],
        *,
        k: int = 6,
        where: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        def _query() -> list[VectorHit]:
            col = self._get_collection(collection)
            kwargs: dict[str, Any] = {
                "n_results": max(k, 1),
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                kwargs["where"] = dict(where)
            raw = col.query(query_embeddings=[vector], **kwargs)
            hits: list[VectorHit] = []
            ids = (raw.get("ids") or [[]])[0]
            docs = (raw.get("documents") or [[]])[0]
            metas = (raw.get("metadatas") or [[]])[0]
            dists = (raw.get("distances") or [[]])[0]
            for idx, item_id in enumerate(ids):
                # cosine distance → similarity
                similarity = 1.0 - float(dists[idx])
                hits.append(
                    VectorHit(
                        id=item_id,
                        document=docs[idx] or "",
                        score=similarity,
                        metadata=dict(metas[idx] or {}),
                    )
                )
            return hits

        return await asyncio.to_thread(_query)

    async def delete_where(self, collection: str, where: dict[str, Any]) -> int:
        def _delete() -> int:
            col = self._get_collection(collection)
            result = col.delete(where=dict(where))
            # chroma returns None; deleted-count is not exposed portably
            return int(result.get("deleted_count", 0)) if isinstance(result, dict) else 0

        return await asyncio.to_thread(_delete)


def _json_safe(metadata: dict[str, Any]) -> dict[str, Any]:
    """Chroma metadata values must be scalars."""
    safe: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe
