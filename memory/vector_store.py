"""
memory/vector_store.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

ChromaDB vector store implementation.
Implements the schema defined in tools/schemas/memory.py (Section A3.3).

Provides:
  - store(content, metadata) → document_id
  - search(query, top_k, filter, min_similarity) → list[SearchResult]
  - delete(document_id)
  - collection stats

Embedding: OpenAI text-embedding-3-small (1536 dims) via chromadb's
           built-in embedding function. Falls back to sentence-transformers
           all-MiniLM-L6-v2 (384 dims, free/local) if no OpenAI key.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    document_id: str
    content: str
    metadata: dict
    similarity: float           # cosine similarity 0.0–1.0

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "content":     self.content,
            "metadata":    self.metadata,
            "similarity":  self.similarity,
        }


# ---------------------------------------------------------------------------
# Embedding function factory
# ---------------------------------------------------------------------------

def _get_embedding_function():
    """
    Return the best available embedding function.
    Priority: OpenAI text-embedding-3-small → sentence-transformers fallback.
    """
    openai_key = settings.llm.anthropic_api_key  # reuse if OpenAI key present
    oai_key = settings.api.__dict__.get("openai_api_key") or None

    try:
        import openai as _oai  # noqa: F401
        if oai_key:
            from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
            logger.info("Embedding: OpenAI text-embedding-3-small")
            return OpenAIEmbeddingFunction(
                api_key=oai_key,
                model_name="text-embedding-3-small",
            )
    except ImportError:
        pass

    # Free local fallback
    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        logger.info("Embedding: sentence-transformers/all-MiniLM-L6-v2 (local)")
        return SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    except ImportError:
        pass

    logger.warning("No embedding function available — using ChromaDB default")
    return None


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class VectorStore:
    """
    ChromaDB-backed vector store implementing the Section A3.3 schema.

    All metadata fields from the schema are stored and filterable:
      ticker, source_type, date, section, confidence, verified,
      session_id, chunk_index
    """

    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(
            path=settings.vector_db.chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        ef = _get_embedding_function()
        self._collection = self._client.get_or_create_collection(
            name=settings.vector_db.collection_name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore ready — collection '%s' (%d docs)",
            settings.vector_db.collection_name,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def store(
        self,
        content: str,
        metadata: dict,
        document_id: str | None = None,
    ) -> str:
        """
        Embed and store a research chunk.

        Args:
            content:     Text to embed (100–500 words recommended).
            metadata:    Must include: ticker, source_type, date, verified.
            document_id: Optional explicit ID (auto-generated if None).

        Returns:
            document_id of the stored chunk.
        """
        doc_id = document_id or self._auto_id(metadata)

        # Sanitise metadata — ChromaDB only accepts str/int/float/bool
        clean_meta = self._sanitise_metadata(metadata)

        self._collection.upsert(
            ids=[doc_id],
            documents=[content],
            metadatas=[clean_meta],
        )
        logger.debug("Stored chunk '%s' (ticker=%s)", doc_id, metadata.get("ticker"))
        return doc_id

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: dict | None = None,
        min_similarity: float | None = None,
    ) -> list[SearchResult]:
        """
        Semantic search with optional metadata filtering.

        Args:
            query:          Natural language search query.
            top_k:          Max results to return.
            filter:         Metadata filter dict (ticker, source_type, etc.).
            min_similarity: Minimum cosine similarity threshold.

        Returns:
            List of SearchResult ordered by similarity descending.
        """
        threshold = min_similarity if min_similarity is not None else settings.vector_db.min_similarity

        where_clause = self._build_where(filter) if filter else None

        try:
            raw = self._collection.query(
                query_texts=[query],
                n_results=min(top_k, max(1, self._collection.count())),
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error("VectorStore search failed: %s", exc)
            return []

        results: list[SearchResult] = []
        ids       = raw["ids"][0]
        docs      = raw["documents"][0]
        metas     = raw["metadatas"][0]
        distances = raw["distances"][0]

        for doc_id, doc, meta, dist in zip(ids, docs, metas, distances):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity: 1 - (dist / 2)
            similarity = round(1.0 - (dist / 2.0), 4)
            if similarity >= threshold:
                results.append(SearchResult(
                    document_id=doc_id,
                    content=doc,
                    metadata=meta,
                    similarity=similarity,
                ))

        results.sort(key=lambda r: r.similarity, reverse=True)
        logger.debug("Search '%s...' → %d results (threshold=%.2f)", query[:40], len(results), threshold)
        return results

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, document_id: str) -> bool:
        try:
            self._collection.delete(ids=[document_id])
            return True
        except Exception as exc:
            logger.error("Delete '%s' failed: %s", document_id, exc)
            return False

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def count(self) -> int:
        return self._collection.count()

    def stats(self) -> dict:
        return {
            "collection":  settings.vector_db.collection_name,
            "total_chunks": self._collection.count(),
            "provider":    "chromadb",
            "path":        settings.vector_db.chroma_path,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_id(metadata: dict) -> str:
        ticker      = metadata.get("ticker", "unknown").lower()
        source_type = metadata.get("source_type", "doc").replace("-", "").lower()
        date        = metadata.get("date", "nodate").replace("-", "")
        chunk_idx   = metadata.get("chunk_index", 0)
        suffix      = uuid.uuid4().hex[:6]
        return f"{ticker}-{source_type}-{date}-{chunk_idx:03d}-{suffix}"

    @staticmethod
    def _sanitise_metadata(meta: dict) -> dict:
        """Convert all values to ChromaDB-safe types (str/int/float/bool)."""
        clean: dict[str, Any] = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif v is None:
                clean[k] = ""
            else:
                clean[k] = str(v)
        return clean

    @staticmethod
    def _build_where(filter: dict) -> dict | None:
        """
        Convert our filter schema to ChromaDB $where syntax.
        Supports: ticker, source_type, verified, date_after, date_before, session_id.
        """
        conditions: list[dict] = []

        if "ticker" in filter:
            conditions.append({"ticker": {"$eq": filter["ticker"]}})
        if "source_type" in filter:
            conditions.append({"source_type": {"$eq": filter["source_type"]}})
        if "verified" in filter:
            conditions.append({"verified": {"$eq": filter["verified"]}})
        if "session_id" in filter:
            conditions.append({"session_id": {"$eq": filter["session_id"]}})
        if "date_after" in filter:
            conditions.append({"date": {"$gte": filter["date_after"]}})
        if "date_before" in filter:
            conditions.append({"date": {"$lte": filter["date_before"]}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Return the singleton VectorStore, creating it on first call."""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
