"""
memory/vector_store.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

ChromaDB vector store implementation.
Implements the schema defined in tools/schemas/memory.py (Section A3.3).

Provides:
  - store(content, metadata) → document_id
  - search(query, top_k, metadata_filter, min_similarity) → list[SearchResult]
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
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

import config.settings as _cfg

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
    # FIX: use getattr instead of __dict__.get — safe even if attribute doesn't exist
    oai_key = getattr(_cfg, "OPENAI_API_KEY", None) or None

    try:
        import openai as _oai  # noqa: F401
        if oai_key:
            from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
            logger.info("Embedding: OpenAI text-embedding-3-small (1536 dims)")
            return OpenAIEmbeddingFunction(
                api_key=oai_key,
                model_name="text-embedding-3-small",
            )
    except ImportError:
        pass

    # Free local fallback — no API key needed
    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        logger.info("Embedding: sentence-transformers/all-MiniLM-L6-v2 (384 dims, local)")
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
            path=getattr(_cfg, "CHROMA_PERSIST_DIR", "./data/chroma"),
            settings=ChromaSettings(anonymized_telemetry=False),  # type: ignore
        )
        ef = _get_embedding_function()
        self._collection = self._client.get_or_create_collection(
            name=getattr(_cfg, "CHROMA_COLLECTION", "ara1_memory"),
            embedding_function=ef,#type: ignore
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore ready — collection '%s' (%d docs)",
            getattr(_cfg, "CHROMA_COLLECTION", "ara1_memory"),
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def store(
        self,
        content: str,
        metadata: dict,
        document_id: Optional[str] = None,
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
        metadata_filter: Optional[dict] = None,   # FIX: renamed from 'filter' (shadows builtin)
        min_similarity: Optional[float] = None,
    ) -> list[SearchResult]:
        """
        Semantic search with optional metadata filtering.

        Args:
            query:           Natural language search query.
            top_k:           Max results to return.
            metadata_filter: Optional dict for filtering by ticker, source_type, etc.
            min_similarity:  Minimum cosine similarity (0.0–1.0).

        Returns:
            List of SearchResult ordered by similarity descending.
        """
        # FIX: guard against empty collection — ChromaDB raises on query if count=0
        if self._collection.count() == 0:
            logger.debug("VectorStore is empty — returning no results.")
            return []

        threshold = (
            min_similarity
            if min_similarity is not None
            else getattr(_cfg, "MIN_SIMILARITY", 0.70)
        )

        where_clause = self._build_where(metadata_filter) if metadata_filter else None

        try:
            raw = self._collection.query(
                query_texts=[query],
                n_results=min(top_k, self._collection.count()),
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error("VectorStore search failed: %s", exc)
            return []

        results: list[SearchResult] = []
        ids       = raw["ids"][0]
        docs      = raw["documents"][0]#type: ignore
        metas     = raw["metadatas"][0]#type: ignore
        distances = raw["distances"][0]#type: ignore

        for doc_id, doc, meta, dist in zip(ids, docs, metas, distances):
            # ChromaDB cosine distance ranges 0–2 (0=identical, 2=opposite).
            # Convert to similarity score 0.0–1.0: similarity = 1 - (distance / 2)
            similarity = round(1.0 - (dist / 2.0), 4)
            if similarity >= threshold:
                results.append(SearchResult(
                    document_id=doc_id,
                    content=doc,
                    metadata=meta,#type: ignore
                    similarity=similarity,
                ))

        results.sort(key=lambda r: r.similarity, reverse=True)
        logger.debug(
            "Search '%s...' → %d results (threshold=%.2f)",
            query[:40], len(results), threshold
        )
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
            "collection":   getattr(_cfg, "CHROMA_COLLECTION", "ara1_memory"),
            "total_chunks": self._collection.count(),
            "provider":     "chromadb",
            "path":         getattr(_cfg, "CHROMA_PERSIST_DIR", "./data/chroma"),
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
    def _build_where(metadata_filter: dict) -> Optional[dict]:
        """
        Convert filter schema to ChromaDB $where syntax.
        Supports: ticker, source_type, verified, date_after, date_before, session_id.
        """
        conditions: list[dict] = []

        if "ticker" in metadata_filter:
            conditions.append({"ticker": {"$eq": metadata_filter["ticker"]}})
        if "source_type" in metadata_filter:
            conditions.append({"source_type": {"$eq": metadata_filter["source_type"]}})
        if "verified" in metadata_filter:
            conditions.append({"verified": {"$eq": metadata_filter["verified"]}})
        if "session_id" in metadata_filter:
            conditions.append({"session_id": {"$eq": metadata_filter["session_id"]}})
        if "date_after" in metadata_filter:
            conditions.append({"date": {"$gte": metadata_filter["date_after"]}})
        if "date_before" in metadata_filter:
            conditions.append({"date": {"$lte": metadata_filter["date_before"]}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Return the singleton VectorStore, creating it on first call."""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
