"""
memory/vector_db_tools.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

Live implementations for the two memory tools:
  - vector_db_search → wraps VectorStore.search()
  - vector_db_store  → wraps VectorStore.store()

These are registered in the ToolRegistry on Day 3, replacing the stubs.
"""

from __future__ import annotations

import logging

from memory.vector_store import get_vector_store

logger = logging.getLogger(__name__)


def vector_db_search(
    query: str,
    top_k: int = 5,
    filter: dict | None = None,
    min_similarity: float = 0.70,
) -> dict:
    """
    Search long-term vector memory for previously researched information.
    Implements the Memory First principle — call this before any external API.
    """
    store = get_vector_store()
    results = store.search(query=query, top_k=top_k, filter=filter, min_similarity=min_similarity)

    hits = [r.to_dict() for r in results]
    cache_hit = len(hits) > 0

    logger.info(
        "vector_db_search: '%s...' → %d hits (top_sim=%.3f)",
        query[:50],
        len(hits),
        hits[0]["similarity"] if hits else 0.0,
    )

    return {
        "tool": "vector_db_search",
        "query": query,
        "cache_hit": cache_hit,
        "results_count": len(hits),
        "results": hits,
        "top_similarity": hits[0]["similarity"] if hits else 0.0,
        "collection_size": store.count(),
    }


def vector_db_store(
    content: str,
    metadata: dict,
    document_id: str | None = None,
) -> dict:
    """
    Store a verified research chunk in long-term vector memory.
    Only call after fact_checker has verified=True for this content.
    """
    if not metadata.get("verified", False):
        logger.warning(
            "vector_db_store called with verified=False — storing anyway but flagging."
        )

    store = get_vector_store()
    doc_id = store.store(content=content, metadata=metadata, document_id=document_id)

    logger.info(
        "vector_db_store: stored '%s' (ticker=%s, source=%s)",
        doc_id, metadata.get("ticker"), metadata.get("source_type"),
    )

    return {
        "tool": "vector_db_store",
        "document_id": doc_id,
        "stored": True,
        "ticker": metadata.get("ticker"),
        "source_type": metadata.get("source_type"),
        "collection_size": store.count(),
    }
