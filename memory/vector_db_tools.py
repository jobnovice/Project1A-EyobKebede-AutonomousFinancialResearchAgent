"""
memory/vector_db_tools.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

Live implementations for the two memory tools:
  - vector_db_search → wraps VectorStore.search()
  - vector_db_store  → wraps VectorStore.store()

These replace the stubs in TOOL_DISPATCH in agent/core.py.
"""

from __future__ import annotations

import logging
from typing import Optional

from memory.vector_store import get_vector_store

logger = logging.getLogger(__name__)


def vector_db_search(
    query: str,
    top_k: int = 5,
    metadata_filter: Optional[dict] = None,   # FIX: renamed from 'filter' (shadows builtin)
    min_similarity: float = 0.70,
) -> dict:
    """
    Search long-term vector memory for previously researched information.
    Implements the Memory First principle — call before any external API.

    Args:
        query:           Natural language search query.
        top_k:           Max results to return.
        metadata_filter: Optional filter dict (ticker, source_type, verified, etc.)
        min_similarity:  Minimum cosine similarity threshold (default 0.70).

    Returns:
        dict with cache_hit, results, top_similarity, collection_size, error.
    """
    try:
        store = get_vector_store()
        results = store.search(
            query=query,
            top_k=top_k,
            metadata_filter=metadata_filter,
            min_similarity=min_similarity,
        )
    except Exception as e:
        # FIX: catch initialisation / search errors — return structured error
        # instead of crashing the agent's execute_step node
        logger.error("vector_db_search failed: %s", e)
        return {
            "tool":            "vector_db_search",
            "query":           query,
            "cache_hit":       False,
            "results_count":   0,
            "results":         [],
            "top_similarity":  0.0,
            "collection_size": 0,
            "error":           str(e),
        }

    hits = [r.to_dict() for r in results]
    cache_hit = len(hits) > 0

    logger.info(
        "vector_db_search: '%s...' → %d hits (top_sim=%.3f)",
        query[:50],
        len(hits),
        hits[0]["similarity"] if hits else 0.0,
    )

    return {
        "tool":            "vector_db_search",
        "query":           query,
        "cache_hit":       cache_hit,
        "results_count":   len(hits),
        "results":         hits,
        "top_similarity":  hits[0]["similarity"] if hits else 0.0,
        "collection_size": store.count(),
        "error":           None,   # FIX: explicit error key for consistency with all other tools
    }


def vector_db_store(
    content: str,
    metadata: dict,
    document_id: Optional[str] = None,
) -> dict:
    """
    Store a verified research chunk in long-term vector memory.
    Should only be called after fact_checker has set verified=True.

    Args:
        content:     Text content to embed and store.
        metadata:    Must include ticker, source_type, date, verified.
        document_id: Optional explicit ID (auto-generated if None).

    Returns:
        dict with document_id, stored flag, collection_size, error.
    """
    if not metadata.get("verified", False):
        logger.warning(
            "vector_db_store called with verified=False for ticker=%s — storing but flagging.",
            metadata.get("ticker", "unknown"),
        )

    try:
        store = get_vector_store()
        doc_id = store.store(
            content=content,
            metadata=metadata,
            document_id=document_id,
        )
    except Exception as e:
        # FIX: catch store errors — return structured error dict
        logger.error("vector_db_store failed: %s", e)
        return {
            "tool":            "vector_db_store",
            "document_id":     None,
            "stored":          False,
            "ticker":          metadata.get("ticker"),
            "source_type":     metadata.get("source_type"),
            "collection_size": 0,
            "error":           str(e),   # FIX: consistent error key
        }

    logger.info(
        "vector_db_store: stored '%s' (ticker=%s, source=%s)",
        doc_id,
        metadata.get("ticker"),
        metadata.get("source_type"),
    )

    return {
        "tool":            "vector_db_store",
        "document_id":     doc_id,
        "stored":          True,
        "ticker":          metadata.get("ticker"),
        "source_type":     metadata.get("source_type"),
        "collection_size": store.count(),
        "error":           None,   # FIX: explicit error key
    }
