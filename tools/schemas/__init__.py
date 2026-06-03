"""
tools/schemas/__init__.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

Central export point for all 12 tool schemas.

Usage:
    from tools.schemas import ALL_SCHEMAS                 # full list (LLM injection)
    from tools.schemas import SEC_FILING_SEARCH           # individual schema
    from tools.schemas import SCHEMA_MAP                  # dict keyed by tool name

Schema groups:
    Group 1 — Data Retrieval (7 tools):   sec_filing_search, financial_data_api,
              earnings_transcript, web_search, news_sentiment, company_profile,
              peer_comparison
    Group 2 — Computation & Verification (3 tools): calculation_engine,
              fact_checker, report_generator
    Group 3 — Memory (2 tools):           vector_db_search, vector_db_store
"""

from tools.schemas.data_retrieval import (
    SEC_FILING_SEARCH,
    FINANCIAL_DATA_API,
    EARNINGS_TRANSCRIPT,
    WEB_SEARCH,
    NEWS_SENTIMENT,
    COMPANY_PROFILE,
    PEER_COMPARISON,
    DATA_RETRIEVAL_SCHEMAS,
)
from tools.schemas.computation import (
    CALCULATION_ENGINE,
    FACT_CHECKER,
    REPORT_GENERATOR,
    COMPUTATION_SCHEMAS,
)
from tools.schemas.memory import (
    VECTOR_DB_SEARCH,
    VECTOR_DB_STORE,
    MEMORY_SCHEMAS,
)

# ---------------------------------------------------------------------------
# Master list — injected verbatim into the LLM system prompt as tool specs
# Order matters: memory tools first (Memory First rule), then data retrieval,
# then computation. This ordering biases the LLM toward checking memory before
# calling external APIs.
# ---------------------------------------------------------------------------
ALL_SCHEMAS: list[dict] = [
    # Group 3: Memory (check first)
    VECTOR_DB_SEARCH,
    VECTOR_DB_STORE,
    # Group 1: Data Retrieval
    SEC_FILING_SEARCH,
    FINANCIAL_DATA_API,
    EARNINGS_TRANSCRIPT,
    WEB_SEARCH,
    NEWS_SENTIMENT,
    COMPANY_PROFILE,
    PEER_COMPARISON,
    # Group 2: Computation & Verification
    CALCULATION_ENGINE,
    FACT_CHECKER,
    REPORT_GENERATOR,
]

# ---------------------------------------------------------------------------
# Lookup map — used by ToolRegistry to route calls by name
# ---------------------------------------------------------------------------
SCHEMA_MAP: dict[str, dict] = {schema["name"]: schema for schema in ALL_SCHEMAS}

# ---------------------------------------------------------------------------
# Tier classification — used by SynthesisEngine for source reliability weighting
# Mirrors Section 7.1 of architecture_specification.md (corrected hierarchy)
# ---------------------------------------------------------------------------
TOOL_TIER: dict[str, int | None] = {
    "sec_filing_search":   1,   # Tier 1: Trust weight 1.00 — legally mandated, audited
    "financial_data_api":  2,   # Tier 2: Trust weight 0.90 — curated, professional controls
    "earnings_transcript": 3,   # Tier 3: Trust weight 0.75 — direct management voice
    "web_search":          4,   # Tier 4: Trust weight 0.60 — professional journalism
    "news_sentiment":      4,   # Tier 4: Trust weight 0.60 — aggregated news signals
    "company_profile":     2,   # Tier 2: Trust weight 0.90 — curated metadata
    "peer_comparison":     2,   # Tier 2: Trust weight 0.90 — curated comparative data
    "calculation_engine":  1,   # Tier 1: Trust weight 1.00 — deterministic local computation
    "fact_checker":        1,   # Tier 1: Trust weight 1.00 — verification orchestrator
    "report_generator":    None, # Not a data source — output formatter
    "vector_db_search":    None, # Not a source tier — memory retrieval
    "vector_db_store":     None, # Not a source tier — memory persistence
}

# ---------------------------------------------------------------------------
# Fallback chain map — used by ToolRegistry for automatic fallback routing
# Mirrors Section 5.3 of architecture_specification.md
# ---------------------------------------------------------------------------
FALLBACK_CHAINS: dict[str, list[str]] = {
    "sec_filing_search":   ["vector_db_search", "web_search"],
    "financial_data_api":  ["sec_filing_search", "vector_db_search"],
    "earnings_transcript": ["web_search", "vector_db_search"],
    "web_search":          ["news_sentiment", "vector_db_search"],
    "news_sentiment":      ["web_search", "vector_db_search"],
    "company_profile":     ["financial_data_api", "web_search", "vector_db_search"],
    "peer_comparison":     ["web_search", "financial_data_api"],
    "calculation_engine":  [],   # Local computation — no external fallback
    "fact_checker":        [],   # Orchestrator — degrades to single-source with flag
    "report_generator":    [],   # Output stage — no fallback
    "vector_db_search":    [],   # Memory layer — no fallback
    "vector_db_store":     [],   # Memory layer — no fallback
}

__all__ = [
    # Individual schemas
    "SEC_FILING_SEARCH",
    "FINANCIAL_DATA_API",
    "EARNINGS_TRANSCRIPT",
    "WEB_SEARCH",
    "NEWS_SENTIMENT",
    "COMPANY_PROFILE",
    "PEER_COMPARISON",
    "CALCULATION_ENGINE",
    "FACT_CHECKER",
    "REPORT_GENERATOR",
    "VECTOR_DB_SEARCH",
    "VECTOR_DB_STORE",
    # Group lists
    "DATA_RETRIEVAL_SCHEMAS",
    "COMPUTATION_SCHEMAS",
    "MEMORY_SCHEMAS",
    # Master collections
    "ALL_SCHEMAS",
    "SCHEMA_MAP",
    "TOOL_TIER",
    "FALLBACK_CHAINS",
]
