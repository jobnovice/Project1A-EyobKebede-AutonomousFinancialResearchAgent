"""
synthesis/narrative.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

Implements the three synthesis techniques from architecture spec §7.3:

  1. narrative_threading        — connect data points into coherent prose
  2. quantitative_triangulation — resolve a metric from multiple sources
  3. sentiment_fact_alignment   — compare qualitative tone vs quantitative reality

All functions are pure: no LLM calls, no network, no side effects.

INTEGRATION CONTRACT (verified against actual tool implementations,
not assumed shapes):

  gathered_data keys follow "{tool_name}_step{step_id}" (see agent/core.py
  node_execute_step: `gathered[key] = tool_result`).

  Each value is the RAW, UNWRAPPED return dict of the tool itself — there is
  no top-level "data" or "metadata" wrapper. Confirmed field-by-field against:

    financial_data_api  -> {"ticker": str, "statement_type": str, "records": [...], ...}
    company_profile     -> {"ticker": str, "company_name": str, ...}
    earnings_transcript -> {"ticker": str, "content": str, "speakers": [...], ...}
    peer_comparison      -> {"ticker": str, "peers": [...], "comparison_matrix": {...}, ...}
    web_search           -> {"query": str, "results": [...], ...}   <- NO ticker field
    news_sentiment        -> {"query": str, "aggregate": {...}, ...} <- NO ticker field

  web_search and news_sentiment carry no ticker — they are matched to a
  company by checking whether the ticker or company name appears in their
  "query" string, which is the only company signal they carry.

  financial_data_api has no precomputed "revenue_growth" or margin fields:
    - key_metrics       carries valuation/return ratios only (P/E, ROE, etc.)
    - financial_ratios  carries grossProfitMargin / operatingProfitMargin (FMP's real field names)
    - income_statement  carries raw "revenue" per period — growth must be
                         computed manually across 2+ periods (no FMP growth
                         endpoint is wired into financial_data_api).
  This module computes growth/margin from whichever statement_type is
  actually present rather than assuming a field that does not exist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from synthesis.conflict_resolver import SourceValue, resolve_all, TIER_WEIGHTS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ticker <-> company name (for matching web_search / news_sentiment, which
# carry no ticker field — only a free-text "query" string)
# ---------------------------------------------------------------------------
_TICKER_TO_NAME: dict[str, str] = {
    "MSFT": "microsoft", "AAPL": "apple", "GOOGL": "google",
    "AMZN": "amazon", "META": "meta", "TSLA": "tesla",
    "NVDA": "nvidia", "NFLX": "netflix", "CRM": "salesforce",
    "ADBE": "adobe", "INTC": "intel", "JPM": "jpmorgan",
    "GS": "goldman", "V": "visa", "WMT": "walmart", "PFE": "pfizer",
}

# Tool name -> known top-level field that carries the ticker (None = no field,
# must match on free-text query instead)
_TICKER_FIELD_BY_TOOL: dict[str, Optional[str]] = {
    "financial_data_api":  "ticker",
    "company_profile":     "ticker",
    "earnings_transcript": "ticker",
    "peer_comparison":     "ticker",
    "sec_filing_search":   "ticker",
    "web_search":          None,
    "news_sentiment":      None,
}

# FMP /ratios real field names (confirmed against financial_api.py FMP_ENDPOINTS)
_MARGIN_FIELDS = {
    "gross_margin":     "grossProfitMargin",
    "operating_margin": "operatingProfitMargin",
    "net_margin":       "netProfitMargin",
}

TOOL_TIERS: dict[str, int] = {
    "sec_filing_search":   1,
    "financial_data_api":  2,
    "company_profile":     2,
    "peer_comparison":     2,
    "earnings_transcript": 3,
    "web_search":          4,
    "news_sentiment":      4,
}


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NarrativeThread:
    """A prose narrative for one company/theme."""
    company:      str
    theme:        str
    narrative:    str
    sources_used: list[str]
    confidence:   float


@dataclass
class TriangulationResult:
    """Result of quantitative triangulation for one metric."""
    metric_name: str
    value:       Optional[float]
    confidence:  float
    method:      str
    disclosure:  str
    range_min:   Optional[float] = None
    range_max:   Optional[float] = None


@dataclass
class AlignmentResult:
    """Result of sentiment-fact alignment for one company."""
    company:          str
    alignment_label:  str
    finding:          str
    confidence:       float
    sentiment_label:  Optional[str] = None
    sentiment_score:  Optional[float] = None
    financial_trend:  Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers: parse gathered_data keys and match by company
# ---------------------------------------------------------------------------

def _tool_name_from_key(key: str) -> str:
    """'financial_data_api_step3' -> 'financial_data_api'"""
    return key.split("_step")[0] if "_step" in key else key


def _record_matches_company(tool_name: str, record: dict, company: str) -> bool:
    """
    Determine whether a gathered_data record belongs to `company`.

    Tools that carry a 'ticker' field match directly.
    Tools that don't (web_search, news_sentiment) are matched by checking
    whether the ticker or its company name appears in the 'query' string —
    the only company signal those tools' output actually contains.
    """
    ticker_field = _TICKER_FIELD_BY_TOOL.get(tool_name)

    if ticker_field:
        return record.get(ticker_field, "").upper() == company.upper()

    query = str(record.get("query", "")).lower()
    company_name = _TICKER_TO_NAME.get(company.upper(), "")
    return company.lower() in query or (bool(company_name) and company_name in query)


def _extract_company_data(gathered: dict[str, Any], company: str) -> dict[str, list]:
    """
    Partition gathered_data into per-tool-type buckets for one company,
    using each tool's REAL output shape (no assumed 'data'/'metadata' wrapper).
    """
    buckets: dict[str, list] = {
        "financial_data_api":  [],
        "company_profile":     [],
        "earnings_transcript": [],
        "peer_comparison":     [],
        "web_search":          [],
        "news_sentiment":      [],
        "sec_filing_search":   [],
    }

    for key, record in gathered.items():
        if not isinstance(record, dict):
            continue
        tool_name = _tool_name_from_key(key)
        if tool_name not in buckets:
            continue
        if record.get("error"):
            continue   # skip failed tool calls
        if _record_matches_company(tool_name, record, company):
            buckets[tool_name].append(record)

    return buckets


# ---------------------------------------------------------------------------
# Financial metric derivation — computed from whichever statement_type
# is actually present, rather than assuming a field that doesn't exist
# ---------------------------------------------------------------------------

def _derive_revenue_growth(financial_records: list[dict]) -> Optional[float]:
    """
    Compute YoY revenue growth from income_statement records.
    Requires at least 2 periods of 'revenue' in the records list.
    Returns None if insufficient data — never fabricates a value.
    """
    for rec in financial_records:
        if rec.get("statement_type") != "income_statement":
            continue
        periods = rec.get("records", [])
        revenues = [
            (p.get("date", ""), p.get("revenue"))
            for p in periods
            if p.get("revenue") is not None
        ]
        if len(revenues) < 2:
            continue
        revenues.sort(key=lambda x: x[0], reverse=True)   # most recent first
        current = revenues[0][1]
        prior   = revenues[1][1]
        if prior:
            return (current - prior) / abs(prior)
    return None


def _derive_margin(financial_records: list[dict], margin_type: str = "operating_margin") -> Optional[float]:
    """
    Extract a margin from financial_ratios records.
    margin_type: 'gross_margin' | 'operating_margin' | 'net_margin'
    Returns None if financial_ratios was never fetched — never fabricates a value.
    """
    fmp_field = _MARGIN_FIELDS.get(margin_type)
    if not fmp_field:
        return None

    for rec in financial_records:
        if rec.get("statement_type") != "financial_ratios":
            continue
        periods = rec.get("records", [])
        if periods and fmp_field in periods[0]:
            val = periods[0][fmp_field]
            return float(val) if val is not None else None
    return None


def _derive_pe_ratio(financial_records: list[dict]) -> Optional[float]:
    """Extract P/E ratio from key_metrics records (real field: peRatio)."""
    for rec in financial_records:
        if rec.get("statement_type") != "key_metrics":
            continue
        periods = rec.get("records", [])
        if periods and "peRatio" in periods[0]:
            val = periods[0]["peRatio"]
            return float(val) if val is not None else None
    return None


# ---------------------------------------------------------------------------
# 1. Narrative Threading
# ---------------------------------------------------------------------------

def narrative_threading(
    gathered_data: dict[str, Any],
    companies: list[str],
    query_type: str = "general",
) -> list[NarrativeThread]:
    """
    Generate a prose narrative per company by connecting data from multiple
    sources actually present in gathered_data.

    Args:
        gathered_data: ResearchState's gathered_data dict (raw tool outputs).
        companies:     Tickers to build narratives for.
        query_type:    Adjusts thematic focus ("risk" emphasises risk language).

    Returns:
        One NarrativeThread per company. Never fabricates data — if a metric
        was not actually fetched, it is simply omitted from the narrative.
    """
    threads: list[NarrativeThread] = []

    for company in companies:
        buckets = _extract_company_data(gathered_data, company)
        sources_used: set[str] = set()
        parts: list[str] = []

        # --- Company profile context (if fetched) ---
        if buckets["company_profile"]:
            profile = buckets["company_profile"][0]
            name    = profile.get("company_name") or company
            sector  = profile.get("sector")
            if sector:
                parts.append(f"{name} operates in the {sector} sector.")
            sources_used.add("company_profile")

        # --- Revenue growth (only if income_statement was actually fetched) ---
        rev_growth = _derive_revenue_growth(buckets["financial_data_api"])
        if rev_growth is not None:
            parts.append(f"Revenue grew {rev_growth*100:.1f}% year-over-year.")
            sources_used.add("financial_data_api")

        # --- Margin (only if financial_ratios was actually fetched) ---
        op_margin = _derive_margin(buckets["financial_data_api"], "operating_margin")
        if op_margin is not None:
            health = "healthy" if op_margin > 0.20 else "compressed"
            parts.append(f"Operating margin stands at {op_margin*100:.1f}%, indicating {health} profitability.")
            sources_used.add("financial_data_api")

        # --- Sentiment (only if news_sentiment was actually fetched) ---
        sentiment_label = None
        sentiment_score = None
        for rec in buckets["news_sentiment"]:
            agg = rec.get("aggregate", {})
            if agg.get("overall_label"):
                sentiment_label = agg["overall_label"]
                sentiment_score = agg.get("mean_score")
                sources_used.add("news_sentiment")
                break
        if sentiment_label:
            parts.append(
                f"Recent news sentiment is {sentiment_label} "
                f"(mean score {sentiment_score:.2f})."
            )

        # --- Earnings call presence (factual, no fabricated content) ---
        if buckets["earnings_transcript"]:
            t = buckets["earnings_transcript"][0]
            speakers = t.get("speakers", [])
            if speakers:
                parts.append(
                    f"The most recent earnings call ({t.get('quarter','')} {t.get('year','')}) "
                    f"included remarks from {', '.join(speakers[:2])}."
                )
            sources_used.add("earnings_transcript")

        # --- Peer comparison context ---
        if buckets["peer_comparison"]:
            pc = buckets["peer_comparison"][0]
            peers = pc.get("peers", [])
            if peers:
                parts.append(f"Key industry peers include {', '.join(peers[:3])}.")
            sources_used.add("peer_comparison")

        # --- Risk theme additions (only when query_type signals it, and only
        #     citing data that was actually retrieved) ---
        if "risk" in query_type.lower() and buckets["web_search"]:
            parts.append(
                "Recent web coverage was reviewed for emerging risk factors "
                "as part of this assessment."
            )
            sources_used.add("web_search")

        if not parts:
            parts.append(
                f"Insufficient data was retrieved to construct a narrative for {company}."
            )

        confidence = min(1.0, 0.3 + 0.15 * len(sources_used)) if sources_used else 0.0

        threads.append(NarrativeThread(
            company=company,
            theme="risk_assessment" if "risk" in query_type.lower() else "general",
            narrative=" ".join(parts),
            sources_used=sorted(sources_used),
            confidence=round(confidence, 2),
        ))

    return threads


# ---------------------------------------------------------------------------
# 2. Quantitative Triangulation
#
# FIX: Routed through synthesis.conflict_resolver.resolve_all() instead of
# duplicating tier-weighting and 5%-tolerance logic. This keeps a single
# source of truth for conflict resolution across the codebase rather than
# two parallel, divergent implementations.
# ---------------------------------------------------------------------------

def quantitative_triangulation(
    metric_name: str,
    source_values: list[SourceValue],
) -> TriangulationResult:
    """
    Triangulate a numerical metric from multiple sources using the same
    6-step protocol as conflict_resolver, rather than a separate algorithm.

    Per architecture spec §7.3:
      - 2/3 (or more) sources agreeing within 5% -> high confidence
      - All sources disagree -> report range, low confidence
      - Single source -> report value, explicit low-confidence flag

    Args:
        metric_name:   Name of the metric being triangulated.
        source_values: SourceValue objects from however many sources are available.

    Returns:
        TriangulationResult with the resolved value, confidence, method, and
        a disclosure string built from conflict_resolver's own disclosures.
    """
    valid = [s for s in source_values if s.value is not None]

    if not valid:
        return TriangulationResult(
            metric_name=metric_name, value=None, confidence=0.0,
            method="no_sources",
            disclosure=f"No valid sources available for metric '{metric_name}'.",
        )

    if len(valid) == 1:
        s = valid[0]
        return TriangulationResult(
            metric_name=metric_name, value=s.value, confidence=0.30,
            method="single_source",
            disclosure=(
                f"Only one source available ({s.source_name}, Tier {s.source_tier}): "
                f"{s.value:,.4f}. Flagged low confidence — independent verification recommended."
            ),
        )

    # Delegate to conflict_resolver's tournament resolver — single source of truth
    final_value, resolutions = resolve_all(metric_name, valid)

    values = [s.value for s in valid]
    range_min, range_max = min(values), max(values)

    if not resolutions:
        # All sources agreed within 5% — no conflicts needed resolving
        confidence = 0.90
        method = "consensus_agreement"
        disclosure = (
            f"{len(valid)} sources all agreed within 5% tolerance for '{metric_name}': "
            f"{', '.join(f'{s.source_name}={s.value:,.4f}' for s in valid)}. "
            f"Using {final_value:,.4f}."
        )
    else:
        # Conflicts existed — confidence reflects how many had to be overridden
        # vs total comparisons; tier-based resolution is inherently more
        # trustworthy than simple averaging since it respects source authority
        real_conflicts = [r for r in resolutions if r.is_real_conflict]
        confidence = max(0.40, 0.85 - 0.10 * len(real_conflicts))
        method = "tier_resolved" if real_conflicts else "temporal_disambiguation"
        disclosure_lines = [r.disclosure_note for r in resolutions]
        disclosure = " ".join(disclosure_lines)

    return TriangulationResult(
        metric_name=metric_name,
        value=final_value,
        confidence=round(confidence, 3),
        method=method,
        disclosure=disclosure,
        range_min=range_min,
        range_max=range_max,
    )


# ---------------------------------------------------------------------------
# 3. Sentiment-Fact Alignment
#
# FIX: removed the .get("data",{}).get("aggregate",{}) double-unwrap —
# news_sentiment's real output has "aggregate" at the top level.
# FIX: financial_data is now the actual financial_data_api gathered records
# list, and trend is derived via the same _derive_revenue_growth helper
# used in narrative_threading (no fabricated field names).
# ---------------------------------------------------------------------------

def sentiment_fact_alignment(
    sentiment_record: dict[str, Any],
    financial_records: list[dict[str, Any]],
    company: str = "unknown",
) -> AlignmentResult:
    """
    Compare qualitative sentiment tone against quantitative financial reality.

    Args:
        sentiment_record:  The RAW news_sentiment() tool output
                           (top-level "aggregate" key, no wrapper).
        financial_records: List of RAW financial_data_api() tool outputs
                           for this company (any statement_type).
        company:           Ticker, used only for labelling the result since
                           news_sentiment's own output carries no ticker field.

    Returns:
        AlignmentResult — aligned / divergent / mixed / inconclusive.
    """
    aggregate       = sentiment_record.get("aggregate", {})
    sentiment_label = aggregate.get("overall_label")
    sentiment_score = aggregate.get("mean_score")

    if not sentiment_label or sentiment_score is None:
        return AlignmentResult(
            company=company, alignment_label="inconclusive",
            finding="Insufficient sentiment data for comparison.",
            confidence=0.0,
        )

    rev_growth = _derive_revenue_growth(financial_records)
    if rev_growth is None:
        return AlignmentResult(
            company=company, alignment_label="inconclusive",
            finding="No revenue growth data available for comparison "
                    "(requires income_statement with 2+ periods).",
            confidence=0.0,
            sentiment_label=sentiment_label, sentiment_score=sentiment_score,
        )

    trend = "positive" if rev_growth > 0 else "negative" if rev_growth < 0 else "neutral"

    if trend == sentiment_label:
        alignment_label = "aligned"
        finding = (
            f"Sentiment ({sentiment_label}, score {sentiment_score:.2f}) aligns with "
            f"{trend} revenue trend ({rev_growth*100:.1f}% YoY)."
        )
        confidence = 0.80
    elif {trend, sentiment_label} == {"positive", "negative"}:
        alignment_label = "divergent"
        finding = (
            f"Sentiment is {sentiment_label} (score {sentiment_score:.2f}) while the "
            f"revenue trend is {trend} ({rev_growth*100:.1f}% YoY). This divergence may "
            f"signal market overreaction, forward-looking concerns not yet in financials, "
            f"or sentiment driven by non-financial news."
        )
        confidence = 0.70
    else:
        alignment_label = "mixed"
        finding = (
            f"Sentiment is {sentiment_label} (score {sentiment_score:.2f}) and revenue "
            f"trend is {trend} ({rev_growth*100:.1f}% YoY) — no strong signal either way."
        )
        confidence = 0.40

    return AlignmentResult(
        company=company,
        alignment_label=alignment_label,
        finding=finding,
        confidence=round(confidence, 2),
        sentiment_label=sentiment_label,
        sentiment_score=sentiment_score,
        financial_trend=trend,
    )
