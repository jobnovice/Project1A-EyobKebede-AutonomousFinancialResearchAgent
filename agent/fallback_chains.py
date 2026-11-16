"""
agent/fallback_chains.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

Fallback tool chains — orchestrates "try the primary tool, then each fallback
in order" with retry, circuit-breaker awareness, parameter translation, and
transparent disclosure.

Source of truth for WHICH tools back up which: FALLBACK_CHAINS in
tools/schemas/__init__.py (Day 2). This module adds the missing HOW:

  1. Parameter translation   — a fallback tool has a different signature
                               (financial_data_api(ticker, statement_type) ->
                               sec_filing_search(ticker, filing_type)), so every
                               (primary, fallback) pair has an explicit translator.
  2. Error-dict conversion   — our tools catch their own exceptions and return
                               {"error": "..."} dicts. retry_with_backoff only
                               retries on exceptions, so we convert error dicts
                               back into failures; otherwise retries never fire.
  3. Usable-data check       — a tool can "succeed" with nothing in it (empty
                               search results, cache miss). That is not a win.
  4. Circuit breaker hook    — optional; open circuits are skipped, not retried.
                               Any object with allow/record_success/record_failure
                               works (agent/circuit_breaker.py implements it).
  5. Transparent disclosure  — fallback use and total failure are reported in the
                               result and in format_degradation_notes().

CORE INTEGRATION:
  core.py sends real calls through execute_with_fallbacks(), counts physical
  retries against the global call budget, and stores each result under the
  actual source tool's name via gathered_key(). This is required because the
  synthesis layer identifies sources from the gathered_data key.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from agent.error_handler import with_graceful_degradation, _build_degradation_stub
from tools.schemas import FALLBACK_CHAINS

logger = logging.getLogger(__name__)

MIN_FALLBACKS_PER_EXTERNAL_TOOL = 2

# Tools with no external dependency to fail over from (empty chain by design)
LOCAL_TOOLS = {
    "calculation_engine", "fact_checker", "report_generator",
    "vector_db_search", "vector_db_store",
}

# Reliability tier per tool, used only for human-readable disclosure text
TOOL_TIER: dict[str, Optional[int]] = {
    "sec_filing_search": 1, "financial_data_api": 2, "company_profile": 2,
    "peer_comparison": 2, "earnings_transcript": 3, "web_search": 4,
    "news_sentiment": 4, "vector_db_search": None,
}


class BreakerLike(Protocol):
    """Interface the circuit breaker must provide (see agent/circuit_breaker.py)."""
    def allow(self, tool_name: str) -> bool: ...
    def record_success(self, tool_name: str) -> None: ...
    def record_failure(self, tool_name: str, error_type: Optional[str] = None) -> None: ...


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ToolAttempt:
    tool:          str
    status:        str                   # success | failed | no_data |
                                         # skipped_circuit_open | skipped_not_implemented |
                                         # skipped_call_budget
    attempts:      int = 0
    error_type:    Optional[str] = None
    error_message: Optional[str] = None
    params:        dict = field(default_factory=dict)
    attempt_log:   list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ChainResult:
    primary_tool: str
    source_tool:  Optional[str]          # tool whose data is in `result`; None if degraded
    result:       dict
    success:      bool
    used_fallback:bool
    degraded:     bool
    attempts:     list[ToolAttempt]
    chain:        list[str]
    disclosure:   Optional[str] = None   # None when the primary tool succeeded cleanly


def gathered_key(chain_result: ChainResult, step_id: Any) -> str:
    """Key to file this result under in gathered_data (source tool, not primary)."""
    return f"{chain_result.source_tool or chain_result.primary_tool}_step{step_id}"


# ---------------------------------------------------------------------------
# Parameter translation: (primary, fallback) -> fallback's params
# ---------------------------------------------------------------------------

_LOOKBACK_FOR_RANGE = {"past_day": 1, "past_week": 7, "past_month": 30,
                       "past_year": 365, "any": 30}


def _ticker(p: dict, c: dict) -> str:
    return str(p.get("ticker") or c.get("company") or "").upper()


def _query(p: dict, c: dict) -> str:
    return str(p.get("query") or c.get("query") or "")


def _range_for_days(days: Any) -> str:
    try:
        d = int(days)
    except (TypeError, ValueError):
        d = 30
    if d <= 1:
        return "past_day"
    if d <= 7:
        return "past_week"
    if d <= 30:
        return "past_month"
    return "past_year"


def _vec(subject: Callable[[dict, dict], str]) -> Callable[[dict, dict], dict]:
    """Translator factory for vector_db_search fallbacks."""
    def _t(p: dict, c: dict) -> dict:
        tk = _ticker(p, c)
        out = {
            "query": f"{tk} {subject(p, c)}".strip() or _query(p, c),
            "top_k": 5,
            "min_similarity": 0.6,   # looser than precheck: a stale hit beats nothing
        }
        if tk:
            out["metadata_filter"] = {"ticker": tk}
        return out
    return _t


def _yr(p: dict) -> str:
    return str(p.get("year") or "")


_TRANSLATORS: dict[tuple[str, str], Callable[[dict, dict], dict]] = {
    ("sec_filing_search", "vector_db_search"):
        _vec(lambda p, c: f"{p.get('filing_type', '10-K')} {_yr(p)} annual report"),
    ("sec_filing_search", "web_search"):
        lambda p, c: {"query": f"{_ticker(p, c)} {p.get('filing_type', '10-K')} SEC filing {_yr(p)}".strip(),
                      "num_results": 5, "date_range": "any"},

    ("financial_data_api", "sec_filing_search"):
        lambda p, c: {"ticker": _ticker(p, c),
                      "filing_type": "10-Q" if p.get("period") == "quarterly" else "10-K"},
    ("financial_data_api", "vector_db_search"):
        _vec(lambda p, c: f"{p.get('statement_type', 'key_metrics')} financials"),

    ("earnings_transcript", "web_search"):
        lambda p, c: {"query": f"{_ticker(p, c)} {p.get('quarter', '')} {_yr(p)} earnings call transcript".strip(),
                      "num_results": 5, "date_range": "past_year"},
    ("earnings_transcript", "vector_db_search"):
        _vec(lambda p, c: f"{p.get('quarter', '')} {_yr(p)} earnings call"),

    ("web_search", "news_sentiment"):
        lambda p, c: {"query": _query(p, c), "num_articles": 10,
                      "lookback_days": _LOOKBACK_FOR_RANGE.get(p.get("date_range", "any"), 30)},
    ("web_search", "vector_db_search"):
        _vec(_query),

    ("news_sentiment", "web_search"):
        lambda p, c: {"query": _query(p, c), "num_results": 10,
                      "date_range": _range_for_days(p.get("lookback_days", 30))},
    ("news_sentiment", "vector_db_search"):
        _vec(_query),

    ("company_profile", "financial_data_api"):
        lambda p, c: {"ticker": _ticker(p, c), "statement_type": "key_metrics",
                      "period": "annual", "limit": 1},
    ("company_profile", "web_search"):
        lambda p, c: {"query": f"{_ticker(p, c)} company overview business sector CEO",
                      "num_results": 5, "date_range": "any"},
    ("company_profile", "vector_db_search"):
        _vec(lambda p, c: "company profile overview"),

    ("peer_comparison", "web_search"):
        lambda p, c: {"query": f"{_ticker(p, c)} industry peers competitors comparison",
                      "num_results": 5, "date_range": "past_year"},
    ("peer_comparison", "financial_data_api"):
        lambda p, c: {"ticker": _ticker(p, c), "statement_type": "key_metrics",
                      "period": "annual", "limit": 1},
}

# Honest limitations of each substitute, surfaced in the disclosure text
_SUBSTITUTE_NOTES: dict[tuple[str, str], str] = {
    ("financial_data_api", "sec_filing_search"):
        "returns raw filing text rather than structured statements; figures still need extracting",
    ("company_profile", "financial_data_api"):
        "valuation and return metrics only; no sector, executive or description fields",
    ("peer_comparison", "financial_data_api"):
        "the company's own metrics only; no peer set was compared",
}


def translate_params(primary: str, fallback: str, params: dict,
                     context: Optional[dict] = None) -> dict:
    """Build the fallback tool's params from the primary's params + run context."""
    context = context or {}
    translator = _TRANSLATORS.get((primary, fallback))
    if translator is None:
        logger.warning("No translator for %s -> %s; using generic params.", primary, fallback)
        return {"query": _query(params, context) or _ticker(params, context)}
    return translator(params, context)


# ---------------------------------------------------------------------------
# Usable-data check
# ---------------------------------------------------------------------------

_USABLE: dict[str, Callable[[dict], bool]] = {
    "vector_db_search":    lambda r: r.get("results_count", len(r.get("results", []))) > 0,
    "web_search":          lambda r: bool(r.get("results")),
    "news_sentiment":      lambda r: bool(r.get("articles")),
    "financial_data_api":  lambda r: bool(r.get("records")),
    "sec_filing_search":   lambda r: bool(r.get("filing_text") or r.get("document_url")),
    "earnings_transcript": lambda r: bool(r.get("content")),
    "company_profile":     lambda r: bool(r.get("company_name")),
    "peer_comparison":     lambda r: bool(r.get("comparison_matrix")),
}


def has_usable_data(tool_name: str, result: Any) -> bool:
    """True only if the tool returned real content (not an error, stub, or empty shell)."""
    if not isinstance(result, dict) or result.get("error") or result.get("degraded"):
        return False
    check = _USABLE.get(tool_name)
    return check(result) if check else True


def _raise_on_error(fn: Callable) -> Callable:
    """Turn a tool's {"error": ...} return into an exception so retry/classification applies."""
    def _wrapped(**kwargs):
        result = fn(**kwargs)
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(str(result["error"]))
        return result
    return _wrapped


# ---------------------------------------------------------------------------
# Chain executor
# ---------------------------------------------------------------------------

def _tier_label(tool: str) -> str:
    tier = TOOL_TIER.get(tool)
    return f"Tier {tier}" if tier else "cached memory, may be stale"


def _reason(a: ToolAttempt) -> str:
    detail = (a.error_message or "").strip()[:120]
    return f"{a.status}" + (f" ({a.error_type}: {detail})" if a.error_type or detail else "")


def execute_with_fallbacks(
    tool_name: str,
    params: dict,
    dispatch: dict[str, Callable],
    context: Optional[dict] = None,
    breaker: Optional[BreakerLike] = None,
    max_retries: int = 2,
    chains: Optional[dict[str, list[str]]] = None,
    max_tool_calls: Optional[int] = None,
) -> ChainResult:
    """
    Run `tool_name`, then each tool in its fallback chain, until one returns
    usable data. Never raises.

    Args:
        tool_name:   Primary tool.
        params:      Primary tool's params (translated for each fallback).
        dispatch:    {tool_name: callable} of implemented tools.
        context:     Optional {"company": ticker, "query": original query}.
        breaker:     Optional circuit breaker (open circuits are skipped).
        max_retries: Retries per tool for retryable errors (network, rate limit).
        chains:      Override FALLBACK_CHAINS (tests).
        max_tool_calls: Optional maximum number of physical calls across the chain.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if max_tool_calls is not None and max_tool_calls < 0:
        raise ValueError("max_tool_calls must be non-negative")

    context = context or {}
    chains = chains if chains is not None else FALLBACK_CHAINS
    chain = [tool_name] + [t for t in chains.get(tool_name, []) if t != tool_name]
    log: list[ToolAttempt] = []
    calls_made = 0

    for idx, tool in enumerate(chain):
        remaining_calls = (
            max_tool_calls - calls_made if max_tool_calls is not None else None
        )
        if remaining_calls is not None and remaining_calls <= 0:
            log.append(ToolAttempt(tool, "skipped_call_budget"))
            break

        fn = dispatch.get(tool)
        if fn is None:
            log.append(ToolAttempt(tool, "skipped_not_implemented"))
            continue
        if breaker is not None and not breaker.allow(tool):
            log.append(ToolAttempt(tool, "skipped_circuit_open"))
            logger.warning("[%s] circuit open — skipping.", tool)
            continue

        tool_params = params if idx == 0 else translate_params(tool_name, tool, params, context)
        tool_retries = (
            min(max_retries, remaining_calls - 1)
            if remaining_calls is not None else max_retries
        )
        gr = with_graceful_degradation(
            tool, _raise_on_error(fn), tool_params, context, tool_retries
        )
        calls_made += gr.attempts

        if gr.success and has_usable_data(tool, gr.result):
            if breaker is not None:
                breaker.record_success(tool)
            log.append(ToolAttempt(
                tool, "success", gr.attempts, params=tool_params,
                attempt_log=gr.attempt_log,
            ))
            return _build_success(tool_name, tool, gr.result, log, chain)

        if gr.success:
            # The tool responded, so clear a half-open probe even though its
            # empty result means the chain should continue to the next source.
            if breaker is not None:
                breaker.record_success(tool)
            log.append(ToolAttempt(
                tool, "no_data", gr.attempts,
                error_message="returned no usable data", params=tool_params,
                attempt_log=gr.attempt_log,
            ))
        else:
            if breaker is not None:
                breaker.record_failure(tool, gr.error_type)
            log.append(ToolAttempt(tool, "failed", gr.attempts, gr.error_type,
                                   gr.error_message, tool_params, gr.attempt_log))

    return _build_degraded(tool_name, log, chain, context)


def _build_success(primary: str, source: str, result: dict,
                   log: list[ToolAttempt], chain: list[str]) -> ChainResult:
    if source == primary:
        return ChainResult(primary, source, result, True, False, False, log, chain, None)

    failed = [a for a in log if a.tool != source and a.status != "success"]
    why = "; ".join(_reason(a) if a.tool == primary else f"{a.tool}: {_reason(a)}" for a in failed)
    note = _SUBSTITUTE_NOTES.get((primary, source))
    if note is None and source == "vector_db_search":
        note = "served from long-term memory and may not reflect the latest data"
    if note is None and source == "web_search":
        note = "web snippets (Tier 4), not verified against filings"
    disclosure = (
        f"`{primary}` was unavailable ({why}). Data was sourced from `{source}` "
        f"({_tier_label(source)}) instead" + (f"; limitation: {note}." if note else ".")
    )
    annotated = {**result, "fallback": {
        "used": True, "primary_tool": primary, "source_tool": source,
        "reason": why, "limitation": note,
    }}
    logger.warning("Fallback used: %s -> %s", primary, source)
    return ChainResult(primary, source, annotated, True, True, False, log, chain, disclosure)


def _build_degraded(primary: str, log: list[ToolAttempt], chain: list[str],
                    context: dict) -> ChainResult:
    errors = [a.error_message for a in log if a.error_message]
    stub = _build_degradation_stub(
        primary, errors[-1] if errors else "all tools in the chain were unavailable",
        context.get("company", ""), context.get("query", ""),
    )
    tried = "; ".join(f"{a.tool}: {_reason(a)}" for a in log)
    stub["degradation_note"] = (
        f"{primary} could not be completed and no fallback produced usable data. Tried: {tried}."
    )
    stub["chain_attempts"] = [
        {"tool": a.tool, "status": a.status, "error": a.error_message} for a in log
    ]
    logger.error("Chain exhausted for %s: %s", primary, tried)
    return ChainResult(primary, None, stub, False, False, True, log, chain,
                       f"[DATA UNAVAILABLE] {stub['degradation_note']}")


# ---------------------------------------------------------------------------
# Output transparency (Day 9 Task 3)
# ---------------------------------------------------------------------------

def format_degradation_notes(chain_results: list[ChainResult]) -> str:
    """Markdown 'Data Availability Notes' section, or '' if everything ran cleanly."""
    lines: list[str] = []
    for cr in chain_results:
        if cr.degraded:
            disclosure = cr.disclosure or cr.result.get(
                "degradation_note", "No fallback produced usable data."
            )
            lines.append(
                f"- ❌ **{cr.primary_tool}** — [DATA UNAVAILABLE]. "
                f"{disclosure.removeprefix('[DATA UNAVAILABLE] ')}"
            )
        elif cr.used_fallback:
            lines.append(f"- ⚠️ {cr.disclosure or f'Fallback used for {cr.primary_tool}.'}")
    if not lines:
        return ""
    return "\n".join(["## Data Availability Notes",
                      "Some requested data could not be retrieved as planned:", ""] + lines) + "\n"


# ---------------------------------------------------------------------------
# Configuration validation (Day 9 Task 2: >= 2 fallbacks per primary tool)
# ---------------------------------------------------------------------------

def validate_chains(chains: Optional[dict[str, list[str]]] = None,
                    dispatch: Optional[dict[str, Callable]] = None) -> list[str]:
    """Return a list of configuration problems (empty list = valid)."""
    chains = chains if chains is not None else FALLBACK_CHAINS
    problems: list[str] = []
    for tool, fallbacks in chains.items():
        if tool not in LOCAL_TOOLS and len(fallbacks) < MIN_FALLBACKS_PER_EXTERNAL_TOOL:
            problems.append(f"{tool}: needs >= {MIN_FALLBACKS_PER_EXTERNAL_TOOL} fallbacks, has {len(fallbacks)}")
        if tool in fallbacks:
            problems.append(f"{tool}: lists itself as a fallback")
        if len(set(fallbacks)) != len(fallbacks):
            problems.append(f"{tool}: duplicate fallbacks")
        for fb in fallbacks:
            if fb not in chains:
                problems.append(f"{tool}: fallback '{fb}' is not a known tool")
            if (tool, fb) not in _TRANSLATORS:
                problems.append(f"{tool} -> {fb}: no parameter translator")
            if dispatch is not None and fb not in dispatch:
                problems.append(f"{tool} -> {fb}: '{fb}' not in dispatch (unimplemented)")
    return problems
