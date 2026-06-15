"""
tools/peer_comparison.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

Peer comparison tool — identifies industry peers and builds a comparative
metrics matrix across the peer group.

    Primary:  FMP /stock_peers    → peer identification (if explicit_peers not provided)
              FMP /key-metrics    → per-peer metric retrieval
    Fallback: FMP /profile        → basic metrics when key-metrics unavailable

Tier: 2 (weight 0.90) — curated financial data.
Critical for Challenge 4 (comparative analysis) and any competitive
positioning section.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

FMP_BASE   = "https://financialmodelingprep.com/stable"
RATE_DELAY = 0.25   # respect FMP rate limits between per-peer calls

# Default metrics returned when caller doesn't specify
DEFAULT_METRICS = [
    "revenue_growth",
    "gross_margin",
    "operating_margin",
    "pe_ratio",
    "ev_ebitda",
    "roe",
    "debt_to_equity",
    "market_cap",
]

# Map our metric names → FMP key-metrics field names
METRIC_FIELD_MAP = {
    "revenue_growth":     "revenueGrowth",
    "gross_margin":       "grossProfitMargin",
    "operating_margin":   "operatingProfitMargin",
    "net_margin":         "netProfitMargin",
    "pe_ratio":           "peRatio",
    "ev_ebitda":          "enterpriseValueOverEBITDA",
    "ev_fcf":             "evToFreeCashFlow",
    "roe":                "returnOnEquity",
    "roa":                "returnOnAssets",
    "debt_to_equity":     "debtToEquity",
    "current_ratio":      "currentRatio",
    "market_cap":         "marketCap",
    "fcf_per_share":      "freeCashFlowPerShare",
    "dividend_yield":     "dividendYield",
    "price_to_book":      "priceToBookRatio",
    "price_to_sales":     "priceToSalesRatio",
}


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _get_json(url: str, timeout: int = 15) -> list | dict:
    time.sleep(RATE_DELAY)
    req = Request(url, headers={"User-Agent": "ARA-1/1.0 ara1@project.com"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {url}") from e
    except URLError as e:
        raise RuntimeError(f"Network error: {e.reason}") from e


# ---------------------------------------------------------------------------
# Peer discovery
# ---------------------------------------------------------------------------

def _get_peers(ticker: str, num_peers: int, api_key: str) -> list[str]:
    """
    Fetch industry peers from FMP /stock_peers.
    Returns a list of ticker symbols (excluding the primary company).
    """
    url = f"{FMP_BASE}/stock_peers?{urlencode({'symbol': ticker, 'apikey': api_key})}"
    logger.info("FMP /stock_peers: %s", ticker)
    data = _get_json(url)

    if not data:
        raise RuntimeError(f"No peers found for {ticker}.")

    # FMP returns: [{"symbol": "MSFT", "peersList": ["AAPL", "GOOGL", ...]}]
    record = data[0] if isinstance(data, list) else data
    peers  = record.get("peersList", [])

    if not peers:
        raise RuntimeError(f"FMP returned empty peers list for {ticker}.")

    # Exclude the primary ticker from its own peer list (sometimes included)
    peers = [p for p in peers if p.upper() != ticker.upper()]

    return peers[:num_peers]


# ---------------------------------------------------------------------------
# Per-company metric retrieval
# ---------------------------------------------------------------------------

def _get_metrics_for_ticker(
    ticker: str,
    requested_metrics: list[str],
    api_key: str,
) -> dict:
    """
    Fetch key metrics for a single ticker from FMP /key-metrics.
    Returns a flat dict of {our_metric_name: value}.
    Falls back to /profile for market_cap if key-metrics is empty.
    """
    url = f"{FMP_BASE}/key-metrics/{ticker}?{urlencode({'limit': 1, 'apikey': api_key})}"
    result: dict[str, Optional[float]] = {m: None for m in requested_metrics}

    try:
        data = _get_json(url)
        if data and isinstance(data, list):
            record = data[0]
            for metric in requested_metrics:
                fmp_field = METRIC_FIELD_MAP.get(metric)
                if fmp_field and fmp_field in record:
                    val = record[fmp_field]
                    result[metric] = round(float(val), 4) if val is not None else None
    except RuntimeError as e:
        logger.warning("key-metrics failed for %s: %s", ticker, e)

    # Fallback for market_cap via /profile (always available)
    if "market_cap" in requested_metrics and result.get("market_cap") is None:
        try:
            prof_url = f"{FMP_BASE}/profile/{ticker}?{urlencode({'apikey': api_key})}"
            prof = _get_json(prof_url)
            if prof and isinstance(prof, list):
                mkt_cap = prof[0].get("mktCap")
                if mkt_cap:
                    result["market_cap"] = round(float(mkt_cap), 0)
        except RuntimeError:
            pass

    return result


# ---------------------------------------------------------------------------
# Comparison analysis
# ---------------------------------------------------------------------------

def _build_comparison_analysis(
    matrix: dict[str, dict],
    metrics: list[str],
) -> dict:
    """
    Derive relative rankings and sector averages from the comparison matrix.
    Returns a dict with per-metric analysis (sector avg, leader, laggard).
    """
    analysis: dict[str, dict] = {}

    for metric in metrics:
        values = {
            ticker: row[metric]
            for ticker, row in matrix.items()
            if row.get(metric) is not None
        }
        if not values:
            analysis[metric] = {"sector_avg": None, "leader": None, "laggard": None}
            continue

        tickers   = list(values.keys())
        vals      = list(values.values())
        sector_avg = round(sum(vals) / len(vals), 4)

        # For debt_to_equity: lower is better; for all others: higher is better
        reverse = metric != "debt_to_equity"
        ranked  = sorted(values.items(), key=lambda x: x[1], reverse=reverse)

        analysis[metric] = {
            "sector_avg": sector_avg,
            "leader":     ranked[0][0],
            "leader_val": ranked[0][1],
            "laggard":    ranked[-1][0],
            "laggard_val": ranked[-1][1],
        }

    return analysis


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------

def peer_comparison(
    ticker: str,
    num_peers: int = 5,
    metrics: Optional[list[str]] = None,
    explicit_peers: Optional[list[str]] = None,
    **kwargs,
) -> dict:
    """
    Identify industry peers and return a comparative metrics matrix.

    Args:
        ticker:         Primary company ticker (e.g. 'MSFT').
        num_peers:      Number of peers to include (2–10, default 5).
        metrics:        List of metric names to compare. Defaults to 8 standard metrics.
        explicit_peers: Optional explicit list of specific ticker symbols to use as peers.

    Returns:
        dict with keys:
            ticker, peers, metrics, comparison_matrix, analysis,
            source_tier, source_weight, data_source, retrieved_at, error
    """
    ticker    = ticker.upper().strip()
    num_peers = max(2, min(10, num_peers))
    metrics   = metrics or DEFAULT_METRICS

    # Validate metric names
    unknown = [m for m in metrics if m not in METRIC_FIELD_MAP]
    if unknown:
        logger.warning("Unknown metrics requested (will return None): %s", unknown)

    api_key = os.getenv("FMP_API_KEY", "")
    if not api_key:
        return _error_response(ticker, "FMP_API_KEY not configured.")

    # Step 1: Discover or Clean Peers
    if explicit_peers:
        peers = [p.upper().strip() for p in explicit_peers if p.upper().strip() != ticker]
        if len(peers) > num_peers:
            peers = peers[:num_peers]
        logger.info("Using explicit peers for %s: %s", ticker, peers)
        data_source_label = "Explicit Peers + FMP /key-metrics"
    else:
        try:
            peers = _get_peers(ticker, num_peers, api_key)
            logger.info("Peers for %s: %s", ticker, peers)
            data_source_label = "FMP /stock_peers + /key-metrics"
        except RuntimeError as e:
            return _error_response(ticker, f"Peer discovery failed: {e}")

    # Step 2: build comparison matrix — primary + each peer
    all_tickers   = [ticker] + peers
    matrix: dict[str, dict] = {}
    failed_tickers: list[str] = []

    for t in all_tickers:
        try:
            matrix[t] = _get_metrics_for_ticker(t, metrics, api_key)
        except Exception as e:
            logger.warning("Metrics fetch failed for %s: %s", t, e)
            matrix[t] = {m: None for m in metrics}
            failed_tickers.append(t)

    # Step 3: relative analysis
    analysis = _build_comparison_analysis(matrix, metrics)

    return {
        "ticker":            ticker,
        "peers":             peers,
        "metrics":           metrics,
        "comparison_matrix": matrix,
        "analysis":          analysis,
        "failed_tickers":    failed_tickers,
        "data_source":       data_source_label,
        "source_tier":       2,
        "source_weight":     0.90,
        "retrieved_at":      _now(),
        "error":             None,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error_response(ticker: str, message: str) -> dict:
    logger.error("peer_comparison error [%s]: %s", ticker, message)
    return {
        "ticker":            ticker,
        "peers":             [],
        "metrics":           [],
        "comparison_matrix": {},
        "analysis":          {},
        "failed_tickers":    [],
        "data_source":       None,
        "source_tier":       2,
        "source_weight":     0.90,
        "retrieved_at":      _now(),
        "error":             message,
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()