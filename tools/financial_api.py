"""
tools/financial_api.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

Financial data tool — structured statements and metrics via:
    Primary:  Financial Modeling Prep (FMP) — 250 req/day free tier
    Fallback: Alpha Vantage              — 25 req/day free tier

Tier: 2 (weight 0.90) — curated from primary sources with data quality controls.
      Always cross-reference material numbers against sec_filing_search (Tier 1).

Supports: income_statement, balance_sheet, cash_flow,
          key_metrics, financial_ratios, earnings
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FMP_BASE   = "https://financialmodelingprep.com/api/v3"
AV_BASE    = "https://www.alphavantage.co/query"
RATE_DELAY = 0.25   # ~4 req/sec — conservative for free tier

# FMP endpoint map per statement_type
FMP_ENDPOINTS = {
    "income_statement":  "/income-statement/{ticker}",
    "balance_sheet":     "/balance-sheet-statement/{ticker}",
    "cash_flow":         "/cash-flow-statement/{ticker}",
    "key_metrics":       "/key-metrics/{ticker}",
    "financial_ratios":  "/ratios/{ticker}",
    "earnings":          "/earnings/{ticker}",
}

# Alpha Vantage function map per statement_type
AV_FUNCTIONS = {
    "income_statement": "INCOME_STATEMENT",
    "balance_sheet":    "BALANCE_SHEET",
    "cash_flow":        "CASH_FLOW",
    "earnings":         "EARNINGS",
    # key_metrics and financial_ratios not available on AV free tier
}


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _get_json(url: str, timeout: int = 15) -> dict | list:
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
# FMP fetcher
# ---------------------------------------------------------------------------

def _fetch_fmp(
    ticker: str,
    statement_type: str,
    period: str,
    limit: int,
    api_key: str,
) -> list[dict]:
    """Fetch data from FMP and return a list of period records."""
    endpoint = FMP_ENDPOINTS[statement_type].format(ticker=ticker)
    params = urlencode({"period": period, "limit": limit, "apikey": api_key})
    url = f"{FMP_BASE}{endpoint}?{params}"

    logger.info("FMP request: %s %s %s", ticker, statement_type, period)
    data = _get_json(url)

    # FMP returns a list; key_metrics/ratios return a list too
    if isinstance(data, dict) and "Error Message" in data:
        raise RuntimeError(f"FMP error: {data['Error Message']}")
    if not data:
        raise RuntimeError(f"FMP returned empty response for {ticker} {statement_type}")

    return data if isinstance(data, list) else [data]


# ---------------------------------------------------------------------------
# Alpha Vantage fallback fetcher
# ---------------------------------------------------------------------------

def _fetch_alpha_vantage(
    ticker: str,
    statement_type: str,
    period: str,
    api_key: str,
) -> list[dict]:
    """Fetch data from Alpha Vantage. Maps to FMP-compatible field names where possible."""
    av_function = AV_FUNCTIONS.get(statement_type)
    if not av_function:
        raise RuntimeError(
            f"Alpha Vantage does not support '{statement_type}' on the free tier. "
            "No fallback available for this statement type."
        )

    params = urlencode({"function": av_function, "symbol": ticker, "apikey": api_key})
    url = f"{AV_BASE}?{params}"
    logger.info("Alpha Vantage fallback: %s %s", ticker, statement_type)

    data = _get_json(url)

    if "Note" in data:
        raise RuntimeError("Alpha Vantage rate limit hit.")
    if isinstance(data, dict) and "Information" in data:
        raise RuntimeError(f"Alpha Vantage API issue: {data.get('Information')}")

    # AV returns annual/quarterly reports under different keys
    period_key = "annualReports" if period == "annual" else "quarterlyReports"
    if not isinstance(data, dict):
        raise RuntimeError("Alpha Vantage returned unexpected data format")
    reports = data.get(period_key, [])

    if not reports:
        raise RuntimeError(f"Alpha Vantage returned no {period} data for {ticker}")

    # Tag each record with the source
    for r in reports:
        r["_source"] = "alpha_vantage"

    return reports


# ---------------------------------------------------------------------------
# Key metrics assembler (FMP-specific composite)
# ---------------------------------------------------------------------------

def _assemble_key_metrics(raw: list[dict], limit: int) -> list[dict]:
    """
    Select and rename the most analysis-relevant fields from FMP key_metrics.
    Keeps analyst-facing names consistent across periods.
    """
    keep = [
        "date", "period", "revenuePerShare", "netIncomePerShare",
        "peRatio", "priceToBookRatio", "priceToSalesRatio",
        "enterpriseValueOverEBITDA", "evToFreeCashFlow",
        "debtToEquity", "currentRatio", "returnOnEquity",
        "returnOnAssets", "freeCashFlowPerShare",
        "operatingCashFlowPerShare", "earningsYield",
        "dividendYield", "payoutRatio",
    ]
    return [{k: r.get(k) for k in keep if k in r} for r in raw[:limit]]


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------

def financial_data_api(
    ticker: str,
    statement_type: str = "key_metrics",
    period: str = "annual",
    limit: int = 4,
) -> dict:
    """
    Retrieve structured financial statements or metrics for a US-listed company.

    Args:
        ticker:          Stock ticker symbol (e.g. 'MSFT', 'AAPL').
        statement_type:  One of: income_statement | balance_sheet | cash_flow |
                         key_metrics | financial_ratios | earnings
        period:          'annual' or 'quarterly'
        limit:           Number of periods to return (default 4 = 4 years or quarters)

    Returns:
        dict with keys:
            ticker, statement_type, period, records (list of period dicts),
            record_count, data_source, source_tier, source_weight,
            retrieved_at, error
    """
    ticker = ticker.upper().strip()
    if statement_type not in FMP_ENDPOINTS:
        return _error_response(
            ticker, statement_type,
            f"Unsupported statement_type '{statement_type}'. "
            f"Choose from: {list(FMP_ENDPOINTS.keys())}"
        )

    fmp_key = os.getenv("FMP_API_KEY", "")
    av_key  = os.getenv("ALPHA_VANTAGE_KEY", "")

    records: list[dict] = []
    data_source = ""
    error_msg: Optional[str] = None

    # --- Primary: FMP ---
    if fmp_key:
        try:
            raw = _fetch_fmp(ticker, statement_type, period, limit, fmp_key)
            records = (
                _assemble_key_metrics(raw, limit)
                if statement_type == "key_metrics"
                else raw[:limit]
            )
            data_source = "Financial Modeling Prep"
            logger.info("FMP success: %d %s records for %s", len(records), statement_type, ticker)
        except RuntimeError as e:
            logger.warning("FMP failed for %s %s: %s — trying Alpha Vantage", ticker, statement_type, e)
            error_msg = str(e)
    else:
        logger.warning("FMP_API_KEY not set — skipping FMP, trying Alpha Vantage")
        error_msg = "FMP_API_KEY not configured."

    # --- Fallback: Alpha Vantage ---
    if not records:
        if av_key:
            try:
                records = _fetch_alpha_vantage(ticker, statement_type, period, av_key)[:limit]
                data_source = "Alpha Vantage"
                error_msg = None
                logger.info("Alpha Vantage fallback success: %d records for %s", len(records), ticker)
            except RuntimeError as e:
                error_msg = f"FMP: {error_msg} | Alpha Vantage: {e}"
                logger.error("Both FMP and AV failed for %s: %s", ticker, error_msg)
        else:
            error_msg = (error_msg or "") + " ALPHA_VANTAGE_KEY not configured."
            logger.error("No API keys configured. Set FMP_API_KEY or ALPHA_VANTAGE_KEY.")

    if not records:
        return _error_response(ticker, statement_type, error_msg or "No data retrieved.")

    return {
        "ticker":         ticker,
        "statement_type": statement_type,
        "period":         period,
        "records":        records,
        "record_count":   len(records),
        "data_source":    data_source,
        "source_tier":    2,
        "source_weight":  0.90,
        "retrieved_at":   _now(),
        "error":          None,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error_response(ticker: str, statement_type: str, message: str) -> dict:
    logger.error("financial_data_api error [%s %s]: %s", ticker, statement_type, message)
    return {
        "ticker":         ticker,
        "statement_type": statement_type,
        "period":         None,
        "records":        [],
        "record_count":   0,
        "data_source":    None,
        "source_tier":    2,
        "source_weight":  0.90,
        "retrieved_at":   _now(),
        "error":          message,
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
