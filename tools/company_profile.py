"""
tools/company_profile.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

Company profile tool — retrieves company metadata from FMP.
    Primary:  Financial Modeling Prep /profile endpoint
    Fallback: FMP /quote endpoint (lighter, fewer fields)

Tier: 2 (weight 0.90) — curated from primary sources.
Use as the FIRST tool call on any new company — provides context
(sector, industry, market cap, executives) that anchors all downstream analysis.
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

# Updated to the modern, non-legacy stable routing pipeline
FMP_BASE = "https://financialmodelingprep.com/stable"
RATE_DELAY = 0.25


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
# FMP /profile (primary)
# ---------------------------------------------------------------------------

def _fetch_profile(ticker: str, api_key: str) -> dict:
    # Modernized URL signature converts ticker from a path variable into a query parameter
    params = urlencode({"symbol": ticker, "apikey": api_key})
    url = f"{FMP_BASE}/profile?{params}"
    
    logger.info("FMP /profile: %s", ticker)
    data = _get_json(url)

    if not data or not isinstance(data, list):
        raise RuntimeError(f"FMP /profile returned no data for {ticker}.")

    p = data[0]

    # Normalise into a clean, analysis-ready schema
    return {
        "ticker":           p.get("symbol", ticker),
        "company_name":     p.get("companyName"),
        "exchange":         p.get("exchangeShortName"),
        "sector":           p.get("sector"),
        "industry":         p.get("industry"),
        "market_cap":       p.get("mktCap"),
        "price":            p.get("price"),
        "beta":             p.get("beta"),
        "employees":        p.get("fullTimeEmployees"),
        "ceo":              p.get("ceo"),
        "founded":          p.get("ipoDate", "")[:4] or None,
        "headquarters": {
            "city":    p.get("city"),
            "state":   p.get("state"),
            "country": p.get("country"),
        },
        "description":      p.get("description"),
        "website":          p.get("website"),
        "isin":             p.get("isin"),
        "currency":         p.get("currency"),
        "is_etf":           p.get("isEtf", False),
        "is_actively_trading": p.get("isActivelyTrading", True),
        "_raw_source":      "FMP /profile",
    }


# ---------------------------------------------------------------------------
# FMP /quote (fallback — fewer fields but more reliable on free tier)
# ---------------------------------------------------------------------------

def _fetch_quote_fallback(ticker: str, api_key: str) -> dict:
    # Modernized URL signature converts ticker from a path variable into a query parameter
    params = urlencode({"symbol": ticker, "apikey": api_key})
    url = f"{FMP_BASE}/quote?{params}"
    
    logger.info("FMP /quote fallback: %s", ticker)
    data = _get_json(url)

    if not data or not isinstance(data, list):
        raise RuntimeError(f"FMP /quote returned no data for {ticker}.")

    q = data[0]
    return {
        "ticker":       q.get("symbol", ticker),
        "company_name": q.get("name"),
        "exchange":     q.get("exchange"),
        "sector":       None,        # not available in /quote
        "industry":     None,
        "market_cap":   q.get("marketCap"),
        "price":        q.get("price"),
        "beta":         None,
        "employees":    None,
        "ceo":          None,
        "founded":      None,
        "headquarters": {"city": None, "state": None, "country": None},
        "description":  None,
        "website":      None,
        "isin":         None,
        "currency":     None,
        "is_etf":       False,
        "is_actively_trading": True,
        "_raw_source":  "FMP /quote (fallback — limited fields)",
    }


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------

def company_profile(ticker: str, include_fields: Optional[list] = None, **kwargs) -> dict:
    """
    Retrieve company metadata for a US-listed company.
    Use as the first tool call when beginning research on any new company.

    Args:
        ticker: Stock ticker symbol (e.g. 'MSFT', 'AAPL').
        include_fields: (Ignored) Subset of fields to return. Kept for schema compatibility.
        **kwargs: Additional parameters (ignored).

    Returns:
        dict with keys:
            ticker, company_name, exchange, sector, industry,
            market_cap, price, beta, employees, ceo, founded,
            headquarters, description, website, isin, currency,
            source_tier, source_weight, retrieved_at, error
    """
    if include_fields is None:
        include_fields = []
    ticker = ticker.upper().strip()
    api_key = os.getenv("FMP_API_KEY", "")

    if not api_key:
        return _error_response(ticker, "FMP_API_KEY not configured.")

    profile: Optional[dict] = None
    error_msg: Optional[str] = None

    # Primary: /profile endpoint
    try:
        profile = _fetch_profile(ticker, api_key)
        logger.info("company_profile OK: %s — %s", ticker, profile.get("company_name"))
    except RuntimeError as e:
        logger.warning("FMP /profile failed for %s: %s — trying /quote fallback", ticker, e)
        error_msg = str(e)

    # Fallback: /quote endpoint
    if not profile:
        try:
            profile = _fetch_quote_fallback(ticker, api_key)
            error_msg = None
            logger.info("company_profile fallback OK: %s", ticker)
        except RuntimeError as e:
            error_msg = f"Profile: {error_msg} | Quote fallback: {e}"
            logger.error("Both FMP endpoints failed for %s: %s", ticker, error_msg)

    if not profile:
        return _error_response(ticker, error_msg or "No data retrieved.")

    return {
        **profile,
        "source_tier":   2,
        "source_weight": 0.90,
        "retrieved_at":  _now(),
        "error":         None,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error_response(ticker: str, message: str) -> dict:
    logger.error("company_profile error [%s]: %s", ticker, message)
    return {
        "ticker":        ticker,
        "company_name":  None,
        "exchange":      None,
        "sector":        None,
        "industry":      None,
        "market_cap":    None,
        "price":         None,
        "beta":          None,
        "employees":     None,
        "ceo":           None,
        "founded":       None,
        "headquarters":  {"city": None, "state": None, "country": None},
        "description":   None,
        "website":       None,
        "isin":          None,
        "currency":      None,
        "is_etf":        False,
        "is_actively_trading": None,
        "source_tier":   2,
        "source_weight": 0.90,
        "retrieved_at":  _now(),
        "error":         message,
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()