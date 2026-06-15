"""
tools/earnings.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

Earnings call transcript tool.
    Primary:  Financial Modeling Prep /earning_call_transcript endpoint (stable base)
    Fallback: FMP /earning_call (summary) endpoint

Tier: 3 (weight 0.75) — direct management voice, valuable for forward
      guidance and qualitative insight, but subject to spin.
      Always cross-reference numerical claims against Tier 1/2 sources.

Returns: full transcript text with speaker attribution + structured
         metadata (date, fiscal quarter, speakers identified).
"""

import json
import logging
import os
import re
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
FMP_BASE   = "https://financialmodelingprep.com/stable"   # ✅ updated to stable
AV_BASE    = "https://www.alphavantage.co/query"
RATE_DELAY = 0.25


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------
def _get_json(url: str, timeout: int = 20) -> list | dict:
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
# Speaker extraction
# ---------------------------------------------------------------------------
def _extract_speakers(content: str) -> list[str]:
    """Extract unique speaker names from transcript content."""
    pattern = re.compile(r'^([A-Z][a-zA-Z\s\(\)]+?):', re.MULTILINE)
    matches = pattern.findall(content)
    seen = set()
    speakers = []
    for name in matches:
        name = name.strip()
        if name and name not in seen and len(name) < 60:
            seen.add(name)
            speakers.append(name)
    return speakers[:20]


# ---------------------------------------------------------------------------
# FMP primary — full transcript
# ---------------------------------------------------------------------------
def _fetch_full_transcript(
    ticker: str,
    quarter: str,
    year: int,
    api_key: str,
) -> dict:
    params = urlencode({"quarter": quarter[-1], "year": year, "apikey": api_key})
    url = f"{FMP_BASE}/earning_call_transcript/{ticker}?{params}"
    logger.info("FMP /earning_call_transcript: %s %s %d", ticker, quarter, year)

    data = _get_json(url)
    if not data:
        raise RuntimeError(f"No transcript found for {ticker} {quarter} {year}.")

    record = data[0] if isinstance(data, list) else data
    content = record.get("content", "")
    if not content:
        raise RuntimeError(f"Empty transcript content for {ticker} {quarter} {year}.")

    return {
        "content": content,
        "date": record.get("date", ""),
        "quarter": record.get("quarter", quarter),
        "year": record.get("year", year),
    }


# ---------------------------------------------------------------------------
# FMP fallback — earnings call summary
# ---------------------------------------------------------------------------
def _fetch_summary_fallback(ticker: str, api_key: str) -> dict:
    params = urlencode({"apikey": api_key})
    url = f"{FMP_BASE}/earning_call/{ticker}?{params}"
    logger.info("FMP /earning_call fallback: %s", ticker)

    data = _get_json(url)
    if not data or not isinstance(data, list):
        raise RuntimeError(f"FMP /earning_call returned no data for {ticker}.")

    record = data[0]
    content = (
        f"Earnings call summary for {ticker}\n"
        f"Date: {record.get('date', 'N/A')}\n"
        f"EPS Estimated: {record.get('epsEstimated', 'N/A')}\n"
        f"EPS Actual: {record.get('eps', 'N/A')}\n"
        f"Revenue Estimated: {record.get('revenueEstimated', 'N/A')}\n"
        f"Revenue Actual: {record.get('revenue', 'N/A')}\n"
        f"Note: Full transcript not available via fallback endpoint."
    )
    return {
        "content": content,
        "date": record.get("date", ""),
        "quarter": "",
        "year": "",
    }


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------
def earnings_transcript(
    ticker: str,
    quarter: str = "Q4",
    year: Optional[int] = None,
    **kwargs  # for schema compatibility
) -> dict:
    """
    Retrieve the earnings call transcript for a company.

    Args:
        ticker:  Stock ticker symbol (e.g. 'AAPL', 'MSFT').
        quarter: Fiscal quarter — 'Q1', 'Q2', 'Q3', or 'Q4'.
        year:    Calendar year (e.g. 2024). Defaults to current year.

    Returns:
        dict with keys:
            ticker, quarter, year, transcript_date, content, text_length,
            speakers, source_tier, source_weight, data_source,
            retrieved_at, error
    """
    ticker = ticker.upper().strip()
    quarter = quarter.upper().strip()
    if year is None:
        year = datetime.now(timezone.utc).year

    if quarter not in ("Q1", "Q2", "Q3", "Q4"):
        return _error_response(
            ticker, quarter, year,
            f"Invalid quarter '{quarter}'. Must be Q1, Q2, Q3, or Q4."
        )

    api_key = os.getenv("FMP_API_KEY", "")
    if not api_key:
        return _error_response(ticker, quarter, year, "FMP_API_KEY not configured.")

    transcript_data: Optional[dict] = None
    data_source = ""
    error_msg: Optional[str] = None

    # --- Primary: full transcript ---
    try:
        transcript_data = _fetch_full_transcript(ticker, quarter, year, api_key)
        data_source = "FMP /earning_call_transcript"
        logger.info("Transcript fetched: %s %s %d (%d chars)",
                    ticker, quarter, year, len(transcript_data["content"]))
    except RuntimeError as e:
        logger.warning("Full transcript failed: %s — trying fallback", e)
        error_msg = str(e)

    # --- Fallback: earnings summary ---
    if not transcript_data:
        try:
            transcript_data = _fetch_summary_fallback(ticker, api_key)
            data_source = "FMP /earning_call (summary fallback)"
            error_msg = None
            logger.info("Earnings summary fallback OK: %s", ticker)
        except RuntimeError as e:
            error_msg = f"Full transcript: {error_msg} | Summary fallback: {e}"
            logger.error("Both FMP endpoints failed for %s: %s", ticker, error_msg)

    if not transcript_data:
        return _error_response(ticker, quarter, year, error_msg or "No data retrieved.")

    content = transcript_data["content"]
    speakers = _extract_speakers(content)

    return {
        "ticker":          ticker,
        "quarter":         quarter,
        "year":            year,
        "transcript_date": transcript_data["date"],
        "content":         content,
        "text_length":     len(content),
        "speakers":        speakers,
        "data_source":     data_source,
        "source_tier":     3,
        "source_weight":   0.75,
        "retrieved_at":    _now(),
        "error":           None,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _error_response(ticker: str, quarter: str, year: int, message: str) -> dict:
    logger.error("earnings_transcript error [%s %s %d]: %s", ticker, quarter, year, message)
    return {
        "ticker":          ticker,
        "quarter":         quarter,
        "year":            year,
        "transcript_date": None,
        "content":         "",
        "text_length":     0,
        "speakers":        [],
        "data_source":     None,
        "source_tier":     3,
        "source_weight":   0.75,
        "retrieved_at":    _now(),
        "error":           message,
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()