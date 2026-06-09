"""
tools/sec_edgar.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

SEC EDGAR tool — fetches official regulatory filings + full text via the free
EDGAR EFTS API. No API key required. Rate limit: 10 req/sec.

Source:  https://efts.sec.gov / https://data.sec.gov
Tier:    1 (weight 1.00) — ground truth for all numerical financial data
Covers:  10-K (annual), 10-Q (quarterly), 8-K (material events), DEF 14A (proxy)

Returns: filing metadata + filing_text (full plain-text content stripped of HTML)
         so the RAG pipeline can chunk, embed, and store directly.
"""

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUBMISSIONS_BASE = "https://data.sec.gov/submissions"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
HEADERS = {
    # SEC requires a descriptive User-Agent identifying your app and contact
    "User-Agent": "ARA-1 Research Agent ara1@project.com",
    "Accept": "application/json",
}
RATE_LIMIT_DELAY = 0.12          # 10 req/sec → 100ms min; use 120ms for safety
TEXT_CHAR_LIMIT = 100_000        # Truncate very large filings to stay within context

FILING_TYPE_MAP = {
    "10-K":    "10-K",
    "10-Q":    "10-Q",
    "8-K":     "8-K",
    "DEF 14A": "DEF 14A",
}

# Session-level CIK cache (ticker → zero-padded CIK string)
_cik_cache: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Internal HTTP helper
# ---------------------------------------------------------------------------

def _get(url: str, accept: str = "application/json", timeout: int = 20) -> bytes:
    """Make a rate-limited GET request and return raw bytes."""
    time.sleep(RATE_LIMIT_DELAY)
    req = Request(url, headers={**HEADERS, "Accept": accept})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} from EDGAR: {url}") from e
    except URLError as e:
        raise RuntimeError(f"Network error reaching EDGAR: {e.reason}") from e


# ---------------------------------------------------------------------------
# CIK resolution
# ---------------------------------------------------------------------------

def _ticker_to_cik(ticker: str) -> str:
    """Resolve ticker → zero-padded 10-digit CIK using SEC's official list."""
    if ticker in _cik_cache:
        return _cik_cache[ticker]

    data = json.loads(_get(TICKERS_URL))
    for entry in data.values():
        if entry.get("ticker", "").upper() == ticker.upper():
            cik = str(entry["cik_str"]).zfill(10)
            _cik_cache[ticker] = cik
            logger.info("CIK resolved: %s → %s", ticker, cik)
            return cik

    raise ValueError(f"Ticker '{ticker}' not found in SEC EDGAR company list.")


# ---------------------------------------------------------------------------
# Filing discovery
# ---------------------------------------------------------------------------

def _find_filings(cik: str, form_type: str, year: Optional[int] = None) -> list[dict]:
    """
    Return a list of filing metadata dicts for a given CIK and form type,
    sorted most-recent first. Optionally filtered to a specific year.

    Each dict contains: form_type, filing_date, accession_number,
                        primary_document, document_url
    """
    raw = json.loads(_get(f"{SUBMISSIONS_BASE}/CIK{cik}.json"))
    recent = raw.get("filings", {}).get("recent", {})

    forms      = recent.get("form", [])
    dates      = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    docs       = recent.get("primaryDocument", [])

    results = []
    for i, form in enumerate(forms):
        if form != form_type:
            continue
        filing_year = int(dates[i][:4]) if dates[i] else None
        if year and filing_year != year:
            continue
        acc_clean = accessions[i].replace("-", "")
        results.append({
            "form_type":        form,
            "filing_date":      dates[i],
            "accession_number": accessions[i],
            "primary_document": docs[i],
            "document_url": (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{acc_clean}/{docs[i]}"
            ),
        })

    return results  # already in reverse-chronological order from EDGAR


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def _fetch_filing_text(document_url: str) -> str:
    """
    Fetch and extract plain text from an SEC filing HTML/XML page.

    Strategy:
      1. Fetch the raw HTML bytes.
      2. Strip <script> and <style> blocks entirely.
      3. Strip all remaining HTML tags.
      4. Collapse whitespace.
      5. Truncate to TEXT_CHAR_LIMIT to stay within LLM context budgets.
    """
    try:
        html = _get(document_url, accept="text/html").decode("utf-8", errors="replace")
    except RuntimeError as e:
        logger.warning("Could not fetch filing text from %s: %s", document_url, e)
        return ""

    # Remove scripts and styles (content not useful for analysis)
    html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>",  " ", html, flags=re.DOTALL | re.IGNORECASE)

    # Strip all remaining tags
    text = re.sub(r"<[^>]+>", " ", html)

    # Collapse whitespace and strip
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) > TEXT_CHAR_LIMIT:
        logger.info(
            "Filing text truncated from %d to %d chars (context limit).",
            len(text), TEXT_CHAR_LIMIT
        )
        text = text[:TEXT_CHAR_LIMIT]

    return text


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------

def sec_filing_search(
    ticker: str,
    filing_type: str = "10-K",
    year: Optional[int] = None,
) -> dict:
    """
    Search and retrieve SEC EDGAR filings for a US publicly traded company.
    Returns filing metadata AND full plain-text content for RAG ingestion.

    Args:
        ticker:       Stock ticker symbol (e.g. 'MSFT', 'AAPL').
        filing_type:  One of '10-K', '10-Q', '8-K', 'DEF 14A'.
        year:         Optional — retrieve filing from a specific year.
                      If None, returns the most recent filing.

    Returns:
        dict with keys:
            ticker, cik, form_type, filing_date, accession_number,
            document_url, filing_text, text_length,
            source_tier, source_weight, retrieved_at, error
    """
    ticker = ticker.upper().strip()
    form = FILING_TYPE_MAP.get(filing_type, filing_type)
    logger.info("SEC EDGAR: ticker=%s form=%s year=%s", ticker, form, year)

    # Step 1: Resolve ticker to CIK
    try:
        cik = _ticker_to_cik(ticker)
    except ValueError as e:
        return _error_response(ticker, form, str(e))
    except RuntimeError as e:
        return _error_response(ticker, form, f"Network error during CIK lookup: {e}")

    # Step 2: Find matching filings
    try:
        filings = _find_filings(cik, form, year=year)
    except RuntimeError as e:
        return _error_response(ticker, form, f"Failed to retrieve filing index: {e}")

    if not filings:
        msg = (
            f"No {form} filings found for {ticker} (CIK {cik})"
            + (f" in year {year}." if year else ".")
        )
        return _error_response(ticker, form, msg)

    # Step 3: Use the most recent matching filing
    filing = filings[0]
    if len(filings) > 1:
        logger.info(
            "Found %d %s filings for %s — using most recent (%s).",
            len(filings), form, ticker, filing["filing_date"]
        )

    # Step 4: Fetch the full filing text
    logger.info("Fetching filing text from %s", filing["document_url"])
    filing_text = _fetch_filing_text(filing["document_url"])

    return {
        "ticker":           ticker,
        "cik":              cik,
        "form_type":        form,
        "filing_date":      filing["filing_date"],
        "accession_number": filing["accession_number"],
        "document_url":     filing["document_url"],
        "filing_text":      filing_text,                # ← full text for RAG pipeline
        "text_length":      len(filing_text),
        "source_tier":      1,
        "source_weight":    1.00,
        "retrieved_at":     _now(),
        "error":            None,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error_response(ticker: str, form: str, message: str) -> dict:
    logger.error("sec_filing_search error [%s %s]: %s", ticker, form, message)
    return {
        "ticker":           ticker,
        "form_type":        form,
        "filing_date":      None,
        "accession_number": None,
        "document_url":     None,
        "filing_text":      "",
        "text_length":      0,
        "source_tier":      1,
        "source_weight":    1.00,
        "retrieved_at":     _now(),
        "error":            message,
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

