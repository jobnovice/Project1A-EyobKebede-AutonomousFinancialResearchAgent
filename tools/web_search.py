"""
tools/web_search.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

Web search tool — current news, analyst commentary, recent events.
    Primary:  Tavily API  — 1,000 req/month free tier
    Fallback: SerpAPI     — 100 req/month free tier

Tier: 4 (weight 0.60) — professional journalism with editorial oversight.
      Never use as sole source for numerical claims. Always cross-reference
      material numbers against Tier 1 (SEC) or Tier 2 (financial_data_api).
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
TAVILY_BASE = "https://api.tavily.com/search"
SERP_BASE   = "https://serpapi.com/search"
RATE_DELAY  = 0.5   # conservative — both services have monthly caps

# Map schema date_range values to Tavily's days parameter
DATE_RANGE_DAYS = {
    "past_day":   1,
    "past_week":  7,
    "past_month": 30,
    "past_year":  365,
    "any":        None,
}


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _post_json(url: str, payload: dict, timeout: int = 20) -> dict:
    time.sleep(RATE_DELAY)
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        url, data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "ARA-1/1.0 ara1@project.com",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {url}") from e
    except URLError as e:
        raise RuntimeError(f"Network error: {e.reason}") from e


def _get_json(url: str, timeout: int = 20) -> dict:
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
# Tavily (primary)
# ---------------------------------------------------------------------------

def _fetch_tavily(
    query: str,
    num_results: int,
    date_range: str,
    api_key: str,
) -> list[dict]:
    payload = {
        "api_key":              api_key,
        "query":                query,
        "search_depth":         "advanced",
        "max_results":          num_results,
        "include_answer":       False,
        "include_raw_content":  False,
    }
    days = DATE_RANGE_DAYS.get(date_range)
    if days:
        payload["days"] = days

    logger.info("Tavily search: %s | range=%s", query[:80], date_range)
    data = _post_json(TAVILY_BASE, payload)

    results = data.get("results", [])
    if not results:
        raise RuntimeError("Tavily returned no results.")

    return [
        {
            "title":     r.get("title", ""),
            "url":       r.get("url", ""),
            "snippet":   r.get("content", "")[:500],
            "score":     round(r.get("score", 0.0), 3),
            "published": r.get("published_date", None),
            "source":    "tavily",
        }
        for r in results
    ]


# ---------------------------------------------------------------------------
# SerpAPI fallback
# ---------------------------------------------------------------------------

def _fetch_serpapi(
    query: str,
    num_results: int,
    date_range: str,
    api_key: str,
) -> list[dict]:
    # Map date_range to SerpAPI tbs parameter
    tbs_map = {
        "past_day":   "qdr:d",
        "past_week":  "qdr:w",
        "past_month": "qdr:m",
        "past_year":  "qdr:y",
        "any":        None,
    }
    params = {
        "q":        query,
        "api_key":  api_key,
        "engine":   "google",
        "num":      num_results,
        "hl":       "en",
    }
    tbs = tbs_map.get(date_range)
    if tbs:
        params["tbs"] = tbs

    url = f"{SERP_BASE}?{urlencode(params)}"
    logger.info("SerpAPI fallback: %s", query[:80])
    data = _get_json(url)

    organic = data.get("organic_results", [])
    if not organic:
        raise RuntimeError("SerpAPI returned no organic results.")

    return [
        {
            "title":     r.get("title", ""),
            "url":       r.get("link", ""),
            "snippet":   r.get("snippet", "")[:500],
            "score":     None,
            "published": r.get("date", None),
            "source":    "serpapi",
        }
        for r in organic[:num_results]
    ]


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------

def web_search(
    query: str,
    num_results: int = 10,
    date_range: str = "any",
    search_type: str = "news",
    **kwargs
) -> dict:
    """
    Search the web for current news, analyst commentary, and recent events.

    Args:
        query:       Specific search query including company name and topic.
        num_results: Number of results to return (1–20, default 10).
        date_range:  One of: past_day | past_week | past_month | past_year | any
        search_type: (Ignored) One of 'news', 'general', 'financial' – kept for schema compatibility.
        **kwargs:    Additional parameters (ignored).

    Returns:
        dict with keys:
            query, results (list of result dicts), result_count,
            data_source, source_tier, source_weight, retrieved_at, error
    """
    num_results = max(1, min(20, num_results))
    if date_range not in DATE_RANGE_DAYS:
        date_range = "any"

    tavily_key = os.getenv("TAVILY_API_KEY", "")
    serp_key   = os.getenv("SERPAPI_KEY", "")

    results: list[dict] = []
    data_source = ""
    error_msg: Optional[str] = None

    # --- Primary: Tavily ---
    if tavily_key:
        try:
            results = _fetch_tavily(query, num_results, date_range, tavily_key)
            data_source = "Tavily"
            logger.info("Tavily: %d results for '%s'", len(results), query[:60])
        except RuntimeError as e:
            logger.warning("Tavily failed: %s — trying SerpAPI", e)
            error_msg = str(e)
    else:
        logger.warning("TAVILY_API_KEY not set — trying SerpAPI")
        error_msg = "TAVILY_API_KEY not configured."

    # --- Fallback: SerpAPI ---
    if not results:
        if serp_key:
            try:
                results = _fetch_serpapi(query, num_results, date_range, serp_key)
                data_source = "SerpAPI"
                error_msg = None
                logger.info("SerpAPI: %d results for '%s'", len(results), query[:60])
            except RuntimeError as e:
                error_msg = f"Tavily: {error_msg} | SerpAPI: {e}"
                logger.error("Both search APIs failed: %s", error_msg)
        else:
            error_msg = (error_msg or "") + " SERPAPI_KEY not configured."

    if not results:
        return _error_response(query, error_msg or "No results retrieved.")

    return {
        "query":        query,
        "results":      results,
        "result_count": len(results),
        "data_source":  data_source,
        "source_tier":  4,
        "source_weight": 0.60,
        "retrieved_at": _now(),
        "error":        None,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error_response(query: str, message: str) -> dict:
    logger.error("web_search error: %s", message)
    return {
        "query":        query,
        "results":      [],
        "result_count": 0,
        "data_source":  None,
        "source_tier":  4,
        "source_weight": 0.60,
        "retrieved_at": _now(),
        "error":        message,
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()