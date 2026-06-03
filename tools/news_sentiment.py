"""
tools/news_sentiment.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

News sentiment tool — fetches articles via NewsAPI then scores them using
VADER (primary) with TextBlob fallback. Both are local NLP — no extra API cost.

    News source:  NewsAPI.org — 100 req/day free tier
    Scoring:      VADER (vaderSentiment) — financial text optimised
    Fallback NLP: TextBlob — if vaderSentiment not installed

Tier: 4 (weight 0.60) — same tier as web_search; use for aggregate sentiment
      signals and Sentiment-Fact Alignment, never for factual numerical claims.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NEWSAPI_BASE = "https://newsapi.org/v2/everything"
RATE_DELAY   = 0.5

# Sentiment label thresholds (VADER compound score: -1.0 to +1.0)
POSITIVE_THRESHOLD =  0.05
NEGATIVE_THRESHOLD = -0.05


# ---------------------------------------------------------------------------
# Sentiment scorer — VADER primary, TextBlob fallback, simple fallback
# ---------------------------------------------------------------------------

def _load_scorer():
    """Return a scoring function: text -> compound float in [-1, 1]."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        logger.info("Sentiment scorer: VADER loaded")
        return lambda text: analyzer.polarity_scores(text)["compound"]
    except ImportError:
        pass

    try:
        from textblob import TextBlob
        logger.info("Sentiment scorer: TextBlob loaded (VADER not installed)")
        # TextBlob polarity is -1 to +1 — compatible range
        return lambda text: TextBlob(text).sentiment.polarity  # type: ignore

    except ImportError:
        pass

    # Final fallback: keyword-based scorer (no dependencies needed)
    logger.warning("Neither VADER nor TextBlob installed — using keyword scorer")
    POSITIVE = {"growth", "beats", "record", "strong", "surge", "profit",
                "upgrade", "gain", "rally", "bullish", "outperform", "raised"}
    NEGATIVE = {"decline", "loss", "misses", "cut", "risk", "lawsuit", "debt",
                "downgrade", "recall", "fraud", "bearish", "underperform", "lowered"}

    def _keyword_score(text: str) -> float:
        words = set(text.lower().split())
        pos = len(words & POSITIVE)
        neg = len(words & NEGATIVE)
        total = pos + neg
        return round((pos - neg) / total, 3) if total > 0 else 0.0

    return _keyword_score


_score_text = _load_scorer()


# ---------------------------------------------------------------------------
# NewsAPI fetcher
# ---------------------------------------------------------------------------

def _fetch_articles(
    query: str,
    num_articles: int,
    lookback_days: int,
    api_key: str,
) -> list[dict]:
    from_date = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    params = urlencode({
        "q":        query,
        "apiKey":   api_key,
        "pageSize": min(num_articles, 100),
        "sortBy":   "publishedAt",
        "language": "en",
        "from":     from_date,
    })
    url = f"{NEWSAPI_BASE}?{params}"

    time.sleep(RATE_DELAY)
    req = Request(url, headers={"User-Agent": "ARA-1/1.0 ara1@project.com"})
    try:
        with urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        raise RuntimeError(f"NewsAPI HTTP {e.code}") from e
    except URLError as e:
        raise RuntimeError(f"NewsAPI network error: {e.reason}") from e

    if data.get("status") != "ok":
        raise RuntimeError(f"NewsAPI error: {data.get('message', 'unknown')}")

    articles = data.get("articles", [])
    if not articles:
        raise RuntimeError("NewsAPI returned zero articles for this query.")

    return articles


# ---------------------------------------------------------------------------
# Sentiment aggregation
# ---------------------------------------------------------------------------

def _score_articles(articles: list[dict]) -> list[dict]:
    """Score each article and return structured sentiment records."""
    scored = []
    for a in articles:
        title       = a.get("title") or ""
        description = a.get("description") or ""
        text        = f"{title}. {description}".strip()

        compound = round(_score_text(text), 4)

        if compound >= POSITIVE_THRESHOLD:
            label = "positive"
        elif compound <= NEGATIVE_THRESHOLD:
            label = "negative"
        else:
            label = "neutral"

        scored.append({
            "title":        title[:200],
            "url":          a.get("url", ""),
            "source":       a.get("source", {}).get("name", ""),
            "published_at": a.get("publishedAt", ""),
            "sentiment_score":  compound,
            "sentiment_label":  label,
            "snippet":      description[:300] if description else "",
        })

    return scored


def _aggregate_sentiment(scored: list[dict]) -> dict:
    """Compute aggregate stats across all scored articles."""
    if not scored:
        return {"mean_score": 0.0, "positive_pct": 0, "negative_pct": 0, "neutral_pct": 0}

    scores     = [a["sentiment_score"] for a in scored]
    mean_score = round(sum(scores) / len(scores), 4)
    pos = sum(1 for a in scored if a["sentiment_label"] == "positive")
    neg = sum(1 for a in scored if a["sentiment_label"] == "negative")
    neu = sum(1 for a in scored if a["sentiment_label"] == "neutral")
    n   = len(scored)

    # Overall label
    if mean_score >= POSITIVE_THRESHOLD:
        overall_label = "positive"
    elif mean_score <= NEGATIVE_THRESHOLD:
        overall_label = "negative"
    else:
        overall_label = "neutral"

    return {
        "mean_score":    mean_score,
        "overall_label": overall_label,
        "positive_pct":  round(pos / n * 100, 1),
        "negative_pct":  round(neg / n * 100, 1),
        "neutral_pct":   round(neu / n * 100, 1),
        "article_count": n,
    }


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------

def news_sentiment(
    query: str,
    num_articles: int = 20,
    lookback_days: int = 30,
) -> dict:
    """
    Fetch and sentiment-score recent news articles about a company or topic.

    Args:
        query:         Company name, ticker, or topic string.
        num_articles:  Number of articles to analyze (5–50, default 20).
        lookback_days: Days of history to search (1–365, default 30).

    Returns:
        dict with keys:
            query, aggregate (summary stats), articles (list of scored dicts),
            scorer_used, data_source, source_tier, source_weight,
            retrieved_at, error
    """
    num_articles  = max(5, min(50, num_articles))
    lookback_days = max(1, min(365, lookback_days))

    newsapi_key = os.getenv("NEWSAPI_KEY", "")

    if not newsapi_key:
        return _error_response(query, "NEWSAPI_KEY not configured.")

    # Determine scorer name for transparency
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        scorer_used = "VADER"
    except ImportError:
        try:
            from textblob import TextBlob
            scorer_used = "TextBlob"
        except ImportError:
            scorer_used = "keyword_fallback"

    try:
        articles = _fetch_articles(query, num_articles, lookback_days, newsapi_key)
    except RuntimeError as e:
        return _error_response(query, str(e))

    scored     = _score_articles(articles)
    aggregate  = _aggregate_sentiment(scored)

    logger.info(
        "Sentiment [%s]: %d articles | mean=%.3f | overall=%s | scorer=%s",
        query[:60], len(scored), aggregate["mean_score"],
        aggregate["overall_label"], scorer_used,
    )

    return {
        "query":         query,
        "aggregate":     aggregate,
        "articles":      scored,
        "scorer_used":   scorer_used,
        "data_source":   "NewsAPI.org",
        "source_tier":   4,
        "source_weight": 0.60,
        "retrieved_at":  _now(),
        "error":         None,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error_response(query: str, message: str) -> dict:
    logger.error("news_sentiment error [%s]: %s", query[:60], message)
    return {
        "query":         query,
        "aggregate":     {},
        "articles":      [],
        "scorer_used":   None,
        "data_source":   None,
        "source_tier":   4,
        "source_weight": 0.60,
        "retrieved_at":  _now(),
        "error":         message,
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
