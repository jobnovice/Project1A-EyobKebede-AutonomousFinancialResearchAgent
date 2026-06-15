"""
tools/fact_checker.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

Fact checker tool — cross-references a claim against multiple authoritative sources.
Uses LLM and available data APIs to validate factual and numerical claims.

Input:
    claim: str                  # The statement to verify (e.g., "Microsoft revenue grew 15% in FY2024")
    sources: list[str] | None   # Optional pre-fetched source texts (if None, tool auto-searches)
    required_confidence: float  # Minimum confidence to treat as verified (default 0.70)

Output:
    verification_status: str    # 'SUPPORTED' | 'CONTRADICTED' | 'INCONCLUSIVE'
    confidence: float           # 0.0–1.0
    verified: bool              # confidence >= required_confidence
    supporting_evidence: list[str]  # Snippets or quotes that support the claim
    contradicting_evidence: list[str] # Snippets that contradict
    explanation: str            # Short reasoning
    error: str | None
    source_tier, source_weight, retrieved_at, sources_used, required_confidence

Tier: composite (tier 2 – curated financial data sources)
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional, List, Tuple, Any

# Dynamic tool loading to avoid hard dependencies
try:
    from tools.web_search import web_search
    from tools.financial_api import financial_data_api
    from tools.sec_edgar import sec_filing_search
except ImportError:
    web_search = None
    financial_data_api = None
    sec_filing_search = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Common uppercase words that are NOT company tickers
NON_TICKER_STOPWORDS = {
    "Q1", "Q2", "Q3", "Q4", "FY", "EPS", "US", "UK", "EU", "IPO", "SEC",
    "CEO", "CFO", "TTM", "YOY", "INC", "LTD", "LLC", "THE", "AND", "FOR",
    "REVENUE", "PROFIT", "GROWTH", "MARGIN", "DEBT"
}


# ---------------------------------------------------------------------------
# Helper: extract numerical values from text (with unit scaling)
# ---------------------------------------------------------------------------
def _extract_numbers(text: str) -> List[Tuple[float, str]]:
    """Extract all numbers (including percentages, millions/billions) from text."""
    pattern = r'(\d+(?:\.\d+)?)\s*(%|percent|million|billion|trillion|M|B|T)?'
    matches = re.findall(pattern, text, re.IGNORECASE)
    results = []
    for match in matches:
        try:
            num = float(match[0])
            suffix = match[1].lower()
            if suffix in ('million', 'm'):
                num *= 1_000_000
            elif suffix in ('billion', 'b'):
                num *= 1_000_000_000
            elif suffix in ('trillion', 't'):
                num *= 1_000_000_000_000
            # suffix '%' or 'percent' left unchanged
            results.append((num, match[0] + ' ' + match[1] if match[1] else match[0]))
        except ValueError:
            continue
    return results


# ---------------------------------------------------------------------------
# Auto-fetch sources from web, SEC filings, and financial API
# ---------------------------------------------------------------------------
def _fetch_sources(claim: str, max_sources: int = 3) -> List[str]:
    """Retrieve relevant source snippets automatically."""
    sources = []

    # 1. Web search
    if web_search:
        try:
            result = web_search(query=claim, num_results=2, date_range="any")
            if result and not result.get("error"):
                for item in result.get("results", []):
                    snippet = item.get("snippet", "")
                    if snippet:
                        sources.append(f"[Web] {snippet[:500]}")
        except Exception as e:
            logger.warning("Web search failed in fact_checker: %s", e)

    # 2. Extract potential ticker (ignore non-ticker uppercase words)
    ticker_match = re.search(r'\b([A-Z]{1,5})\b', claim)
    ticker = None
    if ticker_match and ticker_match.group(1) not in NON_TICKER_STOPWORDS:
        ticker = ticker_match.group(1)

    # 3. SEC filing search (10‑K) with keyword window
    if ticker and sec_filing_search:
        try:
            result = sec_filing_search(ticker=ticker, filing_type="10-K")
            if result and not result.get("error"):
                text = result.get("filing_text", "")
                if text:
                    # Find best position using claim numbers or common keywords
                    search_terms = [orig for _, orig in _extract_numbers(claim)]
                    search_terms += ["revenue", "net income", "operating income", "earnings"]
                    best_index = 0
                    for term in search_terms:
                        idx = text.lower().find(term.lower())
                        if idx != -1:
                            best_index = idx
                            break
                    start = max(0, best_index - 100)
                    end = min(len(text), start + 500)
                    sources.append(f"[10-K Snippet] ...{text[start:end]}...")
        except Exception as e:
            logger.warning("SEC filing search failed: %s", e)

    # 4. Financial API for key metrics
    if ticker and financial_data_api:
        try:
            # Our financial_data_api accepts 'years' parameter (mapped to limit)
            result = financial_data_api(
                ticker=ticker,
                statement_type="key_metrics",
                period="annual",
                years=1
            )
            if result and not result.get("error"):
                records = result.get("records", [])
                if records:
                    sources.append(f"[Financials Data] {json.dumps(records[0])[:500]}")
        except Exception as e:
            logger.warning("Financial API failed: %s", e)

    return sources[:max_sources]


# ---------------------------------------------------------------------------
# LLM-based verification
# ---------------------------------------------------------------------------
def _verify_with_llm(claim: str, source_texts: List[str]) -> dict:
    """Use LLM to evaluate claim against source texts."""
    try:
        from agent.core import _call_llm
    except ImportError:
        _call_llm = None

    if not _call_llm:
        numerical = _verify_numerical_claim(claim, source_texts)
        if numerical:
            return numerical
        return {
            "status": "INCONCLUSIVE",
            "confidence": 0.3,
            "supporting": [],
            "contradicting": [],
            "explanation": "LLM unavailable and numerical fallback inconclusive."
        }

    system_prompt = (
        "You are a meticulous financial fact checker. Evaluate the claim strictly against "
        "the provided source snippets. Determine if the claim is SUPPORTED, CONTRADICTED, "
        "or INCONCLUSIVE.\n\n"
        "CRITICAL RULES:\n"
        "1. If a numerical claim deviates from source figures by more than 5%, flag CONTRADICTED.\n"
        "2. If within 5% tolerance or exact parity, flag SUPPORTED.\n"
        "3. Output **plain JSON** (no markdown code fences) with the exact schema:\n"
        '{"status": "SUPPORTED|CONTRADICTED|INCONCLUSIVE", "confidence": 0.0-1.0,\n'
        ' "supporting_evidence": ["snippet"], "contradicting_evidence": ["snippet"],\n'
        ' "explanation": "brief rationale"}'
    )
    user_content = f"Claim: {claim}\n\nSources:\n" + "\n---\n".join(source_texts)

    try:
        response = _call_llm(system_prompt, user_content, temperature=0.1)
        # Strip markdown fences if any
        response = re.sub(r"^```json\s*", "", response)
        response = re.sub(r"```\s*$", "", response)
        data = json.loads(response)

        status = data.get("status", "INCONCLUSIVE").upper()
        if status not in ("SUPPORTED", "CONTRADICTED", "INCONCLUSIVE"):
            status = "INCONCLUSIVE"

        return {
            "status": status,
            "confidence": float(data.get("confidence", 0.5)),
            "supporting": data.get("supporting_evidence", []),
            "contradicting": data.get("contradicting_evidence", []),
            "explanation": data.get("explanation", ""),
        }
    except Exception as e:
        logger.error("LLM verification failed: %s", e)
        return {
            "status": "INCONCLUSIVE",
            "confidence": 0.2,
            "supporting": [],
            "contradicting": [],
            "explanation": f"LLM error: {e}",
        }


# ---------------------------------------------------------------------------
# Programmatic numerical fallback with 5% tolerance
# ---------------------------------------------------------------------------
def _verify_numerical_claim(claim: str, sources: List[str]) -> Optional[dict]:
    """Check claim numbers against source numbers with 5% tolerance."""
    claim_nums = _extract_numbers(claim)
    if not claim_nums:
        return None

    source_text = " ".join(sources)
    source_nums = _extract_numbers(source_text)
    if not source_nums:
        return None

    supporting = []
    contradicting = []

    for c_val, c_orig in claim_nums:
        matched = False
        for s_val, s_orig in source_nums:
            if c_val == 0 and s_val == 0:
                rel_diff = 0.0
            elif c_val != 0:
                rel_diff = abs(c_val - s_val) / c_val
            else:
                continue
            if rel_diff <= 0.05:
                supporting.append(f"'{c_orig}' matches source '{s_orig}' (diff {rel_diff*100:.1f}%)")
                matched = True
                break
        if not matched:
            contradicting.append(f"'{c_orig}' not found within 5% tolerance")

    if supporting and not contradicting:
        return {
            "status": "SUPPORTED",
            "confidence": 0.70,
            "supporting": supporting,
            "contradicting": [],
            "explanation": "Numerical check within 5% tolerance (fallback)."
        }
    elif contradicting:
        return {
            "status": "CONTRADICTED",
            "confidence": 0.65,
            "supporting": supporting,
            "contradicting": contradicting,
            "explanation": "Some claim numbers exceed 5% deviation."
        }
    return None


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------
def fact_checker(
    claim: str,
    sources: Optional[List[str]] = None,
    max_sources: int = 3,
    required_confidence: float = 0.70,
    **kwargs   # absorbs extra parameters from LLM for robustness
) -> dict:
    """
    Verify a factual or numerical claim against multiple authoritative sources.

    Args:
        claim:               Statement to verify.
        sources:             Optional pre‑fetched source texts.
        max_sources:         Maximum number of auto‑fetched sources.
        required_confidence: Minimum confidence to set verified=True (default 0.70).
        **kwargs:            Absorbs unexpected parameters (defensive).

    Returns:
        dict with verification details, including verified boolean.
    """
    claim = claim.strip()
    if not claim:
        return _error_response(claim, "Claim cannot be empty.")

    # Fetch sources if not provided
    if sources is None:
        sources_used = _fetch_sources(claim, max_sources=max_sources)
    else:
        sources_used = sources[:max_sources] if sources else []

    if not sources_used:
        # No sources: still return tier 2 (tool's inherent authority) but mark error
        return {
            "claim": claim,
            "verification_status": "INCONCLUSIVE",
            "confidence": 0.0,
            "verified": False,
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "explanation": "No sources could be retrieved automatically.",
            "sources_used": [],
            "source_tier": 2,
            "source_weight": 0.90,
            "required_confidence": required_confidence,
            "retrieved_at": _now(),
            "error": "No sources to verify against.",
        }

    # Run verification (LLM + numerical fallback)
    result = _verify_with_llm(claim, sources_used)
    confidence = result.get("confidence", 0.0)
    verified = confidence >= required_confidence

    return {
        "claim": claim,
        "verification_status": result.get("status", "INCONCLUSIVE"),
        "confidence": confidence,
        "verified": verified,
        "supporting_evidence": result.get("supporting", []),
        "contradicting_evidence": result.get("contradicting", []),
        "explanation": result.get("explanation", ""),
        "sources_used": sources_used,
        "source_tier": 2,
        "source_weight": 0.90,
        "required_confidence": required_confidence,
        "retrieved_at": _now(),
        "error": None,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _error_response(claim: str, message: str) -> dict:
    logger.error("fact_checker error for claim '%s': %s", claim[:50], message)
    return {
        "claim": claim,
        "verification_status": "INCONCLUSIVE",
        "confidence": 0.0,
        "verified": False,
        "supporting_evidence": [],
        "contradicting_evidence": [],
        "explanation": f"Error: {message}",
        "sources_used": [],
        "source_tier": 2,
        "source_weight": 0.90,
        "required_confidence": 0.70,
        "retrieved_at": _now(),
        "error": message,
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()