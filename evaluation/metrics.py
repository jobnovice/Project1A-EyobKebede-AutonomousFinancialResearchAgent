"""
evaluation/metrics.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

Computes all 22 quality metrics from architecture spec §10.1.

Five categories:
  FA — Factual Accuracy     (5 metrics)
  CO — Completeness         (4 metrics)
  AD — Analytical Depth     (4 metrics)
  CS — Coherence & Structure(4 metrics)
  AB — Agent Behaviour      (5 metrics)
  AX — Additional           (4 metrics)

Each metric is a standalone function returning a MetricResult.
The main entry point compute_all(state) runs every metric and
returns an EvaluationReport — this replaces the stub in node_evaluate.

LLM-as-judge metrics (AD-1, AD-2, AD-4, CS-1, AB-3) accept an optional
llm_call_fn. If None, they return a score of None with a note that
LLM evaluation was skipped — they never block the other 17 automated metrics.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Targets from architecture spec §10.1
# ---------------------------------------------------------------------------
TARGETS: dict[str, Any] = {
    "FA-1": 0.98,    # >98% numerical accuracy
    "FA-2": 1.00,    # 100% citation accuracy
    "FA-3": 1.00,    # 100% temporal accuracy
    "FA-4": 1.00,    # 100% entity accuracy
    "FA-5": 0,       # 0 hallucinations
    "CO-1": 1.00,    # 100% section coverage
    "CO-2": 4,       # ≥4 distinct source types
    "CO-3": 3,       # ≥3 years temporal coverage
    "CO-4": 0.80,    # ≥80% risk factor coverage
    "AD-1": 3,       # ≥3 insights per page
    "AD-2": 5,       # ≥5 cross-source syntheses per report
    "AD-3": 10,      # ≥10 original calculations
    "AD-4": 2,       # ≥2 forward-looking sections
    "CS-1": 0,       # 0 logical gaps
    "CS-2": 0,       # 0 contradictions
    "CS-3": None,    # qualitative
    "CS-4": 1.00,    # pass/fail formatting
    "AB-1": 0.70,    # ≥70% tool efficiency
    "AB-2": 0.90,    # ≥90% error recovery
    "AB-3": None,    # qualitative
    "AB-4": 0.30,    # ≥0.30 memory utilization (memory_hits/total_api_calls)
    "AB-5": 300,     # <300 seconds (5 min)
    "AX-1": 0.70,    # >0.7 Pearson r confidence calibration
    "AX-2": 0.95,    # >95% conflict detection rate
    "AX-3": 1.00,    # 100% fallback transparency
    "AX-4": None,    # maximize token efficiency
}

# Required sections for CO-1
REQUIRED_SECTIONS = [
    "overview", "key metrics", "methodology",
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    metric_id:   str
    name:        str
    category:    str
    value:       Any              # float | int | bool | None
    target:      Any
    passed:      Optional[bool]   # None = not evaluable (LLM judge skipped)
    note:        str = ""
    requires_llm:bool = False


@dataclass
class EvaluationReport:
    session_id:     str
    query:          str
    companies:      list[str]
    metrics:        list[MetricResult]
    overall_score:  float          # weighted average of automatable passed metrics
    passed_count:   int
    failed_count:   int
    skipped_count:  int            # LLM-judge metrics when llm_call_fn=None
    evaluated_at:   str = field(default_factory=lambda: _now())

    def to_dict(self) -> dict:
        return {
            "session_id":    self.session_id,
            "query":         self.query,
            "companies":     self.companies,
            "overall_score": self.overall_score,
            "passed":        self.passed_count,
            "failed":        self.failed_count,
            "skipped":       self.skipped_count,
            "evaluated_at":  self.evaluated_at,
            "metrics": {
                m.metric_id: {
                    "name":    m.name,
                    "value":   m.value,
                    "target":  m.target,
                    "passed":  m.passed,
                    "note":    m.note,
                }
                for m in self.metrics
            },
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _count_numerical_claims(text: str) -> list[str]:
    """Extract all numerical claims from report text."""
    pattern = r'\b\d+(?:\.\d+)?(?:\s*%|\s*billion|\s*million|\s*trillion|\s*B\b|\s*M\b|\s*T\b)?'
    return re.findall(pattern, text, re.IGNORECASE)


def _word_count(text: str) -> int:
    return len(text.split()) if text else 0


def _pages(text: str) -> float:
    """Approximate page count (250 words per page)."""
    return max(1.0, _word_count(text) / 250)


def _tool_tiers_used(tool_call_log: list[dict]) -> set[int]:
    """Map tool names to tiers and return set of unique tiers used."""
    tier_map = {
        "sec_filing_search":   1,
        "financial_data_api":  2,
        "company_profile":     2,
        "peer_comparison":     2,
        "earnings_transcript": 3,
        "web_search":          4,
        "news_sentiment":      4,
    }
    tiers = set()
    for entry in tool_call_log:
        tool = entry.get("tool", "")
        tier = tier_map.get(tool)
        if tier:
            tiers.add(tier)
    return tiers


def _llm_judge(
    llm_call_fn: Optional[Callable],
    metric_id: str,
    metric_name: str,
    report_section: str,
) -> tuple[Optional[float], str]:
    """
    Run LLM-as-judge for qualitative metrics.
    Returns (score_0_to_5_or_None, note).
    """
    if not llm_call_fn:
        return None, "LLM judge not available — pass llm_call_fn to enable."

    import json
    prompt_system = (
        "You are an expert financial research quality evaluator. "
        f"Assess the following research report on '{metric_name}'. "
        "Scoring rubric (1-5): "
        "5=exceeds professional standard, "
        "4=meets professional standard, "
        "3=adequate but missing key elements, "
        "2=below standard, "
        "1=fails minimum requirements. "
        "Return ONLY valid JSON: "
        '{"score": N, "reasoning": "...", "examples": [...]}'
    )
    prompt_user = f"Report excerpt:\n\n{report_section[:3000]}"

    try:
        raw = llm_call_fn(prompt_system, prompt_user, temperature=0.1)
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        data = json.loads(cleaned)
        score = float(data.get("score", 0))
        note = data.get("reasoning", "")[:200]
        return score, note
    except Exception as e:
        return None, f"LLM judge error: {e}"


# ===========================================================================
# CATEGORY FA — Factual Accuracy
# ===========================================================================

def fa1_numerical_accuracy(state: dict) -> MetricResult:
    """
    FA-1: % of numerical claims supported by source data.
    Proxy: verified_claims / (verified_claims + hallucination_count).
    Exact value requires fact_checker — this is the post-verify approximation.
    """
    verified   = state.get("verified_claims", [])
    flagged    = state.get("flagged_claims", [])
    hallucinations = state.get("hallucination_count", 0)

    verified_count = sum(v.get("count", 1) for v in verified) if verified else 0
    flagged_count  = len(flagged) if flagged else 0
    total = verified_count + max(hallucinations, flagged_count)

    value = round(verified_count / total, 3) if total > 0 else None
    target = TARGETS["FA-1"]
    passed = (value >= target) if value is not None else None

    return MetricResult(
        metric_id="FA-1", name="Numerical Accuracy",
        category="Factual Accuracy", value=value, target=target, passed=passed,
        note=f"Verified: {verified_count} | Flagged/hallucinated: {flagged_count}",
    )


def fa2_citation_accuracy(state: dict) -> MetricResult:
    """
    FA-2: % of tool calls that returned non-error data.
    Proxy for citation validity — real URLs validated in Day 14 final check.
    """
    tool_log = state.get("tool_call_log", [])
    total = len(tool_log)
    if total == 0:
        return MetricResult("FA-2", "Citation Accuracy", "Factual Accuracy",
                            None, TARGETS["FA-2"], None, "No tool calls logged.")

    valid = sum(
        1 for e in tool_log
        if e.get("result_summary") and "error" not in str(e.get("result_summary", "")).lower()
    )
    value = round(valid / total, 3)
    target = TARGETS["FA-2"]
    return MetricResult(
        metric_id="FA-2", name="Citation Accuracy",
        category="Factual Accuracy", value=value, target=target,
        passed=value >= target,
        note=f"{valid}/{total} calls returned valid data.",
    )


def fa3_temporal_accuracy(state: dict) -> MetricResult:
    """
    FA-3: Checks whether dates in tool call log span reasonable fiscal periods.
    Proxy: confirms at least one tool call has a date-bearing result.
    Full NER date extraction deferred to Day 14.
    """
    tool_log = state.get("tool_call_log", [])
    date_pattern = re.compile(r'\b(20\d{2})\b')
    years_found = set()

    for entry in tool_log:
        summary = str(entry.get("result_summary", ""))
        for yr in date_pattern.findall(summary):
            years_found.add(int(yr))

    report = state.get("draft_report", "")
    for yr in date_pattern.findall(report):
        years_found.add(int(yr))

    has_temporal = bool(years_found)
    value = 1.0 if has_temporal else 0.0
    return MetricResult(
        metric_id="FA-3", name="Temporal Accuracy",
        category="Factual Accuracy", value=value, target=TARGETS["FA-3"],
        passed=has_temporal,
        note=f"Years referenced: {sorted(years_found)} "
             f"(full date attribution validated at submission).",
    )


def fa4_entity_accuracy(state: dict) -> MetricResult:
    """
    FA-4: Checks company tickers/names appear consistently in the report.
    Proxy: all identified companies appear in the draft report.
    """
    companies = state.get("companies", [])
    report    = state.get("draft_report", "")

    if not companies or not report:
        return MetricResult("FA-4", "Entity Accuracy", "Factual Accuracy",
                            None, TARGETS["FA-4"], None, "No companies or report.")

    found = [c for c in companies if c in report and c != "UNKNOWN"]
    value = round(len(found) / len(companies), 3) if companies else None
    passed = value == 1.0 if value is not None else None
    return MetricResult(
        metric_id="FA-4", name="Entity Accuracy",
        category="Factual Accuracy", value=value, target=TARGETS["FA-4"],
        passed=passed,
        note=f"{len(found)}/{len(companies)} companies referenced in report.",
    )


def fa5_hallucination_count(state: dict) -> MetricResult:
    """FA-5: Count of claims with no source — direct from verifier output."""
    count  = state.get("hallucination_count", 0)
    target = TARGETS["FA-5"]
    # -1 means verification failed/was skipped
    if count == -1:
        return MetricResult("FA-5", "Hallucination Count", "Factual Accuracy",
                            None, target, None, "Verification pass did not run.")
    return MetricResult(
        metric_id="FA-5", name="Hallucination Count",
        category="Factual Accuracy", value=count, target=target,
        passed=count == 0,
        note=f"{count} unverified claim(s) flagged.",
    )


# ===========================================================================
# CATEGORY CO — Completeness
# ===========================================================================

def co1_section_coverage(state: dict) -> MetricResult:
    """CO-1: % of required sections present in draft report."""
    report = (state.get("draft_report") or "").lower()
    found  = [s for s in REQUIRED_SECTIONS if s.lower() in report]
    value  = round(len(found) / len(REQUIRED_SECTIONS), 3)
    return MetricResult(
        metric_id="CO-1", name="Section Coverage",
        category="Completeness", value=value, target=TARGETS["CO-1"],
        passed=value >= TARGETS["CO-1"],
        note=f"Found: {found} | Missing: {[s for s in REQUIRED_SECTIONS if s not in found]}",
    )


def co2_source_diversity(state: dict) -> MetricResult:
    """CO-2: Number of distinct source types used."""
    tool_log = state.get("tool_call_log", [])
    unique_sources = {e.get("tool") for e in tool_log if e.get("tool")} - {"vector_db_search", "vector_db_store"}
    value  = len(unique_sources)
    target = TARGETS["CO-2"]
    return MetricResult(
        metric_id="CO-2", name="Source Diversity",
        category="Completeness", value=value, target=target,
        passed=value >= target,
        note=f"Sources used: {sorted(unique_sources)}",
    )


def co3_temporal_coverage(state: dict) -> MetricResult:
    """CO-3: Number of distinct years referenced across all sources."""
    year_pat = re.compile(r'\b(20\d{2})\b')
    report   = state.get("draft_report", "")
    years = {int(y) for y in year_pat.findall(report)}
    span  = (max(years) - min(years) + 1) if len(years) >= 2 else len(years)
    target = TARGETS["CO-3"]
    return MetricResult(
        metric_id="CO-3", name="Temporal Coverage",
        category="Completeness", value=span, target=target,
        passed=span >= target,
        note=f"Years referenced: {sorted(years)} (span: {span})",
    )


def co4_risk_factor_coverage(state: dict) -> MetricResult:
    """
    CO-4: % of standard risk categories addressed in the report.
    Proxy: checks for standard risk keyword categories in the draft.
    Full 10-K cross-reference requires sec_filing_search output.
    """
    risk_categories = [
        "competition", "regulatory", "market", "operational",
        "financial", "supply chain", "cybersecurity", "macro",
    ]
    report = (state.get("draft_report") or "").lower()
    found  = [r for r in risk_categories if r in report]
    value  = round(len(found) / len(risk_categories), 3)
    target = TARGETS["CO-4"]
    return MetricResult(
        metric_id="CO-4", name="Risk Factor Coverage",
        category="Completeness", value=value, target=target,
        passed=value >= target,
        note=f"Risk categories found: {found}",
    )


# ===========================================================================
# CATEGORY AD — Analytical Depth
# ===========================================================================

def ad1_insight_density(
    state: dict,
    llm_call_fn: Optional[Callable] = None,
) -> MetricResult:
    """
    AD-1: Non-obvious observations per page (LLM-as-judge).
    Automated proxy: numerical claims per page as floor estimate.
    """
    report = state.get("draft_report") or ""
    pages  = _pages(report)
    nums   = _count_numerical_claims(report)
    auto_density = round(len(nums) / pages, 2) if pages > 0 else 0

    score, note = _llm_judge(llm_call_fn, "AD-1", "Insight Density", report)
    value  = score if score is not None else auto_density
    target = TARGETS["AD-1"]

    return MetricResult(
        metric_id="AD-1", name="Insight Density",
        category="Analytical Depth", value=value, target=target,
        passed=(value >= target) if value is not None else None,
        note=note or f"Auto proxy: {auto_density} numerical claims/page (≈{pages:.1f} pages).",
        requires_llm=True,
    )


def ad2_cross_source_synthesis(
    state: dict,
    llm_call_fn: Optional[Callable] = None,
) -> MetricResult:
    """
    AD-2: # of insights connecting ≥2 sources (LLM-as-judge).
    Automated proxy: count distinct source tools used in same step.
    """
    tool_log = state.get("tool_call_log", [])
    # Group by step_id and count steps that used 2+ tools
    steps: dict[Any, set] = {}
    for e in tool_log:
        sid = e.get("step_id", 0)
        steps.setdefault(sid, set()).add(e.get("tool", ""))
    multi_source_steps = sum(1 for tools in steps.values() if len(tools) >= 2)

    score, note = _llm_judge(llm_call_fn, "AD-2", "Cross-Source Synthesis", state.get("draft_report") or "")
    value  = score if score is not None else multi_source_steps
    target = TARGETS["AD-2"]

    return MetricResult(
        metric_id="AD-2", name="Cross-Source Synthesis",
        category="Analytical Depth", value=value, target=target,
        passed=(value >= target) if value is not None else None,
        note=note or f"Auto proxy: {multi_source_steps} steps drew from 2+ tools.",
        requires_llm=True,
    )


def ad3_quantitative_reasoning(state: dict) -> MetricResult:
    """
    AD-3: # of original calculations in the report (automated).
    Detects computation patterns: %, ratios, growth, margins.
    """
    report = state.get("draft_report") or ""
    # Patterns that suggest derived calculations (not just copied numbers)
    calc_patterns = [
        r'\d+\.?\d*\s*%',               # percentages
        r'\d+\.?\d*x',                  # multiples
        r'grew\s+\d+',                  # growth statement
        r'margin.*?\d+',                # margin statement
        r'ratio.*?\d+',                 # ratio statement
        r'CAGR',                        # compound growth
        r'year.over.year',              # YoY reference
        r'compared\s+to.*?\d+',         # comparison
    ]
    count = sum(len(re.findall(p, report, re.IGNORECASE)) for p in calc_patterns)
    target = TARGETS["AD-3"]
    return MetricResult(
        metric_id="AD-3", name="Quantitative Reasoning",
        category="Analytical Depth", value=count, target=target,
        passed=count >= target,
        note=f"{count} quantitative reasoning patterns detected.",
    )


def ad4_forward_looking(
    state: dict,
    llm_call_fn: Optional[Callable] = None,
) -> MetricResult:
    """
    AD-4: # of sections with reasoned projections (LLM-as-judge).
    Proxy: count forward-looking keyword signals.
    """
    report  = state.get("draft_report") or ""
    fwd_kws = ["guidance", "expect", "project", "forecast", "outlook",
               "anticipated", "next quarter", "next year", "going forward"]
    count = sum(1 for kw in fwd_kws if kw.lower() in report.lower())
    score, note = _llm_judge(llm_call_fn, "AD-4", "Forward-Looking Analysis", report)
    value  = score if score is not None else count
    target = TARGETS["AD-4"]
    return MetricResult(
        metric_id="AD-4", name="Forward-Looking Analysis",
        category="Analytical Depth", value=value, target=target,
        passed=(value >= target) if value is not None else None,
        note=note or f"{count} forward-looking signals detected.",
        requires_llm=True,
    )


# ===========================================================================
# CATEGORY CS — Coherence and Structure
# ===========================================================================

def cs1_logical_flow(
    state: dict,
    llm_call_fn: Optional[Callable] = None,
) -> MetricResult:
    """CS-1: Logical flow quality (LLM-as-judge)."""
    score, note = _llm_judge(llm_call_fn, "CS-1", "Logical Flow", state.get("draft_report") or "")
    target = TARGETS["CS-1"]   # 0 logical gaps
    return MetricResult(
        metric_id="CS-1", name="Logical Flow",
        category="Coherence & Structure",
        value=score, target=target,
        passed=(score is not None and score >= 3.0),
        note=note or "LLM judge not run.",
        requires_llm=True,
    )


def cs2_internal_consistency(state: dict) -> MetricResult:
    """
    CS-2: Detect obvious self-contradictions.
    Automated: check if both 'increased' and 'decreased' apply to same metric.
    Full LLM contradiction detection deferred to Day 14.
    """
    report = state.get("draft_report") or ""
    contradiction_count = 0

    # Check for contradictory direction statements near same metrics
    for metric_kw in ["revenue", "margin", "profit", "growth"]:
        has_up   = bool(re.search(rf'{metric_kw}.*?(grew|increased|improved|rose)', report, re.IGNORECASE))
        has_down = bool(re.search(rf'{metric_kw}.*?(declined|decreased|fell|dropped)', report, re.IGNORECASE))
        if has_up and has_down:
            contradiction_count += 1  # may be legitimate (different periods), flag for review

    value  = contradiction_count
    target = TARGETS["CS-2"]
    return MetricResult(
        metric_id="CS-2", name="Internal Consistency",
        category="Coherence & Structure", value=value, target=target,
        passed=value == 0,
        note=f"{value} potential contradictions flagged (may be different time periods).",
    )


def cs3_executive_summary(state: dict) -> MetricResult:
    """
    CS-3: Checks whether an executive summary / overview section is present.
    Automated proxy — ROUGE alignment deferred to Day 14.
    """
    report   = (state.get("draft_report") or "").lower()
    has_exec = "overview" in report or "executive summary" in report or "summary" in report
    return MetricResult(
        metric_id="CS-3", name="Executive Summary Quality",
        category="Coherence & Structure", value=1.0 if has_exec else 0.0,
        target=TARGETS["CS-3"], passed=has_exec,
        note="Overview/summary section present." if has_exec else "No overview section found.",
    )


def cs4_professional_formatting(state: dict) -> MetricResult:
    """
    CS-4: Check structural formatting — headers, bold metrics, methodology.
    """
    report = state.get("draft_report") or ""
    checks = {
        "has_h2_headers":   bool(re.search(r'^##\s', report, re.MULTILINE)),
        "has_bold_text":    "**" in report,
        "has_methodology":  "methodology" in report.lower(),
        "non_empty":        len(report.strip()) > 100,
    }
    passed_checks = sum(checks.values())
    value = round(passed_checks / len(checks), 2)
    return MetricResult(
        metric_id="CS-4", name="Professional Formatting",
        category="Coherence & Structure", value=value,
        target=TARGETS["CS-4"], passed=value >= TARGETS["CS-4"],
        note=str(checks),
    )


# ===========================================================================
# CATEGORY AB — Agent Behaviour
# ===========================================================================

def ab1_tool_efficiency(state: dict) -> MetricResult:
    """
    AB-1: Useful tool calls / total tool calls.
    'Useful' = result_summary is non-empty and does not indicate an error.
    """
    tool_log = state.get("tool_call_log", [])
    total = len(tool_log)
    if total == 0:
        return MetricResult("AB-1", "Tool Efficiency", "Agent Behaviour",
                            None, TARGETS["AB-1"], None, "No tool calls.")
    useful = sum(
        1 for e in tool_log
        if e.get("result_summary") and
        "not yet implemented" not in str(e.get("result_summary", "")).lower() and
        "error" not in str(e.get("result_summary", "")).lower()
    )
    value = round(useful / total, 3)
    return MetricResult(
        metric_id="AB-1", name="Tool Efficiency",
        category="Agent Behaviour", value=value, target=TARGETS["AB-1"],
        passed=value >= TARGETS["AB-1"],
        note=f"{useful}/{total} calls returned useful data.",
    )


def ab2_error_recovery(state: dict) -> MetricResult:
    """
    AB-2: % of errors that were recovered (fallback triggered and succeeded).
    Approximated from conflicts_detected vs conflicts_resolved.
    """
    error_log     = state.get("error_log", [])
    detected      = state.get("conflicts_detected", [])
    resolved      = state.get("conflicts_resolved", [])
    tool_log      = state.get("tool_call_log", [])

    errors   = len(error_log) + len(detected)
    # Count steps where a fallback was triggered and a subsequent tool call succeeded
    fallback_steps = sum(1 for e in tool_log if "fallback" in str(e.get("thought", "")).lower())
    recoveries = len(resolved) + min(fallback_steps, errors)

    if errors == 0:
        return MetricResult("AB-2", "Error Recovery", "Agent Behaviour",
                            1.0, TARGETS["AB-2"], True, "No errors encountered.")
    value = round(min(1.0, recoveries / errors), 3)
    return MetricResult(
        metric_id="AB-2", name="Error Recovery",
        category="Agent Behaviour", value=value, target=TARGETS["AB-2"],
        passed=value >= TARGETS["AB-2"],
        note=f"{recoveries}/{errors} errors recovered.",
    )


def ab3_planning_quality(
    state: dict,
    llm_call_fn: Optional[Callable] = None,
) -> MetricResult:
    """AB-3: Planning quality (LLM-as-judge)."""
    plan = state.get("research_plan", [])
    plan_summary = f"Plan had {len(plan)} steps. Revisions: {state.get('plan_revision_count', 0)}."
    draft = (state.get("draft_report") or "")
    score, note = _llm_judge(
        llm_call_fn, "AB-3", "Planning Quality",
        plan_summary + "\n\n" + draft[:1000]
    )
    return MetricResult(
        metric_id="AB-3", name="Planning Quality",
        category="Agent Behaviour", value=score, target=TARGETS["AB-3"],
        passed=(score is not None and score >= 3.0),
        note=note or f"{plan_summary} LLM judge not run.",
        requires_llm=True,
    )


def ab4_memory_utilization(state: dict) -> MetricResult:
    """
    AB-4: memory_hits / total_api_calls.
    IMPORTANT: division, not multiplication (Error 002 from ERROR_LOG.md).
    """
    total_calls  = state.get("total_api_calls", 0)
    memory_hits  = state.get("memory_hits", 0)
    value = round(memory_hits / total_calls, 3) if total_calls > 0 else 0.0
    target = TARGETS["AB-4"]
    return MetricResult(
        metric_id="AB-4", name="Memory Utilization",
        category="Agent Behaviour", value=value, target=target,
        passed=value >= target,
        note=f"{memory_hits} memory hits / {total_calls} total calls = {value:.3f}",
    )


def ab5_latency(state: dict) -> MetricResult:
    """AB-5: Total query-to-delivery latency in seconds."""
    metrics  = state.get("evaluation_metrics", {})
    value    = metrics.get("AB-5_latency_seconds")
    target   = TARGETS["AB-5"]
    passed   = (value is not None and value < target)
    return MetricResult(
        metric_id="AB-5", name="Latency",
        category="Agent Behaviour", value=value, target=target,
        passed=passed,
        note=f"{value}s (target: <{target}s)" if value else "Latency not yet recorded.",
    )


# ===========================================================================
# CATEGORY AX — Additional
# ===========================================================================

def ax1_confidence_calibration(state: dict) -> MetricResult:
    """
    AX-1: Correlation between stated confidence and actual accuracy.
    Proxy: average confidence score from verifier.
    """
    conf_scores = state.get("confidence_scores", {})
    overall = conf_scores.get("overall")
    if overall is None:
        return MetricResult("AX-1", "Confidence Calibration", "Additional",
                            None, TARGETS["AX-1"], None, "No confidence score recorded.")
    target = TARGETS["AX-1"]
    return MetricResult(
        metric_id="AX-1", name="Confidence Calibration",
        category="Additional", value=overall, target=target,
        passed=overall >= target,
        note=f"Verifier confidence: {overall:.2f}",
    )


def ax2_conflict_detection_rate(state: dict) -> MetricResult:
    """
    AX-2: % of actual conflicts correctly identified and documented.
    Proxy: if conflicts_resolved is populated, detection+resolution occurred.
    """
    detected = state.get("conflicts_detected", [])
    resolved = state.get("conflicts_resolved", [])

    if not detected and not resolved:
        return MetricResult("AX-2", "Conflict Detection Rate", "Additional",
                            1.0, TARGETS["AX-2"], True,
                            "No conflicts detected or resolved (no discrepancies found).")

    total    = len(detected) + len(resolved)
    handled  = len(resolved)
    value    = round(handled / total, 3) if total > 0 else 1.0
    return MetricResult(
        metric_id="AX-2", name="Conflict Detection Rate",
        category="Additional", value=value, target=TARGETS["AX-2"],
        passed=value >= TARGETS["AX-2"],
        note=f"{len(detected)} detected | {len(resolved)} resolved.",
    )


def ax3_fallback_transparency(state: dict) -> MetricResult:
    """
    AX-3: % of fallback uses disclosed in the report.
    Proxy: checks if error_log entries are referenced in the report.
    """
    error_log = state.get("error_log", [])
    report    = state.get("draft_report", "")

    if not error_log:
        return MetricResult("AX-3", "Fallback Transparency", "Additional",
                            1.0, TARGETS["AX-3"], True, "No fallbacks needed.")

    disclosed = sum(
        1 for err in error_log
        if any(word in report.lower() for word in ["fallback", "error", "unavailable", "failed"])
    )
    value = round(disclosed / len(error_log), 3)
    return MetricResult(
        metric_id="AX-3", name="Fallback Transparency",
        category="Additional", value=value, target=TARGETS["AX-3"],
        passed=value >= TARGETS["AX-3"],
        note=f"{disclosed}/{len(error_log)} fallbacks disclosed in report.",
    )


def ax4_token_efficiency(state: dict) -> MetricResult:
    """
    AX-4: Report quality per 1000 tokens consumed.
    Proxy: report length / total_api_calls (higher = more output per call).
    Full token count requires tiktoken — this is an approximation.
    """
    report      = state.get("draft_report") or ""
    total_calls = state.get("total_api_calls", 1)
    report_words = _word_count(report)
    value = round(report_words / max(total_calls, 1), 1)
    return MetricResult(
        metric_id="AX-4", name="Token Efficiency",
        category="Additional", value=value, target=TARGETS["AX-4"],
        passed=None,   # no fixed threshold — higher is better
        note=f"{report_words} words in report / {total_calls} tool calls = {value} words/call.",
    )


# ===========================================================================
# Main entry point
# ===========================================================================

def compute_all(
    state: dict,
    llm_call_fn: Optional[Callable] = None,
) -> EvaluationReport:
    """
    Compute all 22 metrics and return an EvaluationReport.

    Args:
        state:       The full ResearchState dict (or evaluation_metrics subset).
        llm_call_fn: Optional callable for LLM-as-judge metrics.
                     If None, 5 LLM-judge metrics will be skipped (value=None).

    Returns:
        EvaluationReport with all metrics, overall score, pass/fail counts.
    """
    metrics: list[MetricResult] = [
        # FA
        fa1_numerical_accuracy(state),
        fa2_citation_accuracy(state),
        fa3_temporal_accuracy(state),
        fa4_entity_accuracy(state),
        fa5_hallucination_count(state),
        # CO
        co1_section_coverage(state),
        co2_source_diversity(state),
        co3_temporal_coverage(state),
        co4_risk_factor_coverage(state),
        # AD
        ad1_insight_density(state, llm_call_fn),
        ad2_cross_source_synthesis(state, llm_call_fn),
        ad3_quantitative_reasoning(state),
        ad4_forward_looking(state, llm_call_fn),
        # CS
        cs1_logical_flow(state, llm_call_fn),
        cs2_internal_consistency(state),
        cs3_executive_summary(state),
        cs4_professional_formatting(state),
        # AB
        ab1_tool_efficiency(state),
        ab2_error_recovery(state),
        ab3_planning_quality(state, llm_call_fn),
        ab4_memory_utilization(state),
        ab5_latency(state),
        # AX
        ax1_confidence_calibration(state),
        ax2_conflict_detection_rate(state),
        ax3_fallback_transparency(state),
        ax4_token_efficiency(state),
    ]

    passed  = sum(1 for m in metrics if m.passed is True)
    failed  = sum(1 for m in metrics if m.passed is False)
    skipped = sum(1 for m in metrics if m.passed is None)

    # Overall score: average of evaluable metrics (excludes skipped)
    evaluable = [m for m in metrics if m.passed is not None]
    overall = round(sum(1 for m in evaluable if m.passed) / len(evaluable), 3) if evaluable else 0.0

    report = EvaluationReport(
        session_id=state.get("session_id", "unknown"),
        query=state.get("raw_query", ""),
        companies=state.get("companies", []),
        metrics=metrics,
        overall_score=overall,
        passed_count=passed,
        failed_count=failed,
        skipped_count=skipped,
    )

    logger.info(
        "Evaluation complete: %d passed | %d failed | %d skipped | score=%.1f%%",
        passed, failed, skipped, overall * 100,
    )
    return report
