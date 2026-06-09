"""
tests/test_challenges_1_2.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

Day 5 — Challenge 1 and Challenge 2 test suite.

Modes (auto-detected):
  REAL MODE:  (ANTHROPIC_API_KEY or DEEPSEEK_API_KEY or GEMINI_API_KEY) + FMP_API_KEY present
              → live LLM + real tools
  MOCK MODE:  Keys missing → stub LLM + mock tool data, no external calls

Challenge 1: Microsoft company profile   (Difficulty 1/5)
Challenge 2: Apple earnings analysis     (Difficulty 2/5)

Run from repo root:
  python tests/test_challenges_1_2.py
  source .env && python tests/test_challenges_1_2.py   # real mode
"""

import sys
import os
import types
import json
import logging
from typing import Any
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()   # loads .env file from current directory or parent


logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")

# ---------------------------------------------------------------------------
# Mode detection — must happen before any agent imports
# ---------------------------------------------------------------------------
HAS_DEEPSEEK  = bool(os.getenv("DEEPSEEK_API_KEY"))
HAS_ANTHROPIC = bool(os.getenv("ANTHROPIC_API_KEY"))
HAS_GEMINI    = bool(os.getenv("GEMINI_API_KEY"))
HAS_FMP       = bool(os.getenv("FMP_API_KEY"))
HAS_TAVILY    = bool(os.getenv("TAVILY_API_KEY"))
HAS_NEWSAPI   = bool(os.getenv("NEWSAPI_KEY"))
HAS_GROQAPI   = bool(os.getenv("GROQ_API_KEY"))

# Real mode if any LLM key is present AND FMP key is present
REAL_MODE = (HAS_ANTHROPIC or HAS_DEEPSEEK or HAS_GEMINI) and HAS_FMP

print("\n" + "="*60)
print("ARA-1 — Challenge 1 & 2 Test Suite")
print("="*60)
print(f"  Mode:              {'🟢 REAL API' if REAL_MODE else '🟡 MOCK (no API keys)'}")
print(f"  GROQ_API_KEY:      {'✓' if HAS_GROQAPI else '✗ missing'}")
print(f"  ANTHROPIC_API_KEY: {'✓' if HAS_ANTHROPIC else '✗ missing'}")
print(f"  DEEPSEEK_API_KEY:  {'✓' if HAS_DEEPSEEK else '✗ missing'}")
print(f"  GEMINI_API_KEY:    {'✓' if HAS_GEMINI else '✗ missing'}")
print(f"  FMP_API_KEY:       {'✓' if HAS_FMP else '✗ missing'}")
print(f"  TAVILY_API_KEY:    {'✓' if HAS_TAVILY else '✗ missing'}")
print(f"  NEWSAPI_KEY:       {'✓' if HAS_NEWSAPI else '✗ missing'}")
print("="*60)

# ---------------------------------------------------------------------------
# Install stubs BEFORE importing agent.core — use a swappable fn pointer
# so we can change LLM behaviour between challenges without re-importing
# ---------------------------------------------------------------------------
_llm_fn: dict[str, Any] = {"current": None}   # mutable container — swapped per challenge

if not REAL_MODE:
    # anthropic stub
    ant: Any = types.ModuleType("anthropic")

    class _Content:
        def __init__(self, t): self.text = t

    class _Resp:
        def __init__(self, t): self.content = [_Content(t)]

    class _Messages:
        @staticmethod
        def create(model, max_tokens, temperature, system, messages, **kw):
            fn = _llm_fn["current"]
            return _Resp(fn(system, messages[-1]["content"]))

    class _Anthropic:
        messages = _Messages()

    ant.Anthropic = _Anthropic
    sys.modules["anthropic"] = ant

    # langgraph stub
    lg: Any  = types.ModuleType("langgraph")
    lgg: Any = types.ModuleType("langgraph.graph")

    class _SG:
        def __init__(self, s):
            self._nodes, self._cond = {}, {}
        def add_node(self, n, f):          self._nodes[n] = f
        def set_entry_point(self, n):      pass
        def add_edge(self, a, b):          pass
        def add_conditional_edges(self, s, f, m): self._cond[s] = (f, m)
        def compile(self):                 return _CG(self)

    class _CG:
        def __init__(self, g): self._g = g
        def invoke(self, state):
            nodes = self._g._nodes
            cond  = self._g._cond
            for n in ["analyze_query", "plan", "memory_precheck"]:
                state = nodes[n](state)
            for _ in range(25):
                state = nodes["execute_step"](state)
                fn, mapping = cond["execute_step"]
                if mapping[fn(state)] == "synthesize":
                    break
            for n in ["synthesize", "verify", "generate_report", "evaluate"]:
                state = nodes[n](state)
            return state

    lgg.StateGraph = _SG
    lgg.END = "__end__"
    lg.graph = lgg
    sys.modules["langgraph"] = lg
    sys.modules["langgraph.graph"] = lgg

# Now safe to import agent modules
from agent.core import run_agent

# ---------------------------------------------------------------------------
# LLM response functions (mock mode)
# ---------------------------------------------------------------------------

def _llm_msft(system: str, user: str) -> str:
    if "produce a structured JSON research plan" in system:
        return json.dumps({
            "plan_id": "plan-MSFT-factual-2024-001",
            "query_summary": "Comprehensive Microsoft Corporation company profile.",
            "query_type": "factual",
            "companies": ["MSFT"],
            "estimated_tool_calls": 4,
            "steps": [
                {"step_id": 1, "description": "Check memory for Microsoft",
                 "tool": "vector_db_search",
                 "params": {"query": "Microsoft MSFT company profile", "top_k": 5},
                 "depends_on": [], "fallback_if_fails": None, "optional": False},
                {"step_id": 2, "description": "Get Microsoft company profile",
                 "tool": "company_profile",
                 "params": {"ticker": "MSFT"},
                 "depends_on": [1], "fallback_if_fails": "web_search", "optional": False},
                {"step_id": 3, "description": "Get Microsoft key financial metrics",
                 "tool": "financial_data_api",
                 "params": {"ticker": "MSFT", "statement_type": "key_metrics",
                            "period": "annual", "limit": 4},
                 "depends_on": [1], "fallback_if_fails": "sec_filing_search", "optional": False},
                {"step_id": 4, "description": "Search recent Microsoft news",
                 "tool": "web_search",
                 "params": {"query": "Microsoft Corporation latest developments 2024",
                            "num_results": 8},
                 "depends_on": [], "fallback_if_fails": "news_sentiment", "optional": True},
            ],
        })
    if "identify hallucinations" in system:
        return json.dumps({"hallucination_count": 0, "unsupported_claims": [],
                           "verified_claims_count": 10, "confidence_score": 0.93})
    # executor — read the tool name directly from the plan step JSON in the user message
    import re as _re
    m = _re.search(r'"tool":\s*"([^"]+)"', user)
    tool = m.group(1) if m else "web_search"
    params_map = {
        "vector_db_search": {"query": "Microsoft MSFT", "top_k": 5},
        "company_profile":  {"ticker": "MSFT"},
        "financial_data_api": {"ticker": "MSFT", "statement_type": "key_metrics",
                               "period": "annual", "limit": 4},
        "web_search": {"query": "Microsoft latest news 2024", "num_results": 8},
    }
    params = params_map.get(tool, {"ticker": "MSFT"})
    return (
        f"THOUGHT: Executing step for {tool}.\n\n"
        f'ACTION:\n{{"tool": "{tool}", "params": {json.dumps(params)}}}\n\n'
        f"OBSERVATION: Data retrieved.\n\nDECIDE: step_complete"
    )


def _llm_aapl(system: str, user: str) -> str:
    if "produce a structured JSON research plan" in system:
        return json.dumps({
            "plan_id": "plan-AAPL-analytical-2024-001",
            "query_summary": "Apple Inc. most recent quarterly earnings analysis.",
            "query_type": "analytical",
            "companies": ["AAPL"],
            "estimated_tool_calls": 5,
            "steps": [
                {"step_id": 1, "description": "Check memory for Apple earnings",
                 "tool": "vector_db_search",
                 "params": {"query": "Apple AAPL quarterly earnings 2024", "top_k": 5},
                 "depends_on": [], "fallback_if_fails": None, "optional": False},
                {"step_id": 2, "description": "Retrieve Apple quarterly income statement",
                 "tool": "financial_data_api",
                 "params": {"ticker": "AAPL", "statement_type": "income_statement",
                            "period": "quarterly", "limit": 4},
                 "depends_on": [1], "fallback_if_fails": "sec_filing_search", "optional": False},
                {"step_id": 3, "description": "Retrieve Apple earnings call transcript",
                 "tool": "earnings_transcript",
                 "params": {"ticker": "AAPL", "quarter": "Q4", "year": 2024},
                 "depends_on": [1], "fallback_if_fails": "web_search", "optional": False},
                {"step_id": 4, "description": "Search analyst earnings coverage",
                 "tool": "web_search",
                 "params": {"query": "Apple AAPL Q4 2024 earnings beat miss estimates",
                            "num_results": 8},
                 "depends_on": [], "fallback_if_fails": "news_sentiment", "optional": False},
                {"step_id": 5, "description": "Score market sentiment around earnings",
                 "tool": "news_sentiment",
                 "params": {"query": "Apple AAPL earnings Q4 2024",
                            "num_articles": 20, "lookback_days": 14},
                 "depends_on": [4], "fallback_if_fails": None, "optional": True},
            ],
        })
    if "identify hallucinations" in system:
        return json.dumps({"hallucination_count": 0, "unsupported_claims": [],
                           "verified_claims_count": 14, "confidence_score": 0.91})
    tools = ["vector_db_search", "financial_data_api",
             "earnings_transcript", "web_search", "news_sentiment"]
    import re
    m = re.search(r"step \((\d+)/", user)
    idx = int(m.group(1)) - 1 if m else 0
    tool = tools[min(idx, len(tools) - 1)]
    params = {"ticker": "AAPL"} if tool != "news_sentiment" else \
             {"query": "Apple earnings", "num_articles": 20, "lookback_days": 14}
    return (
        f"THOUGHT: Executing step.\n\n"
        f'ACTION:\n{{"tool": "{tool}", "params": {json.dumps(params)}}}\n\n'
        f"OBSERVATION: Data retrieved.\n\nDECIDE: step_complete"
    )


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

MSFT_MOCK = {
    "vector_db_search": {"results": [], "cache_hit": False, "error": None},
    "company_profile": {
        "ticker": "MSFT", "company_name": "Microsoft Corporation",
        "exchange": "NASDAQ", "sector": "Technology",
        "industry": "Software—Infrastructure",
        "market_cap": 3_120_000_000_000, "price": 418.50, "beta": 0.90,
        "employees": 228_000, "ceo": "Satya Nadella", "founded": "1975",
        "headquarters": {"city": "Redmond", "state": "WA", "country": "US"},
        "description": (
            "Microsoft develops software, services, devices and solutions worldwide. "
            "Key segments: Productivity & Business Processes (Office 365, LinkedIn), "
            "Intelligent Cloud (Azure, GitHub), More Personal Computing (Windows, Xbox)."
        ),
        "website": "https://www.microsoft.com",
        "source_tier": 2, "source_weight": 0.90, "error": None,
    },
    "financial_data_api": {
        "ticker": "MSFT", "statement_type": "key_metrics", "period": "annual",
        "records": [{"date": "2024-06-30", "peRatio": 37.2,
                     "returnOnEquity": 0.389, "debtToEquity": 0.35,
                     "freeCashFlowPerShare": 9.94, "enterpriseValueOverEBITDA": 27.4}],
        "record_count": 1, "data_source": "FMP (mock)",
        "source_tier": 2, "source_weight": 0.90, "error": None,
    },
    "web_search": {
        "results": [
            {"title": "Microsoft Q1 FY2025: Revenue $65.6B, Azure +33%",
             "snippet": "Microsoft beat estimates; Azure grew 33% YoY.",
             "url": "https://reuters.com/msft-q1-2025", "published": "2024-10-30"},
            {"title": "Microsoft Copilot Reaches 600+ Enterprise Customers",
             "snippet": "AI adoption accelerates across Microsoft 365.",
             "url": "https://ft.com/msft-copilot", "published": "2024-11-01"},
        ],
        "result_count": 2, "source_tier": 4, "source_weight": 0.60, "error": None,
    },
}

AAPL_MOCK = {
    "vector_db_search": {"results": [], "cache_hit": False, "error": None},
    "financial_data_api": {
        "ticker": "AAPL", "statement_type": "income_statement", "period": "quarterly",
        "records": [
            {"date": "2024-09-28", "revenue": 94_930_000_000,
             "grossProfit": 43_880_000_000, "netIncome": 14_736_000_000,
             "eps": 0.97, "revenueGrowth": 0.061},
            {"date": "2024-06-29", "revenue": 85_780_000_000,
             "grossProfit": 39_680_000_000, "netIncome": 21_448_000_000,
             "eps": 1.40, "revenueGrowth": 0.049},
        ],
        "record_count": 2, "data_source": "FMP (mock)",
        "source_tier": 2, "source_weight": 0.90, "error": None,
    },
    "earnings_transcript": {
        "ticker": "AAPL", "quarter": "Q4", "year": 2024,
        "transcript": (
            "Tim Cook: We are pleased to report record September quarter revenue of "
            "$94.9 billion, up 6% year over year. Services revenue hit an all-time high "
            "of $24.2 billion. iPhone revenue was $46.2 billion. "
            "Luca Maestri: Gross margin was 46.2%, above our guidance. We returned "
            "$29 billion to shareholders. For Q1 FY2025, we expect low to mid single "
            "digit revenue growth year over year."
        ),
        "speakers": ["Tim Cook (CEO)", "Luca Maestri (CFO)"],
        "source_tier": 3, "source_weight": 0.75,
        "note": "Mock — earnings_transcript tool scheduled for Day 7",
        "error": None,
    },
    "web_search": {
        "results": [
            {"title": "Apple Q4 2024: Revenue $94.9B Beats, EPS $0.97 Slight Miss",
             "snippet": "Revenue beat $94.4B estimate; EPS of $0.97 vs $1.01 consensus.",
             "url": "https://bloomberg.com/apple-q4-2024", "published": "2024-10-31"},
            {"title": "Apple Services Record $24.2B — Key Q4 Takeaway",
             "snippet": "Services +12% YoY is Apple's most reliable growth engine.",
             "url": "https://ft.com/apple-services", "published": "2024-10-31"},
            {"title": "Apple Q1 FY2025 Guidance: Low-to-Mid Single Digit Growth",
             "snippet": "Management cited currency headwinds and iPhone 16 ramp.",
             "url": "https://wsj.com/apple-2025", "published": "2024-11-01"},
        ],
        "result_count": 3, "source_tier": 4, "source_weight": 0.60, "error": None,
    },
    "news_sentiment": {
        "query": "Apple AAPL earnings Q4 2024",
        "aggregate": {"mean_score": 0.12, "overall_label": "positive",
                      "positive_pct": 55.0, "negative_pct": 20.0,
                      "neutral_pct": 25.0, "article_count": 20},
        "articles": [], "scorer_used": "VADER (mock)",
        "data_source": "NewsAPI (mock)",
        "source_tier": 4, "source_weight": 0.60, "error": None,
    },
}


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def _write_result(filename: str, title: str, query: str, state: dict, difficulty: str,
                  expected_tools: set, note: str = ""):
    os.makedirs("results", exist_ok=True)
    metrics  = state.get("evaluation_metrics", {})
    tool_log = state.get("tool_call_log", [])
    tools_used = {e["tool"] for e in tool_log}
    mode_tag = "REAL API" if REAL_MODE else "mock data"

    rows = ""
    for e in tool_log:
        rows += f"| {e.get('step_id')} | `{e.get('tool')}` | {str(e.get('params',''))[:60]} |\n"

    content = f"""# {title}
## ARA-1 Autonomous Financial Research Agent
**Query:** {query}
**Session:** {state['session_id']}
**Completed:** {state['completed_at']}
**Mode:** {mode_tag} | **Difficulty:** {difficulty}
{f'**Note:** {note}' if note else ''}

---

## Agent Research Output

{state.get('final_report', 'No report generated.')}

---

## Evaluation Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| FA-5 Hallucination Count | {metrics.get('FA-5_hallucination_count','N/A')} | 0 | {'✅' if metrics.get('FA-5_hallucination_count')==0 else '❌'} |
| AB-1 Tool Efficiency | {metrics.get('AB-1_tool_efficiency','N/A')} | ≥0.70 | {'✅' if (metrics.get('AB-1_tool_efficiency') or 0)>=0.70 else '⚠️'} |
| AB-4 Memory Utilization | {metrics.get('AB-4_memory_utilization','N/A')} | ≥0.30 | {'⚠️ first run' if (metrics.get('AB-4_memory_utilization') or 0)<0.30 else '✅'} |
| AB-5 Latency | {metrics.get('AB-5_latency_seconds','N/A')}s | <300s | {'✅' if (metrics.get('AB-5_latency_seconds') or 0)<300 else '❌'} |
| CO-2 Source Types | {metrics.get('CO-2_source_types_used','N/A')} | ≥{len(expected_tools)-1} | {'✅' if (metrics.get('CO-2_source_types_used') or 0)>=len(expected_tools)-1 else '⚠️'} |

**Expected tools:** {', '.join(f'`{t}`' for t in sorted(expected_tools))}
**Tools called:**   {', '.join(f'`{t}`' for t in sorted(tools_used))}

---

## Tool Call Trace

| Step | Tool | Params |
|------|------|--------|
{rows}
---
*Generated by ARA-1 | Mode: {mode_tag}*
"""
    with open(f"results/{filename}", "w") as f:
        f.write(content)
    print(f"  ✓ Written → results/{filename}")


# ---------------------------------------------------------------------------
# Challenge runners
# ---------------------------------------------------------------------------

CHALLENGE_1_QUERY = (
    "Create a comprehensive profile of Microsoft Corporation including "
    "business overview, financial summary, key executives, and recent developments."
)
CHALLENGE_2_QUERY = (
    "Analyze Apple Inc.'s most recent quarterly earnings. "
    "Compare actual results to consensus estimates and identify "
    "key takeaways from the earnings call."
)


def run_challenge_1() -> dict:
    print("\n" + "="*60)
    print("CHALLENGE 1 — Microsoft Company Profile  (Difficulty 1/5)")
    print(f"Mode: {'REAL API' if REAL_MODE else 'MOCK'}")
    print(f"  LLM keys: Anthropic={'✓' if HAS_ANTHROPIC else '✗'} | DeepSeek={'✓' if HAS_DEEPSEEK else '✗'} | Gemini={'✓' if HAS_GEMINI else '✗'}")
    print("="*60)

    if not REAL_MODE:
        _llm_fn["current"] = _llm_msft

    mock = None if REAL_MODE else MSFT_MOCK
    state = run_agent(query=CHALLENGE_1_QUERY, mock_data=mock)

    assert state["final_report"],  "No final report generated"
    assert state["hallucination_count"] == 0, f"Hallucinations: {state['hallucination_count']}"
    assert "MSFT" in state["companies"], f"MSFT not extracted: {state['companies']}"

    tools_used = {e["tool"] for e in state["tool_call_log"]}
    expected   = {"company_profile", "financial_data_api", "web_search"}
    assert expected & tools_used, f"No expected tools called. Got: {tools_used}"

    m = state["evaluation_metrics"]
    print(f"  ✓ Companies:       {state['companies']}")
    print(f"  ✓ Hallucinations:  {state['hallucination_count']} (target 0)")
    print(f"  ✓ Tools called:    {tools_used}")
    print(f"  ✓ Tool efficiency: {m.get('AB-1_tool_efficiency')}")
    print(f"  ✓ Latency:         {m.get('AB-5_latency_seconds')}s")
    print(f"  ✓ Source types:    {m.get('CO-2_source_types_used')}")

    _write_result("challenge_1.md", "Challenge 1: Microsoft Corporation — Company Profile",
                  CHALLENGE_1_QUERY, state, "1/5", expected)  # type: ignore
    print("  CHALLENGE 1 PASSED ✅")
    return state  # type: ignore


def run_challenge_2() -> dict:
    print("\n" + "="*60)
    print("CHALLENGE 2 — Apple Earnings Analysis  (Difficulty 2/5)")
    print(f"Mode: {'REAL API (earnings_transcript: mock)' if REAL_MODE else 'MOCK'}")
    print(f"  LLM keys: Anthropic={'✓' if HAS_ANTHROPIC else '✗'} | DeepSeek={'✓' if HAS_DEEPSEEK else '✗'} | Gemini={'✓' if HAS_GEMINI else '✗'}")
    print("="*60)

    if not REAL_MODE:
        _llm_fn["current"] = _llm_aapl

    # Always mock earnings_transcript — real impl scheduled Day 7
    mock = AAPL_MOCK if not REAL_MODE else {"earnings_transcript": AAPL_MOCK["earnings_transcript"]}
    state = run_agent(query=CHALLENGE_2_QUERY, mock_data=mock)

    assert state["final_report"],  "No final report generated"
    assert state["hallucination_count"] == 0, f"Hallucinations: {state['hallucination_count']}"
    assert "AAPL" in state["companies"], f"AAPL not extracted: {state['companies']}"

    tools_used = {e["tool"] for e in state["tool_call_log"]}
    expected   = {"financial_data_api", "earnings_transcript", "web_search", "news_sentiment"}
    assert expected & tools_used, f"No expected tools called. Got: {tools_used}"

    m = state["evaluation_metrics"]
    print(f"  ✓ Companies:       {state['companies']}")
    print(f"  ✓ Hallucinations:  {state['hallucination_count']} (target 0)")
    print(f"  ✓ Tools called:    {tools_used}")
    print(f"  ✓ Tool efficiency: {m.get('AB-1_tool_efficiency')}")
    print(f"  ✓ Latency:         {m.get('AB-5_latency_seconds')}s")
    print(f"  ✓ Source types:    {m.get('CO-2_source_types_used')}")

    _write_result("challenge_2.md", "Challenge 2: Apple Inc. — Earnings Analysis",
                  CHALLENGE_2_QUERY, state, "2/5", expected,  # type: ignore
                  note="`earnings_transcript` uses mock data — real tool scheduled for Day 7.")
    print("  CHALLENGE 2 PASSED ✅")
    return state  # type: ignore


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    passed, failed = 0, 0

    for name, fn in [("Challenge 1", run_challenge_1), ("Challenge 2", run_challenge_2)]:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"\n  ✗ {name} FAILED: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"RESULTS: {passed}/2 challenges passed | {failed} failed")
    if failed == 0:
        print("Day 5 Challenges — COMPLETE ✅")
        print("Outputs: results/challenge_1.md  results/challenge_2.md")
    else:
        print("Some challenges FAILED ❌ — check output above.")
    print("="*60)
    sys.exit(0 if failed == 0 else 1)