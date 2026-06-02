"""
tests/test_day4_challenge1.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

Day 4 — Task 5: Test the agent with Challenge 1 (Microsoft company profile)
using mock tool data. No real API calls made.

Run with:
    python tests/test_day4_challenge1.py
"""

import sys
import types
import json
import logging
from typing import Any, cast

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("test_day4")

# ---------------------------------------------------------------------------
# Stub external dependencies (anthropic + langgraph)
# Type annotated as Any to prevent Pylance "Unknown Attribute" exceptions
# ---------------------------------------------------------------------------

def _install_stubs():
    """Install minimal stubs for anthropic and langgraph."""

    # --- anthropic stub ---
    # FIX: Cast to Any so Pylance allows dynamic runtime property assignments
    anthropic_mod: Any = types.ModuleType("anthropic")

    class _Content:
        def __init__(self, text): self.text = text

    class _Response:
        def __init__(self, text): self.content = [_Content(text)]

    class _Messages:
        @staticmethod
        def create(model, max_tokens, temperature, system, messages, **kw):
            user_content = messages[-1]["content"]

            # Planner: return a valid research plan for MSFT
            if "produce a structured JSON research plan" in system:
                plan = {
                    "plan_id": "plan-MSFT-factual-2024-001",
                    "query_summary": "Retrieve Microsoft Corporation company profile.",
                    "query_type": "factual",
                    "companies": ["MSFT"],
                    "estimated_tool_calls": 4,
                    "steps": [
                        {
                            "step_id": 1,
                            "description": "Check long-term memory for cached Microsoft data",
                            "tool": "vector_db_search",
                            "params": {"query": "Microsoft company profile MSFT", "top_k": 5},
                            "depends_on": [],
                            "fallback_if_fails": None,
                            "optional": False,
                        },
                        {
                            "step_id": 2,
                            "description": "Retrieve Microsoft company profile from FMP",
                            "tool": "company_profile",
                            "params": {"ticker": "MSFT"},
                            "depends_on": [1],
                            "fallback_if_fails": "web_search",
                            "optional": False,
                        },
                        {
                            "step_id": 3,
                            "description": "Retrieve Microsoft key financial metrics",
                            "tool": "financial_data_api",
                            "params": {"ticker": "MSFT", "statement_type": "key_metrics"},
                            "depends_on": [1],
                            "fallback_if_fails": "sec_filing_search",
                            "optional": False,
                        },
                        {
                            "step_id": 4,
                            "description": "Search for recent Microsoft news",
                            "tool": "web_search",
                            "params": {"query": "Microsoft MSFT latest news 2024", "num_results": 5},
                            "depends_on": [],
                            "fallback_if_fails": "news_sentiment",
                            "optional": True,
                        },
                    ],
                }
                return _Response(json.dumps(plan))

            # Verifier: return clean verification
            if "identify hallucinations" in system:
                verification = {
                    "hallucination_count": 0,
                    "unsupported_claims": [],
                    "verified_claims_count": 8,
                    "confidence_score": 0.92,
                }
                return _Response(json.dumps(verification))

            # Executor: alternate step_complete and continue decisions
            react = (
                "THOUGHT: Memory cache is empty for Microsoft. "
                "I will call the requested tool to retrieve the data.\n\n"
                'ACTION:\n{"tool": "company_profile", "params": {"ticker": "MSFT"}}\n\n'
                "OBSERVATION: Tool result will be populated after execution.\n\n"
                "DECIDE: step_complete"
            )
            return _Response(react)

    class _Anthropic:
        messages = _Messages()

    anthropic_mod.Anthropic = _Anthropic
    sys.modules["anthropic"] = anthropic_mod

    # --- langgraph stub ---
    # FIX: Cast graph modules to Any to satisfy dynamic execution parameters
    lg_mod: Any = types.ModuleType("langgraph")
    lg_graph_mod: Any = types.ModuleType("langgraph.graph")

    class _StateGraph:
        def __init__(self, schema):
            self._nodes = {}
            self._entry = None
            self._edges = []
            self._cond_edges = {}

        def add_node(self, name, fn):
            self._nodes[name] = fn

        def set_entry_point(self, name):
            self._entry = name

        def add_edge(self, a, b):
            self._edges.append((a, b))

        def add_conditional_edges(self, src, fn, mapping):
            self._cond_edges[src] = (fn, mapping)

        def compile(self):
            return _CompiledGraph(self)

    class _CompiledGraph:
        def __init__(self, graph): self._g = graph

        def invoke(self, state):
            nodes = self._g._nodes
            cond = self._g._cond_edges

            order = ["analyze_query", "plan", "memory_precheck"]

            for node_name in order:
                state = nodes[node_name](state)

            for _ in range(20):
                state = nodes["execute_step"](state)
                route_fn, mapping = cond["execute_step"]
                next_node = mapping[route_fn(state)]
                if next_node == "synthesize":
                    break

            for node_name in ["synthesize", "verify", "generate_report", "evaluate"]:
                state = nodes[node_name](state)

            return state

    lg_graph_mod.StateGraph = _StateGraph
    lg_graph_mod.END = "__end__"
    lg_mod.graph = lg_graph_mod
    sys.modules["langgraph"] = lg_mod
    sys.modules["langgraph.graph"] = lg_graph_mod


_install_stubs()

# ---------------------------------------------------------------------------
# Now import the actual agent modules
# ---------------------------------------------------------------------------
from agent.core import run_agent, MAX_TOOL_CALLS, MAX_ITERATIONS, MAX_CYCLES_PER_STEP
from agent.prompts import build_planner_prompt, build_executor_prompt
from agent.parser import parse_plan, parse_react_step, parse_verification
from tools.schemas import ALL_SCHEMAS

# ---------------------------------------------------------------------------
# Mock tool data — realistic Microsoft data
# ---------------------------------------------------------------------------
MSFT_MOCK_DATA = {
    "vector_db_search": {
        "results": [],
        "cache_hit": False,
        "note": "No prior Microsoft research in memory.",
    },
    "company_profile": {
        "name": "Microsoft Corporation",
        "ticker": "MSFT",
        "exchange": "NASDAQ",
        "sector": "Technology",
        "industry": "Software—Infrastructure",
        "market_cap": 3_120_000_000_000,
        "market_cap_formatted": "$3.12T",
        "ceo": "Satya Nadella",
        "employees": 228_000,
        "founded": 1975,
        "headquarters": "Redmond, Washington, USA",
        "description": (
            "Microsoft Corporation develops and supports software, services, "
            "devices, and solutions worldwide. Segments: Productivity and Business "
            "Processes, Intelligent Cloud, and More Personal Computing."
        ),
        "website": "https://www.microsoft.com",
        "isin": "US5949181045",
    },
    "financial_data_api": {
        "ticker": "MSFT",
        "period": "FY2024",
        "revenue": 245_122_000_000,
        "revenue_growth_yoy": 0.1599,
        "operating_income": 109_433_000_000,
        "operating_margin": 0.4465,
        "net_income": 88_136_000_000,
        "eps": 11.80,
        "pe_ratio": 37.2,
        "debt_to_equity": 0.35,
        "return_on_equity": 0.389,
        "free_cash_flow": 74_071_000_000,
        "source": "Financial Modeling Prep",
    },
    "web_search": {
        "results": [
            {
                "title": "Microsoft Q1 FY2025 Earnings Beat Estimates",
                "snippet": "Microsoft reported Q1 FY2025 revenue of $65.6B, up 16% YoY, driven by Azure cloud growth of 33%.",
                "url": "https://reuters.com/technology/microsoft-earnings-2024",
                "date": "2024-10-30",
            },
            {
                "title": "Microsoft Copilot AI Integration Expands Across Products",
                "snippet": "Microsoft continues integrating its Copilot AI assistant across Office 365, Azure, and Windows.",
                "url": "https://ft.com/microsoft-copilot-2024",
                "date": "2024-11-01",
            },
        ]
    },
}

# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_task1_reasoning_loop():
    """Task 1: Verify the full reasoning loop executes all 8 nodes."""
    print("\n" + "="*60)
    print("TASK 1 — Reasoning loop (Plan-Execute-Verify cycle)")
    print("="*60)

    state = run_agent(
        query="Microsoft MSFT company profile overview",
        mock_data=MSFT_MOCK_DATA,
    )

    assert state["final_report"] is not None, "No final report generated"
    assert state["completed_at"] is not None, "No completion timestamp"
    assert len(state["research_plan"]) > 0, "No research plan generated"
    assert state["current_step_index"] > 0, "No steps executed"

    print(f"  ✓ Research plan: {len(state['research_plan'])} steps")
    print(f"  ✓ Steps executed: {state['current_step_index']}")
    print(f"  ✓ Tool calls made: {state['total_api_calls']}")
    print(f"  ✓ Final report: {len(state['final_report'])} chars")
    print(f"  ✓ Completed at: {state['completed_at']}")
    print("  PASSED")
    return state


def test_task2_prompt_injection():
    """Task 2: Verify tool registry is injected into the system prompt."""
    print("\n" + "="*60)
    print("TASK 2 — System prompt tool registry injection")
    print("="*60)

    planner_prompt = build_planner_prompt(ALL_SCHEMAS)
    executor_prompt = build_executor_prompt(ALL_SCHEMAS, gathered_data=None)

    tool_names = [s["name"] for s in ALL_SCHEMAS]
    for name in tool_names:
        assert name in planner_prompt, f"Tool '{name}' missing from planner prompt"

    assert "Tier 1" in planner_prompt, "Source tiers not injected into planner"
    assert "Tier 1" in executor_prompt, "Source tiers not injected into executor"
    assert "Memory First" in executor_prompt, "Memory First rule missing from executor"
    assert "Never invent data" in executor_prompt, "Anti-hallucination rule missing"

    print(f"  ✓ All {len(tool_names)} tool schemas injected into planner prompt")
    print(f"  ✓ Source reliability tiers injected (Planner + Executor)")
    print(f"  ✓ Memory First rule present in executor")
    print(f"  ✓ Anti-hallucination rule present in executor")
    print("  PASSED")


def test_task3_parsing():
    """Task 3: Verify LLM response parsing extracts tool calls and traces."""
    print("\n" + "="*60)
    print("TASK 3 — LLM response parsing")
    print("="*60)

    plan_json = json.dumps({
        "plan_id": "plan-MSFT-factual-2024-001",
        "query_summary": "Microsoft company profile.",
        "query_type": "factual",
        "companies": ["MSFT"],
        "estimated_tool_calls": 4,
        "steps": [
            {"step_id": 1, "description": "Check memory", "tool": "vector_db_search",
             "params": {"query": "MSFT"}, "depends_on": [], "fallback_if_fails": None, "optional": False},
        ],
    })
    plan = parse_plan(plan_json)
    assert plan.plan_id == "plan-MSFT-factual-2024-001"
    assert plan.companies == ["MSFT"]
    assert len(plan.steps) == 1
    print(f"  ✓ parse_plan: plan_id={plan.plan_id} | {len(plan.steps)} step(s)")

    react_text = (
        "THOUGHT: I need to check memory before making an external call.\n\n"
        'ACTION:\n{"tool": "vector_db_search", "params": {"query": "Microsoft profile", "top_k": 5}}\n\n'
        "OBSERVATION: Cache miss — no prior data found.\n\n"
        "DECIDE: continue"
    )
    step = parse_react_step(react_text)
    assert step.tool_name == "vector_db_search"
    assert step.tool_params["query"] == "Microsoft profile"
    assert step.decision == "continue"
    print(f"  ✓ parse_react_step: tool={step.tool_name} | decision={step.decision}")

    verify_json = json.dumps({
        "hallucination_count": 0,
        "unsupported_claims": [],
        "verified_claims_count": 8,
        "confidence_score": 0.92,
    })
    result = parse_verification(verify_json)
    assert result.hallucination_count == 0
    assert result.confidence_score == 0.92
    print(f"  ✓ parse_verification: hallucinations={result.hallucination_count} | confidence={result.confidence_score}")

    print("  PASSED")


def test_task4_limits():
    """Task 4: Verify iteration limits and termination conditions."""
    print("\n" + "="*60)
    print("TASK 4 — Iteration limits and termination conditions")
    print("="*60)

    from agent.core import should_continue_execution

    # FIX: Wrap dictionary literals with a typing bypass cast to resolve Pylance's ResearchState conflict flags
    # Limit 1: all steps complete → synthesize
    s = cast(Any, {"current_step_index": 4, "research_plan": [{}]*4,
                   "total_api_calls": 5, "iteration_count": 5})
    assert should_continue_execution(s) == "synthesize"
    print(f"  ✓ Steps exhausted → routes to synthesize")

    # Limit 2: tool call ceiling → synthesize
    s2 = cast(Any, {"current_step_index": 0, "research_plan": [{}]*4,
                    "total_api_calls": MAX_TOOL_CALLS, "iteration_count": 0})
    assert should_continue_execution(s2) == "synthesize"
    print(f"  ✓ MAX_TOOL_CALLS={MAX_TOOL_CALLS} reached → routes to synthesize")

    # Limit 3: iteration ceiling → synthesize
    s3 = cast(Any, {"current_step_index": 0, "research_plan": [{}]*4,
                    "total_api_calls": 0, "iteration_count": MAX_ITERATIONS})
    assert should_continue_execution(s3) == "synthesize"
    print(f"  ✓ MAX_ITERATIONS={MAX_ITERATIONS} reached → routes to synthesize")

    # Limit 4: steps still remain → execute_step
    s4 = cast(Any, {"current_step_index": 1, "research_plan": [{}]*4,
                    "total_api_calls": 3, "iteration_count": 3})
    assert should_continue_execution(s4) == "execute_step"
    print(f"  ✓ Steps remaining → continues execute_step loop")

    print(f"  ✓ MAX_CYCLES_PER_STEP={MAX_CYCLES_PER_STEP} (enforced inside execute_step)")
    print("  PASSED")


def test_task5_challenge1(state):
    """Task 5: Full Challenge 1 — Microsoft company profile with mock data."""
    print("\n" + "="*60)
    print("TASK 5 — Challenge 1: Microsoft Company Profile (mock data)")
    print("="*60)

    assert state["query_type"] in ("factual", "analytical", "comparative")
    assert "MSFT" in state["companies"]
    assert state["hallucination_count"] == 0, (
        f"Unexpected hallucinations: {state['hallucination_count']}"
    )
    assert state["evaluation_metrics"]["FA-5_hallucination_count"] == 0
    assert state["evaluation_metrics"]["AB-4_memory_utilization"] >= 0

    total_calls = state["total_api_calls"]
    memory_hits = state["memory_hits"]
    expected_ab4 = round(memory_hits / total_calls, 3) if total_calls > 0 else 0.0
    assert state["evaluation_metrics"]["AB-4_memory_utilization"] == expected_ab4, \
        "AB-4 formula error — check division vs multiplication"

    print(f"  ✓ Query type detected: {state['query_type']}")
    print(f"  ✓ Companies extracted: {state['companies']}")
    print(f"  ✓ Hallucination count: {state['hallucination_count']} (target: 0)")
    print(f"  ✓ AB-4 memory utilization: {state['evaluation_metrics']['AB-4_memory_utilization']}")
    print(f"  ✓ AB-1 tool efficiency: {state['evaluation_metrics']['AB-1_tool_efficiency']}")
    print(f"  ✓ Latency: {state['evaluation_metrics']['AB-5_latency_seconds']}s")
    print(f"  ✓ Source types used: {state['evaluation_metrics']['CO-2_source_types_used']}")

    _write_challenge_result(state)
    print("  PASSED")


def _write_challenge_result(state):
    """Write Challenge 1 result to results/challenge_1.md"""
    import os
    
    # FIX: Replaced absolute root path '/home/claude/results' with local relative path 'results'
    # This prevents runtime OS Permission Denied errors on macOS
    os.makedirs("results", exist_ok=True)

    report = state.get("final_report", "No report generated.")
    metrics = state.get("evaluation_metrics", {})

    content = f"""# Challenge 1: Microsoft Company Profile
## ARA-1 Autonomous Financial Research Agent
**Session:** {state['session_id']}
**Completed:** {state['completed_at']}
**Query:** Microsoft MSFT company profile overview

---

## Agent Output

{report}

---

## Evaluation Metrics

| Metric | Value | Target | Pass? |
|--------|-------|--------|-------|
| FA-5 Hallucination Count | {metrics.get('FA-5_hallucination_count', 'N/A')} | 0 | {'✅' if metrics.get('FA-5_hallucination_count') == 0 else '❌'} |
| AB-1 Tool Efficiency | {metrics.get('AB-1_tool_efficiency', 'N/A')} | ≥0.70 | {'✅' if (metrics.get('AB-1_tool_efficiency') or 0) >= 0.70 else '⚠️'} |
| AB-4 Memory Utilization | {metrics.get('AB-4_memory_utilization', 'N/A')} | ≥0.30 | {'⚠️ (first run — no cache yet)' if metrics.get('AB-4_memory_utilization', 0) < 0.3 else '✅'} |
| AB-5 Latency | {metrics.get('AB-5_latency_seconds', 'N/A')}s | <300s | {'✅' if (metrics.get('AB-5_latency_seconds') or 0) < 300 else '❌'} |
| CO-2 Source Types | {metrics.get('CO-2_source_types_used', 'N/A')} | ≥4 | {'✅' if (metrics.get('CO-2_source_types_used') or 0) >= 4 else '⚠️'} |

**Note:** AB-4 Memory Utilization is expected to be 0 on first run — no prior Microsoft
research exists in the vector DB. Subsequent runs will improve this metric.

---

## Tool Call Trace

| Step | Tool | Decision |
|------|------|----------|
"""
    for entry in state.get("tool_call_log", []):
        content += f"| {entry.get('step_id')} | {entry.get('tool')} | recorded |\n"

    content += "\n---\n*Generated by ARA-1 Day 4 test suite — mock data only*\n"

    # FIX: Relative file write path string routing safely inside your workspace directory tree
    with open("results/challenge_1.md", "w") as f:
        f.write(content)
    print("  ✓ Challenge 1 result written to results/challenge_1.md")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ARA-1 — Day 4 Validation Suite")
    print("Challenge 1: Microsoft Company Profile")
    print("="*60)

    passed = 0
    failed = 0

    tests = [
        ("Task 1 — Reasoning loop", lambda s: test_task1_reasoning_loop()),
        ("Task 2 — Prompt injection", lambda s: test_task2_prompt_injection()),
        ("Task 3 — LLM parsing",     lambda s: test_task3_parsing()),
        ("Task 4 — Iteration limits",lambda s: test_task4_limits()),
    ]

    state = None
    for name, fn in tests:
        try:
            result = fn(state)
            if result:
                state = result
            passed += 1
        except Exception as e:
            print(f"\n  ✗ FAILED: {e}")
            failed += 1

    if state:
        try:
            test_task5_challenge1(state)
            passed += 1
        except Exception as e:
            print(f"\n  ✗ Task 5 FAILED: {e}")
            failed += 1

    print("\n" + "="*60)
    print(f"RESULTS: {passed}/5 tasks passed | {failed} failed")
    if failed == 0:
        print("Day 4 — ALL TASKS COMPLETE ✅")
    else:
        print("Day 4 — SOME TASKS FAILED ❌")
    print("="*60)
    sys.exit(0 if failed == 0 else 1)