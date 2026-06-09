"""
agent/core.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

The main ARA-1 agent — a LangGraph StateGraph implementing the
Hybrid Plan-Execute + ReAct reasoning loop.

Graph nodes:
    analyze_query     — classify query type, extract tickers
    plan              — planner LLM call → structured research plan
    memory_precheck   — check vector DB before any external calls
    execute_step      — ReAct micro-loop for one plan step
    synthesize        — weight sources, resolve conflicts (stub → Day 9)
    verify            — hallucination check on draft report
    generate_report   — format final output (stub → Day 10)
    evaluate          — compute 20+ metrics (stub → Day 11)

Edges:
    analyze_query → plan
    plan → memory_precheck → execute_step
    execute_step → execute_step  (loop while steps remain)
    execute_step → synthesize    (when all steps done)
    synthesize → verify → generate_report → evaluate → END
"""

import json
import logging
import re
import time
from datetime import datetime
from typing import Any, Optional

import anthropic
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from agent.prompts import (
    build_planner_prompt,
    build_executor_prompt,
    build_verifier_prompt,
    build_disambiguation_prompt,
)
from agent.parser import (
    parse_plan,
    parse_react_step,
    parse_verification,
    ResearchPlan,
    ReActStep,
)
from tools.schemas import ALL_SCHEMAS

# Real tool implementations (Days 5–7)
# Only import tools that are fully implemented — stubs stay as None
from tools.sec_edgar          import sec_filing_search
from tools.financial_api      import financial_data_api
from tools.web_search         import web_search
from tools.news_sentiment     import news_sentiment
from tools.company_profile    import company_profile
from memory.vector_db_tools   import vector_db_search, vector_db_store
from memory.context_manager   import ContextManager
from memory.episodic           import get_episodic_memory

# Dispatch table: tool_name → callable
# Stubs (not yet built) fall through to the stub response in _execute_tool
TOOL_DISPATCH: dict[str, Any] = {
    "sec_filing_search":  sec_filing_search,
    "financial_data_api": financial_data_api,
    "web_search":         web_search,
    "news_sentiment":     news_sentiment,
    "company_profile":    company_profile,
    "vector_db_search":   vector_db_search,   # Day 6
    "vector_db_store":    vector_db_store,    # Day 6
    # Day 7: peer_comparison, earnings, calculator, fact_checker, report_gen
}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Design constraints (from architecture spec Section 1.4)
# ---------------------------------------------------------------------------
MAX_TOOL_CALLS = 20
MAX_ITERATIONS = 15
MAX_CYCLES_PER_STEP = 3          # ReAct micro-loop limit per plan step
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096


# ---------------------------------------------------------------------------
# Shared State — passed between every graph node
# ---------------------------------------------------------------------------
class ResearchState(TypedDict):
    # Input
    raw_query: str
    query_type: str
    companies: list[str]
    clarifications: Optional[str]

    # Planning
    research_plan: list[dict]
    current_step_index: int
    plan_revision_count: int

    # Execution
    gathered_data: dict[str, Any]
    tool_call_log: list[dict]
    memory_hits: int
    total_api_calls: int
    iteration_count: int

    # Synthesis
    draft_report: Optional[str]
    conflicts_detected: list[dict]
    conflicts_resolved: list[dict]

    # Verification
    verified_claims: list[dict]
    flagged_claims: list[dict]
    hallucination_count: int

    # Output
    final_report: Optional[str]
    confidence_scores: dict[str, float]
    evaluation_metrics: dict[str, Any]
    session_id: str
    completed_at: Optional[str]
    error_log: list[str]


# ---------------------------------------------------------------------------
# LLM client helper
# ---------------------------------------------------------------------------

def _call_llm(system: str, user: str, temperature: float = 0.2) -> str:
    """Single LLM call. Returns the text content of the first message block."""
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text#type: ignore


# ---------------------------------------------------------------------------
# Tool executor — real dispatch with mock override for testing
# ---------------------------------------------------------------------------

def _execute_tool(tool_name: str, params: dict, mock_data: Optional[dict] = None) -> dict:
    """
    Execute a tool call.

    Priority:
      1. If mock_data contains tool_name → return mock result (test mode)
      2. If tool_name is in TOOL_DISPATCH → call the real implementation
      3. Otherwise → return stub response (tool not yet implemented)

    All real tool calls are wrapped in try/except so a failure returns a
    structured error dict instead of crashing the agent. The executor's
    fallback logic then handles the error via the fallback chain.
    """
    # 1. Mock override (testing / offline runs)
    if mock_data and tool_name in mock_data:
        logger.info("MOCK tool call: %s | params=%s", tool_name, params)
        return {"source": tool_name, "data": mock_data[tool_name], "mock": True}

    # 2. Real tool dispatch
    if tool_name in TOOL_DISPATCH:
        fn = TOOL_DISPATCH[tool_name]
        logger.info("REAL tool call: %s | params=%s", tool_name, params)
        try:
            result = fn(**params)
            # All real tools return a dict with an 'error' key
            if result.get("error"):
                logger.warning("Tool %s returned error: %s", tool_name, result["error"])
            return result
        except TypeError as e:
            # Param mismatch — log and return structured error
            logger.error("Tool %s param error: %s | params=%s", tool_name, e, params)
            return _tool_error(tool_name, f"Parameter error: {e}")
        except Exception as e:
            logger.error("Tool %s raised unexpected error: %s", tool_name, e)
            return _tool_error(tool_name, str(e))

    # 3. Stub — tool not yet implemented
    logger.info("STUB tool call: %s | params=%s", tool_name, params)
    return {
        "source": tool_name,
        "data": None,
        "mock": False,
        "error": None,
        "note": f"Tool '{tool_name}' not yet implemented. Scheduled for Days 6–7.",
    }


def _tool_error(tool_name: str, message: str) -> dict:
    """Structured error response — signals the executor to try a fallback."""
    return {
        "source":  tool_name,
        "data":    None,
        "mock":    False,
        "error":   message,
        "note":    f"Tool '{tool_name}' failed: {message}",
    }


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------

def node_analyze_query(state: ResearchState) -> ResearchState:
    """
    Classify the query type and extract company tickers.
    Uses simple heuristics; full NER added in query_analyzer.py (Day 10).
    """
    query = state["raw_query"].lower()

    # Simple query type classification
    if any(w in query for w in ["compare", "vs", "versus", "between"]):
        query_type = "comparative"
    elif any(w in query for w in ["why", "how", "explain", "analyse", "analyze"]):
        query_type = "analytical"
    elif "?" not in query and len(query.split()) < 8:
        query_type = "factual"
    else:
        query_type = "factual"

    # Company name → ticker mapping (handles natural language queries without tickers)
    NAME_TO_TICKER = {
        "microsoft": "MSFT", "apple": "AAPL", "google": "GOOGL",
        "alphabet": "GOOGL", "amazon": "AMZN", "meta": "META",
        "tesla": "TSLA", "nvidia": "NVDA", "netflix": "NFLX",
        "salesforce": "CRM", "adobe": "ADBE", "intel": "INTC",
        "ibm": "IBM", "oracle": "ORCL", "qualcomm": "QCOM",
        "jpmorgan": "JPM", "goldman": "GS", "visa": "V",
        "walmart": "WMT", "pfizer": "PFE",
    }
    query_lower = state["raw_query"].lower()
    companies = [ticker for name, ticker in NAME_TO_TICKER.items() if name in query_lower]

    # Also catch explicit ticker symbols (uppercase 1–5 letter words)
    stopwords = {"A", "I", "AI", "CEO", "CFO", "US", "UK", "Q1", "Q2", "Q3", "Q4",
                 "YOY", "TTM", "IPO", "EPS", "DCF", "SEC", "FOR", "AND", "THE", "INC"}
    explicit = [t for t in re.findall(r'\b[A-Z]{1,5}\b', state["raw_query"])
                if t not in stopwords and t not in companies]
    companies = list(dict.fromkeys(companies + explicit))

    logger.info("Query analyzed: type=%s | companies=%s", query_type, companies)

    return {
        **state,
        "query_type": query_type,
        "companies": companies if companies else ["UNKNOWN"],
        "iteration_count": 0,
    }


def node_plan(state: ResearchState) -> ResearchState:
    """
    Call the Planner LLM to generate a structured research plan.
    """
    logger.info("Planning for query: %s", state["raw_query"])

    system_prompt = build_planner_prompt(ALL_SCHEMAS)

    # Include cache summary from memory_precheck so planner skips cached steps
    cache_note = state.get("clarifications", "") or "No prior data in memory cache."

    # Inject episodic memory — past strategies for this query type
    try:
        episodic_hint = get_episodic_memory().get_summary(state["query_type"])
    except Exception as _e:
        episodic_hint = "No episodic memory available."
        logger.debug("Episodic memory unavailable: %s", _e)

    user_msg = (
        f"Query: {state['raw_query']}\n"
        f"Query type: {state['query_type']}\n"
        f"Companies identified: {state['companies']}\n"
        f"Memory precheck results: {cache_note}\n"
        f"Episodic memory (past sessions): {episodic_hint}"
    )

    try:
        llm_response = _call_llm(system_prompt, user_msg)
        plan = parse_plan(llm_response)
        steps = plan.steps
        logger.info("Plan created: %s | %d steps", plan.plan_id, len(steps))
    except Exception as e:
        logger.error("Planning failed: %s — using fallback single-step plan", e)
        state["error_log"].append(f"Planning LLM error: {e}")
        steps = [{
            "step_id": 1,
            "description": "Fallback: retrieve company profile via web search",
            "tool": "web_search",
            "params": {"query": state["raw_query"], "num_results": 5},
            "depends_on": [],
            "fallback_if_fails": None,
            "optional": False,
        }]

    return {
        **state,
        "research_plan": steps,
        "current_step_index": 0,
    }


# Similarity threshold for treating a memory hit as sufficient to skip an API call
MEMORY_HIT_THRESHOLD = 0.85


def node_memory_precheck(state: ResearchState) -> ResearchState:
    """
    Search long-term vector memory BEFORE the planner makes any external API calls.
    Implements the Memory First principle from architecture spec Section 2.2.

    What this does:
      1. Formulates a retrieval query from the raw query + identified companies.
      2. Calls vector_db_search for each company to find cached research.
      3. Any result with similarity >= MEMORY_HIT_THRESHOLD is loaded directly
         into state["gathered_data"] — the planner will skip redundant API calls.
      4. Increments state["memory_hits"] for the AB-4 metric.
      5. Attaches a cache_summary to state so the planner knows what is
         already available and can skip those steps in the research plan.
    """
    query    = state["raw_query"]
    companies = state.get("companies", [])

    gathered      = dict(state.get("gathered_data", {}))
    memory_hits   = state.get("memory_hits", 0)
    cache_summary: list[str] = []

    for ticker in companies:
        # Build a targeted retrieval query per company
        search_query = f"{ticker} {query}"

        try:
            result = TOOL_DISPATCH["vector_db_search"](
                query=search_query,
                top_k=5,
                metadata_filter={"ticker": ticker},
                min_similarity=MEMORY_HIT_THRESHOLD,
            )
        except Exception as e:
            logger.warning("Memory precheck failed for %s: %s", ticker, e)
            continue

        hits = result.get("results", [])
        if not hits:
            logger.info("Memory precheck: no cache hit for %s (collection=%d)",
                        ticker, result.get("collection_size", 0))
            continue

        # Load each hit into gathered_data with a memory source tag
        for hit in hits:
            key = f"memory_{ticker}_{hit.get('metadata', {}).get('source_type', 'doc')}"
            gathered[key] = {
                "source":      "vector_db_search",
                "data":        hit["content"],
                "metadata":    hit.get("metadata", {}),
                "similarity":  hit["similarity"],
                "memory_hit":  True,
            }
            memory_hits += 1

        # Build a one-line summary for the planner
        top_sim   = result.get("top_similarity", 0.0)
        hit_count = len(hits)
        source_types = list({
            h.get("metadata", {}).get("source_type", "unknown") for h in hits
        })
        summary = (
            f"{ticker}: {hit_count} cached chunk(s) found "
            f"(top similarity {top_sim:.2f}, types: {source_types})"
        )
        cache_summary.append(summary)
        logger.info("Memory precheck HIT: %s", summary)

    # Attach cache_summary to state so node_plan can inject it into the
    # planner prompt — the planner will then skip already-cached steps
    clarifications = state.get("clarifications") or ""
    if cache_summary:
        lines = [f"  - {s}" for s in cache_summary]
        cache_note = "MEMORY CACHE AVAILABLE:" + "\n" + "\n".join(lines)
        clarifications = (clarifications + "\n" + cache_note).strip()

    return {
        **state,
        "gathered_data":  gathered,
        "memory_hits":    memory_hits,
        "clarifications": clarifications,
    }


def node_execute_step(state: ResearchState, mock_data: Optional[dict] = None, context_manager: Optional["ContextManager"] = None) -> ResearchState:
    """
    Execute the current plan step via a ReAct micro-loop.
    Max MAX_CYCLES_PER_STEP iterations before moving to the next step.
    """
    plan = state["research_plan"]
    idx = state["current_step_index"]

    # Termination: all steps complete
    if idx >= len(plan):
        logger.info("All plan steps complete.")
        return {**state, "current_step_index": idx}

    step = plan[idx]
    logger.info("Executing step %d/%d: %s", idx + 1, len(plan), step["description"])

    gathered = state.get("gathered_data", {})
    tool_log = state.get("tool_call_log", [])
    total_calls = state.get("total_api_calls", 0)
    iteration_count = state.get("iteration_count", 0)

    # Guard: global iteration limit
    if iteration_count >= MAX_ITERATIONS:
        logger.warning("Max iterations (%d) reached — terminating execution.", MAX_ITERATIONS)
        state["error_log"].append("Max iteration limit reached.")
        return {**state, "current_step_index": len(plan)}

    # Guard: global tool call limit
    if total_calls >= MAX_TOOL_CALLS:
        logger.warning("Max tool calls (%d) reached — skipping remaining steps.", MAX_TOOL_CALLS)
        state["error_log"].append("Max tool call limit reached.")
        return {**state, "current_step_index": len(plan)}

    # Context window check — compress if approaching limit
    # Use session-scoped CM from build_graph so context accumulates across all steps
    _cm = context_manager or ContextManager(session_id=state.get("session_id", "unknown"))
    ctx_usage = _cm.update(state)
    if _cm.should_summarise():
        logger.warning(
            "Context at %.1f%% — compressing TAO traces before step %d.",
            ctx_usage["pct_used"], idx + 1,
        )
        tool_log = _cm.summarise(
            state, llm_call_fn=_call_llm, keep_recent=3
        )
    if not _cm.has_headroom():
        logger.error("No headroom for step %d — halting.", idx + 1)
        state["error_log"].append("Context window full — execution halted.")
        return {**state, "current_step_index": len(plan)}

    # ReAct micro-loop (max MAX_CYCLES_PER_STEP cycles per step)
    system_prompt = build_executor_prompt(ALL_SCHEMAS, gathered)
    step_complete = False

    for cycle in range(MAX_CYCLES_PER_STEP):
        user_msg = (
            f"Current plan step ({idx + 1}/{len(plan)}):\n"
            f"{json.dumps(step, indent=2)}\n\n"
            f"Cycle: {cycle + 1}/{MAX_CYCLES_PER_STEP}\n"
            f"Total tool calls so far: {total_calls}/{MAX_TOOL_CALLS}"
        )

        try:
            llm_response = _call_llm(system_prompt, user_msg)
            react = parse_react_step(llm_response)
        except Exception as e:
            logger.error("ReAct parse error on step %d cycle %d: %s", idx, cycle, e)
            state["error_log"].append(f"ReAct parse error step {idx} cycle {cycle}: {e}")
            break

        # Execute the tool
        tool_result = {}
        if react.tool_name:
            tool_result = _execute_tool(react.tool_name, react.tool_params, mock_data)
            total_calls += 1
            iteration_count += 1

            # Log the tool call
            tool_log.append({
                "step_id": step["step_id"],
                "cycle": cycle + 1,
                "tool": react.tool_name,
                "params": react.tool_params,
                "thought": react.thought,
                "result_summary": str(tool_result.get("data", ""))[:200],
                "timestamp": datetime.utcnow().isoformat(),
            })

            # Store result in gathered_data keyed by tool + step
            key = f"{react.tool_name}_step{step['step_id']}"
            gathered[key] = tool_result

        # Evaluate decision
        decision = react.decision
        if decision == "step_complete":
            logger.info("Step %d complete after %d cycle(s).", idx + 1, cycle + 1)
            step_complete = True
            break
        elif decision == "fallback_needed":
            fallback = step.get("fallback_if_fails")
            if fallback:
                logger.info("Fallback triggered: switching to %s", fallback)
                step = {**step, "tool": fallback, "fallback_if_fails": None}
            else:
                logger.warning("No fallback available for step %d.", idx + 1)
                state["error_log"].append(f"Step {idx + 1} failed, no fallback.")
                step_complete = True  # Move on regardless
                break
        elif decision == "plan_revision_needed":
            logger.info("Plan revision triggered at step %d.", idx + 1)
            state["plan_revision_count"] = state.get("plan_revision_count", 0) + 1
            break
        elif decision == "conflict_detected":
            logger.warning("Conflict detected at step %d — logging for synthesis.", idx + 1)
            state["conflicts_detected"].append({
                "step_id": step["step_id"],
                "tool": react.tool_name,
                "note": react.thought,
            })
            step_complete = True
            break
        # "continue" — loop to next cycle

    return {
        **state,
        "gathered_data": gathered,
        "tool_call_log": tool_log,
        "total_api_calls": total_calls,
        "iteration_count": iteration_count,
        "current_step_index": idx + 1,
    }


def node_synthesize(state: ResearchState) -> ResearchState:
    """
    Combine all gathered data into a draft report.
    Full synthesis engine implemented in synthesis/engine.py (Day 9).
    This stub assembles gathered data into a structured draft.
    """
    logger.info("Synthesizing %d data sources...", len(state["gathered_data"]))

    gathered = state["gathered_data"]
    companies = state.get("companies", ["UNKNOWN"])
    query = state["raw_query"]

    # Build a simple draft from gathered data
    sections = [
        f"# Financial Research Report",
        f"**Query:** {query}",
        f"**Companies:** {', '.join(companies)}",
        f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Session:** {state['session_id']}",
        "",
        "## Data Sources Retrieved",
    ]

    for key, result in gathered.items():
        data = result.get("data")
        source = result.get("source", key)
        is_mock = result.get("mock", False)
        tag = " *(mock)*" if is_mock else ""
        if data:
            sections.append(f"- **{source}**{tag}: {str(data)[:300]}")
        else:
            note = result.get("note", "No data returned.")
            sections.append(f"- **{source}**{tag}: {note}")

    sections += [
        "",
        "## Research Methodology Notes",
        f"- Total tool calls made: {state['total_api_calls']}",
        f"- Memory hits: {state['memory_hits']}",
        f"- Plan steps executed: {state['current_step_index']}",
        f"- Conflicts detected: {len(state['conflicts_detected'])}",
        "",
        "*Full synthesis with source weighting and conflict resolution: Day 9.*",
    ]

    draft = "\n".join(sections)
    logger.info("Draft report assembled (%d chars).", len(draft))

    return {**state, "draft_report": draft}


def node_verify(state: ResearchState) -> ResearchState:
    """
    Run hallucination check on the draft report.
    Calls the verifier LLM with the draft + source context.
    """
    draft = state.get("draft_report", "")
    if not draft:
        logger.warning("No draft to verify.")
        return {**state, "hallucination_count": 0}

    logger.info("Running verification pass...")
    system_prompt = build_verifier_prompt()
    user_msg = (
        f"DRAFT REPORT:\n{draft}\n\n"
        f"SOURCE CONTEXT:\n{json.dumps(state['gathered_data'], indent=2)[:3000]}"
    )

    try:
        llm_response = _call_llm(system_prompt, user_msg)
        result = parse_verification(llm_response)
        return {
            **state,
            "hallucination_count": result.hallucination_count,
            "flagged_claims": result.unsupported_claims,
            "verified_claims": [{"count": result.verified_claims_count}],
            "confidence_scores": {"overall": result.confidence_score},
        }
    except Exception as e:
        logger.error("Verification failed: %s", e)
        state["error_log"].append(f"Verification error: {e}")
        return {**state, "hallucination_count": -1}


def node_generate_report(state: ResearchState) -> ResearchState:
    """
    Finalize and format the report.
    Full template engine in report_gen.py (Day 7).
    """
    logger.info("Generating final report...")
    draft = state.get("draft_report", "No data retrieved.")
    hallucination_count = state.get("hallucination_count", 0)

    if hallucination_count > 0:
        disclaimer = (
            f"\n\n---\n"
            f"⚠️ **Verification Warning:** {hallucination_count} claim(s) could not be "
            f"verified against source data and have been flagged for review.\n"
        )
        final = draft + disclaimer#type: ignore
    else:
        final = draft

    return {
        **state,
        "final_report": final,
        "completed_at": datetime.utcnow().isoformat(),
    }


def node_evaluate(state: ResearchState) -> ResearchState:
    """
    Compute evaluation metrics.
    Full 20+ metric suite in evaluation/metrics.py (Day 11).
    """
    total_calls = state.get("total_api_calls", 0)
    memory_hits = state.get("memory_hits", 0)
    tool_log = state.get("tool_call_log", [])

    # AB-4: Memory Utilization = memory_hits / total_api_calls (division, not multiplication)
    ab4 = round(memory_hits / total_calls, 3) if total_calls > 0 else 0.0

    # AB-1: Tool Efficiency — stub (all calls counted as useful until real tool data available)
    ab1 = round(len(tool_log) / max(total_calls, 1), 3)

    metrics = {
        "AB-1_tool_efficiency": ab1,
        "AB-4_memory_utilization": ab4,
        "AB-5_latency_seconds": None,          # computed externally
        "FA-5_hallucination_count": state.get("hallucination_count", 0),
        "CO-2_source_types_used": len({e["tool"] for e in tool_log}),
        "overall_confidence": state.get("confidence_scores", {}).get("overall", None),
        "plan_steps_completed": state.get("current_step_index", 0),
        "total_tool_calls": total_calls,
        "note": "Full 20+ metric suite: Day 11",
    }

    logger.info("Evaluation complete: %s", metrics)
    return {**state, "evaluation_metrics": metrics}


# ---------------------------------------------------------------------------
# Routing edges
# ---------------------------------------------------------------------------

def should_continue_execution(state: ResearchState) -> str:
    """Route back to execute_step if steps remain, else to synthesize."""
    idx = state["current_step_index"]
    plan = state["research_plan"]
    total_calls = state.get("total_api_calls", 0)
    iteration_count = state.get("iteration_count", 0)

    if (idx < len(plan)
            and total_calls < MAX_TOOL_CALLS
            and iteration_count < MAX_ITERATIONS):
        return "execute_step"
    return "synthesize"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(mock_data: Optional[dict] = None, session_cm: Optional["ContextManager"] = None) -> Any:
    """
    Construct and compile the ARA-1 LangGraph StateGraph.

    Args:
        mock_data: Optional dict of {tool_name: mock_result} for testing.

    Returns:
        Compiled LangGraph app.
    """
    graph = StateGraph(ResearchState)

    # Use the CM passed in from run_agent (created fresh per query in run_agent).
    # Fallback to a new instance only when build_graph is called directly (e.g. tests).
    _session_cm = session_cm or ContextManager(
        session_id=f"session-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    )

    # Wrap execute_step to pass mock_data and the session-scoped CM
    def _execute(state):
        return node_execute_step(state, mock_data=mock_data, context_manager=_session_cm)

    # Register nodes
    graph.add_node("analyze_query", node_analyze_query)
    graph.add_node("plan", node_plan)
    graph.add_node("memory_precheck", node_memory_precheck)
    graph.add_node("execute_step", _execute)
    graph.add_node("synthesize", node_synthesize)
    graph.add_node("verify", node_verify)
    graph.add_node("generate_report", node_generate_report)
    graph.add_node("evaluate", node_evaluate)

    # Entry point
    graph.set_entry_point("analyze_query")

    # Linear edges
    graph.add_edge("analyze_query", "memory_precheck")
    graph.add_edge("memory_precheck", "plan")
    graph.add_edge("plan", "execute_step")

    # Conditional loop: keep executing steps or move to synthesis
    graph.add_conditional_edges(
        "execute_step",
        should_continue_execution,
        {"execute_step": "execute_step", "synthesize": "synthesize"},
    )

    # Post-execution pipeline
    graph.add_edge("synthesize", "verify")
    graph.add_edge("verify", "generate_report")
    graph.add_edge("generate_report", "evaluate")
    graph.add_edge("evaluate", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------

def run_agent(query: str, mock_data: Optional[dict] = None) -> ResearchState:
    """
    Run ARA-1 on a research query.

    Args:
        query:     Natural language research query.
        mock_data: Optional {tool_name: result} dict for offline testing.

    Returns:
        Final ResearchState with completed report and evaluation metrics.
    """
    session_id = f"session-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    logger.info("ARA-1 starting | session=%s | query=%s", session_id, query)
    start = time.time()

    initial_state: ResearchState = {
        "raw_query": query,
        "query_type": "factual",
        "companies": [],
        "clarifications": None,
        "research_plan": [],
        "current_step_index": 0,
        "plan_revision_count": 0,
        "gathered_data": {},
        "tool_call_log": [],
        "memory_hits": 0,
        "total_api_calls": 0,
        "iteration_count": 0,
        "draft_report": None,
        "conflicts_detected": [],
        "conflicts_resolved": [],
        "verified_claims": [],
        "flagged_claims": [],
        "hallucination_count": 0,
        "final_report": None,
        "confidence_scores": {},
        "evaluation_metrics": {},
        "session_id": session_id,
        "completed_at": None,
        "error_log": [],
    }

    # Create a fresh ContextManager per query — never reuse across sessions
    session_cm = ContextManager(session_id=session_id)
    app = build_graph(mock_data=mock_data, session_cm=session_cm)
    final_state = app.invoke(initial_state)

    elapsed = round(time.time() - start, 2)
    final_state["evaluation_metrics"]["AB-5_latency_seconds"] = elapsed
    logger.info("ARA-1 complete | session=%s | %.2fs", session_id, elapsed)

    # Log completed session to episodic memory for future planner use
    try:
        get_episodic_memory().log_session(final_state)
        logger.info("Session logged to episodic memory.")
    except Exception as _e:
        logger.warning("Episodic memory log failed (non-critical): %s", _e)

    return final_state