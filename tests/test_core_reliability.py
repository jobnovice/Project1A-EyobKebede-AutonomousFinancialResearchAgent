from agent import core
from agent.circuit_breaker import CircuitBreaker
from agent.parser import ReActStep
from memory.context_manager import ContextManager
from memory.episodic import _compute_tool_performance


class _ContextManager(ContextManager):
    def __init__(self):
        self.session_id = "test"

    def update(self, state: dict) -> dict[str, int]:
        del state
        return {"pct_used": 0}

    def should_summarise(self):
        return False

    def has_headroom(self):
        return True


def _state() -> core.ResearchState:
    return {
        "raw_query": "Find recent MSFT news",
        "query_type": "factual",
        "companies": ["MSFT"],
        "clarifications": None,
        "research_plan": [{
            "step_id": 1,
            "description": "Search recent news",
            "tool": "web_search",
            "params": {"query": "MSFT news", "num_results": 5},
            "fallback_if_fails": "news_sentiment",
        }],
        "current_step_index": 0,
        "plan_revision_count": 0,
        "gathered_data": {},
        "tool_call_log": [],
        "chain_results": [],
        "memory_hits": 0,
        "total_api_calls": 0,
        "iteration_count": 0,
        "draft_report": None,
        "error_log": [],
        "conflicts_detected": [],
        "conflicts_resolved": [],
        "verified_claims": [],
        "flagged_claims": [],
        "hallucination_count": 0,
        "final_report": None,
        "confidence_scores": {},
        "evaluation_metrics": {},
        "session_id": "test",
        "completed_at": None,
    }


def _run_step(monkeypatch, *, mock_data=None, decision="step_complete"):
    react = ReActStep(
        thought="Search news",
        tool_name="web_search",
        tool_params={"query": "MSFT news", "num_results": 5},
        observation="",
        decision=decision,
        raw="test",
    )
    monkeypatch.setattr(core, "build_executor_prompt", lambda *_: "")
    monkeypatch.setattr(core, "_call_llm", lambda *_: "test")
    monkeypatch.setattr(core, "parse_react_step", lambda _: react)
    return core.node_execute_step(
        _state(),
        mock_data=mock_data,
        context_manager=_ContextManager(),
    )


def test_core_retries_then_falls_back_and_tracks_source(monkeypatch):
    primary_calls = []
    fallback_calls = []

    def failing_search(**params):
        primary_calls.append(params)
        raise RuntimeError("Simulated network connection error")

    def news_fallback(**params):
        fallback_calls.append(params)
        return {"articles": [{"title": "MSFT news"}]}

    monkeypatch.setattr(core, "TOOL_DISPATCH", {
        "web_search": failing_search,
        "news_sentiment": news_fallback,
    })
    monkeypatch.setattr(core, "_CIRCUIT_BREAKER", CircuitBreaker(failure_threshold=1))
    monkeypatch.setattr(core, "MAX_RETRIES_PER_TOOL", 2)
    monkeypatch.setattr("agent.error_handler.time.sleep", lambda _: None)

    result = _run_step(monkeypatch, decision="fallback_needed")

    assert len(primary_calls) == 3
    assert len(fallback_calls) == 1
    assert result["total_api_calls"] == 4
    assert "news_sentiment_step1" in result["gathered_data"]
    assert "web_search_step1" not in result["gathered_data"]
    assert [entry["status"] for entry in result["tool_call_log"]] == [
        "failed", "failed", "failed", "success"
    ]
    assert all(entry["tool"] == "web_search" for entry in result["tool_call_log"][:3])
    assert result["tool_call_log"][-1]["tool"] == "news_sentiment"
    assert result["chain_results"][0].used_fallback

    report = core.node_generate_report({
        **result,
        "draft_report": "Research summary",
        "hallucination_count": 0,
        "confidence_scores": {"overall": 0.8},
    })
    assert report["final_report"] is not None
    assert "Data Availability Notes" in report["final_report"]
    assert "web_search" in report["final_report"]
    assert report["confidence_scores"]["overall"] == 0.8
    assert report["confidence_scores"]["data_availability"] == 1.0

    recovered = _run_step(monkeypatch, decision="step_complete")
    assert len(primary_calls) == 3
    assert len(fallback_calls) == 2
    assert recovered["total_api_calls"] == 1
    assert recovered["tool_call_log"][0]["status"] == "skipped_circuit_open"


def test_core_preserves_explicit_mock_mode(monkeypatch):
    monkeypatch.setattr(
        core,
        "execute_with_fallbacks",
        lambda **_: (_ for _ in ()).throw(AssertionError("real path was called")),
    )

    result = _run_step(monkeypatch, mock_data={
        "web_search": {"results": [{"title": "Mock news"}]},
    })

    assert result["total_api_calls"] == 1
    assert result["gathered_data"]["web_search_step1"]["mock"] is True
    assert result["chain_results"] == []


def test_core_retry_attempts_respect_global_tool_budget(monkeypatch):
    def failing_search(**_):
        raise RuntimeError("Simulated network connection error")

    monkeypatch.setattr(core, "TOOL_DISPATCH", {"web_search": failing_search})
    monkeypatch.setattr(core, "_CIRCUIT_BREAKER", CircuitBreaker(failure_threshold=1))
    monkeypatch.setattr(core, "MAX_RETRIES_PER_TOOL", 4)
    monkeypatch.setattr(core, "MAX_TOOL_CALLS", 2)
    monkeypatch.setattr("agent.error_handler.time.sleep", lambda _: None)

    result = _run_step(monkeypatch)

    assert result["total_api_calls"] == 2
    assert [entry["status"] for entry in result["tool_call_log"]] == [
        "failed", "failed", "skipped_call_budget"
    ]
    performance = _compute_tool_performance(result["tool_call_log"])
    assert performance["web_search"]["calls"] == 2
    assert result["chain_results"][0].degraded
    report = core.node_generate_report({
        **result,
        "draft_report": "Research summary",
        "hallucination_count": 0,
        "confidence_scores": {"overall": 0.8},
    })
    assert report["final_report"] is not None
    assert "[DATA UNAVAILABLE]" in report["final_report"]
    assert report["confidence_scores"]["overall"] == 0.0
    assert report["confidence_scores"]["data_availability"] == 0.0
