# tests/test_agent.py
import pytest
from agent.core import run_agent

# Sample mock data for Challenge 1: Microsoft company profile
MOCK_DATA = {
    "company_profile": {
        "name": "Microsoft Corporation",
        "ticker": "MSFT",
        "sector": "Technology",
        "industry": "Software—Infrastructure",
        "market_cap": 3100000000000,
        "description": "Microsoft develops, licenses, and supports software, services, devices, and solutions worldwide.",
        "ceo": "Satya Nadella"
    },
    "web_search": {
        "results": [
            {"title": "Microsoft Cloud Revenue Growth", "snippet": "Azure grew 30% YoY"},
            {"title": "Microsoft AI Investments", "snippet": "$10B in OpenAI partnership"}
        ]
    },
    "financial_data_api": {
        "revenue_2024": 245000000000,
        "net_income_2024": 88000000000,
        "pe_ratio": 35.2,
        "roi": 18.5
    }
}

def test_challenge1_company_profile():
    """Challenge 1: Create a company profile for Microsoft using mock data."""
    query = "Create a comprehensive profile of Microsoft Corporation including business overview, financial summary, key executives, and recent developments."
    
    # Run the agent with mock data
    final_state = run_agent(query, mock_data=MOCK_DATA)
    
    # ---- Assertions ----
    # 1. A final report was generated
    assert final_state["final_report"] is not None, "No final report generated"
    assert len(final_state["final_report"]) > 100, "Report too short"
    
    # 2. The planner created steps (at least 1)
    assert len(final_state["research_plan"]) > 0, "No research plan created"
    
    # 3. Tool calls were made
    assert final_state["total_api_calls"] > 0, "No tool calls executed"
    
    # 4. No critical errors in error_log
    error_log = final_state.get("error_log", [])
    critical_errors = [e for e in error_log if "LLM error" in e or "parse failed" in e]
    assert len(critical_errors) == 0, f"LLM or parsing errors: {critical_errors}"
    
    # 5. Hallucination count should be 0 or small (mock data is consistent)
    assert final_state.get("hallucination_count", 0) < 3, f"Too many hallucinations: {final_state.get('hallucination_count')}"
    
    # 6. Basic content checks (presence of expected words)
    report = final_state["final_report"].lower()
    assert "microsoft" in report, "Company name not found in report"
    assert any(word in report for word in ["revenue", "financial", "profile"]), "Missing key sections"
    
    print("✅ Challenge 1 test passed")
    print(f"Report preview: {final_state['final_report'][:500]}...")

def test_max_iteration_limit():
    """Verify that the agent respects MAX_ITERATIONS limit."""
    # Very complex query to potentially trigger many steps (but still mock)
    query = "Analyze everything about Apple, Google, Amazon, Tesla, Microsoft, and NVIDIA"
    final_state = run_agent(query, mock_data=MOCK_DATA)
    
    # Should never exceed the limit (15 in core.py)
    assert final_state["iteration_count"] <= 15, f"Iteration limit exceeded: {final_state['iteration_count']}"
    assert final_state["total_api_calls"] <= 20, f"Tool call limit exceeded: {final_state['total_api_calls']}"

def test_no_hallucinations_in_simple_query():
    """Simple factual query should produce zero hallucinations."""
    query = "What is the market cap of Microsoft?"
    final_state = run_agent(query, mock_data=MOCK_DATA)
    
    # Hallucination count should be 0 because mock data is clean
    assert final_state.get("hallucination_count", -1) == 0, f"Hallucinations found: {final_state.get('flagged_claims', [])}"