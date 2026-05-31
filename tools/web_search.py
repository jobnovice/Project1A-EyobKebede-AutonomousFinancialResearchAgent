"""
src/tools/web_search.py
Stub implementation for web_search tool.
"""
from typing import Any, Dict

def web_search(query: str, max_results: int = 3) -> Dict[str, Any]:
    """Simulates Tavily/Reuters broad semantic search execution for unstructured alpha data."""
    return {
        "query": query,
        "results": [
            {
                "title": "Reuters: Tech Giants Adjust Guidance Amid Regulatory Squeeze",
                "snippet": "Major corporations signal potential shifts in margin tracking as regulatory frameworks tighten internationally.",
                "url": "https://www.reuters.com/business/tech-guidance-2026",
                "source_tier": 4
            },
            {
                "title": "Bloomberg: Enterprise Infrastructure Spending Beats Estimates",
                "snippet": "New telemetry records show an aggregate increase in hardware capital expenses among Fortune 500 companies.",
                "url": "https://www.bloomberg.com/news/enterprise-capex",
                "source_tier": 4
            }
        ][:max_results]
    }