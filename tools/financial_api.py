"""
src/tools/financial_api.py
Stub implementation for financial_data_api tool.
"""
from typing import Any, Dict

def financial_data_api(ticker: str, statement_type: str, period: str = "annual", limit: int = 1) -> Dict[str, Any]:
    """Simulates loading normalized rows from the Financial Modeling Prep (FMP) API (Tier 2)."""
    ticker_upper = ticker.upper()
    
    if statement_type == "income_statement":
        return {
            "ticker": ticker_upper,
            "statement": "income_statement",
            "period": period,
            "data": [{
                "date": "2025-12-31",
                "revenue": 96773000000,
                "costOfRevenue": 79113000000,
                "grossProfit": 17660000000,
                "operatingIncome": 8890000000,
                "netIncome": 7500000000,
                "eps": 2.35,
                "currency": "USD"
            }]
        }
    elif statement_type == "ratios":
        return {
            "ticker": ticker_upper,
            "statement": "ratios",
            "data": [{
                "currentRatio": 1.65,
                "quickRatio": 1.22,
                "debtToEquity": 0.45,
                "returnOnEquity": 0.185,
                "grossProfitMargin": 0.1825
            }]
        }
    return {"ticker": ticker_upper, "statement": statement_type, "data": []}