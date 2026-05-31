"""
tools/schemas/data_retrieval.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

JSON schemas for Group 1: Data Retrieval Tools (7 tools).
All schemas follow the OpenAI function-calling specification,
which is also compatible with Anthropic Claude tool use.

Schema contract:
    - name: snake_case, no spaces
    - description: 1-3 sentences; the LLM reads this to decide WHEN to use the tool
    - parameters: JSON Schema object (type, properties, required, defaults)
"""

# ---------------------------------------------------------------------------
# Tool 1: sec_filing_search
# ---------------------------------------------------------------------------
SEC_FILING_SEARCH: dict = {
    "name": "sec_filing_search",
    "description": (
        "Search and retrieve SEC EDGAR filings for a publicly traded US company. "
        "Use this tool when you need official regulatory disclosures including annual "
        "reports (10-K), quarterly reports (10-Q), material event reports (8-K), or "
        "proxy statements (DEF 14A). Always prefer this tool over web_search for "
        "authoritative financial data — these filings are Tier 1 ground truth."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": (
                    "Stock ticker symbol of the company (e.g., 'AAPL', 'MSFT', 'TSLA'). "
                    "Must be the official NYSE/NASDAQ ticker."
                ),
            },
            "filing_type": {
                "type": "string",
                "enum": ["10-K", "10-Q", "8-K", "DEF 14A"],
                "description": (
                    "Type of SEC filing to retrieve. "
                    "10-K = annual report, 10-Q = quarterly report, "
                    "8-K = material event disclosure, DEF 14A = proxy statement."
                ),
            },
            "year": {
                "type": "integer",
                "description": (
                    "Filing year (e.g., 2024). Defaults to the most recent available "
                    "filing year if not specified. Use this to retrieve historical filings."
                ),
                "minimum": 1993,
                "maximum": 2025,
            },
            "accession_number": {
                "type": "string",
                "description": (
                    "Optional EDGAR accession number in the format XXXXXXXXXX-YY-ZZZZZZ. "
                    "Use when you need a specific filing version rather than the most recent."
                ),
            },
        },
        "required": ["ticker", "filing_type"],
    },
}

# ---------------------------------------------------------------------------
# Tool 2: financial_data_api
# ---------------------------------------------------------------------------
FINANCIAL_DATA_API: dict = {
    "name": "financial_data_api",
    "description": (
        "Retrieve structured financial statements and key metrics for a publicly traded "
        "company from curated financial data APIs (Financial Modeling Prep primary, "
        "Alpha Vantage fallback). Use for income statements, balance sheets, cash flow "
        "statements, and pre-computed financial ratios. Prefer this over sec_filing_search "
        "when you need machine-readable numerical data rather than full filing text. "
        "This is a Tier 2 source — always cross-reference material numbers against "
        "the 10-K via sec_filing_search."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol (e.g., 'AAPL', 'MSFT').",
            },
            "statement_type": {
                "type": "string",
                "enum": [
                    "income_statement",
                    "balance_sheet",
                    "cash_flow",
                    "key_metrics",
                    "financial_ratios",
                    "earnings",
                ],
                "description": (
                    "The financial statement or metric set to retrieve. "
                    "'key_metrics' returns a pre-computed bundle (P/E, EV/EBITDA, ROE, etc.). "
                    "'financial_ratios' returns efficiency, liquidity, and leverage ratios."
                ),
            },
            "period": {
                "type": "string",
                "enum": ["annual", "quarterly"],
                "description": (
                    "Reporting period. Use 'annual' for year-over-year trend analysis; "
                    "'quarterly' for recent performance and sequential comparisons."
                ),
                "default": "annual",
            },
            "years": {
                "type": "integer",
                "description": (
                    "Number of historical periods to retrieve (1-10). "
                    "Default is 3 years / 12 quarters. "
                    "Metric CO-3 requires at least 3 years of data."
                ),
                "minimum": 1,
                "maximum": 10,
                "default": 3,
            },
        },
        "required": ["ticker", "statement_type"],
    },
}

# ---------------------------------------------------------------------------
# Tool 3: earnings_transcript
# ---------------------------------------------------------------------------
EARNINGS_TRANSCRIPT: dict = {
    "name": "earnings_transcript",
    "description": (
        "Retrieve the full earnings call transcript for a specific company, quarter, "
        "and year, with speaker-level attribution. Use this tool to extract management "
        "commentary, forward guidance, and analyst Q&A sentiment — data not available "
        "in financial statements. This is a Tier 3 source: valuable for qualitative "
        "insight but subject to spin and selective disclosure. Always cross-reference "
        "management claims against 10-K/financial_data_api numerical data."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol (e.g., 'AAPL', 'TSLA').",
            },
            "quarter": {
                "type": "string",
                "enum": ["Q1", "Q2", "Q3", "Q4"],
                "description": (
                    "Fiscal quarter of the earnings call. "
                    "Note: company fiscal quarters may differ from calendar quarters."
                ),
            },
            "year": {
                "type": "integer",
                "description": "Calendar year of the earnings call (e.g., 2024).",
                "minimum": 2010,
                "maximum": 2025,
            },
            "sections": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["prepared_remarks", "qa_session", "full"],
                },
                "description": (
                    "Which sections of the transcript to return. "
                    "Default returns 'full'. Use 'qa_session' when focused on "
                    "analyst questions and management responses."
                ),
                "default": ["full"],
            },
        },
        "required": ["ticker", "quarter", "year"],
    },
}

# ---------------------------------------------------------------------------
# Tool 4: web_search
# ---------------------------------------------------------------------------
WEB_SEARCH: dict = {
    "name": "web_search",
    "description": (
        "Perform a web search for current news, analyst commentary, and recent events "
        "about a company or financial topic that may not yet appear in SEC filings or "
        "earnings transcripts. Use as a Tier 4 source — helpful for emerging risks "
        "and market context, but never as the sole source for numerical claims. "
        "Always check vector_db_search first (Memory First rule) before calling this tool. "
        "Do NOT use for structured financial data — use financial_data_api instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Search query string. Be specific and include the company name, "
                    "ticker, and the specific topic (e.g., 'Tesla TSLA supply chain "
                    "risks 2024 2025'). Vague queries return low-quality results."
                ),
            },
            "num_results": {
                "type": "integer",
                "description": "Number of search results to return (1-20). Default is 10.",
                "minimum": 1,
                "maximum": 20,
                "default": 10,
            },
            "date_range": {
                "type": "string",
                "enum": ["past_day", "past_week", "past_month", "past_year", "any"],
                "description": (
                    "Filter results by recency. Use 'past_month' for current events; "
                    "'past_year' for broader context. Default is 'any'."
                ),
                "default": "any",
            },
            "search_type": {
                "type": "string",
                "enum": ["news", "general", "financial"],
                "description": (
                    "Type of search to optimize result ranking. "
                    "'news' prioritizes press sources; 'financial' prioritizes "
                    "analyst reports and financial publications."
                ),
                "default": "news",
            },
        },
        "required": ["query"],
    },
}

# ---------------------------------------------------------------------------
# Tool 5: news_sentiment
# ---------------------------------------------------------------------------
NEWS_SENTIMENT: dict = {
    "name": "news_sentiment",
    "description": (
        "Analyze the aggregate sentiment of recent news articles about a company or "
        "financial topic using NLP scoring (TextBlob/VADER). Returns sentiment scores "
        "(positive/negative/neutral), article summaries, and trend over time. "
        "Use to quantify market narrative direction and identify divergence between "
        "news sentiment and fundamental financial data (Sentiment-Fact Alignment technique). "
        "This supplements web_search — use when you need scored, aggregated sentiment "
        "rather than individual article content."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Company name, ticker, or topic to analyze "
                    "(e.g., 'Tesla electric vehicle competition', 'NVDA AI chips')."
                ),
            },
            "num_articles": {
                "type": "integer",
                "description": "Number of articles to analyze for sentiment (5-50). Default is 20.",
                "minimum": 5,
                "maximum": 50,
                "default": 20,
            },
            "lookback_days": {
                "type": "integer",
                "description": (
                    "Number of days to look back for articles (1-365). "
                    "Default is 30 days. Use 7 for very recent sentiment; "
                    "90+ for trend analysis."
                ),
                "minimum": 1,
                "maximum": 365,
                "default": 30,
            },
            "return_breakdown": {
                "type": "boolean",
                "description": (
                    "If true, returns per-article sentiment scores in addition to "
                    "aggregate scores. Useful for identifying outlier articles driving "
                    "overall sentiment."
                ),
                "default": False,
            },
        },
        "required": ["query"],
    },
}

# ---------------------------------------------------------------------------
# Tool 6: company_profile
# ---------------------------------------------------------------------------
COMPANY_PROFILE: dict = {
    "name": "company_profile",
    "description": (
        "Retrieve basic company metadata including sector, industry classification, "
        "market capitalization, key executives, business description, exchange listing, "
        "and founding year. Use this as the first tool call when beginning research on "
        "any company — it provides the contextual foundation for all subsequent analysis "
        "and is required for Challenge 1 (company profile) and entity validation. "
        "Also use to verify that a ticker symbol maps to the correct company before "
        "making other tool calls."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": (
                    "Stock ticker symbol (e.g., 'AAPL', 'MSFT'). "
                    "For companies listed on multiple exchanges, use the primary "
                    "US listing ticker."
                ),
            },
            "include_fields": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "basic",
                        "executives",
                        "description",
                        "market_data",
                        "all",
                    ],
                },
                "description": (
                    "Subset of profile fields to return. 'basic' returns name, ticker, "
                    "exchange, sector, industry. 'all' returns the complete profile. "
                    "Default is 'all'."
                ),
                "default": ["all"],
            },
        },
        "required": ["ticker"],
    },
}

# ---------------------------------------------------------------------------
# Tool 7: peer_comparison
# ---------------------------------------------------------------------------
PEER_COMPARISON: dict = {
    "name": "peer_comparison",
    "description": (
        "Identify industry peer companies for a given ticker and retrieve a comparative "
        "matrix of financial metrics across the peer group. Use for competitive positioning "
        "analysis, relative valuation, and benchmarking. Required for Challenge 4 "
        "(industry comparison) and any report section discussing competitive dynamics. "
        "The peer set is automatically identified from the same GICS sector/industry; "
        "you can override with explicit peers if needed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The primary company ticker to find peers for.",
            },
            "num_peers": {
                "type": "integer",
                "description": "Number of peer companies to include (2-10). Default is 5.",
                "minimum": 2,
                "maximum": 10,
                "default": 5,
            },
            "metrics": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "List of financial metrics to compare across the peer group. "
                    "Examples: ['revenue_growth', 'gross_margin', 'operating_margin', "
                    "'pe_ratio', 'ev_ebitda', 'roe', 'debt_to_equity', 'market_cap']. "
                    "Default returns a standard set of 8 key metrics."
                ),
                "default": [
                    "revenue_growth",
                    "gross_margin",
                    "operating_margin",
                    "pe_ratio",
                    "ev_ebitda",
                    "roe",
                    "debt_to_equity",
                    "market_cap",
                ],
            },
            "explicit_peers": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional list of specific ticker symbols to use as peers instead of "
                    "auto-identified peers (e.g., ['MSFT', 'GOOGL', 'META']). "
                    "Use when the auto-identified peer set is inappropriate for the analysis."
                ),
            },
        },
        "required": ["ticker"],
    },
}

# ---------------------------------------------------------------------------
# Export: all Group 1 schemas as an ordered list
# ---------------------------------------------------------------------------
DATA_RETRIEVAL_SCHEMAS: list[dict] = [
    SEC_FILING_SEARCH,
    FINANCIAL_DATA_API,
    EARNINGS_TRANSCRIPT,
    WEB_SEARCH,
    NEWS_SENTIMENT,
    COMPANY_PROFILE,
    PEER_COMPARISON,
]
