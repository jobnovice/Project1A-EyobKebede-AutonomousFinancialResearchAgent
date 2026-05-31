"""
tools/schemas/computation.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

JSON schemas for Group 2: Computation and Verification Tools (3 tools).

These tools operate on data already gathered by Group 1 tools —
they derive, verify, and format rather than retrieve.
"""

# ---------------------------------------------------------------------------
# Tool 8: calculation_engine
# ---------------------------------------------------------------------------
CALCULATION_ENGINE: dict = {
    "name": "calculation_engine",
    "description": (
        "Perform financial calculations on retrieved raw data, including discounted "
        "cash flow (DCF) valuation, growth rate computation, financial ratio derivation, "
        "margin analysis, and statistical comparisons. Use whenever the research requires "
        "original quantitative reasoning beyond pre-computed API figures — this is the "
        "primary driver of metric AD-3 (Quantitative Reasoning). This tool runs locally "
        "with no external API calls and has no rate limit. Always show intermediate "
        "calculation steps in the inputs so the result is auditable."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "calculation_type": {
                "type": "string",
                "enum": [
                    "dcf_valuation",
                    "growth_rate",
                    "cagr",
                    "margin",
                    "ratio",
                    "yoy_change",
                    "peer_percentile",
                    "statistical_summary",
                    "custom",
                ],
                "description": (
                    "Type of calculation to perform. "
                    "'dcf_valuation': requires free_cash_flows, wacc, terminal_growth_rate. "
                    "'growth_rate' / 'cagr': requires start_value, end_value, periods. "
                    "'margin': requires numerator, denominator (e.g., gross_profit, revenue). "
                    "'ratio': requires numerator, denominator with optional label. "
                    "'yoy_change': requires current_value, prior_value. "
                    "'peer_percentile': requires company_value, peer_values list. "
                    "'statistical_summary': requires values list (mean, median, std, min, max). "
                    "'custom': provide formula string and named variable dict."
                ),
            },
            "inputs": {
                "type": "object",
                "description": (
                    "Named input variables required for the selected calculation_type. "
                    "All monetary values must be in the same unit (millions or billions). "
                    "Example for cagr: {'start_value': 50000, 'end_value': 97000, 'periods': 4}. "
                    "Example for margin: {'numerator': 43000, 'denominator': 97000, "
                    "'label': 'gross_margin'}."
                ),
            },
            "output_format": {
                "type": "string",
                "enum": ["value_only", "with_steps", "full_breakdown"],
                "description": (
                    "Level of detail in the returned result. "
                    "'value_only': returns the final number. "
                    "'with_steps': returns the result plus key intermediate steps. "
                    "'full_breakdown': returns all intermediate values for full auditability. "
                    "Default is 'with_steps' — required for transparent reporting."
                ),
                "default": "with_steps",
            },
            "unit": {
                "type": "string",
                "enum": ["percentage", "ratio", "currency_millions", "currency_billions", "raw"],
                "description": (
                    "Expected unit of the output result. Used for formatting and sanity checks. "
                    "For margin/growth calculations use 'percentage'; for DCF use 'currency_millions'."
                ),
                "default": "raw",
            },
        },
        "required": ["calculation_type", "inputs"],
    },
}

# ---------------------------------------------------------------------------
# Tool 9: fact_checker
# ---------------------------------------------------------------------------
FACT_CHECKER: dict = {
    "name": "fact_checker",
    "description": (
        "Cross-reference a specific factual claim against multiple authoritative sources "
        "to verify accuracy before including it in the final report. Use for every "
        "material numerical claim per the Verify Before You Assert principle — this is "
        "the primary mechanism for achieving metric FA-1 (>98% numerical accuracy) and "
        "FA-5 (0 hallucinations). Internally orchestrates calls to other tools. "
        "A 'material' claim is any number cited in the executive summary, financial "
        "analysis, or risk assessment sections — revenue, earnings, margins, growth rates."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "claim": {
                "type": "string",
                "description": (
                    "The specific factual claim to verify, stated as precisely as possible. "
                    "Include the company, metric, value, time period, and unit. "
                    "Example: 'Apple revenue was $97.3 billion in Q1 FY2024' or "
                    "'Tesla gross margin was 17.4% in FY2023'."
                ),
            },
            "claimed_value": {
                "type": ["number", "string"],
                "description": (
                    "The specific value being claimed, extracted from the claim text. "
                    "Providing this enables automated numerical comparison against sources."
                ),
            },
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional list of source tool names already consulted "
                    "(['sec_filing_search', 'financial_data_api']). "
                    "The fact_checker will consult additional independent sources "
                    "beyond those listed here."
                ),
            },
            "ticker": {
                "type": "string",
                "description": "Company ticker for targeted source retrieval.",
            },
            "time_period": {
                "type": "string",
                "description": (
                    "The time period the claim refers to (e.g., 'Q1 FY2024', 'FY2023', "
                    "'TTM as of Oct 2024'). Critical for avoiding temporal misattribution."
                ),
            },
            "required_confidence": {
                "type": "number",
                "description": (
                    "Minimum confidence threshold (0.0-1.0) to mark a claim as verified. "
                    "Default is 0.85. Claims below this threshold are flagged for manual review."
                ),
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.85,
            },
        },
        "required": ["claim"],
    },
}

# ---------------------------------------------------------------------------
# Tool 10: report_generator
# ---------------------------------------------------------------------------
REPORT_GENERATOR: dict = {
    "name": "report_generator",
    "description": (
        "Format all gathered, synthesized, and verified research data into a structured "
        "professional investment research report in markdown, following a specified "
        "template. Call this tool only AFTER all research, conflict resolution, and "
        "fact-checking steps are complete — it is the final output generation step. "
        "The template enforces the required sections: Executive Summary, Company Overview, "
        "Financial Analysis, Risk Assessment, Competitive Position, and Research "
        "Methodology Notes, with inline source citations."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "template": {
                "type": "string",
                "enum": [
                    "company_profile",
                    "earnings_analysis",
                    "risk_assessment",
                    "industry_comparison",
                    "investigative_report",
                    "sector_analysis",
                    "full_investment_report",
                    "executive_brief",
                ],
                "description": (
                    "Report template to use, matched to the challenge type. "
                    "'full_investment_report' is used for Challenge 8 and covers all sections. "
                    "'executive_brief' produces a condensed 2-3 page summary."
                ),
            },
            "company": {
                "type": "string",
                "description": (
                    "Primary company name and ticker being reported on "
                    "(e.g., 'Tesla Inc. (TSLA)'). For multi-company reports, "
                    "list all companies."
                ),
            },
            "sections": {
                "type": "object",
                "description": (
                    "Dictionary mapping section names to their content. Each section "
                    "should be pre-synthesized text with inline citations. Keys correspond "
                    "to template sections (e.g., 'executive_summary', 'financial_analysis', "
                    "'risk_assessment', 'competitive_position', 'research_methodology')."
                ),
            },
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Ordered list of all sources cited in the report, in citation format: "
                    "'[Source type] [Company], [Date], [Tool used]'. "
                    "Example: 'SEC 10-K TSLA, FY2024, sec_filing_search'."
                ),
            },
            "confidence_scores": {
                "type": "object",
                "description": (
                    "Per-section confidence scores (0.0-1.0) derived from fact-checking. "
                    "Keys are section names; values are confidence floats. "
                    "Sections with scores below 0.7 will be flagged with a disclaimer."
                ),
            },
            "flags": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "List of data gaps, unresolved conflicts, or tool failures to "
                    "transparently disclose in the Research Methodology Notes section. "
                    "Never omit flags — graceful degradation requires full transparency."
                ),
            },
        },
        "required": ["template", "company", "sections", "sources"],
    },
}

# ---------------------------------------------------------------------------
# Export: all Group 2 schemas as an ordered list
# ---------------------------------------------------------------------------
COMPUTATION_SCHEMAS: list[dict] = [
    CALCULATION_ENGINE,
    FACT_CHECKER,
    REPORT_GENERATOR,
]
