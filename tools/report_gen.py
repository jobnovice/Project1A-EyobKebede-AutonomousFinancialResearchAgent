"""
tools/report_gen.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

Report generator — formats research findings into a structured markdown report.

Input:
    template: str               # Name of the report template (see TEMPLATES below)
    sections: dict              # Mapping of section names to markdown content strings
    sources: list               # List of source citations (strings or dicts with 'source', 'url', etc.)

Output:
    report: str                 # Complete markdown report
    error: str | None

Templates:
    company_profile
    earnings_analysis
    risk_assessment
    comparative_analysis
    full_research
"""

import logging
from datetime import datetime, timezone
from typing import Optional, List, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template definitions: section order and required section names
# ---------------------------------------------------------------------------
TEMPLATES = {
    "company_profile": {
        "name": "Company Profile Report",
        "sections": [
            "executive_summary",
            "company_overview",
            "financial_summary",
            "key_executives",
            "recent_developments",
            "research_methodology_notes",
        ],
        "description": "Comprehensive company overview and financial snapshot."
    },
    "earnings_analysis": {
        "name": "Earnings Analysis Report",
        "sections": [
            "executive_summary",
            "earnings_overview",
            "actual_vs_consensus",
            "segment_performance",
            "earnings_call_transcript_analysis",
            "forward_guidance",
            "research_methodology_notes",
        ],
        "description": "Quarterly earnings review with consensus comparison."
    },
    "risk_assessment": {
        "name": "Risk Assessment Report",
        "sections": [
            "executive_summary",
            "financial_risks",
            "operational_risks",
            "regulatory_risks",
            "competitive_risks",
            "risk_quantification",
            "research_methodology_notes",
        ],
        "description": "Comprehensive risk factor analysis."
    },
    "comparative_analysis": {
        "name": "Competitive Analysis Report",
        "sections": [
            "executive_summary",
            "comparison_methodology",
            "financial_metrics_comparison",
            "market_position",
            "competitive_advantages",
            "peer_analysis",
            "research_methodology_notes",
        ],
        "description": "Multi‑company comparative analysis."
    },
    "full_research": {
        "name": "Investment Research Report",
        "sections": [
            "executive_summary",
            "company_overview",
            "financial_analysis",
            "risk_assessment",
            "competitive_positioning",
            "valuation_analysis",
            "outlook",
            "research_methodology_notes",
        ],
        "description": "Complete investment research report."
    }
}

DEFAULT_TEMPLATE = "full_research"

# ---------------------------------------------------------------------------
# Helper: format sources as markdown references
# ---------------------------------------------------------------------------

def _format_sources(sources: List[Union[str, dict]]) -> str:
    """Convert a list of source references into markdown reference list."""
    if not sources:
        return ""
    lines = ["## References\n"]
    for i, src in enumerate(sources, 1):
        if isinstance(src, dict):
            # Attempt to build a markdown link or plain text
            title = src.get("title", src.get("source", f"Source {i}"))
            url = src.get("url")
            if url:
                lines.append(f"{i}. [{title}]({url})")
            else:
                lines.append(f"{i}. {title}")
        else:
            lines.append(f"{i}. {src}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper: generate report header
# ---------------------------------------------------------------------------

def _generate_header(
    template_name: str,
    session_id: Optional[str] = None,
    custom_title: Optional[str] = None
) -> str:
    """Generate markdown header with metadata."""
    template_info = TEMPLATES.get(template_name, TEMPLATES[DEFAULT_TEMPLATE])
    title = custom_title or template_info["name"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = f"# {title}\n\n"
    header += f"**Generated:** {now}\n"
    if session_id:
        header += f"**Session ID:** {session_id}\n"
    header += "\n---\n\n"
    return header


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------

def report_generator(
    template: str,
    sections: dict,
    sources: Optional[List[Union[str, dict]]] = None,
    session_id: Optional[str] = None,
    custom_title: Optional[str] = None,
    **kwargs
) -> dict:
    """
    Format research findings into a structured markdown report.

    Args:
        template:   Name of the template to use (see TEMPLATES keys).
        sections:   Dictionary mapping section names (as defined in template) to markdown content strings.
        sources:    Optional list of source references (strings or dicts with 'title'/'url').
        session_id: Optional session identifier for metadata.
        custom_title: Override the default report title.

    Returns:
        dict with keys:
            report (str): The formatted markdown report,
            template (str),
            sections_used (list),
            error (str | None)
    """
    if not isinstance(sections, dict):
        return _error_response(template, "sections must be a dictionary")

    # Validate template exists
    if template not in TEMPLATES:
        logger.warning(
            "Unknown template '%s', using default '%s'",
            template, DEFAULT_TEMPLATE
        )
        template = DEFAULT_TEMPLATE

    template_info = TEMPLATES[template]
    required_sections = template_info["sections"]

    # Build the report
    report = _generate_header(template, session_id, custom_title)

    # Add each required section (if content provided)
    for sec_name in required_sections:
        content = sections.get(sec_name, "")
        if content:
            # Convert section name to title case and replace underscores with spaces
            title = sec_name.replace("_", " ").title()
            report += f"## {title}\n\n{content}\n\n"

    # Add any extra sections (not in template) at the end
    extra_sections = set(sections.keys()) - set(required_sections)
    for sec_name in extra_sections:
        content = sections[sec_name]
        if content:
            title = sec_name.replace("_", " ").title()
            report += f"## {title}\n\n{content}\n\n"

    # Append sources if provided
    if sources:
        report += _format_sources(sources)

    return {
        "report": report,
        "template": template,
        "sections_used": list(sections.keys()),
        "error": None,
    }


# ---------------------------------------------------------------------------
# Error helper
# ---------------------------------------------------------------------------

def _error_response(template: str, message: str) -> dict:
    logger.error("report_generator error for template '%s': %s", template, message)
    return {
        "report": "",
        "template": template,
        "sections_used": [],
        "error": message,
    }