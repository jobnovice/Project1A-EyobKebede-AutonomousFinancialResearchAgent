"""
agent/parser.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

Parses raw LLM text responses into structured Python objects.

Three parsers:
    parse_plan()        — extracts JSON research plan from planner response
    parse_react_step()  — extracts Thought/Action/Decide from executor response
    parse_verification()— extracts hallucination report from verifier response
"""

import json
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output dataclasses — typed containers for parsed results
# ---------------------------------------------------------------------------

@dataclass
class ResearchPlan:
    plan_id: str
    query_summary: str
    query_type: str
    companies: list[str]
    estimated_tool_calls: int
    steps: list[dict]
    raw: str = field(repr=False)          # original LLM output, kept for trace log


@dataclass
class ReActStep:
    thought: str
    tool_name: Optional[str]
    tool_params: dict
    observation: str                       # empty string until tool executes
    decision: str                          # continue | step_complete | fallback_needed | conflict_detected | plan_revision_needed
    raw: str = field(repr=False)


@dataclass
class VerificationResult:
    hallucination_count: int
    unsupported_claims: list[dict]
    verified_claims_count: int
    confidence_score: float
    raw: str = field(repr=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """
    Extract the first valid JSON object from a string.
    Handles LLM responses that wrap JSON in markdown fences or add prose.
    """
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "")

    # Try parsing the whole string first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Find the first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Found JSON-like block but could not parse it: {e}") from e

    raise ValueError("No JSON object found in LLM response.")


def _extract_section(text: str, label: str) -> str:
    """
    Extract the text following a labelled section header (e.g. 'THOUGHT:').
    Returns empty string if the label is not found.
    """
    pattern = rf"{label}:\s*(.*?)(?=\n[A-Z]{{2,}}:|$)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


# ---------------------------------------------------------------------------
# Parser 1: Research Plan
# ---------------------------------------------------------------------------

def parse_plan(llm_response: str) -> ResearchPlan:
    """
    Parse the planner LLM response into a ResearchPlan.

    Raises:
        ValueError  if the JSON is missing required fields
    """
    try:
        data = _extract_json(llm_response)
    except ValueError as e:
        logger.error("plan parse failed: %s | raw=%s", e, llm_response[:200])
        raise

    required = {"plan_id", "query_summary", "query_type", "companies",
                "estimated_tool_calls", "steps"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Research plan missing required fields: {missing}")

    if not isinstance(data["steps"], list) or len(data["steps"]) == 0:
        raise ValueError("Research plan has no steps.")

    # Validate each step has minimum required keys
    for i, step in enumerate(data["steps"]):
        step_required = {"step_id", "description", "tool", "params"}
        step_missing = step_required - step.keys()
        if step_missing:
            raise ValueError(f"Step {i} missing fields: {step_missing}")
        # Set defaults for optional step keys
        step.setdefault("depends_on", [])
        step.setdefault("fallback_if_fails", None)
        step.setdefault("optional", False)

    logger.info(
        "Plan parsed: %s | %d steps | est. %d tool calls",
        data["plan_id"], len(data["steps"]), data["estimated_tool_calls"]
    )

    return ResearchPlan(
        plan_id=data["plan_id"],
        query_summary=data["query_summary"],
        query_type=data["query_type"],
        companies=data["companies"],
        estimated_tool_calls=data["estimated_tool_calls"],
        steps=data["steps"],
        raw=llm_response,
    )


# ---------------------------------------------------------------------------
# Parser 2: ReAct Step (Thought / Action / Decide)
# ---------------------------------------------------------------------------

VALID_DECISIONS = {
    "continue",
    "step_complete",
    "fallback_needed",
    "conflict_detected",
    "plan_revision_needed",
}


def parse_react_step(llm_response: str) -> ReActStep:
    """
    Parse one Thought-Action-Observation-Decide cycle from the executor.

    The LLM writes ACTION as a JSON block embedded in free text.
    OBSERVATION is empty at parse time — filled in after tool execution.
    """
    thought = _extract_section(llm_response, "THOUGHT")
    if not thought:
        logger.warning("No THOUGHT section found in executor response.")

    # Extract ACTION JSON block
    tool_name: Optional[str] = None
    tool_params: dict = {}

    action_block = _extract_section(llm_response, "ACTION")
    if action_block:
        try:
            action_data = _extract_json(action_block)
            tool_name = action_data.get("tool")
            tool_params = action_data.get("params", {})
        except ValueError:
            logger.warning("Could not parse ACTION block: %s", action_block[:200])

    # Extract DECIDE
    decide_raw = _extract_section(llm_response, "DECIDE").lower().strip()
    # Normalise — LLM might say "step complete" or "step_complete"
    decide_normalised = decide_raw.replace(" ", "_")
    if decide_normalised not in VALID_DECISIONS:
        logger.warning("Unexpected DECIDE value '%s' — defaulting to 'continue'", decide_raw)
        decide_normalised = "continue"

    logger.info(
        "ReAct step parsed: tool=%s | decision=%s", tool_name, decide_normalised
    )

    return ReActStep(
        thought=thought,
        tool_name=tool_name,
        tool_params=tool_params,
        observation="",          # populated by executor after tool call
        decision=decide_normalised,
        raw=llm_response,
    )


# ---------------------------------------------------------------------------
# Parser 3: Verification Result
# ---------------------------------------------------------------------------

def parse_verification(llm_response: str) -> VerificationResult:
    """
    Parse the verifier LLM response into a VerificationResult.
    """
    try:
        data = _extract_json(llm_response)
    except ValueError as e:
        logger.error("Verification parse failed: %s", e)
        raise

    hallucination_count = int(data.get("hallucination_count", 0))
    unsupported = data.get("unsupported_claims", [])
    verified_count = int(data.get("verified_claims_count", 0))
    confidence = float(data.get("confidence_score", 0.0))

    if hallucination_count > 0:
        logger.warning(
            "%d hallucination(s) detected. Confidence: %.2f",
            hallucination_count, confidence
        )
    else:
        logger.info("Verification passed. Confidence: %.2f", confidence)

    return VerificationResult(
        hallucination_count=hallucination_count,
        unsupported_claims=unsupported,
        verified_claims_count=verified_count,
        confidence_score=confidence,
        raw=llm_response,
    )
