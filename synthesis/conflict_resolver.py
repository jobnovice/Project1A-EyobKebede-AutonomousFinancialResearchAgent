"""
synthesis/conflict_resolver.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

Implements the 6-step Conflict Resolution Protocol from architecture spec §7.2.

When two sources report different values for the same metric, this module:
  Step 1 — IDENTIFY:          detect conflict (>5% threshold)
  Step 2 — ASSESS TIERS:      unconditional preference for higher-tier source
  Step 3 — TEMPORAL ALIGNMENT:check if values cover the same time period
  Step 4 — RESTATEMENTS:      newer source supersedes older on same period
  Step 5 — HIGHEST-TIER RULE: final tiebreaker
  Step 6 — DOCUMENT:          always explain the resolution decision

Source reliability tiers (from architecture spec §7.1):
  Tier 1 (1.00): SEC filings         — ground truth
  Tier 2 (0.90): Financial data APIs — curated structured data
  Tier 3 (0.75): Earnings transcripts— management voice
  Tier 4 (0.60): Major news outlets  — editorial oversight
  Tier 5 (0.20): Social media / anon — sentiment only
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFLICT_THRESHOLD = 0.05   # 5% relative difference triggers conflict detection

TIER_WEIGHTS: dict[int, float] = {
    1: 1.00,
    2: 0.90,
    3: 0.75,
    4: 0.60,
    5: 0.20,
}

PERIOD_KEYWORDS = {
    "ttm":  "trailing_twelve_months",
    "ltm":  "trailing_twelve_months",
    "fy":   "fiscal_year",
    "q1":   "q1", "q2": "q2", "q3": "q3", "q4": "q4",
    "annual": "annual",
    "quarterly": "quarterly",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SourceValue:
    """A single data point from one source."""
    source_name:  str
    source_tier:  int
    value:        float
    metric_name:  str
    period:       Optional[str] = None     # e.g. "FY2024", "Q3 2024", "TTM"
    date:         Optional[str] = None     # ISO date of the source document
    raw_context:  Optional[str] = None     # snippet from the source for citation


@dataclass
class ConflictResolution:
    """Output of the 6-step resolution protocol for one metric conflict."""
    metric_name:      str
    preferred_value:  float
    preferred_source: str
    preferred_tier:   int
    rejected_value:   float
    rejected_source:  str
    rejected_tier:    int
    resolution_step:  int          # which step resolved it (2–5)
    resolution_reason:str          # one-line explanation
    disclosure_note:  str          # full §7.2 Step 6 disclosure for the report
    is_real_conflict: bool         # False = temporal mismatch, not a true discrepancy
    resolved_at:      str = field(default_factory=lambda: _now())


# ---------------------------------------------------------------------------
# Step 1 — Identify
# ---------------------------------------------------------------------------

def detect_conflict(a: SourceValue, b: SourceValue) -> bool:
    """
    Step 1: Detect conflict using the 5% relative threshold.
    abs(A - B) / max(A, B) > 0.05
    """
    max_val = max(abs(a.value), abs(b.value))
    if max_val == 0:
        return False
    relative_diff = abs(a.value - b.value) / max_val
    return relative_diff > CONFLICT_THRESHOLD


# ---------------------------------------------------------------------------
# Step 3 — Temporal alignment
# ---------------------------------------------------------------------------

def _normalise_period(period: Optional[str]) -> Optional[str]:
    """Normalise period string for comparison (case-insensitive, strip spaces)."""
    if not period:
        return None
    p = period.lower().strip()
    for key, val in PERIOD_KEYWORDS.items():
        if key in p:
            return val
    return p


def check_temporal_alignment(a: SourceValue, b: SourceValue) -> bool:
    """
    Step 3: Return True if both values cover the same time period.
    Mismatched periods (e.g. TTM vs FY) are NOT real conflicts.
    """
    pa = _normalise_period(a.period)
    pb = _normalise_period(b.period)

    if pa is None or pb is None:
        return True   # unknown periods — assume aligned, let tier rule decide

    # TTM vs fiscal year are different periods — not a real conflict
    if "trailing" in (pa, pb) and "trailing" not in (pa and pb):
        return False

    return pa == pb


# ---------------------------------------------------------------------------
# Step 4 — Restatement check
# ---------------------------------------------------------------------------

def check_restatement(a: SourceValue, b: SourceValue) -> Optional[SourceValue]:
    """
    Step 4: If both values cover the same period but differ, the newer
    source likely reflects a restatement — prefer it.
    Returns the newer SourceValue, or None if dates are unavailable/equal.
    """
    if not a.date or not b.date:
        return None
    try:
        date_a = datetime.fromisoformat(a.date[:10])
        date_b = datetime.fromisoformat(b.date[:10])
        if date_a > date_b:
            return a
        if date_b > date_a:
            return b
    except (ValueError, TypeError):
        pass
    return None


# ---------------------------------------------------------------------------
# Main resolver
# ---------------------------------------------------------------------------

def resolve(a: SourceValue, b: SourceValue) -> ConflictResolution:
    """
    Run the full 6-step conflict resolution protocol for two conflicting values.

    Always returns a ConflictResolution — caller decides what to do with it.
    The disclosure_note is ready to paste into the report as-is.
    """
    # Ensure consistent ordering: lower tier number = more authoritative
    if a.source_tier > b.source_tier:
        a, b = b, a   # a is always the higher-authority source

    max_val   = max(abs(a.value), abs(b.value))
    rel_diff  = abs(a.value - b.value) / max_val if max_val else 0.0
    diff_pct  = round(rel_diff * 100, 1)

    logger.info(
        "Conflict: %s — %s=%.4f (Tier %d) vs %s=%.4f (Tier %d) [%.1f%% diff]",
        a.metric_name, a.source_name, a.value, a.source_tier,
        b.source_name, b.value, b.source_tier, diff_pct,
    )

    # ------------------------------------------------------------------
    # Step 2: ASSESS TIERS — if tiers differ, higher authority wins
    # ------------------------------------------------------------------
    if a.source_tier != b.source_tier:
        resolution_step   = 2
        resolution_reason = (
            f"Tier {a.source_tier} ({a.source_name}) outranks "
            f"Tier {b.source_tier} ({b.source_name}) unconditionally."
        )
        preferred, rejected = a, b

    else:
        # Same tier — need further disambiguation

        # ------------------------------------------------------------------
        # Step 3: CHECK TEMPORAL ALIGNMENT
        # ------------------------------------------------------------------
        if not check_temporal_alignment(a, b):
            resolution_step   = 3
            resolution_reason = (
                f"Values cover different time periods "
                f"({a.period or 'unknown'} vs {b.period or 'unknown'}) — "
                f"not a real conflict. Using {a.source_name}."
            )
            preferred, rejected = a, b

        else:
            # ------------------------------------------------------------------
            # Step 4: CHECK FOR RESTATEMENTS
            # ------------------------------------------------------------------
            newer = check_restatement(a, b)
            if newer:
                other  = b if newer is a else a
                resolution_step   = 4
                resolution_reason = (
                    f"{newer.source_name} (dated {newer.date}) is more recent "
                    f"and may reflect a restatement of {other.source_name} "
                    f"(dated {other.date})."
                )
                preferred = newer
                rejected  = other

            else:
                # ------------------------------------------------------------------
                # Step 5: HIGHEST-TIER RULE (same tier, same period, no restatement)
                # ------------------------------------------------------------------
                resolution_step   = 5
                resolution_reason = (
                    f"Same tier ({a.source_tier}), same period, no restatement detected. "
                    f"Defaulting to {a.source_name} as primary source."
                )
                preferred, rejected = a, b

    # ------------------------------------------------------------------
    # Step 6: DOCUMENT
    # ------------------------------------------------------------------
    is_real = resolution_step != 3   # temporal mismatch is not a true conflict
    likely_cause = _infer_discrepancy_cause(resolution_step, a, b)

    disclosure_note = (
        f"**Data Note ({preferred.metric_name}):** "
        f"{preferred.source_name} reported {preferred.value:,} while "
        f"{rejected.source_name} reported {rejected.value:,} "
        f"(a {diff_pct}% discrepancy). "
        f"We used {preferred.source_name} (Tier {preferred.source_tier}) "
        f"as it is more authoritative. "
        f"Discrepancy may reflect {likely_cause}."
    )

    logger.info(
        "Resolved at Step %d: preferred=%s (%.4f) | reason: %s",
        resolution_step, preferred.source_name, preferred.value, resolution_reason,
    )

    return ConflictResolution(
        metric_name=      preferred.metric_name,
        preferred_value=  preferred.value,
        preferred_source= preferred.source_name,
        preferred_tier=   preferred.source_tier,
        rejected_value=   rejected.value,
        rejected_source=  rejected.source_name,
        rejected_tier=    rejected.source_tier,
        resolution_step=  resolution_step,
        resolution_reason=resolution_reason,
        disclosure_note=  disclosure_note,
        is_real_conflict= is_real,
    )


# ---------------------------------------------------------------------------
# Batch resolver — for the synthesis engine
# ---------------------------------------------------------------------------

def resolve_all(
    metric_name: str,
    source_values: list[SourceValue],
) -> tuple[float, list[ConflictResolution]]:
    """
    Resolve conflicts across multiple sources for the same metric.

    Uses a tournament approach: compare the current winner against each
    subsequent source, propagating the winner forward.

    Args:
        metric_name:   Name of the metric being resolved.
        source_values: All available values for this metric, any order.

    Returns:
        (final_value, list_of_resolutions)
        final_value is the single authoritative value to use in the report.
        list_of_resolutions contains Step 6 disclosures for all comparisons.
    """
    if not source_values:
        raise ValueError(f"No source values provided for metric '{metric_name}'.")

    if len(source_values) == 1:
        return source_values[0].value, []

    # Sort by tier ascending (Tier 1 first) so tournament starts with best source
    sorted_sources = sorted(source_values, key=lambda s: s.source_tier)
    resolutions:   list[ConflictResolution] = []
    winner = sorted_sources[0]

    for challenger in sorted_sources[1:]:
        if detect_conflict(winner, challenger):
            resolution = resolve(winner, challenger)
            resolutions.append(resolution)
            # Update winner to the preferred source
            winner = next(
                s for s in (winner, challenger)
                if s.source_name == resolution.preferred_source
            )
        else:
            # Values agree within 5% — no conflict, keep current winner
            logger.debug(
                "No conflict: %s %.4f vs %s %.4f (within 5%%)",
                winner.source_name, winner.value,
                challenger.source_name, challenger.value,
            )

    return winner.value, resolutions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_discrepancy_cause(step: int, a: SourceValue, b: SourceValue) -> str:
    """Infer the most likely explanation for the discrepancy."""
    if step == 3:
        return f"different reporting periods ({a.period} vs {b.period})"
    if step == 4:
        return "a financial restatement in the more recent filing"
    if a.source_tier <= 2 and b.source_tier >= 4:
        return "data lag in news sources vs official filings"
    return "different reporting methodologies or data lag"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
