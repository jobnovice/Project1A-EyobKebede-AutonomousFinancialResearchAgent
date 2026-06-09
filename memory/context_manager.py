"""
memory/context_manager.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

Short-term memory (context window) manager.

Responsibilities:
  1. Token counting    — estimate token usage across all state components
  2. Budget tracking   — enforce the 6-bucket allocation from architecture spec §6.2
  3. Summarisation     — compress oldest TAO traces when usage hits 70% of window
  4. Trace archiving   — save raw traces to session log before compression
  5. Headroom check    — gate every LLM call to ensure enough room for response

Architecture spec §6.2 token budget (200K window):
  System prompt + tool registry :  ~8K   (4%)
  Research plan                 :  ~2K   (1%)
  Active TAO traces             :  ~40K  (20%)
  Gathered data (current step)  :  ~80K  (40%)
  Prior step summaries          :  ~40K  (20%)
  Generation headroom           :  ~30K  (15%)
  ─────────────────────────────────────────────
  Total                         :  200K  (100%)

Summarisation trigger: 70% of window = 140K tokens.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants from architecture spec §6.2
# ---------------------------------------------------------------------------
CONTEXT_WINDOW        = 200_000   # Claude Sonnet 4 context window
SUMMARISE_THRESHOLD   = 0.70      # Trigger compression at 70% usage
HEADROOM_TOKENS       = 30_000    # Always reserve for generation

# Token budget per bucket (soft targets — used for logging / warnings)
BUDGET = {
    "system_prompt":   8_000,
    "research_plan":   2_000,
    "tao_traces":     40_000,
    "gathered_data":  80_000,
    "step_summaries": 40_000,
    "headroom":       30_000,
}

TRACE_LOG_DIR = Path("logs/traces")


# ---------------------------------------------------------------------------
# Token estimation (no tiktoken required — approximation is sufficient)
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """
    Estimate token count without requiring tiktoken.
    Rule of thumb: ~4 characters per token for English prose.
    Slightly conservative (3.8) to avoid underestimating.
    """
    return max(1, int(len(text) / 3.8))


def estimate_dict_tokens(obj: dict | list) -> int:
    """Estimate tokens for a JSON-serialisable object."""
    try:
        return estimate_tokens(json.dumps(obj))
    except (TypeError, ValueError):
        return estimate_tokens(str(obj))


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------

class ContextManager:
    """
    Tracks context window usage and manages compression for a single session.

    Usage:
        cm = ContextManager(session_id="session-20240101-001")
        cm.update(state)              # call after every node execution
        if cm.should_summarise():
            compressed = cm.summarise(state, llm_call_fn)
            # use compressed TAO traces going forward
        if not cm.has_headroom():
            # do not make another LLM call
    """

    def __init__(self, session_id: str) -> None:
        self.session_id   = session_id
        self._usage: dict[str, int] = {k: 0 for k in BUDGET}
        self._total       = 0
        self._archive_path = TRACE_LOG_DIR / f"{session_id}-traces.jsonl"
        TRACE_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Update usage from current ResearchState
    # ------------------------------------------------------------------

    def update(self, state: dict) -> dict[str, int]:
        """
        Recalculate token usage from the current state dict.

        Returns:
            Dict of bucket → token count, plus "total" and "pct_used".
        """
        usage = {
            "system_prompt":  BUDGET["system_prompt"],   # fixed cost — not in state
            "research_plan":  estimate_dict_tokens(state.get("research_plan", [])),
            "tao_traces":     estimate_dict_tokens(state.get("tool_call_log", [])),
            "gathered_data":  estimate_dict_tokens(state.get("gathered_data", {})),
            "step_summaries": estimate_tokens(state.get("draft_report") or ""),
            "headroom":       HEADROOM_TOKENS,
        }
        total = sum(usage.values())
        usage["total"]    = total
        usage["pct_used"] = round(total / CONTEXT_WINDOW * 100, 1)
        self._usage = usage
        self._total = total

        logger.debug(
            "Context usage: %d/%d tokens (%.1f%%) | traces=%d gathered=%d",
            total, CONTEXT_WINDOW, usage["pct_used"],
            usage["tao_traces"], usage["gathered_data"],
        )
        return usage

    # ------------------------------------------------------------------
    # Threshold checks
    # ------------------------------------------------------------------

    def should_summarise(self) -> bool:
        """True when total usage exceeds the 70% summarisation trigger."""
        return self._total >= (CONTEXT_WINDOW * SUMMARISE_THRESHOLD)

    def has_headroom(self, needed: int = HEADROOM_TOKENS) -> bool:
        """True when there is at least `needed` tokens free for generation."""
        return (CONTEXT_WINDOW - self._total) >= needed

    def usage_report(self) -> dict:
        """Return current usage snapshot for logging / evaluation."""
        return dict(self._usage)

    # ------------------------------------------------------------------
    # Summarisation
    # ------------------------------------------------------------------

    def summarise(
        self,
        state: dict,
        llm_call_fn,
        keep_recent: int = 3,
    ) -> list[dict]:
        """
        Compress the oldest TAO traces to free context space.

        Strategy (from architecture spec §6.2):
          1. Archive raw traces to disk (Trace Gallery).
          2. Split tool_call_log into oldest (to compress) + recent (to keep).
          3. Call the LLM summariser on the oldest traces.
          4. Return a single compressed summary entry + the recent traces.

        Args:
            state:        Current ResearchState dict.
            llm_call_fn:  Callable(system, user) → str  (the _call_llm function).
            keep_recent:  Number of most-recent TAO entries to keep verbatim.

        Returns:
            New tool_call_log with oldest traces replaced by one summary entry.
        """
        tool_log = state.get("tool_call_log", [])

        if len(tool_log) <= keep_recent:
            logger.info("Summarisation skipped — only %d traces, nothing to compress.", len(tool_log))
            return tool_log

        to_compress = tool_log[:-keep_recent]
        to_keep     = tool_log[-keep_recent:]

        # Step 1: Archive raw traces before compressing
        self._archive_traces(to_compress, state.get("session_id", self.session_id))

        # Step 2: Build compression prompt
        traces_text = json.dumps(to_compress, indent=2)
        system = (
            "You are a research assistant compressing agent execution traces. "
            "Extract only the KEY FINDINGS from these Thought-Action-Observation traces. "
            "Output a concise bullet-point summary (max 200 words). "
            "Keep all numerical data, company names, dates, and source references. "
            "Discard reasoning steps and intermediate thoughts."
        )
        user = f"Compress these {len(to_compress)} trace entries:\n\n{traces_text[:8000]}"

        # Step 3: LLM compression call
        try:
            summary_text = llm_call_fn(system, user)
        except Exception as e:
            logger.error("Summarisation LLM call failed: %s — keeping traces uncompressed.", e)
            return tool_log

        # Step 4: Replace compressed traces with a single summary entry
        summary_entry = {
            "step_id":    "COMPRESSED",
            "cycle":      0,
            "tool":       "context_summariser",
            "params":     {"entries_compressed": len(to_compress)},
            "thought":    "Context window approaching limit — earlier traces compressed.",
            "result_summary": summary_text[:500],
            "timestamp":  _now(),
            "is_summary": True,
        }

        compressed_log = [summary_entry] + to_keep
        tokens_saved   = estimate_dict_tokens(to_compress) - estimate_tokens(summary_text)

        logger.info(
            "Summarisation complete: %d traces → 1 summary (~%d tokens freed).",
            len(to_compress), max(0, tokens_saved),
        )
        return compressed_log

    # ------------------------------------------------------------------
    # Chunker — used by vector_db_store for optimal chunk sizing
    # ------------------------------------------------------------------

    @staticmethod
    def chunk_text(
        text: str,
        source_type: str = "generic",
        max_tokens: int = 400,
        overlap_tokens: int = 50,
    ) -> list[str]:
        """
        Split text into overlapping chunks sized for embedding.

        Chunking strategy by source_type (from architecture spec §6.4):
          10-K / 10-Q : by paragraph, max 400 tokens, 50-token overlap
          earnings     : by speaker turn (split on \\n\\n)
          news         : by paragraph, always include first paragraph
          generic      : fixed-size with overlap

        Args:
            text:           Full document text.
            source_type:    '10-K', '10-Q', 'earnings_call', 'news', or 'generic'.
            max_tokens:     Target max tokens per chunk.
            overlap_tokens: Overlap between consecutive chunks.

        Returns:
            List of text chunks.
        """
        if not text or not text.strip():
            return []

        # Earnings transcripts: split by speaker turn
        if source_type in ("earnings_call", "earnings_transcript"):
            return _chunk_by_speaker(text, max_tokens)

        # News: paragraph-based, always prepend first paragraph for context
        if source_type == "news":
            return _chunk_news(text, max_tokens)

        # 10-K, 10-Q, generic: paragraph-based with overlap
        return _chunk_paragraphs(text, max_tokens, overlap_tokens)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _archive_traces(self, traces: list[dict], session_id: str) -> None:
        """Append raw traces to the session JSONL archive (Trace Gallery)."""
        try:
            with open(self._archive_path, "a", encoding="utf-8") as f:
                for trace in traces:
                    record = {"session_id": session_id, "archived_at": _now(), **trace}
                    f.write(json.dumps(record) + "\n")
            logger.debug("Archived %d traces to %s", len(traces), self._archive_path)
        except OSError as e:
            logger.warning("Could not archive traces: %s", e)


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

def _chunk_paragraphs(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split by paragraphs, merge short ones, overlap at boundaries."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para)
        if current_tokens + para_tokens > max_tokens and current:
            chunk = "\n\n".join(current)
            chunks.append(chunk)
            # Overlap: keep last paragraph(s) within overlap budget
            overlap: list[str] = []
            overlap_count = 0
            for p in reversed(current):
                t = estimate_tokens(p)
                if overlap_count + t > overlap_tokens:
                    break
                overlap.insert(0, p)
                overlap_count += t
            current = overlap
            current_tokens = overlap_count

        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks if chunks else [text[:max_tokens * 4]]


def _chunk_by_speaker(text: str, max_tokens: int) -> list[str]:
    """Split earnings transcript by speaker turns (double newline boundaries)."""
    turns = [t.strip() for t in text.split("\n\n") if t.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for turn in turns:
        turn_tokens = estimate_tokens(turn)
        # Keep Q&A pairs together — if the turn fits, accumulate
        if current_tokens + turn_tokens > max_tokens and current:
            chunks.append("\n\n".join(current))
            current = []
            current_tokens = 0
        current.append(turn)
        current_tokens += turn_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks if chunks else [text]


def _chunk_news(text: str, max_tokens: int) -> list[str]:
    """Paragraph-based; every chunk includes the first paragraph for context."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [text]

    first_para = paragraphs[0]
    first_tokens = estimate_tokens(first_para)

    if len(paragraphs) == 1:
        return [first_para]

    chunks: list[str] = []
    current = [first_para]
    current_tokens = first_tokens

    for para in paragraphs[1:]:
        para_tokens = estimate_tokens(para)
        if current_tokens + para_tokens > max_tokens:
            chunks.append("\n\n".join(current))
            current = [first_para, para]   # always restart with first para
            current_tokens = first_tokens + para_tokens
        else:
            current.append(para)
            current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
