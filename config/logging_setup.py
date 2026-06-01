"""
config/logging_setup.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

Logging infrastructure that captures:
  - Agent reasoning traces (Thought / Action / Observation)
  - Tool calls (name, params, result, latency)
  - LLM token usage per call
  - Errors with full context

Two handlers:
  1. Console — human-readable, colour-coded by level
  2. File    — JSON-lines format for machine parsing by evaluation framework

Usage:
    from config.logging_setup import setup_logging, get_trace_logger
    setup_logging()                    # call once at startup
    trace = get_trace_logger("ara1")   # per-session trace logger
    trace.thought("I need to retrieve the 10-K for AAPL...")
    trace.action("sec_filing_search", {"ticker": "AAPL", "filing_type": "10-K"})
    trace.observation(result_dict)
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# JSON-lines formatter — every log record is one parseable JSON object
# ---------------------------------------------------------------------------

class JSONLFormatter(logging.Formatter):
    """Emits one JSON object per line — suitable for log analytics."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts":      self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Carry through any extra fields attached to the record
        for key, val in record.__dict__.items():
            if key not in {
                "msg", "args", "levelname", "levelno", "pathname", "filename",
                "module", "exc_info", "exc_text", "stack_info", "lineno",
                "funcName", "created", "msecs", "relativeCreated", "thread",
                "threadName", "processName", "process", "name", "message",
            }:
                payload[key] = val
        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Colour console formatter
# ---------------------------------------------------------------------------

_COLOURS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
}
_RESET = "\033[0m"


class ColourFormatter(logging.Formatter):
    FMT = "%(asctime)s %(levelname)-8s %(name)-20s %(message)s"

    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOURS.get(record.levelname, "")
        record.levelname = f"{colour}{record.levelname}{_RESET}"
        return super().format(record)


# ---------------------------------------------------------------------------
# Public setup function
# ---------------------------------------------------------------------------

def setup_logging(
    level: str = "INFO",
    log_file: str = "./logs/ara1.log",
    traces_dir: str = "./logs/traces",
) -> None:
    """
    Configure root logger with console + rotating file handlers.
    Call once at application startup (main.py or agent entry point).
    """
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    Path(traces_dir).mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove any handlers added by imported libraries
    root.handlers.clear()

    # 1. Console handler — human-readable
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(ColourFormatter(
        fmt="%(asctime)s %(levelname)-8s %(name)-20s %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(console)

    # 2. Rotating file handler — JSON-lines
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(JSONLFormatter())
    root.addHandler(file_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# TraceLogger — structured agent trace events
# ---------------------------------------------------------------------------

class TraceLogger:
    """
    Writes structured Thought / Action / Observation / Token events
    to a per-session JSONL trace file.

    These trace files are consumed by:
      - evaluation/metrics.py  (tool efficiency, memory utilisation)
      - docs/trace_gallery.md  (human-readable trace curation)
    """

    def __init__(self, session_id: str, traces_dir: str = "./logs/traces") -> None:
        self.session_id = session_id
        self._path = Path(traces_dir) / f"{session_id}.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._start = time.time()
        self._step = 0
        self._logger = logging.getLogger(f"trace.{session_id}")

    def _write(self, event_type: str, payload: dict) -> None:
        self._step += 1
        record = {
            "session_id": self.session_id,
            "step":       self._step,
            "event":      event_type,
            "elapsed_s":  round(time.time() - self._start, 3),
            **payload,
        }
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        self._logger.debug("[%s] step=%d %s", event_type, self._step, str(payload)[:120])

    def thought(self, text: str) -> None:
        """Log a Thought step from the LLM reasoning trace."""
        self._write("THOUGHT", {"text": text})

    def action(self, tool_name: str, parameters: dict) -> None:
        """Log a tool call Action."""
        self._write("ACTION", {"tool": tool_name, "parameters": parameters})

    def observation(self, result: Any, tool_name: str = "", success: bool = True) -> None:
        """Log the Observation returned from a tool call."""
        self._write("OBSERVATION", {
            "tool": tool_name,
            "success": success,
            "result_preview": str(result)[:300],
        })

    def token_usage(self, input_tokens: int, output_tokens: int, model: str = "") -> None:
        """Log LLM token consumption for cost tracking."""
        self._write("TOKEN_USAGE", {
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "total_tokens":  input_tokens + output_tokens,
            "model":         model,
        })

    def plan(self, steps: list[str]) -> None:
        """Log the Plan-and-Execute plan at session start."""
        self._write("PLAN", {"steps": steps, "step_count": len(steps)})

    def memory_hit(self, query: str, results_count: int, top_score: float) -> None:
        """Log a vector DB cache hit — used to compute metric AB-4."""
        self._write("MEMORY_HIT", {
            "query": query,
            "results_count": results_count,
            "top_similarity": top_score,
        })

    def fallback(self, original_tool: str, fallback_tool: str, reason: str) -> None:
        """Log a fallback chain activation — used for metric AB-2."""
        self._write("FALLBACK", {
            "original_tool": original_tool,
            "fallback_tool": fallback_tool,
            "reason": reason[:200],
        })

    def error(self, message: str, context: dict | None = None) -> None:
        """Log a handled error."""
        self._write("ERROR", {"message": message, "context": context or {}})

    def final_report(self, word_count: int, sections_completed: int, confidence: float) -> None:
        """Log final report metadata."""
        self._write("FINAL_REPORT", {
            "word_count": word_count,
            "sections_completed": sections_completed,
            "avg_confidence": confidence,
            "total_elapsed_s": round(time.time() - self._start, 1),
        })

    @property
    def trace_path(self) -> str:
        return str(self._path)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def get_trace_logger(session_id: str | None = None, traces_dir: str = "./logs/traces") -> TraceLogger:
    """Create (or retrieve) a TraceLogger for the given session."""
    sid = session_id or f"session-{uuid.uuid4().hex[:8]}"
    return TraceLogger(session_id=sid, traces_dir=traces_dir)


def setup_from_settings() -> None:
    """Convenience wrapper — reads config from settings singleton."""
    from config.settings import settings
    setup_logging(
        level=settings.logging.level,
        log_file=settings.logging.log_file,
        traces_dir=settings.logging.traces_dir,
    )
