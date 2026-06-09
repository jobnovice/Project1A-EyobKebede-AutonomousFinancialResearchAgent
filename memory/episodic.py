"""
memory/episodic.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

Episodic memory — persistent strategy log for agent self-improvement.

Medium:    JSON log (full records) + SQLite index (fast queries)
Scope:     Persistent across all sessions
Used by:   Planner — informs tool prioritisation and strategy selection
           based on what worked in past sessions of the same query type.

Architecture spec §6.5 schema:
  session_id, query_type, companies, tool_performance,
  strategy_notes, errors_encountered, final_quality_score,
  memory_hit_rate

Key queries the Planner uses:
  - get_tool_priority(query_type)    → ranked tool list by success rate
  - get_strategy_notes(query_type)   → past strategy observations
  - get_error_patterns(query_type)   → common errors to avoid
  - get_summary(query_type)          → compact planner-ready text block
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------
EPISODIC_DIR   = Path("logs/episodic")
JSON_LOG_FILE  = EPISODIC_DIR / "sessions.jsonl"
SQLITE_DB_FILE = EPISODIC_DIR / "episodic_index.db"


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------

class EpisodicMemory:
    """
    Logs completed research sessions and answers planning queries.

    Two-layer storage:
      JSON log  — append-only JSONL, one record per session (full fidelity)
      SQLite    — indexed view for fast aggregation queries by the Planner
    """

    def __init__(self) -> None:
        EPISODIC_DIR.mkdir(parents=True, exist_ok=True)
        self._db = self._init_db()
        logger.info("EpisodicMemory ready — %d sessions logged", self.session_count())

    # ------------------------------------------------------------------
    # Write: log a completed session
    # ------------------------------------------------------------------

    def log_session(self, state: dict) -> None:
        """
        Persist a completed research session to both JSON log and SQLite.

        Args:
            state: Final ResearchState dict from run_agent().
        """
        session_id   = state.get("session_id", f"session-{_now_compact()}")
        query_type   = state.get("query_type", "unknown")
        companies    = state.get("companies", [])
        tool_log     = state.get("tool_call_log", [])
        error_log    = state.get("error_log", [])
        metrics      = state.get("evaluation_metrics", {})
        memory_hits  = state.get("memory_hits", 0)
        total_calls  = state.get("total_api_calls", 0)

        # Compute per-tool performance from tool_call_log
        tool_perf = _compute_tool_performance(tool_log)

        # Derive strategy notes from what actually worked
        strategy_notes = _derive_strategy_notes(query_type, tool_perf, metrics)

        record = {
            "session_id":        session_id,
            "query_type":        query_type,
            "companies":         companies,
            "raw_query":         state.get("raw_query", ""),
            "tool_performance":  tool_perf,
            "strategy_notes":    strategy_notes,
            "errors_encountered": error_log,
            "final_quality_score": metrics.get("overall_confidence"),
            "memory_hit_rate":   round(memory_hits / total_calls, 3) if total_calls > 0 else 0.0,
            "hallucination_count": state.get("hallucination_count", 0),
            "total_tool_calls":  total_calls,
            "latency_seconds":   metrics.get("AB-5_latency_seconds"),
            "logged_at":         _now(),
        }

        # Append to JSON log (full fidelity)
        self._append_json(record)

        # Index in SQLite (fast aggregation)
        self._index_sqlite(record)

        logger.info(
            "Session logged: %s | type=%s | companies=%s | quality=%.2f | hit_rate=%.2f",
            session_id, query_type, companies,
            record["final_quality_score"] or 0.0,
            record["memory_hit_rate"],
        )

    # ------------------------------------------------------------------
    # Read: planner queries
    # ------------------------------------------------------------------

    def get_tool_priority(self, query_type: str, top_n: int = 5) -> list[dict]:
        """
        Return tools ranked by success rate for a given query type.
        Used by the Planner to order tool steps optimally.

        Returns:
            List of {tool, avg_success_rate, avg_latency_ms, call_count}
            sorted by success rate descending.
        """
        rows = self._db.execute(
            """
            SELECT tool_name,
                   AVG(success_rate)  AS avg_success,
                   AVG(avg_latency_ms) AS avg_latency,
                   SUM(call_count)    AS total_calls
            FROM   tool_stats
            WHERE  query_type = ?
            GROUP  BY tool_name
            ORDER  BY avg_success DESC, total_calls DESC
            LIMIT  ?
            """,
            (query_type, top_n),
        ).fetchall()

        return [
            {
                "tool":             row[0],
                "avg_success_rate": round(row[1], 3),
                "avg_latency_ms":   round(row[2] or 0, 0),
                "call_count":       int(row[3]),
            }
            for row in rows
        ]

    def get_strategy_notes(self, query_type: str, limit: int = 3) -> list[str]:
        """
        Return the most recent strategy notes for a query type.
        Injected into the Planner prompt as learned guidance.
        """
        rows = self._db.execute(
            """
            SELECT strategy_notes
            FROM   sessions
            WHERE  query_type = ?
              AND  strategy_notes IS NOT NULL
              AND  strategy_notes != ''
            ORDER  BY logged_at DESC
            LIMIT  ?
            """,
            (query_type, limit),
        ).fetchall()
        return [row[0] for row in rows]

    def get_error_patterns(self, query_type: str, limit: int = 5) -> list[str]:
        """
        Return the most common errors for a query type.
        Used to warn the Planner about known failure modes.
        """
        rows = self._db.execute(
            """
            SELECT error_text, COUNT(*) AS freq
            FROM   error_log
            WHERE  query_type = ?
            GROUP  BY error_text
            ORDER  BY freq DESC
            LIMIT  ?
            """,
            (query_type, limit),
        ).fetchall()
        return [row[0] for row in rows]

    def get_summary(self, query_type: str) -> str:
        """
        Return a compact, planner-ready text block summarising past experience.
        Injected directly into the Planner system prompt.
        """
        tool_priority  = self.get_tool_priority(query_type, top_n=5)
        strategy_notes = self.get_strategy_notes(query_type, limit=2)
        error_patterns = self.get_error_patterns(query_type, limit=3)
        session_count  = self._session_count_for_type(query_type)

        if session_count == 0:
            return f"No prior sessions found for query_type='{query_type}'."

        lines = [f"EPISODIC MEMORY ({session_count} prior '{query_type}' sessions):"]

        if tool_priority:
            lines.append("  Top tools by success rate:")
            for t in tool_priority:
                lines.append(
                    f"    {t['tool']}: {t['avg_success_rate']*100:.0f}% success, "
                    f"{int(t['avg_latency_ms'])}ms avg, {t['call_count']} calls"
                )

        if strategy_notes:
            lines.append("  Strategy notes from past sessions:")
            for note in strategy_notes:
                lines.append(f"    - {note}")

        if error_patterns:
            lines.append("  Common errors to avoid:")
            for err in error_patterns:
                lines.append(f"    ! {err}")

        return "\n".join(lines)

    def session_count(self) -> int:
        row = self._db.execute("SELECT COUNT(*) FROM sessions").fetchone()
        return row[0] if row else 0

    def get_recent_sessions(self, limit: int = 10) -> list[dict]:
        """Return the most recent session records for inspection."""
        rows = self._db.execute(
            """
            SELECT session_id, query_type, companies, final_quality_score,
                   memory_hit_rate, latency_seconds, logged_at
            FROM   sessions
            ORDER  BY logged_at DESC
            LIMIT  ?
            """,
            (limit,),
        ).fetchall()
        cols = ["session_id", "query_type", "companies", "final_quality_score",
                "memory_hit_rate", "latency_seconds", "logged_at"]
        return [dict(zip(cols, row)) for row in rows]

    # ------------------------------------------------------------------
    # Private: SQLite setup
    # ------------------------------------------------------------------

    def _init_db(self) -> sqlite3.Connection:
        db = sqlite3.connect(str(SQLITE_DB_FILE))
        db.execute("PRAGMA journal_mode=WAL")   # safe concurrent writes
        db.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id          TEXT PRIMARY KEY,
                query_type          TEXT NOT NULL,
                companies           TEXT,           -- JSON array
                raw_query           TEXT,
                strategy_notes      TEXT,
                final_quality_score REAL,
                memory_hit_rate     REAL,
                hallucination_count INTEGER,
                total_tool_calls    INTEGER,
                latency_seconds     REAL,
                logged_at           TEXT
            );

            CREATE TABLE IF NOT EXISTS tool_stats (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT NOT NULL,
                query_type   TEXT NOT NULL,
                tool_name    TEXT NOT NULL,
                call_count   INTEGER,
                success_count INTEGER,
                success_rate  REAL,
                avg_latency_ms REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS error_log (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT NOT NULL,
                query_type   TEXT NOT NULL,
                error_text   TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_qtype   ON sessions(query_type);
            CREATE INDEX IF NOT EXISTS idx_tool_stats_qtype ON tool_stats(query_type);
            CREATE INDEX IF NOT EXISTS idx_error_qtype      ON error_log(query_type);
        """)
        db.commit()
        return db

    def _append_json(self, record: dict) -> None:
        try:
            with open(JSON_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as e:
            logger.warning("Could not write JSON log: %s", e)

    def _index_sqlite(self, record: dict) -> None:
        sid        = record["session_id"]
        query_type = record["query_type"]

        # Upsert session row
        self._db.execute(
            """
            INSERT OR REPLACE INTO sessions
              (session_id, query_type, companies, raw_query, strategy_notes,
               final_quality_score, memory_hit_rate, hallucination_count,
               total_tool_calls, latency_seconds, logged_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                sid,
                query_type,
                json.dumps(record["companies"]),
                record["raw_query"],
                record["strategy_notes"],
                record["final_quality_score"],
                record["memory_hit_rate"],
                record["hallucination_count"],
                record["total_tool_calls"],
                record["latency_seconds"],
                record["logged_at"],
            ),
        )

        # Insert tool stats rows
        for tool_name, perf in record["tool_performance"].items():
            self._db.execute(
                """
                INSERT INTO tool_stats
                  (session_id, query_type, tool_name, call_count,
                   success_count, success_rate, avg_latency_ms)
                VALUES (?,?,?,?,?,?,?)
                """,
                (
                    sid, query_type, tool_name,
                    perf["calls"], perf["successes"],
                    perf["success_rate"], perf["avg_latency_ms"],
                ),
            )

        # Insert error log rows
        for err in record["errors_encountered"]:
            self._db.execute(
                "INSERT INTO error_log (session_id, query_type, error_text) VALUES (?,?,?)",
                (sid, query_type, str(err)[:500]),
            )

        self._db.commit()

    def _session_count_for_type(self, query_type: str) -> int:
        row = self._db.execute(
            "SELECT COUNT(*) FROM sessions WHERE query_type=?", (query_type,)
        ).fetchone()
        return row[0] if row else 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_tool_performance(tool_log: list[dict]) -> dict[str, dict]:
    """
    Aggregate tool_call_log entries into per-tool performance stats.
    A call is a 'success' if its result_summary is non-empty and not an error.
    """
    stats: dict[str, dict] = {}

    for entry in tool_log:
        tool = entry.get("tool", "unknown")
        if tool not in stats:
            stats[tool] = {"calls": 0, "successes": 0, "latencies_ms": []}

        stats[tool]["calls"] += 1

        result = str(entry.get("result_summary", ""))
        is_error = "error" in result.lower() or "failed" in result.lower() or not result
        if not is_error:
            stats[tool]["successes"] += 1

        # Estimate latency from timestamps if available
        ts = entry.get("timestamp")
        if ts:
            stats[tool]["latencies_ms"].append(0)  # placeholder — real timing in Day 10

    # Compute derived fields
    output = {}
    for tool, s in stats.items():
        calls = s["calls"]
        successes = s["successes"]
        latencies = s["latencies_ms"]
        output[tool] = {
            "calls":         calls,
            "successes":     successes,
            "success_rate":  round(successes / calls, 3) if calls > 0 else 0.0,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 0) if latencies else 0.0,
        }

    return output


def _derive_strategy_notes(
    query_type: str,
    tool_perf: dict,
    metrics: dict,
) -> str:
    """
    Auto-generate a strategy note from session outcomes.
    Kept concise — injected into future planner prompts.
    """
    lines = []

    # Note which tools had 100% success
    reliable = [t for t, p in tool_perf.items() if p["success_rate"] == 1.0 and p["calls"] >= 2]
    if reliable:
        lines.append(f"Reliable tools for {query_type}: {', '.join(reliable)}.")

    # Note any tools that failed
    failing = [t for t, p in tool_perf.items() if p["success_rate"] < 0.5 and p["calls"] >= 2]
    if failing:
        lines.append(f"Tools with low success rate: {', '.join(failing)} — use fallbacks.")

    # Note if memory cache was effective
    confidence = metrics.get("overall_confidence")
    if confidence and confidence >= 0.90:
        lines.append(f"High confidence ({confidence:.2f}) achieved for {query_type} queries.")

    return " ".join(lines) if lines else f"Standard {query_type} research completed."


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_episodic: Optional[EpisodicMemory] = None


def get_episodic_memory() -> EpisodicMemory:
    """Return the singleton EpisodicMemory, creating it on first call."""
    global _episodic
    if _episodic is None:
        _episodic = EpisodicMemory()
    return _episodic
