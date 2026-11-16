"""
agent/error_handler.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

Comprehensive error handling for all external API calls.

Provides:
  retry_with_backoff(fn, *args, **kwargs)
      Wraps any tool call with exponential backoff retry logic.
      Pure stdlib implementation (no tenacity dependency).

  with_graceful_degradation(tool_name, fn, params, context)
      Executes a tool call with full error handling pipeline:
        1. Retry with exponential backoff
        2. On final failure → return structured [DATA UNAVAILABLE] dict
        3. Log the failure for episodic memory

  GracefulResult
      Structured return type that carries both the tool result and
      degradation metadata so the synthesis engine knows what to trust.

Architecture spec §9.1 degradation protocol:
  - Network errors → retry 3x with exponential backoff (1s, 2s, 4s)
  - Rate limit (429) → retry 3x with longer delays (5s, 10s, 20s)
  - Auth errors (401/403) → fail immediately, no retry
  - Data errors (empty/malformed) → use fallback tool (see fallback_chains.py)
  - All-fallbacks-exhausted → mark section [DATA UNAVAILABLE] and continue
"""

from __future__ import annotations

import logging
import re
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------
DEFAULT_MAX_RETRIES    = 3
DEFAULT_BASE_DELAY     = 1.0      # seconds
DEFAULT_MAX_DELAY      = 30.0     # seconds
RATE_LIMIT_BASE_DELAY  = 5.0      # longer backoff for 429s
RATE_LIMIT_MAX_DELAY   = 60.0

# Error substrings that signal a rate limit
RATE_LIMIT_SIGNALS = {"429", "rate limit", "too many requests", "quota exceeded"}

# Error substrings that signal auth failure (no retry)
AUTH_FAILURE_SIGNALS = {"401", "403", "unauthorized", "forbidden", "invalid api key",
                        "api key not configured", "not configured"}

DATA_SIGNALS = {"no data", "returned no", "not found", "no results", "no transcript", "no record",
                "empty response", "empty result", "empty response from", "could not find"}

INVALID_PARAM_SIGNALS = {
    "typeerror",
    "unexpected keyword argument",
    "unexpected positional argument",
    "missing required positional argument",
    "unsupported parameter",
    "invalid parameter",
    "invalid parameters",
    "got an unexpected keyword argument",
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class GracefulResult:
    """
    Wraps a tool call result with error handling metadata.
    Always returned — even on failure, this carries a usable dict.
    """
    tool_name:      str
    result:         dict               # the actual tool output (or degradation stub)
    success:        bool
    attempts:       int
    total_delay_s:  float
    error_type:     Optional[str]      # "network" | "rate_limit" | "auth" | "data" | None
    error_message:  Optional[str]
    degraded:       bool               # True if [DATA UNAVAILABLE] was substituted
    retrieved_at:   str = field(default_factory=lambda: _now())
    attempt_log:    list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def _http_status_code(message: str) -> int | None:
    """Extract a status code such as HTTP 404 or status=503 from the error text."""
    match = re.search(r"(?:http\s+|status(?:\s+code)?\s+|status=)(\d{3})(?:\b|$)", message, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def _classify_error(exc: Exception) -> tuple[str, bool]:
    """
    Classify an exception into an error type and whether it's retryable.
    Returns (error_type, is_retryable).
    """
    msg = str(exc).lower()
    status_code = _http_status_code(msg)

    if isinstance(exc, TypeError) or any(s in msg for s in INVALID_PARAM_SIGNALS):
        return "invalid_params", False

    if status_code is not None:
        if status_code in {401, 403}:
            return "auth", False
        if status_code == 404:
            return "data", False
        if status_code == 422:
            return "invalid_params", False
        if status_code == 429:
            return "rate_limit", True
        if 500 <= status_code <= 599:
            return "network", True

    if any(s in msg for s in AUTH_FAILURE_SIGNALS):
        return "auth", False

    if any(s in msg for s in RATE_LIMIT_SIGNALS):
        return "rate_limit", True

    if any(s in msg for s in ["network", "connection", "timeout", "urlopen", "socket"]):
        return "network", True

    if any(s in msg for s in DATA_SIGNALS) or "no " in msg and " found" in msg:
        return "data", False   # retrying won't help with empty data

    return "unknown", True


def _compute_delay(attempt: int, base: float, max_delay: float, jitter: bool = True) -> float:
    """Exponential backoff: base * 2^attempt, capped at max_delay."""
    delay = min(base * (2 ** attempt), max_delay)
    if jitter:
        import random
        delay *= (0.8 + random.random() * 0.4)   # ±20% jitter
    return round(delay, 2)


# ---------------------------------------------------------------------------
# Core retry logic (pure stdlib — no tenacity dependency required)
# ---------------------------------------------------------------------------

def retry_with_backoff(
    fn: Callable,
    *args,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    tool_name: str = "unknown",
    _delay_tracker: Optional[list[float]] = None,
    _attempt_tracker: Optional[list[dict[str, Any]]] = None,
    **kwargs,
) -> tuple[Any, int, float]:
    """
    Call fn(*args, **kwargs) with exponential backoff retry.

    Args:
        fn:          The callable to retry.
        max_retries: Maximum number of retry attempts (not counting first try).
        base_delay:  Initial backoff delay in seconds.
        max_delay:   Maximum backoff delay in seconds.
        tool_name:   Used for logging only.

    Returns:
        (result, attempts_made, total_delay_seconds)

    Raises:
        The last exception if all retries are exhausted.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")

    last_exc: Optional[Exception] = None
    total_delay = 0.0

    for attempt in range(max_retries + 1):
        try:
            result = fn(*args, **kwargs)
            if _attempt_tracker is not None:
                _attempt_tracker.append({
                    "attempt": attempt + 1,
                    "status": "success",
                    "error_type": None,
                    "error_message": None,
                    "retry_delay_s": 0.0,
                })
            if attempt > 0:
                logger.info(
                    "[%s] Succeeded on attempt %d/%d after %.1fs total delay.",
                    tool_name, attempt + 1, max_retries + 1, total_delay
                )
            return result, attempt + 1, total_delay

        except Exception as exc:
            last_exc = exc
            error_type, retryable = _classify_error(exc)
            retry_delay = 0.0
            if retryable and attempt < max_retries:
                b = RATE_LIMIT_BASE_DELAY if error_type == "rate_limit" else base_delay
                m = RATE_LIMIT_MAX_DELAY if error_type == "rate_limit" else max_delay
                retry_delay = _compute_delay(attempt, b, m)
            if _attempt_tracker is not None:
                _attempt_tracker.append({
                    "attempt": attempt + 1,
                    "status": "failed",
                    "error_type": error_type,
                    "error_message": str(exc),
                    "retryable": retryable,
                    "retry_delay_s": retry_delay,
                })

            if not retryable:
                logger.warning(
                    "[%s] Non-retryable error (%s): %s",
                    tool_name, error_type, exc
                )
                raise

            if attempt < max_retries:
                total_delay += retry_delay
                if _delay_tracker is not None:
                    _delay_tracker.append(retry_delay)
                logger.warning(
                    "[%s] Attempt %d/%d failed (%s): %s. Retrying in %.1fs...",
                    tool_name, attempt + 1, max_retries + 1, error_type, exc, retry_delay
                )
                time.sleep(retry_delay)
            else:
                logger.error(
                    "[%s] All %d attempts failed. Last error: %s",
                    tool_name, max_retries + 1, exc
                )

    if last_exc is None:
        raise RuntimeError("Retry loop completed without an exception")
    raise last_exc


# ---------------------------------------------------------------------------
# Graceful degradation wrapper
# ---------------------------------------------------------------------------

def with_graceful_degradation(
    tool_name: str,
    fn: Callable,
    params: dict,
    context: Optional[dict] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> GracefulResult:
    """
    Execute a tool call with the full error handling pipeline.

    Pipeline:
      1. Retry with exponential backoff (auth errors are not retried)
      2. On success → return GracefulResult(success=True)
      3. On final failure → return GracefulResult(success=False, degraded=True)
         with a [DATA UNAVAILABLE] stub result that the synthesis engine
         and report generator can handle transparently

    Args:
        tool_name:   Name of the tool being called (for logging and stub).
        fn:          The tool callable.
        params:      Parameters to pass as **kwargs.
        context:     Optional dict with 'company', 'query' for richer stubs.
        max_retries: Override default retry count.

    Returns:
        GracefulResult — always, never raises.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")

    context = context or {}
    company = context.get("company", "")
    query   = context.get("query", "")

    # Single retry path: stdlib backoff with error classification
    call_count = [0]
    retry_delays: list[float] = []
    attempt_log: list[dict[str, Any]] = []

    def _counted(**kw):
        call_count[0] += 1
        return fn(**kw)

    try:
        result, attempts, delay = retry_with_backoff(
            _counted, max_retries=max_retries, tool_name=tool_name,
            _delay_tracker=retry_delays, _attempt_tracker=attempt_log, **params
        )
        return GracefulResult(
            tool_name=tool_name, result=result, success=True,
            attempts=attempts, total_delay_s=delay,
            error_type=None, error_message=None, degraded=False,
            attempt_log=attempt_log,
        )

    except Exception as exc:
        error_type, _ = _classify_error(exc)
        error_msg = str(exc)
        tb = traceback.format_exc()

        logger.error(
            "[%s] DEGRADED after %d attempts: %s\n%s",
            tool_name, call_count[0], error_msg, tb[-500:]
        )

        # Build the [DATA UNAVAILABLE] stub result
        stub = _build_degradation_stub(tool_name, error_msg, company, query)

        return GracefulResult(
            tool_name=tool_name, result=stub, success=False,
            attempts=call_count[0],   # real number of calls made (auth errors stop at 1)
            total_delay_s=round(sum(retry_delays), 2),
            error_type=error_type,
            error_message=error_msg,
            degraded=True,
            attempt_log=attempt_log,
        )


def _build_degradation_stub(
    tool_name: str,
    error_msg: str,
    company: str = "",
    query:   str = "",
) -> dict:
    """
    Build a transparent [DATA UNAVAILABLE] stub result.
    The synthesis engine and report generator check for this marker
    and include a disclosure note rather than fabricating data.
    """
    company_note = f" for {company}" if company else ""
    return {
        "tool_name":            tool_name,
        "data":                 None,
        "degraded":             True,
        "degradation_marker":   "[DATA UNAVAILABLE]",
        "degradation_note": (
            f"{tool_name}{company_note} is currently unavailable. "
            f"This section could not be completed. "
            f"Error: {error_msg[:200]}"
        ),
        "source_tier":          None,
        "source_weight":        0.0,
        "retrieved_at":         _now(),
        "error":                error_msg[:500],
    }


# ---------------------------------------------------------------------------
# Failure injection (Day 9 Task 5 — for testing recovery behaviour)
# ---------------------------------------------------------------------------

class FailureInjector:
    """
    Wraps tool calls and injects simulated failures at a configurable rate.
    Used only in tests — never in production.

    Usage:
        injector = FailureInjector(failure_rate=0.5)
        wrapped_fn = injector.wrap(real_tool_fn)
        result = with_graceful_degradation("tool", wrapped_fn, params)
    """

    def __init__(
        self,
        failure_rate: float = 0.3,
        failure_type: str = "network",
        fail_on_attempts: Optional[list[int]] = None,
    ):
        """
        Args:
            failure_rate:     0.0–1.0 probability of injecting a failure.
            failure_type:     "network" | "rate_limit" | "auth" | "data"
            fail_on_attempts: If set, only fail on these specific attempt numbers (1-indexed).
        """
        self.failure_rate     = failure_rate
        self.failure_type     = failure_type
        self.fail_on_attempts = fail_on_attempts or []
        self._attempt_count   = 0

    def wrap(self, fn: Callable) -> Callable:
        """Return a wrapped version of fn that may inject failures."""
        injector = self

        def _wrapped(*args, **kwargs):
            import random
            injector._attempt_count += 1
            attempt = injector._attempt_count

            should_fail = (
                (injector.fail_on_attempts and attempt in injector.fail_on_attempts) or
                (not injector.fail_on_attempts and random.random() < injector.failure_rate)
            )

            if should_fail:
                error_map = {
                    "network":    "Simulated network error: connection refused.",
                    "rate_limit": "Simulated 429 Too Many Requests: rate limit exceeded.",
                    "auth":       "Simulated 401 Unauthorized: invalid API key.",
                    "data":       "Simulated empty response: no data returned.",
                }
                msg = error_map.get(injector.failure_type, "Simulated failure.")
                logger.debug("[INJECTOR] Injecting %s failure on attempt %d", injector.failure_type, attempt)
                raise RuntimeError(msg)

            return fn(*args, **kwargs)

        return _wrapped

    def reset(self):
        self._attempt_count = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
