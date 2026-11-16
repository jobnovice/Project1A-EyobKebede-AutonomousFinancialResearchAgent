"""
Per-tool circuit breakers for preventing repeated calls to failing tools.

Provides:
  CircuitBreaker
      Tracks failures independently for each tool and temporarily blocks calls
      to tools that repeatedly fail.

  allow(tool_name)
      Returns whether a tool call is permitted. An open circuit blocks calls
      until its reset timeout expires; then exactly one half-open probe is
      admitted.

  record_success(tool_name) / record_failure(tool_name, error_type=None)
      Update circuit state after the outcome of an attempted tool call. A
      successful call closes the circuit; a failed half-open probe reopens it.
      Data-level and invalid-parameter failures are treated as non-outage
      responses and do not keep the breaker open across a probe.

State transitions:
  CLOSED → OPEN       after the failure threshold is reached within the window
  OPEN → HALF-OPEN    after the reset timeout; permits one recovery probe
  HALF-OPEN → CLOSED  when the probe succeeds or responds with a non-outage error
  HALF-OPEN → OPEN    when the probe fails with a real outage

Configuration:
  CIRCUIT_BREAKER_THRESHOLD  failures before opening (default: 3)
  CIRCUIT_BREAKER_TIMEOUT   seconds before a probe is allowed (default: 60)
  probe_timeout_s           deadline for an in-flight half-open probe (default: reset timeout)
  failure_window_s          rolling window for counting failures (default: 60)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

CircuitStatus = Literal["CLOSED", "OPEN", "HALF-OPEN"]


@dataclass
class _ToolCircuit:
    status: CircuitStatus = "CLOSED"
    failures: list[float] = field(default_factory=list)
    opened_at: float | None = None
    probe_in_flight: bool = False
    probe_started_at: float | None = None


class CircuitBreaker:
    """Track failures independently per tool and allow a single recovery probe.

    A tool circuit opens after ``failure_threshold`` failures within
    ``failure_window_s``. After ``reset_timeout_s``, one call is admitted as a
    half-open probe. A successful probe closes the circuit; a failed probe
    opens it again.
    """

    def __init__(
        self,
        failure_threshold: int | None = None,
        reset_timeout_s: float | None = None,
        failure_window_s: float = 60.0,
        probe_timeout_s: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.failure_threshold = (
            int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "3"))
            if failure_threshold is None else failure_threshold
        )
        self.reset_timeout_s = (
            float(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "60"))
            if reset_timeout_s is None else reset_timeout_s
        )
        self.probe_timeout_s = (
            self.reset_timeout_s if probe_timeout_s is None else probe_timeout_s
        )
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if self.reset_timeout_s < 0:
            raise ValueError("reset_timeout_s must be non-negative")
        if self.probe_timeout_s < 0:
            raise ValueError("probe_timeout_s must be non-negative")
        if failure_window_s <= 0:
            raise ValueError("failure_window_s must be positive")

        self.failure_window_s = failure_window_s
        self._clock = clock
        self._circuits: dict[str, _ToolCircuit] = {}
        self._lock = threading.Lock()

    def allow(self, tool_name: str) -> bool:
        """Return whether a call is permitted; only one half-open probe may run."""
        with self._lock:
            circuit = self._circuits.setdefault(tool_name, _ToolCircuit())
            now = self._clock()

            if circuit.status == "OPEN":
                assert circuit.opened_at is not None
                if now - circuit.opened_at < self.reset_timeout_s:
                    return False
                circuit.status = "HALF-OPEN"
                circuit.probe_in_flight = True
                circuit.probe_started_at = now
                logger.info("Circuit for %s entered HALF-OPEN; admitting one probe.", tool_name)
                return True

            if circuit.status == "HALF-OPEN":
                if circuit.probe_in_flight:
                    if circuit.probe_started_at is not None and now - circuit.probe_started_at >= self.probe_timeout_s:
                        logger.warning(
                            "Circuit for %s half-open probe timed out; reopening the circuit.",
                            tool_name,
                        )
                        self._open(tool_name, circuit, now)
                        return False
                    return False
                circuit.probe_in_flight = True
                circuit.probe_started_at = now
                return True

            return True

    def record_success(self, tool_name: str) -> None:
        """Close the tool circuit and clear its failure history after success."""
        with self._lock:
            circuit = self._circuits.setdefault(tool_name, _ToolCircuit())
            was_open = circuit.status != "CLOSED"
            circuit.status = "CLOSED"
            circuit.failures.clear()
            circuit.opened_at = None
            circuit.probe_in_flight = False
            circuit.probe_started_at = None
            if was_open:
                logger.info("Circuit for %s closed after a successful call.", tool_name)

    def record_failure(self, tool_name: str, error_type: str | None = None) -> None:
        """Record a failed call and open the circuit when its threshold is reached.

        Data and invalid-parameter responses are treated as non-outage results and
        are not allowed to keep a half-open probe stuck open.
        """
        with self._lock:
            circuit = self._circuits.setdefault(tool_name, _ToolCircuit())
            now = self._clock()

            if error_type in {"data", "invalid_params"}:
                if circuit.status == "HALF-OPEN":
                    circuit.status = "CLOSED"
                    circuit.failures.clear()
                    circuit.opened_at = None
                    circuit.probe_in_flight = False
                    circuit.probe_started_at = None
                    logger.info(
                        "Circuit for %s closed after a non-outage response during half-open.",
                        tool_name,
                    )
                return

            if circuit.status == "HALF-OPEN":
                self._open(tool_name, circuit, now)
                return

            if circuit.status == "OPEN":
                return

            cutoff = now - self.failure_window_s
            circuit.failures = [failure for failure in circuit.failures if failure > cutoff]
            circuit.failures.append(now)
            if len(circuit.failures) >= self.failure_threshold:
                self._open(tool_name, circuit, now)

    def get_state(self, tool_name: str) -> CircuitStatus:
        """Return a tool's current state, defaulting to CLOSED if unseen."""
        with self._lock:
            circuit = self._circuits.get(tool_name)
            return circuit.status if circuit else "CLOSED"

    def _open(self, tool_name: str, circuit: _ToolCircuit, now: float) -> None:
        circuit.status = "OPEN"
        circuit.opened_at = now
        circuit.probe_in_flight = False
        circuit.probe_started_at = None
        logger.warning("Circuit for %s opened after repeated failures.", tool_name)
 