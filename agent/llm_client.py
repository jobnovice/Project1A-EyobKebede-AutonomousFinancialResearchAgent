"""
agent/llm_client.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

Anthropic API wrapper with:
  - Exponential backoff retry (tenacity)
  - Token usage tracking (per-call + session cumulative)
  - Tool-call response parsing (returns structured ToolCallRequest objects)
  - Context window guard (warns before hitting limit)
  - OpenAI fallback hook (Day 3 stub — wired on Day 4 if needed)

Usage:
    from agent.llm_client import LLMClient
    client = LLMClient()
    response = client.chat(messages, tools=ALL_SCHEMAS, system=SYSTEM_PROMPT)
    if response.tool_calls:
        for tc in response.tool_calls:
            result = registry.call(tc.tool_name, tc.parameters)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
    before_sleep_log,
)

from config.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response data structures
# ---------------------------------------------------------------------------

@dataclass
class ToolCallRequest:
    """A single tool invocation requested by the LLM."""
    tool_use_id: str      # Anthropic's internal ID — needed for tool_result messages
    tool_name: str
    parameters: dict


@dataclass
class LLMResponse:
    """Parsed response from a single LLM API call."""
    content_text: str                          # concatenated text blocks
    tool_calls: list[ToolCallRequest]          # may be empty
    input_tokens: int
    output_tokens: int
    stop_reason: str                           # 'end_turn' | 'tool_use' | 'max_tokens'
    raw_content: list                          # original Anthropic content blocks

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class SessionTokenUsage:
    """Cumulative token tracker for one research session."""
    input_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def record(self, response: LLMResponse) -> None:
        self.input_tokens += response.input_tokens
        self.output_tokens += response.output_tokens
        self.api_calls += 1

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total,
            "api_calls": self.api_calls,
            "estimated_cost_usd": round(
                (self.input_tokens * 3 + self.output_tokens * 15) / 1_000_000, 4
            ),  # Claude Sonnet pricing: $3/M input, $15/M output
        }


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Thin, reliable wrapper around the Anthropic Messages API.

    Handles:
      - Retry with exponential backoff + jitter on transient errors
      - Token tracking (per-call + session)
      - Tool-call response parsing
      - Context window warnings
      - Building tool_result messages for multi-turn tool loops
    """

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=settings.llm.anthropic_api_key)
        self._model = settings.llm.anthropic_model
        self._max_tokens = settings.llm.anthropic_max_tokens
        self.session_usage = SessionTokenUsage()
        logger.info("LLMClient initialised — model: %s", self._model)

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        *,
        system: str = "",
        tools: list[dict] | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """
        Send a messages request and return a parsed LLMResponse.

        Args:
            messages:    Anthropic-format message list (role/content pairs).
            system:      System prompt string.
            tools:       List of tool schemas (ALL_SCHEMAS from tools.schemas).
            temperature: Override default temperature.

        Returns:
            LLMResponse with text content, tool calls, and token counts.
        """
        self._check_context_window(messages, system)

        kwargs: dict[str, Any] = {
            "model":      self._model,
            "max_tokens": self._max_tokens,
            "messages":   messages,
            "temperature": temperature if temperature is not None else settings.llm.anthropic_temperature,
        }
        if system:
            kwargs["system"] = system
        if tools:
            # Anthropic tool format: list of {"name", "description", "input_schema"}
            kwargs["tools"] = [self._to_anthropic_tool(t) for t in tools]

        raw = self._call_with_retry(**kwargs)
        response = self._parse_response(raw)
        self.session_usage.record(response)

        logger.debug(
            "LLM call: %d input / %d output tokens | stop=%s | tools=%d",
            response.input_tokens, response.output_tokens,
            response.stop_reason, len(response.tool_calls),
        )
        return response

    # ------------------------------------------------------------------
    # Message builders — helpers for multi-turn tool loops
    # ------------------------------------------------------------------

    @staticmethod
    def tool_result_message(tool_use_id: str, result: Any) -> dict:
        """
        Build the 'tool_result' message to feed back after a tool call.
        The result is serialised to a string if it's a dict/list.
        """
        import json as _json
        content = result if isinstance(result, str) else _json.dumps(result, default=str)
        return {
            "role": "user",
            "content": [{
                "type":        "tool_result",
                "tool_use_id": tool_use_id,
                "content":     content,
            }],
        }

    @staticmethod
    def assistant_tool_use_message(response: LLMResponse) -> dict:
        """
        Build the assistant message containing the tool_use blocks,
        required by Anthropic's API when continuing the conversation.
        """
        return {
            "role":    "assistant",
            "content": response.raw_content,
        }

    # ------------------------------------------------------------------
    # Retry wrapper
    # ------------------------------------------------------------------

    def _call_with_retry(self, **kwargs) -> anthropic.types.Message:
        """Execute API call with tenacity retry on transient errors."""

        @retry(
            retry=retry_if_exception_type((
                anthropic.RateLimitError,
                anthropic.APIConnectionError,
                anthropic.InternalServerError,
            )),
            wait=wait_exponential(
                multiplier=settings.agent.retry_base_delay,
                min=1, max=32
            ) + wait_random(0, 0.5),
            stop=stop_after_attempt(settings.agent.retry_max_attempts),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        def _call():
            return self._client.messages.create(**kwargs)

        return _call()

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(raw: anthropic.types.Message) -> LLMResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []

        for block in raw.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCallRequest(
                    tool_use_id=block.id,
                    tool_name=block.name,
                    parameters=dict(block.input),
                ))

        return LLMResponse(
            content_text=" ".join(text_parts),
            tool_calls=tool_calls,
            input_tokens=raw.usage.input_tokens,
            output_tokens=raw.usage.output_tokens,
            stop_reason=raw.stop_reason or "end_turn",
            raw_content=raw.content,
        )

    @staticmethod
    def _to_anthropic_tool(schema: dict) -> dict:
        """Convert our JSON schema format to Anthropic's tool format."""
        return {
            "name":         schema["name"],
            "description":  schema["description"],
            "input_schema": schema["parameters"],
        }

    def _check_context_window(self, messages: list[dict], system: str) -> None:
        """Rough token estimate — warn if approaching context limit."""
        import json as _json
        rough_chars = len(system) + sum(
            len(_json.dumps(m, default=str)) for m in messages
        )
        rough_tokens = rough_chars // 4   # ~4 chars per token
        if rough_tokens > settings.llm.context_window_warning:
            logger.warning(
                "Context window approaching limit: ~%d estimated tokens "
                "(warning threshold: %d). Consider summarising earlier steps.",
                rough_tokens, settings.llm.context_window_warning,
            )

    def reset_session(self) -> None:
        """Reset token counters between research sessions."""
        self.session_usage = SessionTokenUsage()
