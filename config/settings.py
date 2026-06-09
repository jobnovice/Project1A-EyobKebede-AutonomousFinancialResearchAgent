"""
config/settings.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

Single source of truth for all configuration.
Reads from environment variables (populated from .env via python-dotenv).

Usage:
    from config.settings import settings
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (two levels up from config/)
load_dotenv(Path(__file__).parent.parent / ".env")


@dataclass(frozen=True)
class LLMSettings:
    # Anthropic
    anthropic_api_key: str = field(
        default_factory=lambda: os.environ["ANTHROPIC_API_KEY"])
    anthropic_model: str = field(default_factory=lambda: os.getenv(
        "ANTHROPIC_MODEL", "claude-sonnet-4-5"))
    anthropic_max_tokens: int = field(default_factory=lambda: int(
        os.getenv("ANTHROPIC_MAX_TOKENS", "8192")))
    anthropic_temperature: float = field(
        default_factory=lambda: float(os.getenv("ANTHROPIC_TEMPERATURE", "0.1")))
    groq_api_key: str | None = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY"))
    groq_model: str = field(default_factory=lambda: os.getenv(
        "GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"))
    deepseek_api_key: str | None = field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_KEY"))
    deepseek_model: str = field(default_factory=lambda: os.getenv(
        "DEEPSEEK_MODEL", "deepseek-chat"))
    # OpenAI (optional fallback)
    openai_api_key: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o"))
    gemini_api_key: str | None = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    gemini_model: str = field(default_factory=lambda: os.getenv(
        "GEMINI_MODEL", "gemini-2.5-flash"))
    # Token budget per research session
    max_tokens_per_session: int = field(default_factory=lambda: int(
        os.getenv("MAX_TOKENS_PER_SESSION", "150000")))
    context_window_warning: int = field(default_factory=lambda: int(
        os.getenv("CONTEXT_WINDOW_WARNING", "120000")))


@dataclass(frozen=True)
class VectorDBSettings:
    provider: str = field(default_factory=lambda: os.getenv(
        "VECTOR_DB_PROVIDER", "chroma"))
    chroma_path: str = field(default_factory=lambda: os.getenv(
        "CHROMA_PATH", "./data/chroma_db"))
    collection_name: str = field(default_factory=lambda: os.getenv(
        "CHROMA_COLLECTION", "ara1_research"))
    embedding_model: str = field(default_factory=lambda: os.getenv(
        "EMBEDDING_MODEL", "text-embedding-3-small"))
    embedding_dim: int = field(default_factory=lambda: int(
        os.getenv("EMBEDDING_DIM", "1536")))
    min_similarity: float = field(default_factory=lambda: float(
        os.getenv("MIN_SIMILARITY", "0.70")))

    # Pinecone (optional upgrade)
    pinecone_api_key: str | None = field(
        default_factory=lambda: os.getenv("PINECONE_API_KEY"))
    pinecone_index: str = field(
        default_factory=lambda: os.getenv("PINECONE_INDEX", "ara1"))


@dataclass(frozen=True)
class APISettings:
    # Financial data
    fmp_api_key: str | None = field(
        default_factory=lambda: os.getenv("FMP_API_KEY"))
    alpha_vantage_key: str | None = field(
        default_factory=lambda: os.getenv("ALPHA_VANTAGE_KEY"))

    # Search
    tavily_api_key: str | None = field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY"))
    serpapi_key: str | None = field(
        default_factory=lambda: os.getenv("SERPAPI_KEY"))

    # News
    newsapi_key: str | None = field(
        default_factory=lambda: os.getenv("NEWSAPI_KEY"))

    # SEC EDGAR — no key required, but User-Agent is mandatory
    sec_user_agent: str = field(default_factory=lambda: os.getenv(
        "SEC_USER_AGENT", "Project1A-ARA1 eyob@research.local"
    ))
    sec_rate_limit_rps: int = field(
        default_factory=lambda: int(os.getenv("SEC_RATE_LIMIT_RPS", "8")))


@dataclass(frozen=True)
class AgentSettings:
    max_iterations: int = field(default_factory=lambda: int(
        os.getenv("MAX_ITERATIONS", "20")))
    max_tool_calls: int = field(default_factory=lambda: int(
        os.getenv("MAX_TOOL_CALLS", "20")))
    retry_max_attempts: int = field(
        default_factory=lambda: int(os.getenv("RETRY_MAX_ATTEMPTS", "5")))
    retry_base_delay: float = field(
        default_factory=lambda: float(os.getenv("RETRY_BASE_DELAY", "1.0")))
    circuit_breaker_threshold: int = field(
        default_factory=lambda: int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "3")))
    circuit_breaker_timeout: int = field(default_factory=lambda: int(
        os.getenv("CIRCUIT_BREAKER_TIMEOUT", "60")))
    use_stubs: bool = field(default_factory=lambda: os.getenv(
        "USE_STUBS", "true").lower() == "true")


@dataclass(frozen=True)
class LoggingSettings:
    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    traces_dir: str = field(default_factory=lambda: os.getenv(
        "TRACES_DIR", "./logs/traces"))
    log_file: str = field(default_factory=lambda: os.getenv(
        "LOG_FILE", "./logs/ara1.log"))
    trace_to_file: bool = field(default_factory=lambda: os.getenv(
        "TRACE_TO_FILE", "true").lower() == "true")


@dataclass(frozen=True)
class Settings:
    llm: LLMSettings = field(default_factory=LLMSettings)
    vector_db: VectorDBSettings = field(default_factory=VectorDBSettings)
    api: APISettings = field(default_factory=APISettings)
    agent: AgentSettings = field(default_factory=AgentSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)

    def validate(self) -> list[str]:
        """Return list of missing required config values."""
        issues = []
        if not self.llm.anthropic_api_key:
            issues.append("ANTHROPIC_API_KEY is not set")
        if not self.api.fmp_api_key and not self.agent.use_stubs:
            issues.append("FMP_API_KEY is not set and USE_STUBS=false")
        if not self.api.tavily_api_key and not self.agent.use_stubs:
            issues.append("TAVILY_API_KEY is not set and USE_STUBS=false")
        return issues


# Singleton — import this everywhere
settings = Settings()
