"""
src/tools/tool_registry.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

Central registry for tool execution, validation, circuit-breaker routing,
and operational metrics tracking.
"""
import logging
from typing import Any, Callable, Dict
from jsonschema import validate, ValidationError

from agent.circuit_breaker import CircuitBreaker
from agent.fallback_chains import execute_with_fallbacks
from tools.schemas import SCHEMA_MAP

# Explicitly import the tool functions directly from their respective files
try:
    from tools.sec_edgar import sec_filing_search
    from tools.financial_api import financial_data_api
    from tools.web_search import web_search
    from tools.news_sentiment import news_sentiment
    from tools.company_profile import company_profile
    from tools.peer_comparison import peer_comparison
    from tools.earnings import earnings_transcript
    from tools.calculator import calculation_engine
    from tools.fact_checker import fact_checker
    from tools.report_gen import report_generator
except ImportError as e:
    logging.getLogger(__name__).warning(f"Core tool import partial miss: {e}. Ensure matching function names exist inside files.")

# Fallback placeholders for memory modules if memory layer is still initializing
try:
    from memory.vector_db_tools import vector_db_search, vector_db_store
except ImportError:
    def vector_db_search(*args, **kwargs):
        del args, kwargs
        return {"status": "mock_cache_miss"}

    def vector_db_store(*args, **kwargs):
        del args, kwargs
        return {"status": "mock_stored"}

logger = logging.getLogger(__name__)

class ToolRegistry:
    def __init__(self, circuit_breaker: CircuitBreaker | None = None):
        self.schemas = SCHEMA_MAP
        
        # Explicit mapping dictionary targeting independent tool files
        self.tools: Dict[str, Callable] = {
            "sec_filing_search": sec_filing_search,
            "financial_data_api": financial_data_api,
            "web_search": web_search,
            "news_sentiment": news_sentiment,
            "company_profile": company_profile,
            "peer_comparison": peer_comparison,
            "earnings_transcript": earnings_transcript,
            "calculation_engine": calculation_engine,
            "fact_checker": fact_checker,
            "report_generator": report_generator,
            "vector_db_search": vector_db_search,
            "vector_db_store": vector_db_store
        }

        self.circuit_breaker = (
            circuit_breaker if circuit_breaker is not None else CircuitBreaker()
        )
        
        # Track tool statistics for metric AB-1 (Tool Efficiency)
        self.metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "fallback_recoveries": 0,
            "failed_calls": 0
        }

    def _validate_inputs(self, tool_name: str, params: Dict[str, Any]) -> None:
        """Validates incoming parameters against the strict JSON schema."""
        if tool_name not in self.schemas:
            raise ValueError(f"Tool '{tool_name}' is not registered in schemas.")
        
        schema = self.schemas[tool_name].get("parameters", {})
        try:
            validate(instance=params, schema=schema)
        except ValidationError as e:
            raise ValueError(f"Schema validation failed for {tool_name}: {str(e)}")

    def execute_with_fallback(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate inputs, then delegate retries and fallback routing to the shared
        fallback-chain executor.
        """
        self.metrics["total_calls"] += 1
        
        # 1. Verification & Schema Validation
        try:
            self._validate_inputs(tool_name, params)
        except ValueError as e:
            logger.error(f"Validation error on tool {tool_name}: {str(e)}")
            self.metrics["failed_calls"] += 1
            return {"error": str(e), "status": "failed"}

        chain_result = execute_with_fallbacks(
            tool_name,
            params,
            self.tools,
            breaker=self.circuit_breaker,
        )
        if chain_result.success:
            self.metrics["successful_calls"] += 1
            if chain_result.used_fallback:
                self.metrics["fallback_recoveries"] += 1
            status = "recovered_via_fallback" if chain_result.used_fallback else "success"
        else:
            self.metrics["failed_calls"] += 1
            status = "degraded"

        response: dict[str, Any] = {
            "result": chain_result.result,
            "source": chain_result.source_tool or tool_name,
            "status": status,
            "used_fallback": chain_result.used_fallback,
            "degraded": chain_result.degraded,
            "disclosure": chain_result.disclosure,
        }
        if chain_result.degraded:
            response["error"] = chain_result.disclosure
        return response