"""
src/tools/tool_registry.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

Central registry for tool execution, validation, circuit-breaker routing,
and operational metrics tracking.
"""
import time
import logging
from typing import Any, Dict, Callable, List
from jsonschema import validate, ValidationError

# Direct imports matching actual project tree layout
from tools.schemas import SCHEMA_MAP, FALLBACK_CHAINS

# Explicitly import the tool functions directly from their respective files
try:
    from tools.sec_edgar import sec_filing_search
    from tools.financial_api import financial_data_api
    from tools.web_search import web_search
    from tools.news_sentiment import news_sentiment
    from tools.company_profile import company_profile
    from tools.peer_comparison import peer_comparison
    from tools.calculator import calculation_engine
    from tools.fact_checker import fact_checker
    from tools.report_gen import report_generator
except ImportError as e:
    logging.getLogger(__name__).warning(f"Core tool import partial miss: {e}. Ensure matching function names exist inside files.")

# Fallback placeholders for memory modules if memory layer is still initializing
try:
    from memory.vector_db_tools import vector_db_search, vector_db_store
except ImportError:
    def vector_db_search(*args, **kwargs): return {"status": "mock_cache_miss"}
    def vector_db_store(*args, **kwargs): return {"status": "mock_stored"}

logger = logging.getLogger(__name__)

class ToolExecutionError(Exception):
    """Custom exception for tool failures."""
    pass

class ToolRegistry:
    def __init__(self):
        self.schemas = SCHEMA_MAP
        self.fallbacks = FALLBACK_CHAINS
        
        # Explicit mapping dictionary targeting independent tool files
        self.tools: Dict[str, Callable] = {
            "sec_filing_search": sec_filing_search,
            "financial_data_api": financial_data_api,
            "web_search": web_search,
            "news_sentiment": news_sentiment,
            "company_profile": company_profile,
            "peer_comparison": peer_comparison,
            "calculation_engine": calculation_engine,
            "fact_checker": fact_checker,
            "report_generator": report_generator,
            "vector_db_search": vector_db_search,
            "vector_db_store": vector_db_store
        }
        
        # Day 3 Requirement: Circuit Breaker State Tracking
        # Tracks states: "CLOSED", "OPEN", "HALF-OPEN"
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {
            tool_name: {
                "state": "CLOSED",
                "sequential_failures": [],  # List of timestamps
                "last_state_change": 0.0
            } for tool_name in self.tools
        }
        
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

    def _check_circuit_breaker(self, tool_name: str) -> bool:
        """
        Evaluates the circuit breaker status for a specific tool.
        Returns True if execution can proceed, or False if tripped OPEN.
        """
        cb = self.circuit_breakers.get(tool_name)
        if not cb:
            return True
            
        current_time = time.time()
        
        if cb["state"] == "OPEN":
            # Half-open rule: After 5 minutes (300s), permit exactly one probe request
            if current_time - cb["last_state_change"] > 300:
                logger.info(f"🔧 Circuit breaker for '{tool_name}' entering HALF-OPEN probe state.")
                cb["state"] = "HALF-OPEN"
                cb["last_state_change"] = current_time
                return True
            return False
            
        return True

    def _handle_tool_failure(self, tool_name: str) -> None:
        """Logs sequential failure metrics and trips circuit breaker if thresholds break."""
        cb = self.circuit_breakers.get(tool_name)
        if not cb:
            return
            
        current_time = time.time()
        cb["sequential_failures"].append(current_time)
        
        # Filter failures out older than the 60-second window restriction
        cb["sequential_failures"] = [t for t in cb["sequential_failures"] if current_time - t <= 60]
        
        # If 3 or more consecutive failures register inside the 60s frame, trip the circuit
        if len(cb["sequential_failures"]) >= 3 or cb["state"] == "HALF-OPEN":
            cb["state"] = "OPEN"
            cb["last_state_change"] = current_time
            logger.error(f"🚨 CIRCUIT BREAKER TRIPPED OPEN FOR '{tool_name}'. Routing directly to fallbacks.")

    def _handle_tool_success(self, tool_name: str) -> None:
        """Resets sequential tracking records on a clear execution confirmation."""
        cb = self.circuit_breakers.get(tool_name)
        if cb:
            if cb["state"] == "HALF-OPEN":
                logger.info(f"✅ Tool '{tool_name}' passed probe. Circuit breaker reset to CLOSED.")
            cb["state"] = "CLOSED"
            cb["sequential_failures"] = []

    def execute_with_fallback(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a tool. Instantly diverts to alternative recovery chains if the 
        tool schema validation fails or if its circuit breaker is actively OPEN.
        """
        self.metrics["total_calls"] += 1
        
        # 1. Verification & Schema Validation
        try:
            self._validate_inputs(tool_name, params)
        except ValueError as e:
            logger.error(f"Validation error on tool {tool_name}: {str(e)}")
            self.metrics["failed_calls"] += 1
            return {"error": str(e), "status": "failed"}

        # 2. Circuit Breaker Execution Gate check
        if not self._check_circuit_breaker(tool_name):
            logger.warning(f"⚠️ Circuit OPEN for '{tool_name}'. Bypassing execution, skipping directly to fallback.")
            return self._execute_fallback_chain(tool_name, params)

        # 3. Primary Execution Path
        try:
            if tool_name not in self.tools or self.tools[tool_name] is None:
                raise ToolExecutionError(f"Target function reference missing for tool: {tool_name}")
                
            logger.info(f"Executing {tool_name} with params: {params}")
            result = self.tools[tool_name](**params)
            
            self._handle_tool_success(tool_name)
            self.metrics["successful_calls"] += 1
            return {"result": result, "source": tool_name, "status": "success"}
            
        except Exception as e:
            logger.warning(f"Primary tool {tool_name} execution error: {str(e)}.")
            self._handle_tool_failure(tool_name)
            return self._execute_fallback_chain(tool_name, params)

    def _execute_fallback_chain(self, original_tool: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Iterates sequentially across historical fallback definitions to locate recovery data."""
        chain = self.fallbacks.get(original_tool, [])
        
        for fallback_tool in chain:
            if fallback_tool not in self.tools or self.tools[fallback_tool] is None:
                continue
                
            logger.info(f"Attempting architectural fallback: {fallback_tool}")
            try:
                fallback_result = self.tools[fallback_tool](**params)
                self.metrics["fallback_recoveries"] += 1
                return {
                    "result": fallback_result, 
                    "source": fallback_tool, 
                    "status": "recovered_via_fallback",
                    "original_failure": original_tool
                }
            except Exception as fallback_err:
                logger.warning(f"Fallback option {fallback_tool} also dropped: {str(fallback_err)}")
                continue
        
        # 4. Graceful Degradation Protocol (All paths exhausted)
        self.metrics["failed_calls"] += 1
        logger.error(f"All structural fallbacks completely exhausted for tool: {original_tool}.")
        return {
            "error": f"DATA UNAVAILABLE: {original_tool} and all fallbacks failed.",
            "status": "degraded"
        }