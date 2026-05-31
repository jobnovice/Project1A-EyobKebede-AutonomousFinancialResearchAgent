"""
src/tools/tool_registry.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

Central registry for tool execution, validation, and fallback routing.
"""
import json
import logging
from typing import Any, Dict, Callable
from jsonschema import validate, ValidationError

# Import the schema definitions you created
from tools.schemas import SCHEMA_MAP, FALLBACK_CHAINS
# Import stubs (to be built below)
from tools.stubs import TOOL_STUBS

logger = logging.getLogger(__name__)

class ToolExecutionError(Exception):
    """Custom exception for tool failures."""
    pass

class ToolRegistry:
    def __init__(self):
        self.schemas = SCHEMA_MAP
        self.fallbacks = FALLBACK_CHAINS
        self.tools: Dict[str, Callable] = TOOL_STUBS
        
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
            raise ValueError(f"Tool '{tool_name}' is not registered.")
        
        schema = self.schemas[tool_name]["parameters"]
        try:
            validate(instance=params, schema=schema)
        except ValidationError as e:
            raise ValueError(f"Schema validation failed for {tool_name}: {str(e)}")

    def execute_with_fallback(self, tool_name: str, params: Dict[str, Any], attempt: int = 1) -> Dict[str, Any]:
        """
        Executes a tool. If it fails, automatically routes to the fallback chain
        defined in the architecture specification.
        """
        self.metrics["total_calls"] += 1
        
        # 1. Validation
        try:
            self._validate_inputs(tool_name, params)
        except ValueError as e:
            logger.error(e)
            self.metrics["failed_calls"] += 1
            return {"error": str(e), "status": "failed"}

        # 2. Execution (Primary)
        try:
            logger.info(f"Executing {tool_name} with params: {params}")
            result = self.tools[tool_name](**params)
            self.metrics["successful_calls"] += 1
            return {"result": result, "source": tool_name, "status": "success"}
            
        except Exception as e:
            logger.warning(f"Primary tool {tool_name} failed: {str(e)}. Initiating fallback.")
            
            # 3. Fallback Routing
            chain = self.fallbacks.get(tool_name, [])
            for fallback_tool in chain:
                logger.info(f"Attempting fallback: {fallback_tool}")
                try:
                    # Note: In a production environment, params might need mapping between tools
                    fallback_result = self.tools[fallback_tool](**params)
                    self.metrics["fallback_recoveries"] += 1
                    return {
                        "result": fallback_result, 
                        "source": fallback_tool, 
                        "status": "recovered_via_fallback",
                        "original_failure": tool_name
                    }
                except Exception as fallback_err:
                    logger.warning(f"Fallback {fallback_tool} failed: {str(fallback_err)}")
                    continue
            
            # 4. Graceful Degradation (All fallbacks exhausted)
            self.metrics["failed_calls"] += 1
            logger.error(f"All fallbacks exhausted for {tool_name}.")
            return {
                "error": f"DATA UNAVAILABLE: {tool_name} and fallbacks failed.",
                "status": "degraded"
            }