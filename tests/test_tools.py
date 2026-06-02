"""
tests/test_tools.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

Synchronized test framework validating tool schema firewalls, routing logic,
circuit breaker interventions, and downstream stub data integrity.
"""
import unittest
from tools.tool_registry import ToolRegistry

# Import the actual tool functions to test them directly
from tools.sec_edgar import sec_filing_search
from tools.financial_api import financial_data_api
from tools.calculator import calculation_engine


class TestToolRegistryAndImplementations(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()

    # =========================================================================
    # PART 1: Tool Registry Core Infrastructure Tests
    # =========================================================================

    def test_registry_validation_pass(self):
        """Verifies valid parameters cleanly clear the JSON schema validation pass."""
        payload = {"ticker": "TSLA",
                   "filing_type": "10-K", "section": "Item 7"}
        # FIX: Changed .dispatch() to .execute_with_fallback()
        response = self.registry.execute_with_fallback(
            "sec_filing_search", payload)
        self.assertEqual(response["status"], "success")

    def test_registry_validation_blocks_malformed_input(self):
        """Ensures invalid parameters are caught by the firewall before execution."""
        # Missing required parameter: 'ticker'
        invalid_payload = {"filing_type": "10-K"}
        # FIX: Changed .dispatch() to .execute_with_fallback()
        response = self.registry.execute_with_fallback(
            "sec_filing_search", invalid_payload)

        # FIX: Synced status string and error message matching tool_registry.py
        self.assertEqual(response["status"], "failed")
        self.assertIn("Schema validation failed", response["error"])

    def test_registry_fallback_automatic_routing(self):
        """Forces an execution error to verify automatic diversion down the fallback chain."""
        # FIX: Changed internal state manipulation target from .execution_pool to .tools
        if "sec_filing_search" in self.registry.tools:
            del self.registry.tools["sec_filing_search"]

        # FIX: Kept payload clean and matching schema fields so validation passes, forcing an execution fault instead
        payload = {"ticker": "AAPL",
                   "filing_type": "10-K", "section": "Item 1A"}
        response = self.registry.execute_with_fallback(
            "sec_filing_search", payload)

        # Confirms it automatically fell back according to fallback chains setup
        self.assertEqual(response["status"], "recovered_via_fallback")
        # Checks if it hits the designated target inside FALLBACK_CHAINS
        self.assertTrue(response["source"] in [
                        "web_search", "vector_db_search"])

    def test_registry_circuit_breaker_tripping(self):
        """Day 3 Test: Confirms consecutive faults trip circuit breaker to OPEN."""
        # Force the tool to throw a definitive exception on invocation
        def broken_mock_tool(*args, **kwargs):
            raise RuntimeError("Network timeout simulation")

        self.registry.tools["financial_api"] = broken_mock_tool
        payload = {"ticker": "AAPL",
                   "statement_type": "income_statement", "period": "annual"}

        # Fire 3 consecutive requests to break threshold limits
        self.registry.execute_with_fallback("financial_api", payload)
        self.registry.execute_with_fallback("financial_api", payload)
        response_three = self.registry.execute_with_fallback(
            "financial_api", payload)

        # Inspect state tracking properties
        cb_state = self.registry.circuit_breakers["financial_api"]["state"]
        self.assertEqual(cb_state, "OPEN")

    # =========================================================================
    # PART 2: Explicit Tool Implementation Tests
    # =========================================================================

    def test_tool_implementation_sec_filing_search(self):
        """Test Case 1: Verifies data structure and keys returned by sec_filing_search stub."""
        result = sec_filing_search(
            ticker="MSFT", filing_type="10-K", section="Item 1A")

        self.assertIsInstance(result, dict)
        self.assertEqual(result["ticker"], "MSFT")
        self.assertEqual(result["filing_type"], "10-K")
        self.assertEqual(result["section_retrieved"], "Item 1A")
        self.assertIn("content", result)
        self.assertEqual(result["source_tier"], 1)

    def test_tool_implementation_financial_data_api(self):
        """Test Case 2: Verifies data structure and keys returned by financial_data_api stub."""
        result = financial_data_api(
            ticker="AAPL", statement_type="income_statement", period="annual")

        self.assertIsInstance(result, dict)
        self.assertEqual(result["ticker"], "AAPL")
        self.assertEqual(result["statement_type"], "income_statement")
        self.assertIsInstance(result["data"], list)
        self.assertTrue(len(result["data"]) > 0)

        first_row = result["data"][0]
        self.assertIn("revenue", first_row)
        self.assertIn("net_income", first_row)
        self.assertIn("eps", first_row)

    def test_tool_implementation_calculation_engine(self):
        """Test Case 3: Verifies logic and keys returned by calculation_engine stub."""
        mock_inputs = {"base_fcf": 5000.0, "growth_rate": 0.05, "wacc": 0.08}
        result = calculation_engine(
            calculation_type="dcf_valuation", inputs=mock_inputs, show_steps=True)

        self.assertIsInstance(result, dict)
        self.assertEqual(result["calculation_type"], "dcf_valuation")
        self.assertIn("intrinsic_value", result["computed_outputs"])
        self.assertIn("safety_margin", result["computed_outputs"])
        self.assertTrue(len(result["audit_trail"]) > 0)
        self.assertEqual(result["execution"], "local_deterministic")


if __name__ == "__main__":
    unittest.main()
