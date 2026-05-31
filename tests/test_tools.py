"""
tests/test_tools.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent
"""
import unittest
from tools.tool_registry import ToolRegistry

# Import the actual tool functions to test them directly
from tools.sec_filing_search import sec_filing_search
from tools.financial_data_api import financial_data_api
from tools.calculation_engine import calculation_engine

class TestToolRegistryAndImplementations(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()

    # =========================================================================
    # PART 1: Tool Registry Core Infrastructure Tests
    # =========================================================================

    def test_registry_validation_pass(self):
        """Verifies valid parameters cleanly clear the JSON schema validation pass."""
        payload = {"ticker": "TSLA", "filing_type": "10-K", "section": "Item 7"}
        response = self.registry.dispatch("sec_filing_search", payload)
        self.assertEqual(response["status"], "success")

    def test_registry_validation_blocks_malformed_input(self):
        """Ensures invalid parameters are caught by the firewall before execution."""
        # Missing required parameter: 'ticker'
        invalid_payload = {"filing_type": "10-K"}
        response = self.registry.dispatch("sec_filing_search", invalid_payload)
        self.assertEqual(response["status"], "validation_failure")
        self.assertIn("validation barrier breached", response["error"])

    def test_registry_fallback_automatic_routing(self):
        """Forces an execution error to verify automatic diversion down the fallback chain."""
        # Intentionally break the primary tool reference to simulate a tool failure
        self.registry.execution_pool["sec_filing_search"] = None 
        
        payload = {"ticker": "AAPL", "filing_type": "10-K", "query": "cache_trigger search values"}
        response = self.registry.dispatch("sec_filing_search", payload)
        
        # Confirms it automatically fell back to vector_db_search according to fallback chains
        self.assertEqual(response["status"], "recovered_via_fallback")
        self.assertEqual(response["source"], "vector_db_search")

    # =========================================================================
    # PART 2: Explicit Tool Implementation Tests (Satisfying Objective 5)
    # =========================================================================

    def test_tool_implementation_sec_filing_search(self):
        """Test Case 1: Verifies data structure and keys returned by sec_filing_search stub."""
        result = sec_filing_search(ticker="MSFT", filing_type="10-K", section="Item 1A")
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["ticker"], "MSFT")
        self.assertEqual(result["filing_type"], "10-K")
        self.assertEqual(result["section_retrieved"], "Item 1A")
        self.assertIn("content", result)
        self.assertEqual(result["source_tier"], 1)

    def test_tool_implementation_financial_data_api(self):
        """Test Case 2: Verifies data structure and keys returned by financial_data_api stub."""
        result = financial_data_api(ticker="AAPL", statement_type="income_statement", period="annual")
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["ticker"], "AAPL")
        self.assertEqual(result["statement_type"], "income_statement")
        self.assertIsInstance(result["data"], list)
        self.assertTrue(len(result["data"]) > 0)
        # Check that essential financial line-items are present in the mock data rows
        first_row = result["data"][0]
        self.assertIn("revenue", first_row)
        self.assertIn("net_income", first_row)
        self.assertIn("eps", first_row)

    def test_tool_implementation_calculation_engine(self):
        """Test Case 3: Verifies logic and keys returned by calculation_engine stub."""
        mock_inputs = {"base_fcf": 5000.0, "growth_rate": 0.05, "wacc": 0.08}
        result = calculation_engine(calculation_type="dcf_valuation", inputs=mock_inputs, show_steps=True)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["calculation_type"], "dcf_valuation")
        self.assertIn("intrinsic_value", result["computed_outputs"])
        self.assertIn("safety_margin", result["computed_outputs"])
        self.assertTrue(len(result["audit_trail"]) > 0)
        self.assertEqual(result["execution"], "local_deterministic")

if __name__ == "__main__":
    unittest.main()