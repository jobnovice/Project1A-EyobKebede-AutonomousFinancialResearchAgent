import unittest

from agent.circuit_breaker import CircuitBreaker
from agent.error_handler import _classify_error


class TestCircuitBreaker(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.breaker = CircuitBreaker(
            failure_threshold=3,
            reset_timeout_s=10,
            failure_window_s=60,
            clock=lambda: self.now,
        )

    def test_non_outage_probe_failures_close_circuit(self):
        self.breaker.record_failure("web_search")
        self.breaker.record_failure("web_search")
        self.breaker.record_failure("web_search")
        self.now = 10.0
        self.assertTrue(self.breaker.allow("web_search"))

        self.breaker.record_failure("web_search", "data")
        self.assertEqual(self.breaker.get_state("web_search"), "CLOSED")

    def test_half_open_probe_timeout_reopens_circuit(self):
        self.breaker.record_failure("web_search")
        self.breaker.record_failure("web_search")
        self.breaker.record_failure("web_search")
        self.now = 10.0
        self.assertTrue(self.breaker.allow("web_search"))

        self.now = 20.0
        self.assertFalse(self.breaker.allow("web_search"))
        self.assertEqual(self.breaker.get_state("web_search"), "OPEN")

    def test_opens_after_threshold_and_blocks_calls(self):
        for _ in range(2):
            self.breaker.record_failure("web_search")
            self.assertTrue(self.breaker.allow("web_search"))

        self.breaker.record_failure("web_search")

        self.assertEqual(self.breaker.get_state("web_search"), "OPEN")
        self.assertFalse(self.breaker.allow("web_search"))

    def test_old_failures_expire_from_window(self):
        self.breaker.record_failure("web_search")
        self.breaker.record_failure("web_search")
        self.now = 61.0

        self.breaker.record_failure("web_search")

        self.assertEqual(self.breaker.get_state("web_search"), "CLOSED")
        self.assertTrue(self.breaker.allow("web_search"))

    def test_allows_only_one_half_open_probe(self):
        for _ in range(3):
            self.breaker.record_failure("web_search")
        self.now = 10.0

        self.assertTrue(self.breaker.allow("web_search"))
        self.assertFalse(self.breaker.allow("web_search"))
        self.assertEqual(self.breaker.get_state("web_search"), "HALF-OPEN")

    def test_successful_probe_closes_and_resets_circuit(self):
        for _ in range(3):
            self.breaker.record_failure("web_search")
        self.now = 10.0
        self.assertTrue(self.breaker.allow("web_search"))

        self.breaker.record_success("web_search")

        self.assertEqual(self.breaker.get_state("web_search"), "CLOSED")
        self.assertTrue(self.breaker.allow("web_search"))

    def test_failed_probe_reopens_circuit(self):
        for _ in range(3):
            self.breaker.record_failure("web_search")
        self.now = 10.0
        self.assertTrue(self.breaker.allow("web_search"))

        self.breaker.record_failure("web_search")

        self.assertEqual(self.breaker.get_state("web_search"), "OPEN")
        self.assertFalse(self.breaker.allow("web_search"))

    def test_tools_have_independent_circuits(self):
        for _ in range(3):
            self.breaker.record_failure("web_search")

        self.assertEqual(self.breaker.get_state("web_search"), "OPEN")
        self.assertEqual(self.breaker.get_state("news_sentiment"), "CLOSED")
        self.assertTrue(self.breaker.allow("news_sentiment"))

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            CircuitBreaker(failure_threshold=0)
        with self.assertRaises(ValueError):
            CircuitBreaker(reset_timeout_s=-1)

    def test_error_classification_handles_http_and_invalid_params(self):
        self.assertEqual(_classify_error(TypeError("unexpected keyword argument 'foo'")), ("invalid_params", False))
        self.assertEqual(_classify_error(RuntimeError("HTTP 404 Not Found: no transcript for AAPL")), ("data", False))
        self.assertEqual(_classify_error(RuntimeError("HTTP 503 Service Unavailable")), ("network", True))


if __name__ == "__main__":
    unittest.main()
