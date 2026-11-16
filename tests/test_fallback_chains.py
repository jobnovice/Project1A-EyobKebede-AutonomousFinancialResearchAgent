import unittest

from agent.circuit_breaker import CircuitBreaker
from agent.fallback_chains import execute_with_fallbacks


class TestFallbackChainCircuitBreaker(unittest.TestCase):
    def test_empty_half_open_probe_closes_circuit_and_uses_fallback(self):
        now = 0.0
        breaker = CircuitBreaker(
            failure_threshold=1,
            reset_timeout_s=10,
            clock=lambda: now,
        )
        breaker.record_failure("web_search")
        self.assertEqual(breaker.get_state("web_search"), "OPEN")

        now = 10.0
        primary_calls = 0

        def empty_primary(**kwargs):
            nonlocal primary_calls
            del kwargs
            primary_calls += 1
            return {"results": []}

        def news_fallback(**kwargs):
            del kwargs
            return {"articles": [{"title": "News"}]}

        dispatch = {
            "web_search": empty_primary,
            "news_sentiment": news_fallback,
        }

        result = execute_with_fallbacks(
            "web_search",
            {"query": "company news"},
            dispatch,
            breaker=breaker,
            max_retries=0,
            chains={"web_search": ["news_sentiment"]},
        )

        self.assertTrue(result.success)
        self.assertTrue(result.used_fallback)
        self.assertEqual(result.source_tool, "news_sentiment")
        self.assertEqual(primary_calls, 1)
        self.assertEqual(breaker.get_state("web_search"), "CLOSED")

        execute_with_fallbacks(
            "web_search",
            {"query": "company news"},
            dispatch,
            breaker=breaker,
            max_retries=0,
            chains={"web_search": ["news_sentiment"]},
        )
        self.assertEqual(primary_calls, 2)


if __name__ == "__main__":
    unittest.main()
