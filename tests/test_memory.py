"""
tests/test_memory.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

Day 6 — Memory system test suite.

Covers all four Day 6 deliverables:
  1. VectorStore          — store, search, empty-collection guard, delete, stats
  2. vector_db_tools      — vector_db_search / vector_db_store as tool functions
  3. ContextManager       — token counting, summarisation trigger, chunking strategies
  4. EpisodicMemory       — log session, planner queries, SQLite + JSON persistence

Integration test (Day 6 Task 5):
  Store Challenge 1 (Microsoft) results → retrieve via vector_db_search
  → verify cache hit for a related query (simulating Challenge 7 retrieval).

Run from repo root:
  python tests/test_memory.py
"""

import json
import logging
import shutil
import sys
import types
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")

# ---------------------------------------------------------------------------
# Stub chromadb before any memory imports
# ---------------------------------------------------------------------------

def _install_chromadb_stub():
    """Install an in-memory chromadb stub for unit tests."""
    chroma       = types.ModuleType("chromadb")
    chroma_cfg   = types.ModuleType("chromadb.config")
    chroma_utils = types.ModuleType("chromadb.utils")
    chroma_ef    = types.ModuleType("chromadb.utils.embedding_functions")

    chroma_cfg.Settings = type("Settings", (), {"__init__": lambda s, **k: None})

    class _FakeCollection:
        """In-memory collection that mimics ChromaDB behaviour."""
        def __init__(self):
            self._docs:   dict[str, str]  = {}
            self._metas:  dict[str, dict] = {}
            self._embeds: dict[str, list] = {}

        def count(self):
            return len(self._docs)

        def upsert(self, ids, documents, metadatas, embeddings=None):
            for i, doc_id in enumerate(ids):
                self._docs[doc_id]   = documents[i]
                self._metas[doc_id]  = metadatas[i]

        def query(self, query_texts, n_results, where=None, include=None):
            # Return all docs (simple stub — no real similarity)
            filtered_ids = list(self._docs.keys())

            # Apply simple where filter if present
            if where:
                filtered_ids = [
                    doc_id for doc_id in filtered_ids
                    if _matches_where(self._metas[doc_id], where)
                ]

            result_ids   = filtered_ids[:n_results]
            # Return distance 0.1 (similarity ≈ 0.95) for all hits
            return {
                "ids":       [result_ids],
                "documents": [[self._docs[i] for i in result_ids]],
                "metadatas": [[self._metas[i] for i in result_ids]],
                "distances": [[0.1] * len(result_ids)],
            }

        def delete(self, ids):
            for doc_id in ids:
                self._docs.pop(doc_id, None)
                self._metas.pop(doc_id, None)

    class _FakeClient:
        def __init__(self):
            self._collections: dict[str, _FakeCollection] = {}  # instance-level

        def get_or_create_collection(self, name, embedding_function=None, metadata=None):
            if name not in self._collections:
                self._collections[name] = _FakeCollection()
            return self._collections[name]

    chroma.PersistentClient = lambda path, settings: _FakeClient()

    for mod, obj in [
        ("chromadb", chroma),
        ("chromadb.config", chroma_cfg),
        ("chromadb.utils", chroma_utils),
        ("chromadb.utils.embedding_functions", chroma_ef),
    ]:
        sys.modules[mod] = obj


def _matches_where(meta: dict, where: dict) -> bool:
    """Simple $where evaluator for the stub."""
    if "$and" in where:
        return all(_matches_where(meta, c) for c in where["$and"])
    for field, condition in where.items():
        if isinstance(condition, dict):
            op, val = next(iter(condition.items()))
            mv = meta.get(field)
            if op == "$eq"  and mv != val:   return False
            if op == "$gte" and (mv is None or mv < val):  return False
            if op == "$lte" and (mv is None or mv > val):  return False
        else:
            if meta.get(field) != condition: return False
    return True


_install_chromadb_stub()

# Clean test artefact directories
for _dir in ["logs/episodic", "logs/traces"]:
    shutil.rmtree(_dir, ignore_errors=True)

# Now safe to import memory modules
from memory.vector_store    import VectorStore, SearchResult
from memory.vector_db_tools import vector_db_search, vector_db_store
from memory.context_manager import (
    ContextManager, estimate_tokens, estimate_dict_tokens,
    CONTEXT_WINDOW, SUMMARISE_THRESHOLD, HEADROOM_TOKENS,
)
from memory.episodic import EpisodicMemory, _compute_tool_performance


# ---------------------------------------------------------------------------
# Helper: sample metadata
# ---------------------------------------------------------------------------
def _meta(ticker="MSFT", source_type="10-K", date="2024-06-30",
          verified=True, session_id="session-test-001"):
    return {
        "ticker":       ticker,
        "source_type":  source_type,
        "date":         date,
        "verified":     verified,
        "session_id":   session_id,
        "chunk_index":  0,
        "confidence":   0.95,
    }


# ===========================================================================
# 1. VectorStore
# ===========================================================================

class TestVectorStore(unittest.TestCase):

    def setUp(self):
        self.store = VectorStore()
        # Clear between tests by replacing the collection
        self.store._collection = (
            self.store._client.get_or_create_collection(
                name=f"test_{id(self)}", metadata={"hnsw:space": "cosine"}
            )
        )

    def test_empty_collection_returns_no_results(self):
        """Empty collection must not crash — returns []."""
        results = self.store.search("Microsoft revenue")
        self.assertEqual(results, [])

    def test_store_and_retrieve(self):
        """Store a chunk then retrieve it."""
        doc_id = self.store.store(
            content="Microsoft FY2024 revenue was $245.1 billion, up 16% YoY.",
            metadata=_meta(),
        )
        self.assertIsNotNone(doc_id)
        self.assertEqual(self.store.count(), 1)

        results = self.store.search("Microsoft revenue 2024")
        self.assertEqual(len(results), 1)
        self.assertIn("245.1 billion", results[0].content)

    def test_similarity_conversion(self):
        """ChromaDB distance 0.1 → similarity ≈ 0.95."""
        self.store.store("Apple Q4 2024 earnings beat estimates.", metadata=_meta("AAPL", "news"))
        results = self.store.search("Apple earnings", min_similarity=0.0)
        self.assertGreater(results[0].similarity, 0.9)

    def test_upsert_deduplicates(self):
        """Upserting same doc_id replaces the existing record."""
        self.store.store("Old content.", metadata=_meta(), document_id="msft-001")
        self.store.store("Updated content.", metadata=_meta(), document_id="msft-001")
        self.assertEqual(self.store.count(), 1)
        results = self.store.search("content")
        self.assertIn("Updated", results[0].content)

    def test_metadata_filter_by_ticker(self):
        """Filter should return only results matching the ticker."""
        self.store.store("MSFT data.", metadata=_meta("MSFT"))
        self.store.store("AAPL data.", metadata=_meta("AAPL"))
        results = self.store.search("data", metadata_filter={"ticker": "AAPL"})
        self.assertTrue(all(r.metadata["ticker"] == "AAPL" for r in results))

    def test_delete(self):
        """Delete removes a document."""
        doc_id = self.store.store("Temporary content.", metadata=_meta())
        self.assertEqual(self.store.count(), 1)
        self.store.delete(doc_id)
        self.assertEqual(self.store.count(), 0)

    def test_stats_schema(self):
        """Stats dict has required keys."""
        stats = self.store.stats()
        self.assertIn("total_chunks", stats)
        self.assertIn("provider", stats)

    def test_sanitise_none_metadata(self):
        """None values in metadata must not crash ChromaDB upsert."""
        meta = _meta()
        meta["section"] = None    # None is not allowed by ChromaDB
        doc_id = self.store.store("Content with None metadata.", metadata=meta)
        self.assertIsNotNone(doc_id)


# ===========================================================================
# 2. vector_db_tools
# ===========================================================================

class TestVectorDbTools(unittest.TestCase):

    def setUp(self):
        # Reset the singleton so each test gets a fresh store
        import memory.vector_store as vs_module
        vs_module._store = None

    def test_search_returns_schema_on_empty(self):
        """vector_db_search on empty store returns correct schema, no crash."""
        result = vector_db_search("Microsoft revenue")
        self.assertIn("cache_hit", result)
        self.assertIn("results", result)
        self.assertIn("error", result)
        self.assertFalse(result["cache_hit"])
        self.assertEqual(result["error"], None)

    def test_store_then_search_cache_hit(self):
        """Store a chunk then search — expect cache_hit=True."""
        store_result = vector_db_store(
            content="Microsoft Azure revenue grew 33% in Q1 FY2025.",
            metadata=_meta(),
        )
        self.assertTrue(store_result["stored"])
        self.assertIsNone(store_result["error"])

        search_result = vector_db_search("Microsoft Azure growth", min_similarity=0.0)
        self.assertTrue(search_result["cache_hit"])
        self.assertGreater(search_result["results_count"], 0)

    def test_store_unverified_logs_warning(self):
        """Storing unverified content returns stored=True but flags it."""
        meta = _meta()
        meta["verified"] = False
        result = vector_db_store("Unverified claim.", metadata=meta)
        # Should still store — just logs a warning
        self.assertTrue(result["stored"])

    def test_search_error_key_present(self):
        """error key must always be present in return dict."""
        result = vector_db_search("test query")
        self.assertIn("error", result)

    def test_store_error_key_present(self):
        """error key must always be present in store result."""
        result = vector_db_store("content", metadata=_meta())
        self.assertIn("error", result)

    def test_metadata_filter_passthrough(self):
        """metadata_filter is passed through correctly."""
        vector_db_store("MSFT chunk.", metadata=_meta("MSFT"))
        vector_db_store("AAPL chunk.", metadata=_meta("AAPL"))

        result = vector_db_search(
            "chunk",
            metadata_filter={"ticker": "MSFT"},
            min_similarity=0.0,
        )
        if result["results"]:
            tickers = {r["metadata"]["ticker"] for r in result["results"]}
            self.assertIn("MSFT", tickers)


# ===========================================================================
# 3. ContextManager
# ===========================================================================

class TestContextManager(unittest.TestCase):

    def setUp(self):
        shutil.rmtree("logs/traces", ignore_errors=True)
        self.cm = ContextManager(session_id="ctx-test-001")

    def _base_state(self, **overrides):
        state = {
            "research_plan":  [],
            "tool_call_log":  [],
            "gathered_data":  {},
            "draft_report":   None,
            "session_id":     "ctx-test-001",
        }
        state.update(overrides)
        return state

    def test_estimate_tokens(self):
        self.assertEqual(estimate_tokens("a" * 3800), 1000)
        self.assertGreater(estimate_tokens("hello world"), 0)

    def test_update_returns_usage_dict(self):
        usage = self.cm.update(self._base_state())
        self.assertIn("total", usage)
        self.assertIn("pct_used", usage)
        self.assertGreater(usage["total"], 0)

    def test_should_summarise_false_when_under_threshold(self):
        self.cm.update(self._base_state())
        self.assertFalse(self.cm.should_summarise())

    def test_should_summarise_true_when_over_threshold(self):
        self.cm._total = int(CONTEXT_WINDOW * 0.75)
        self.assertTrue(self.cm.should_summarise())

    def test_has_headroom_true_when_enough_space(self):
        self.cm._total = CONTEXT_WINDOW - HEADROOM_TOKENS - 1000
        self.assertTrue(self.cm.has_headroom())

    def test_has_headroom_false_when_full(self):
        self.cm._total = CONTEXT_WINDOW - HEADROOM_TOKENS + 100
        self.assertFalse(self.cm.has_headroom())

    def test_summarise_skips_when_few_traces(self):
        """Summarise returns unchanged log when traces <= keep_recent."""
        log = [{"step_id": i, "tool": "web_search", "thought": "t",
                "result_summary": "r", "timestamp": "now", "cycle": 1}
               for i in range(3)]
        state = self._base_state(tool_call_log=log)
        result = self.cm.summarise(state, llm_call_fn=lambda s, u: "summary", keep_recent=3)
        self.assertEqual(result, log)

    def test_summarise_compresses_old_traces(self):
        """Summarise replaces old traces with one summary entry."""
        log = [{"step_id": i, "tool": "web_search", "thought": "t",
                "result_summary": f"result {i}", "timestamp": "now", "cycle": 1}
               for i in range(6)]
        state = self._base_state(tool_call_log=log)
        result = self.cm.summarise(
            state,
            llm_call_fn=lambda s, u: "Key findings: revenue 245B, growth 16%.",
            keep_recent=3,
        )
        # 1 summary entry + 3 recent = 4 entries
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0]["tool"], "context_summariser")
        self.assertTrue(result[0].get("is_summary"))

    def test_summarise_archives_to_disk(self):
        """Compressed traces must be archived to the trace log file."""
        log = [{"step_id": i, "tool": "sec_filing_search", "thought": "t",
                "result_summary": f"filing {i}", "timestamp": "now", "cycle": 1}
               for i in range(5)]
        state = self._base_state(tool_call_log=log)
        self.cm.summarise(state, llm_call_fn=lambda s, u: "Summary.", keep_recent=2)
        archive = Path("logs/traces/ctx-test-001-traces.jsonl")
        self.assertTrue(archive.exists())
        lines = archive.read_text().strip().split("\n")
        self.assertEqual(len(lines), 3)   # 5 - 2 = 3 archived

    def test_chunk_text_paragraph(self):
        long_text = "\n\n".join([f"Para {i}: " + "word " * 80 for i in range(8)])
        chunks = self.cm.chunk_text(long_text, source_type="10-K", max_tokens=200)
        self.assertGreater(len(chunks), 1)

    def test_chunk_text_earnings(self):
        transcript = "\n\n".join([
            "Tim Cook: Revenue was 94.9 billion, up 6 percent year over year.",
            "Analyst Q: What drove iPhone growth?",
            "Tim Cook: iPhone 16 launch was strong across all geographies.",
        ])
        chunks = self.cm.chunk_text(transcript, source_type="earnings_call")
        self.assertGreater(len(chunks), 0)

    def test_chunk_text_news_repeats_first_para(self):
        news = "\n\n".join([f"Para {i}: " + "word " * 50 for i in range(4)])
        chunks = self.cm.chunk_text(news, source_type="news", max_tokens=100)
        if len(chunks) > 1:
            first = chunks[0].split("\n\n")[0]
            for chunk in chunks[1:]:
                self.assertTrue(chunk.startswith(first))


# ===========================================================================
# 4. EpisodicMemory
# ===========================================================================

class TestEpisodicMemory(unittest.TestCase):

    def setUp(self):
        shutil.rmtree("logs/episodic", ignore_errors=True)
        # Reset singleton
        import memory.episodic as ep_module
        ep_module._episodic = None
        self.em = EpisodicMemory()

    def _mock_state(self, session_id="s-001", query_type="factual",
                    companies=None, quality=0.90):
        return {
            "session_id":    session_id,
            "query_type":    query_type,
            "companies":     companies or ["MSFT"],
            "raw_query":     "Microsoft company profile",
            "tool_call_log": [
                {"tool": "company_profile",   "result_summary": "Microsoft data",
                 "timestamp": "2024-01-01T00:00:00", "cycle": 1},
                {"tool": "financial_data_api","result_summary": "Revenue 245B",
                 "timestamp": "2024-01-01T00:01:00", "cycle": 1},
                {"tool": "web_search",        "result_summary": "News found",
                 "timestamp": "2024-01-01T00:02:00", "cycle": 1},
                {"tool": "vector_db_search",  "result_summary": "",
                 "timestamp": "2024-01-01T00:03:00", "cycle": 1},
            ],
            "error_log":     ["financial_data_api rate limit — fallback used"],
            "evaluation_metrics": {"overall_confidence": quality,
                                   "AB-5_latency_seconds": 12.0},
            "memory_hits":   0,
            "total_api_calls": 4,
            "hallucination_count": 0,
        }

    def test_fresh_db_has_zero_sessions(self):
        self.assertEqual(self.em.session_count(), 0)

    def test_log_session_increments_count(self):
        self.em.log_session(self._mock_state())
        self.assertEqual(self.em.session_count(), 1)

    def test_log_multiple_sessions(self):
        self.em.log_session(self._mock_state("s-001"))
        self.em.log_session(self._mock_state("s-002"))
        self.assertEqual(self.em.session_count(), 2)

    def test_json_log_written_to_disk(self):
        self.em.log_session(self._mock_state())
        from memory.episodic import JSON_LOG_FILE
        self.assertTrue(JSON_LOG_FILE.exists())
        lines = JSON_LOG_FILE.read_text().strip().split("\n")
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertEqual(record["session_id"], "s-001")

    def test_get_tool_priority_returns_ranked_list(self):
        self.em.log_session(self._mock_state())
        priority = self.em.get_tool_priority("factual", top_n=5)
        self.assertIsInstance(priority, list)
        self.assertGreater(len(priority), 0)
        self.assertIn("tool", priority[0])
        self.assertIn("avg_success_rate", priority[0])
        # company_profile, financial_data_api, web_search should be 1.0
        success_tools = {p["tool"] for p in priority if p["avg_success_rate"] == 1.0}
        self.assertIn("company_profile", success_tools)

    def test_get_tool_priority_empty_for_unknown_type(self):
        result = self.em.get_tool_priority("nonexistent_type")
        self.assertEqual(result, [])

    def test_get_strategy_notes(self):
        self.em.log_session(self._mock_state(quality=0.95))
        notes = self.em.get_strategy_notes("factual")
        self.assertIsInstance(notes, list)

    def test_get_error_patterns(self):
        self.em.log_session(self._mock_state())
        errors = self.em.get_error_patterns("factual")
        self.assertEqual(len(errors), 1)
        self.assertIn("financial_data_api", errors[0])

    def test_get_summary_no_sessions(self):
        summary = self.em.get_summary("analytical")
        self.assertIn("No prior sessions", summary)

    def test_get_summary_with_sessions(self):
        self.em.log_session(self._mock_state())
        summary = self.em.get_summary("factual")
        self.assertIn("EPISODIC MEMORY", summary)
        self.assertIn("factual", summary)
        self.assertIn("success", summary.lower())

    def test_get_recent_sessions_ordering(self):
        self.em.log_session(self._mock_state("s-001"))
        self.em.log_session(self._mock_state("s-002"))
        recent = self.em.get_recent_sessions(limit=5)
        self.assertEqual(recent[0]["session_id"], "s-002")   # most recent first

    def test_compute_tool_performance(self):
        log = [
            {"tool": "web_search", "result_summary": "results found", "timestamp": "t"},
            {"tool": "web_search", "result_summary": "", "timestamp": "t"},
            {"tool": "sec_filing_search", "result_summary": "10-K retrieved", "timestamp": "t"},
        ]
        perf = _compute_tool_performance(log)
        self.assertIn("web_search", perf)
        self.assertEqual(perf["web_search"]["calls"], 2)
        self.assertEqual(perf["web_search"]["successes"], 1)
        self.assertEqual(perf["web_search"]["success_rate"], 0.5)
        self.assertEqual(perf["sec_filing_search"]["success_rate"], 1.0)


# ===========================================================================
# 5. Integration: Store → Retrieve (Challenge 1 → Challenge 7 simulation)
# ===========================================================================

class TestMemoryIntegration(unittest.TestCase):

    def setUp(self):
        # Fresh vector store
        import memory.vector_store as vs_module
        vs_module._store = None

    def test_store_challenge1_then_retrieve(self):
        """
        Simulate Day 6 Task 5:
        Store Challenge 1 (Microsoft) results → run a related query → verify cache hit.
        """
        # Step 1: Store Microsoft research findings (simulating post-Challenge-1 storage)
        chunks = [
            ("Microsoft Corporation (MSFT) is a technology company headquartered in "
             "Redmond, WA. CEO: Satya Nadella. Market cap ~$3.1T. Sectors: Cloud, "
             "Productivity, Personal Computing.",
             _meta("MSFT", "company_profile", "2024-11-01")),

            ("Microsoft FY2024 revenue: $245.1B (+16% YoY). Operating income: $109.4B. "
             "Operating margin: 44.6%. Net income: $88.1B. Free cash flow: $74.1B.",
             _meta("MSFT", "10-K", "2024-06-30")),

            ("Microsoft Azure cloud revenue grew 33% YoY in Q1 FY2025. "
             "Copilot AI now deployed in 600+ enterprise customers. "
             "Management guided for continued double-digit growth.",
             _meta("MSFT", "earnings_call", "2024-10-30")),
        ]

        stored_ids = []
        for content, meta in chunks:
            result = vector_db_store(content=content, metadata=meta)
            self.assertTrue(result["stored"], f"Failed to store: {result['error']}")
            stored_ids.append(result["document_id"])

        self.assertEqual(len(stored_ids), 3)

        # Step 2: Simulate Challenge 7 — a related query that should hit the cache
        search_result = vector_db_search(
            query="What is Microsoft's revenue and market cap?",
            top_k=5,
            min_similarity=0.0,   # low threshold to ensure stub hits register
        )

        self.assertTrue(search_result["cache_hit"],
                        "Expected cache hit after storing Microsoft chunks")
        self.assertGreater(search_result["results_count"], 0)
        self.assertIsNone(search_result["error"])

        # Step 3: Verify the retrieved content contains the right information
        all_content = " ".join(r["content"] for r in search_result["results"])
        self.assertIn("Microsoft", all_content)

        # Step 4: Verify ticker filter works (retrieves MSFT only)
        filtered = vector_db_search(
            query="revenue growth",
            metadata_filter={"ticker": "MSFT"},
            min_similarity=0.0,
        )
        if filtered["results"]:
            tickers = {r["metadata"]["ticker"] for r in filtered["results"]}
            self.assertIn("MSFT", tickers)

        print(f"\n  Integration test: stored {len(stored_ids)} MSFT chunks")
        print(f"  Cache hit: {search_result['cache_hit']} | "
              f"results: {search_result['results_count']} | "
              f"collection size: {search_result['collection_size']}")
        print(f"  Top result snippet: \"{search_result['results'][0]['content'][:80]}...\"")


# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ARA-1 — Day 6 Memory System Test Suite")
    print("="*60)

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    for cls in [
        TestVectorStore,
        TestVectorDbTools,
        TestContextManager,
        TestEpisodicMemory,
        TestMemoryIntegration,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "="*60)
    if result.wasSuccessful():
        print(f"Day 6 Memory Tests — ALL {result.testsRun} TESTS PASSED ✅")
    else:
        print(f"FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
    print("="*60)

    sys.exit(0 if result.wasSuccessful() else 1)
