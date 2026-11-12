# ARA-1: Autonomous Financial Research Agent
## Project1A-EyobKebede-AutonomousFinancialResearchAgent

> 15-Day Agentic AI Project | **Day 9 of 15 in progress** | Version 0.9

---

## What This Project Is

ARA-1 is a purpose-built agentic AI system that replicates the core workflow of a junior financial analyst: finding data from multiple sources, reconciling conflicts, and producing structured research reports. A junior analyst takes 2-5 days to produce a comprehensive report. ARA-1 targets sub-5-minute delivery.

**ARA-1 is not a chatbot.** It does not answer single-turn prompts, make investment recommendations, or fabricate data when sources are unavailable. It degrades gracefully and says so in the output.

---

## Current Status

**Legend:** ✅ built and validated | 🟡 built, not yet wired or fully tested | ⏳ not started | 👤 built outside the assistant sessions, not re-reviewed

### Where we are right now

Day 9 (error handling) is partly done. Retry/degradation and the fallback-chain executor are built and tested in isolation. The circuit breaker is next, followed by wiring all of it into `agent/core.py`. After that comes Challenge 6.

### Progress by day

| Day | Scope | Status |
|-----|-------|--------|
| 1 | Architecture spec, error log, system diagram | ✅ |
| 2 | 12 tool schemas, repo scaffold | ✅ |
| 3 | Environment setup, LLM integration, ChromaDB init | ✅ |
| 4 | Core agent loop (`prompts.py`, `parser.py`, `core.py`) | ✅ Challenge 1 validated with mock data |
| 5 | Real API tools: SEC EDGAR, financial data, web search, news sentiment, company profile | ✅ Challenges 1-2 pass in **mock mode only**; live-API runs not done |
| 6 | Memory: vector store, context manager, episodic memory | ✅ 39 tests in `tests/test_memory.py` |
| 7 | Remaining tools: earnings, peer comparison, calculator, fact checker, report generator | ✅ per owner. Tools not all wired into `TOOL_DISPATCH`; Challenges 3-4 results not recorded |
| 8 | Synthesis: conflict resolver, narrative, engine | ✅ engine wired into `node_synthesize` |
| 9 | Error handling | 🟡 in progress, see below |
| 10 | Query analysis and disambiguation, Challenge 7 | ⏳ |
| 11+ | Evaluation dashboard, remaining challenges, stress testing, documentation (Day 14), final submission | ⏳ `evaluation/metrics.py` was built ahead of schedule |

### Day 9 breakdown

| Deliverable | Status |
|-------------|--------|
| `agent/error_handler.py`: retry with exponential backoff, `[DATA UNAVAILABLE]` degradation, failure injector | ✅ |
| `agent/fallback_chains.py`: chain executor, parameter translation, usable-data check, disclosure notes | ✅ not yet wired into `core.py` |
| `agent/circuit_breaker.py` | ⏳ next. Interface it must provide: `allow(tool)`, `record_success(tool)`, `record_failure(tool)` |
| Wire error handling + fallbacks + breaker into `core.py` | ⏳ |
| `results/challenge_6.md` (ambiguous query handling) | ⏳ |
| Failure injection against the full agent | ⏳ (done against the chain executor only: at a 50% injected failure rate, retries alone recovered 89.7% of calls and retries plus fallbacks recovered 100%, over 300 seeded runs) |

### Validation evidence

| Suite | Result |
|-------|--------|
| `tests/test_memory.py` | 39 tests pass |
| `tests/test_challenges_1_2.py` | Challenges 1-2 pass in mock mode |
| `tests/test_day4_challenge1.py` | Day 4 validation passes |
| Synthesis, evaluation metrics, error handling, fallback chains | Validated by inline runs during development (conflict resolver, narrative, engine, 26 metrics, 8 error-handler checks, 14 fallback-chain checks). **Not yet saved as test files.** |

Test files are script-style runners that stub external packages (`chromadb`, `anthropic`, `langgraph`) at import time. Run them one at a time, not together under a single `pytest` session.

### Known open items

1. **`.env.example` key mismatch:** it lists `ALPHA_VANTAGE_API_KEY`, but `tools/financial_api.py` reads `ALPHA_VANTAGE_KEY`. Until one is renamed, the Alpha Vantage fallback is silently disabled.
2. **Tools not yet in `TOOL_DISPATCH`:** `peer_comparison`, `calculation_engine`, `fact_checker`, `report_generator`. Wiring is deferred until all tools are done.
3. **No automatic memory write:** the graph has no step that stores verified findings in long-term memory at session end. `vector_db_store` is only used if the plan calls it, so cross-session cache hits depend on that. Challenge 7 (memory-driven research) needs this.
4. **Fallback results must be filed under the tool that produced them** (use `gathered_key()` from `fallback_chains.py`). Synthesis parses each record by the tool name in its key.
5. **`node_evaluate` and `node_generate_report` are still basic.** `evaluation/metrics.py` (26 metrics) and `tools/report_gen.py` are not yet connected to them.
6. **`fact_checker` (👤):** confirm it calls `financial_data_api(..., limit=1)` (not `years=`) and returns a `verified` boolean. `vector_db_store` depends on that flag.
7. **Live-API runs:** Challenges 1-2 have only been run with mock data.
8. **Embedding model:** without an OpenAI key, `vector_store.py` falls back to a local 384-dimension model. Do not mix embedding models in one ChromaDB collection.

---

## Architecture at a Glance

```
User Query
    │
    ▼
analyze_query              ← heuristic today; query_analyzer + disambiguation arrive Day 10
    │
    ▼
memory_precheck            ← vector_db_search per company; hits (>= 0.85 similarity) load into gathered_data
    │
    ▼
plan                       ← Planner LLM, sees cache summary + episodic memory of past sessions
    │
    ▼
execute_step  ⟲            ← ReAct loop per plan step (max 20 tool calls, 15 iterations)
    │                         ContextManager compresses old traces at 70% of the context window
    ├── Tool registry (12 tools)
    └── Error handling: retry → fallback chain → circuit breaker → degrade   (built, wiring pending)
    │
    ▼
synthesize                 ← synthesis engine: conflict resolution, narrative threading,
    │                         quantitative triangulation, sentiment-fact alignment
    ▼
verify → generate_report → evaluate
    │
    ▼
episodic.log_session       ← strategy log for future planners
```

**Key principles:** Memory First (check long-term memory before any external API call), source reliability weighting, and no fabrication (missing data is marked `[DATA UNAVAILABLE]` and disclosed).

---

## Repository Structure

```
Project1A-EyobKebede-AutonomousFinancialResearchAgent/
│
├── README.md
├── ERROR_LOG.md                       ✅ 7 deliberate errors identified (Day 1)
├── architecture_specification.md      ✅ moves to docs/…_final.md on Day 14
├── requirements.txt, .env.example
│
├── agent/
│   ├── core.py                        ✅ LangGraph StateGraph, ResearchState, tool dispatch
│   ├── prompts.py                     ✅ planner, executor, verifier, disambiguator prompts
│   ├── parser.py                      ✅ plan / ReAct / verification parsing
│   ├── error_handler.py               ✅ retry, backoff, degradation, failure injector
│   ├── fallback_chains.py             ✅ fallback chain executor (not wired yet)
│   ├── circuit_breaker.py             ⏳
│   ├── query_analyzer.py              ⏳ Day 10
│   └── disambiguation.py              ⏳ Day 10
│
├── tools/
│   ├── schemas/                       ✅ 12 LLM-facing schemas, ALL_SCHEMAS, FALLBACK_CHAINS
│   ├── sec_edgar.py                   ✅ filing metadata + full filing text
│   ├── financial_api.py               ✅ FMP primary, Alpha Vantage fallback
│   ├── web_search.py                  ✅ Tavily primary, SerpAPI fallback
│   ├── news_sentiment.py              ✅ NewsAPI + VADER / TextBlob / keyword scoring
│   ├── company_profile.py             ✅ FMP /profile, /quote fallback
│   ├── earnings.py                    ✅ FMP transcripts
│   ├── peer_comparison.py             ✅ FMP peers + metric matrix
│   ├── calculator.py                  ✅ 9 calculation types, local
│   ├── fact_checker.py                👤
│   ├── report_gen.py                  👤
│   └── tool_registry.py               👤
│
├── memory/
│   ├── vector_store.py                ✅ ChromaDB, cosine similarity, metadata filtering
│   ├── vector_db_tools.py             ✅ vector_db_search / vector_db_store tools
│   ├── context_manager.py             ✅ token budget, summarisation, chunking strategies
│   └── episodic.py                    ✅ JSONL + SQLite strategy log
│
├── synthesis/
│   ├── conflict_resolver.py           ✅ 6-step conflict resolution protocol
│   ├── narrative.py                   ✅ threading, triangulation, sentiment-fact alignment
│   └── engine.py                      ✅ orchestrator, wired into node_synthesize
│
├── evaluation/
│   ├── metrics.py                     ✅ 26 metrics across 5 categories (not yet wired to node_evaluate)
│   └── dashboard.py                   ⏳
│
├── config/                            settings and logging (Day 3)
├── tests/                             test_memory.py, test_challenges_1_2.py, test_day4_challenge1.py
├── results/                           challenge_1.md, challenge_2.md (mock mode); 3-8 pending
└── docs/                              ⏳ Day 14
```

---

## Technology Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| Language | Python 3.11+ | |
| Agent framework | LangGraph | Explicit `StateGraph` over `ResearchState` |
| LLM | Anthropic Claude Sonnet 4 | `core.py` can route to Groq (Llama 3.1 8B Instant) first when a Groq key is configured |
| Vector DB | ChromaDB (local, dev) | Pinecone planned for production |
| Embeddings | `text-embedding-3-small` (1536 dims) | Falls back to local `all-MiniLM-L6-v2` (384 dims) without an OpenAI key |
| Retry | stdlib exponential backoff | No tenacity dependency in the retry path |
| Episodic store | JSONL + SQLite (WAL) | `logs/episodic/` |

---

## Setup

### Prerequisites

- Python 3.11+
- API keys: Anthropic (or Groq), Financial Modeling Prep, Tavily, NewsAPI. Optional: OpenAI (embeddings), SerpAPI and Alpha Vantage (fallbacks).

### Installation

```bash
git clone <your-repo-url>
cd Project1A-EyobKebede-AutonomousFinancialResearchAgent

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env            # then add your keys
```

Key names the code actually reads: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `FMP_API_KEY`, `TAVILY_API_KEY`, `SERPAPI_KEY`, `NEWSAPI_KEY`, `ALPHA_VANTAGE_KEY`.

> Free tiers are enough for development: FMP 250 req/day, Tavily 1000/month, NewsAPI 100/day. SEC EDGAR is free with no key.

### Run the tests

Run each file on its own:

```bash
python tests/test_memory.py
python tests/test_challenges_1_2.py            # mock mode without keys
source .env && python tests/test_challenges_1_2.py   # live mode (FMP + Anthropic keys present)
python tests/test_day4_challenge1.py
```

---

## The 12 Tools

| Tool | Source | Tier | Built | In `TOOL_DISPATCH` |
|------|--------|------|-------|--------------------|
| `sec_filing_search` | SEC EDGAR | 1 | ✅ | ✅ |
| `financial_data_api` | FMP, Alpha Vantage | 2 | ✅ | ✅ |
| `company_profile` | FMP | 2 | ✅ | ✅ |
| `peer_comparison` | FMP | 2 | ✅ | ⏳ |
| `earnings_transcript` | FMP | 3 | ✅ | ✅ |
| `web_search` | Tavily, SerpAPI | 4 | ✅ | ✅ |
| `news_sentiment` | NewsAPI + local NLP | 4 | ✅ | ✅ |
| `calculation_engine` | local | n/a | ✅ | ⏳ |
| `fact_checker` | multi-source | composite | 👤 | ⏳ |
| `report_generator` | local template | n/a | 👤 | ⏳ |
| `vector_db_search` | ChromaDB | n/a | ✅ | ✅ |
| `vector_db_store` | ChromaDB | n/a | ✅ | ✅ |

### Fallback chains (defined in `tools/schemas/__init__.py`)

| Primary | Fallbacks, in order |
|---------|---------------------|
| `sec_filing_search` | `vector_db_search`, `web_search` |
| `financial_data_api` | `sec_filing_search`, `vector_db_search` |
| `company_profile` | `financial_data_api`, `web_search`, `vector_db_search` |
| `earnings_transcript` | `web_search`, `vector_db_search` |
| `web_search` | `news_sentiment`, `vector_db_search` |
| `news_sentiment` | `web_search`, `vector_db_search` |
| `peer_comparison` | `web_search`, `financial_data_api` |

---

## Source Reliability Hierarchy

| Tier | Weight | Source | Used for |
|------|--------|--------|----------|
| 1 | 1.00 | SEC filings (10-K, 10-Q, 8-K) | Ground truth for numerical data |
| 2 | 0.90 | Financial data APIs | Structured metrics, historical series |
| 3 | 0.75 | Earnings call transcripts | Forward guidance, management tone |
| 4 | 0.60 | Major news (Reuters, Bloomberg, FT) | Recent events, market context |
| 5 | 0.20 | Social media, forums | Sentiment signals only, never factual claims |

> Tiers 4 and 5 are corrected from the project brief, which inverted them. See `ERROR_LOG.md`, Error 001.

---

## Error Handling and Degradation

- **Retry:** network and rate-limit errors retry with exponential backoff (rate limits use longer delays). Authentication and missing-key errors are never retried.
- **Fallbacks:** each external tool has at least two fallbacks. A fallback tool's parameters are translated from the primary's, and empty results count as failures.
- **Degradation:** if every tool in a chain fails, the section is marked `[DATA UNAVAILABLE]`, the other sections still complete, and the failure is disclosed in the report. Data is never invented to fill a gap.
- **Disclosure:** fallback use is reported with the substitute's tier and its limitations (for example, memory-sourced data may be stale).

---

## Design Constraints

| Constraint | Target |
|-----------|--------|
| Max tool calls per task | 20 |
| Max agent iterations | 15 |
| Max ReAct cycles per plan step | 3 |
| Target latency | under 5 min |
| Hallucination count | 0 |
| Tool efficiency | at least 70% |
| Error recovery rate | at least 90% |

---

## Error Detection Log

Seven deliberate factual errors were embedded in the project brief. All seven were identified and documented in `ERROR_LOG.md`:

| # | Location | Error | Severity |
|---|----------|-------|----------|
| 001 | A6.2 Source Hierarchy | Tiers 4 and 5 inverted | High |
| 002 | A5.2 Metric AB-4 | Multiply instead of divide | High |
| 003 | A7.3 SCAP | Wrong year (2007) and wrong act | Medium |
| 004 | C4.2 Indian Filing | Form 20-F is SEC, not MCA | Medium |
| 005 | C3.2 Hallucination % | 45-60% unsupported or implausible | Medium |
| 006 | E2.2 Embedding dims | `text-embedding-3-large` is not 1024 dims | Medium |
| 007 | B3.1 Scoring dims | "Five dimensions" but the table has four | Low |

---

## Commit Convention

```
Day-NN: Brief description - EyobKebede
```

Example: `Day-09: Error handler and fallback chains - EyobKebede`

---

## Implementation Notes

1. **AB-4 Memory Utilization** is `memory_hits / total_api_calls` (division, not multiplication). See `ERROR_LOG.md`, Error 002.
2. **`gathered_data` stores each tool's raw output with no wrapper.** Keys are `"{tool_name}_step{step_id}"`. `web_search` and `news_sentiment` carry no ticker, so synthesis matches them to a company through their `query` text.
3. **`metadata_filter`** (not `filter`) is the vector search filter argument, to avoid shadowing the Python builtin.
4. **`ContextManager` is created per `run_agent` call**, never shared across sessions.
5. **Graph order:** `analyze_query → memory_precheck → plan → execute_step`. Memory is checked before planning so the planner knows what is already cached.
6. **No investment recommendations.** The system prompts prohibit buy/sell/hold language. ARA-1 is an analysis tool only.

---

*ARA-1 Autonomous Financial Research Agent | Version 0.9 (Day 9 in progress) | Author: Eyob Kebede*
