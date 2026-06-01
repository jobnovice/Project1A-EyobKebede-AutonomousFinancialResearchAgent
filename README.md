# ARA-1: Autonomous Financial Research Agent
## Project1A-EyobKebede-AutonomousFinancialResearchAgent

> **ZeTheta 15-Day Agentic AI Project**  
> Candidate: Eyob Kebede | Day 3 of 15 | Status: Foundation Complete

---

## What This Project Is

ARA-1 is a purpose-built agentic AI system that replicates the core workflow of a junior financial analyst — finding data from multiple sources, reconciling conflicts, and producing structured investment research reports — at a fraction of the cost and time.

A junior analyst costs $80K–$120K/year and takes 2–5 days to produce a comprehensive research report. ARA-1 targets sub-5-minute delivery with comparable output quality.

**ARA-1 is not a chatbot.** It does not respond to single-turn prompts, make investment recommendations, or fabricate data when sources are unavailable. It degrades gracefully and transparently.

---

## Architecture at a Glance

```
User Query
    │
    ▼
Query Analyzer + Disambiguation
    │
    ▼
Planner LLM (Claude Sonnet 4)    ← generates structured research plan
    │
    ▼
Executor (ReAct Micro-Loop)      ← Memory First → Tool → Observe → Adapt
    │
    ├── Tool Registry (12 tools, fallback chains, circuit breakers)
    ├── Memory System (short-term context + long-term vector DB + episodic)
    └── Error Handler (exponential backoff, graceful degradation)
    │
    ▼
Synthesis Engine                 ← source reliability weighting, conflict resolution
    │
    ▼
Verification Pipeline            ← fact-checking all numerical claims
    │
    ▼
Report Generator + Evaluation    ← 20+ quality metrics computed before delivery
    │
    ▼
vector_db_store                  ← findings persisted for future sessions
```

**Core design:** Hybrid Plan-Execute + ReAct. The Planner creates a complete roadmap before any tool is invoked. Each plan step runs a local Thought-Action-Observation loop for adaptive recovery.

**Key principle:** Memory First. Before every external API call, the agent checks its long-term vector memory. This reduces cost, latency, and rate-limit exposure.

---

## Repository Structure

```
Project1A-EyobKebede-AutonomousFinancialResearchAgent/
│
├── README.md                          ← This file
├── ERROR_LOG.md                       ← 7 deliberate errors identified (Day 1)
├── architecture_specification.md      ← Full system design (Day 1)
├── requirements.txt
├── setup.py
│
├── agent/                             ← Core agent logic
│   ├── core.py                        ← LangGraph StateGraph, ResearchState
│   ├── prompts.py                     ← System prompts (planner, executor, verifier)
│   ├── parser.py                      ← LLM output parsing
│   ├── query_analyzer.py              ← Query classification and entity extraction
│   ├── disambiguation.py             ← Ambiguous query resolution
│   ├── error_handler.py               ← Retry, backoff, graceful degradation
│   ├── fallback_chains.py             ← Per-tool fallback sequences
│   └── circuit_breaker.py             ← Open/half-open/closed circuit logic
│
├── tools/                             ← 12-tool registry
│   ├── tool_registry.py               ← ToolRegistry class (routing, caching, rate limits)
│   ├── schemas/                       ← LLM-facing JSON schemas for all 12 tools
│   │   ├── data_retrieval.py          ← sec_filing_search, financial_data_api, ...
│   │   ├── computation.py             ← calculation_engine, fact_checker, report_gen
│   │   └── memory.py                  ← vector_db_search, vector_db_store
│   ├── sec_edgar.py                   ← EDGAR EFTS API (free, no auth)
│   ├── financial_api.py               ← Financial Modeling Prep + Alpha Vantage
│   ├── web_search.py                  ← Tavily (primary) + SerpAPI (fallback)
│   ├── news_sentiment.py              ← NewsAPI + TextBlob/VADER scoring
│   ├── earnings.py                    ← Earnings call transcripts (FMP)
│   ├── company_profile.py             ← Company metadata (FMP)
│   ├── peer_comparison.py             ← Industry peers + comparative metrics
│   ├── calculator.py                  ← DCF, ratios, margins (local, no API)
│   ├── fact_checker.py                ← Cross-reference claims across sources
│   └── report_gen.py                  ← Structured markdown report formatter
│
├── memory/                            ← Three-layer memory architecture
│   ├── vector_store.py                ← ChromaDB (dev) / Pinecone (prod)
│   ├── context_manager.py             ← Sliding window summarization at 70% capacity
│   └── episodic.py                    ← JSON + SQLite strategy log
│
├── synthesis/                         ← Multi-source synthesis engine
│   ├── engine.py                      ← Source weighting, conflict resolution
│   ├── conflict_resolver.py           ← 6-step conflict resolution protocol
│   └── narrative.py                   ← Narrative threading, quantitative triangulation
│
├── evaluation/                        ← 20+ quality metrics
│   ├── metrics.py                     ← FA, CO, AD, CS, AB metric categories
│   ├── dashboard.py                   ← Evaluation summary (4 simulation dimensions)
│   └── benchmarks/                    ← Baseline comparisons
│
├── config/
│   ├── settings.py                    ← Pydantic settings, API key management
│   ├── llm_client.py                  ← Claude Sonnet 4 primary, GPT-4o-mini fallback
│   └── logging_config.py              ← Structured JSON logging
│
├── tests/
│   ├── test_tools.py                  ← Tool registry + 3 core tool unit tests
│   ├── test_memory.py                 ← Vector store and context manager tests
│   ├── test_agent.py                  ← End-to-end agent workflow tests
│   └── test_synthesis.py              ← Conflict resolution and synthesis tests
│
├── results/                           ← Simulation challenge outputs (Days 11–12)
│   ├── challenge_1.md
│   ├── ...
│   └── challenge_8.md
│
└── docs/                              ← Final documentation (Day 14)
    ├── architecture_specification_final.md
    ├── trace_gallery.md
    └── optimization_log.md
```

---

## Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Language | Python 3.11+ | Best LLM library ecosystem |
| Agent Framework | LangGraph 0.1+ | Stateful graph, checkpointing, explicit state transitions |
| Primary LLM | Claude Sonnet 4 (Anthropic) | 200K context window, strong tool use |
| Fallback LLM | GPT-4o-mini (OpenAI) | Cost-efficient for summarization sub-tasks |
| Vector DB (dev) | ChromaDB 0.4+ | Zero setup, local, free |
| Vector DB (prod) | Pinecone | Managed scaling, metadata filtering |
| Embeddings | text-embedding-3-small (OpenAI) | 1536 dims native, $0.02/1M tokens |
| Retry Logic | Tenacity 8.2+ | Declarative exponential backoff |
| Testing | Pytest 8.0+ | Standard Python testing |

---

## Setup

### Prerequisites

- Python 3.11+
- API keys for: Anthropic, OpenAI, Financial Modeling Prep, Tavily, NewsAPI

### Installation

```bash
# Clone the repository
git clone https://github.com/jobnovice/Project1A-EyobKebede-AutonomousFinancialResearchAgent.git
cd Project1A-EyobKebede-AutonomousFinancialResearchAgent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configure API Keys

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your keys:
# ANTHROPIC_API_KEY=your_key_here
# OPENAI_API_KEY=your_key_here
# FMP_API_KEY=your_key_here
# TAVILY_API_KEY=your_key_here
# NEWSAPI_KEY=your_key_here
```

> **Free tiers are sufficient for development.** FMP: 250 req/day. Tavily: 1000/month. NewsAPI: 100/day. SEC EDGAR is completely free.

### Run Tests

```bash
pytest tests/ -v
```

---

## The 12 Tools

### Data Retrieval (7 tools)

| Tool | Source | Free Tier | Use Case |
|------|--------|-----------|----------|
| `sec_filing_search` | EDGAR EFTS | ∞ free | 10-K, 10-Q, 8-K — ground truth |
| `financial_data_api` | FMP + Alpha Vantage | 250/day | Income statement, ratios, history |
| `earnings_transcript` | FMP | 250/day | Management commentary, guidance |
| `web_search` | Tavily + SerpAPI | 1000/month | Recent news, emerging risks |
| `news_sentiment` | NewsAPI + VADER | 100/day | Aggregate sentiment trends |
| `company_profile` | FMP | 250/day | Sector, market cap, executives |
| `peer_comparison` | FMP | 250/day | Competitive positioning |

### Computation & Verification (3 tools)

| Tool | Notes |
|------|-------|
| `calculation_engine` | Local only — DCF, growth rates, margins. No API, no rate limits. |
| `fact_checker` | Orchestrates 2+ sources to verify any numerical claim before report inclusion |
| `report_generator` | Template engine + LLM formatting call → structured markdown |

### Memory (2 tools)

| Tool | Notes |
|------|-------|
| `vector_db_search` | Always called first — Memory First principle |
| `vector_db_store` | Called at session end — persists verified findings |

---

## Source Reliability Hierarchy

The synthesis engine weights all claims by source tier:

| Tier | Weight | Source | Used For |
|------|--------|--------|----------|
| 1 | 1.00 | SEC filings (10-K, 10-Q, 8-K) | Ground truth for all numerical data |
| 2 | 0.90 | Financial data APIs (FMP, FactSet) | Structured metrics, historical series |
| 3 | 0.75 | Earnings call transcripts | Forward guidance, management tone |
| 4 | 0.60 | Major news (Reuters, Bloomberg, FT) | Recent events, market context |
| 5 | 0.20 | Social media, forums | Sentiment signals only — never factual claims |

> Note: Tiers 4 and 5 are corrected from the project document, which erroneously inverted them. See ERROR_LOG.md Error 001.

---

## Design Constraints

| Constraint | Target | Rationale |
|-----------|--------|-----------|
| Max tool calls / task | 20 | Prevent runaway API costs |
| Max agent iterations | 15 | Prevent infinite loops |
| Target latency | < 5 min | Usable SLA for urgent requests |
| Hallucination rate | < 2% | Financial safety requirement |
| Tool efficiency | > 70% | Cost efficiency |
| Error recovery rate | > 90% | Production reliability |

---

## Error Detection Log

Seven deliberate factual errors were embedded in the project document. All 7 were identified and documented in `ERROR_LOG.md`. Summary:

| # | Location | Error | Severity |
|---|----------|-------|----------|
| 001 | A6.2 Source Hierarchy | Tiers 4 & 5 inverted | High |
| 002 | A5.2 Metric AB-4 | Multiply → divide | High |
| 003 | A7.3 SCAP | Wrong year (2007) + wrong act | Medium |
| 004 | C4.2 Indian Filing | Form 20-F is SEC, not MCA | Medium |
| 005 | C3.2 Hallucination % | 45–60% unsupported/implausible | Medium |
| 006 | E2.2 Embedding dims | text-embedding-3-large ≠ 1024 dims | Medium |
| 007 | B3.1 Scoring dims | "Five dimensions" but table has four | Low |

---

## Project Timeline

| Day | Deliverable | Status |
|-----|-------------|--------|
| 1 | Architecture spec, error log, system diagram | ✅ Complete |
| 2 | Tool schemas (12 tools), tool registry, unit tests, project scaffold | ✅ Complete |
| 3 | Environment setup, LLM integration, ChromaDB init | ✅ Complete |
| 4 | Agent core (LangGraph StateGraph, planner, executor) | 🔲 Next |
| 5–7 | Tool implementations | 🔲 |
| 8 | Memory architecture | 🔲 |
| 9 | Synthesis engine | 🔲 |
| 10 | Error handling and circuit breakers | 🔲 |
| 11–12 | 8 simulation challenges | 🔲 |
| 13 | Stress testing and optimization | 🔲 |
| 14 | Documentation finalization | 🔲 |
| 15 | Final submission | 🔲 |

---

## Commit Convention

```
Day-NN: Brief description - EyobKebede
```

Example: `Day-02: Tool schemas and registry scaffold - EyobKebede`

---

## Important Implementation Notes

1. **AB-4 Memory Utilization** is computed as `memory_hits / total_api_calls` (division, not multiplication). See ERROR_LOG.md Error 002.

2. **Embedding dimensions**: `text-embedding-3-small` has **1536 native dimensions**. The vector store schema is initialized accordingly. Do not use `text-embedding-3-large` at default settings without updating the schema.

3. **No investment recommendations**: The system prompt explicitly prohibits buy/sell/hold recommendations. ARA-1 is an analysis tool only.

4. **Graceful degradation**: When all fallbacks fail, the agent marks the section `[DATA UNAVAILABLE]` and completes all other sections. It never fabricates data to fill gaps.

---

*Project: ARA-1 Autonomous Financial Research Agent | Version: 0.3 (Day 3) | Author: Eyob Kebede*
