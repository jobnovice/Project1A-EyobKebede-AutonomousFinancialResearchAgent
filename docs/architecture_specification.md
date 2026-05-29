# Architecture Specification: Project1A-EyobKebede-AutonomousFinancialResearchAgent
## Autonomous Financial Research Agent with Multi-Source Synthesis
### Project1A-EyobKebede-AutonomousFinancialResearchAgent

**Version:** 1.0  
**Date:** Day 1  
**Status:** Foundation Draft  
**Author:** Eyob Kebede

---

## Table of Contents

1. [Executive Overview](#1-executive-overview)
2. [Design Philosophy: First Principles](#2-design-philosophy-first-principles)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Reasoning Loop Design](#4-reasoning-loop-design)
5. [Tool Registry Architecture](#5-tool-registry-architecture)
6. [Memory Architecture](#6-memory-architecture)
7. [Multi-Source Synthesis Engine](#7-multi-source-synthesis-engine)
8. [Error Handling and Graceful Degradation](#8-error-handling-and-graceful-degradation)
9. [RAG Pipeline Design](#9-rag-pipeline-design)
10. [Evaluation Framework](#10-evaluation-framework)
11. [Technology Stack and Rationale](#11-technology-stack-and-rationale)
12. [Data Flow Diagrams](#12-data-flow-diagrams)
13. [Security and Compliance Considerations](#13-security-and-compliance-considerations)
14. [Known Limitations and Future Extensions](#14-known-limitations-and-future-extensions)

---

## 1. Executive Overview

### 1.1 Problem Statement

A junior financial analyst at an investment bank spends the majority of their time on three activities:
finding data, reconciling conflicting data from different sources, and reformatting findings into
structured reports. This workflow is expensive (~$80K–$120K annual cost per analyst), slow (a
comprehensive research report takes 2–5 days), and inconsistent (quality varies by analyst skill and
available time).

The core question this project answers: **Can an AI system replicate this workflow with comparable
output quality at a fraction of the cost and time?**

### 1.2 What This Agent Is

ARA-1 (Autonomous Research Agent — Version 1), the core engine of Project1A-EyobKebede-AutonomousFinancialResearchAgent, is a purpose-built agentic AI system that:

- Receives a natural-language research query from a human stakeholder
- Independently formulates a structured research plan
- Gathers data from multiple authoritative sources using a registry of 10+ tools
- Resolves conflicting data using a tiered source reliability hierarchy
- Synthesizes all findings into a professional-grade investment research report
- Self-evaluates output quality against 20+ metrics before delivery

The agent operates without step-by-step human guidance once given an initial query.

### 1.3 What This Agent Is Not

ARA-1 is not a chatbot. It does not respond to single-turn prompts. It does not make investment
recommendations. It does not fabricate data when sources are unavailable — it degrades gracefully
and transparently instead.

### 1.4 Design Constraints

| Constraint | Value | Rationale |
|-----------|-------|-----------|
| Max tool calls per task | 20 | Prevents runaway API costs |
| Max agent iterations | 15 | Prevents infinite loops |
| Target latency | <5 min (single company) | Usable SLA for urgent requests |
| Hallucination rate target | <2% | Financial safety requirement |
| Tool efficiency target | >70% | Cost efficiency |
| Error recovery rate target | >90% | Production reliability |

---

## 2. Design Philosophy: First Principles

### 2.1 Why First Principles?

Rather than starting from "what does LangChain support," we start from "what does the problem
require." Three fundamental questions drive every architectural decision:

**Question 1: What is the minimum structure an agent needs to produce reliable research?**

An agent needs: a goal (the query), a plan (what to find), tools (how to find it), memory (what it
already knows), a synthesis mechanism (how to combine findings), and a verification step (how to
check its own work). Every component in this architecture maps to one of these six primitives.

**Question 2: Where does failure most commonly occur in research workflows?**

Based on the Alpha Research Labs case study and general agentic AI literature, failure occurs in
this order of frequency:
1. Hallucination (generating plausible-but-wrong data) — most dangerous
2. Missing verification (accepting a single source without cross-referencing)
3. No error recovery (giving up when a tool fails)
4. No memory (re-researching what was already known)
5. Poor tool selection (using web search when a structured API would be more accurate)

Every architectural choice is designed to prevent failures in this priority order.

**Question 3: What does "good research" look like, and how do we measure it?**

Good research is: factually accurate, comprehensive, analytically insightful, internally consistent,
and professionally presented. These map directly to the five evaluation categories in Section 10.

### 2.2 Core Architectural Principles

**Principle 1 — Verify Before You Assert**  
No claim appears in the final report without a source. Every numerical claim passes through a
fact-verification sub-process against at least two independent sources.

**Principle 2 — Memory First, API Second**  
Before making any external API call, the agent checks its long-term vector memory. If sufficient
information exists from a prior research session, it is reused. This reduces cost, latency, and
rate-limit exposure.

**Principle 3 — Degrade Gracefully, Never Silently**  
When a tool fails, the agent tries fallbacks in sequence. When all fallbacks fail, the report
explicitly states what could not be obtained and why — it never fabricates data to fill the gap.

**Principle 4 — Plan Globally, Adapt Locally**  
The agent creates a complete research plan upfront (reducing redundant tool calls) but runs each
plan step via a local ReAct micro-loop (allowing adaptation when intermediate results are unexpected).

**Principle 5 — Radical Transparency**  
Every tool call is logged with: the reasoning that triggered it, the parameters used, the result
received, and the confidence assigned. The full trace is always available for human review.

---

## 3. System Architecture Overview

### 3.1 High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        QUERY INGESTION                          │
│  User Query → Query Analyzer → Disambiguation (if needed)       │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                      PLANNER (LLM)                              │
│  Generates structured research plan with steps + dependencies   │
│  Classifies query type: factual / analytical / comparative /    │
│  contradictory / ambiguous                                      │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                   EXECUTOR (ReAct Micro-Loop)                   │
│  For each plan step:                                            │
│    1. Check long-term memory (vector DB)                        │
│    2. If cache miss → select tool → execute with fallback chain │
│    3. Observe result → update state → decide: continue / revise │
└──────┬──────────────────────┬────────────────────────┬──────────┘
       │                      │                        │
┌──────▼──────┐    ┌──────────▼──────────┐    ┌───────▼──────────┐
│ TOOL        │    │  MEMORY SYSTEM      │    │  ERROR HANDLER   │
│ REGISTRY    │    │  ┌─────────────┐    │    │  ┌────────────┐  │
│ 10+ tools   │    │  │ Short-term  │    │    │  │ Retry      │  │
│ with schemas│    │  │ (context)   │    │    │  │ Backoff    │  │
│ + fallback  │    │  ├─────────────┤    │    │  ├────────────┤  │
│ chains      │    │  │ Long-term   │    │    │  │ Fallback   │  │
│             │    │  │ (vector DB) │    │    │  │ Chain      │  │
│             │    │  ├─────────────┤    │    │  ├────────────┤  │
│             │    │  │ Episodic    │    │    │  │ Circuit    │  │
│             │    │  │ (strategy)  │    │    │  │ Breaker    │  │
└─────────────┘    │  └─────────────┘    │    │  └────────────┘  │
                   └─────────────────────┘    └──────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                   SYNTHESIS ENGINE                              │
│  Source Reliability Weighting → Conflict Resolution →           │
│  Narrative Threading → Quantitative Triangulation              │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                 VERIFICATION PIPELINE                           │
│  Extract all claims → Cross-reference vs. sources →             │
│  Flag discrepancies → Attach confidence scores                  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│               REPORT GENERATOR + EVALUATION                     │
│  Format → Structured markdown report → Compute 20+ metrics →   │
│  Store findings in vector DB for future sessions                │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 LangGraph State Object

The entire agent is driven by a single shared state object passed between nodes in the LangGraph
`StateGraph`. This makes every component's inputs and outputs explicit.

```python
from typing import TypedDict, List, Optional, Dict, Any
from datetime import datetime

class ResearchState(TypedDict):
    # --- Input ---
    raw_query: str                        # Original user query
    query_type: str                       # factual/analytical/comparative/ambiguous
    assumed_companies: List[str]          # Tickers/names identified in query
    clarifications: Optional[str]        # Disambiguation output if needed

    # --- Planning ---
    research_plan: List[Dict[str, Any]]   # Ordered steps with dependencies
    current_step_index: int              # Which plan step we're on
    plan_revision_count: int             # How many times the plan was revised

    # --- Execution ---
    gathered_data: Dict[str, Any]        # Keyed by source type + company
    tool_call_log: List[Dict[str, Any]]  # Full trace of every tool call
    memory_hits: int                     # How many times long-term memory was used
    total_api_calls: int                 # Total external calls made

    # --- Synthesis ---
    draft_report: Optional[str]          # Pre-verification draft
    conflicts_detected: List[Dict]       # Any data conflicts found
    conflicts_resolved: List[Dict]       # Resolution decisions made

    # --- Verification ---
    verified_claims: List[Dict]          # Claims that passed verification
    flagged_claims: List[Dict]           # Claims needing manual review
    hallucination_count: int             # Claims with no source support

    # --- Output ---
    final_report: Optional[str]          # Verified, formatted report
    confidence_scores: Dict[str, float] # Per-section confidence
    evaluation_metrics: Dict[str, Any]  # All 20+ metrics computed
    session_id: str                      # For episodic memory logging
    completed_at: Optional[datetime]
```

---

## 4. Reasoning Loop Design

### 4.1 The Hybrid Plan-Execute + ReAct Architecture

The choice of reasoning loop is the most consequential architectural decision. Three options exist:

| Pattern | Strength | Weakness | Best For |
|---------|----------|----------|----------|
| Pure ReAct | Adaptive, handles surprises | Redundant tool calls, no roadmap | Simple, single-step queries |
| Pure Plan-Execute | Efficient, reviewable | Brittle to unexpected findings | Well-structured, predictable tasks |
| **Hybrid (chosen)** | **Roadmap + adaptability** | **Slightly more complex** | **Complex multi-source research** |

**Decision:** The agent uses a two-level hybrid:

- **Macro level (Plan-Execute):** The Planner LLM generates a complete research plan before any
  tool is invoked. This plan is the "roadmap" — it ensures all necessary research areas are covered
  and prevents the agent from getting stuck on the first interesting finding.

- **Micro level (ReAct):** Each step in the plan is executed via a local Thought-Action-Observation
  loop. This allows the agent to adapt within a step (e.g., try a fallback tool, refine a query)
  without requiring a full plan revision.

### 4.2 The Planner

The Planner is a dedicated LLM call with a specialized system prompt. It receives:
- The user query
- The query classification (factual / analytical / comparative / ambiguous)
- A list of identified companies/tickers
- The current long-term memory index (so it can skip research that's already cached)

It outputs a structured JSON plan:

```json
{
  "plan_id": "plan-tsla-risk-2024-001",
  "query_summary": "Comprehensive risk assessment for Tesla Inc.",
  "estimated_tool_calls": 12,
  "steps": [
    {
      "step_id": 1,
      "description": "Retrieve Tesla's most recent 10-K for official risk factor disclosures",
      "tool": "sec_filing_search",
      "params": {"ticker": "TSLA", "filing_type": "10-K", "year": 2024},
      "depends_on": [],
      "fallback_if_fails": "Use vector_db_search for cached Tesla 10-K data"
    },
    {
      "step_id": 2,
      "description": "Check long-term memory for any prior Tesla research",
      "tool": "vector_db_search",
      "params": {"query": "Tesla risk factors financials", "top_k": 5},
      "depends_on": [],
      "fallback_if_fails": null
    },
    {
      "step_id": 3,
      "description": "Retrieve recent news to identify risks not yet in filings",
      "tool": "web_search",
      "params": {"query": "Tesla risks challenges 2024 2025", "num_results": 10},
      "depends_on": [1],
      "fallback_if_fails": "Use news_sentiment tool as alternative"
    }
  ]
}
```

### 4.3 The Executor (ReAct Micro-Loop)

For each plan step, the Executor runs a local ReAct loop:

```
THOUGHT:  Evaluate current state. What do I have? What does this step require?
          Is the tool available? Have I already retrieved this from memory?

ACTION:   Call the specified tool (or fallback if primary unavailable).

OBSERVE:  Parse the result. Is it what was expected? Is it complete?
          Does it conflict with anything already gathered?

DECIDE:   (a) Result is sufficient → mark step complete, continue to next
          (b) Result is partial → try supplementary tool call within this step
          (c) Tool failed → invoke fallback chain
          (d) Result contradicts prior data → log conflict, invoke conflict resolver
```

Maximum iterations per step: 3 (prevents local infinite loops while allowing recovery attempts).

### 4.4 Plan Revision Triggers

The plan is revised (not just a single step) when:
- A step reveals the company has restructured segments (affects all financial comparison steps)
- A step reveals the company is in an acquisition (all forward-looking analysis must be flagged)
- A step fails entirely including all fallbacks (the affected section is marked incomplete)
- The accumulated token count approaches 70% of context window (triggers summarization + compression)

---

## 5. Tool Registry Architecture

### 5.1 Registry Design

The tool registry is implemented as a Python class that manages all tool metadata, routing, rate
limiting, caching, and fallback chains. The LLM never calls tools directly — it requests a tool
by name, and the registry handles all operational concerns.

```python
class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}
        self.fallback_chains: Dict[str, List[str]] = {}
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.cache: Dict[str, CachedResult] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

    def register(self, tool: ToolDefinition) -> None: ...
    def execute(self, tool_name: str, params: dict) -> ToolResult: ...
    def get_schema_for_llm(self) -> List[dict]: ...  # Injected into system prompt
```

### 5.2 Complete Tool Registry (12 Tools)

#### Group 1: Data Retrieval Tools

**Tool 1: sec_filing_search**
```
Purpose:    Retrieve official SEC filings from EDGAR
Source:     efts.sec.gov (free, no auth required)
Rate limit: 10 req/sec
Fallback:   vector_db_search → web_search (SEC.gov direct links)
Best for:   10-K, 10-Q, 8-K, DEF 14A — authoritative source (Tier 1)
```

**Tool 2: financial_data_api**
```
Purpose:    Structured financial statements, ratios, key metrics
Source:     Financial Modeling Prep (primary), Alpha Vantage (fallback)
Rate limit: 250/day (FMP free), 25/day (Alpha Vantage free)
Fallback:   sec_filing_search (extract from filings) → yfinance → vector_db_search
Best for:   Income statement, balance sheet, cash flow, pre-computed ratios
```

**Tool 3: earnings_transcript**
```
Purpose:    Full earnings call transcripts with speaker attribution
Source:     Financial Modeling Prep API
Rate limit: 250/day
Fallback:   web_search (Seeking Alpha transcript pages) → vector_db_search
Best for:   Management commentary, forward guidance, analyst Q&A sentiment
```

**Tool 4: web_search**
```
Purpose:    Current news, analysis, recent events not in filings
Source:     Tavily API (primary), SerpAPI (fallback)
Rate limit: 1000/month (Tavily), 100/month (SerpAPI)
Fallback:   news_sentiment → vector_db_search
Best for:   Recent news, analyst commentary, emerging risks
```

**Tool 5: news_sentiment**
```
Purpose:    NLP-scored sentiment across recent news articles
Source:     NewsAPI.org + TextBlob/VADER scoring
Rate limit: 100/day
Fallback:   web_search with manual sentiment scoring in synthesis layer
Best for:   Aggregate market sentiment, sentiment trends over time
```

**Tool 6: company_profile**
```
Purpose:    Basic company metadata: sector, market cap, executives, description
Source:     Financial Modeling Prep
Rate limit: 250/day
Fallback:   web_search → vector_db_search
Best for:   Challenge 1 (company profile), entity validation
```

**Tool 7: peer_comparison**
```
Purpose:    Identify industry peers and retrieve comparative metrics
Source:     Financial Modeling Prep (peers endpoint)
Rate limit: 250/day
Fallback:   Manual peer identification via web_search + financial_data_api per peer
Best for:   Competitive positioning, relative valuation
```

#### Group 2: Computation and Verification Tools

**Tool 8: calculation_engine**
```
Purpose:    Financial calculations: DCF, growth rates, ratios, margins
Source:     Local computation (no external API)
Rate limit: None
Fallback:   N/A (local computation always available)
Best for:   Derived metrics not available from APIs, DCF valuation
```

**Tool 9: fact_checker**
```
Purpose:    Cross-reference a specific claim against multiple sources
Source:     Orchestrates calls to other tools internally
Rate limit: Inherits from sub-tools called
Fallback:   Single-source verification with explicit confidence downgrade
Best for:   Any numerical claim before including in final report
```

**Tool 10: report_generator**
```
Purpose:    Format gathered data into structured markdown report
Source:     Local template engine + LLM formatting call
Rate limit: 1 LLM call per report section
Fallback:   Raw markdown assembly without LLM formatting
Best for:   Final output generation
```

#### Group 3: Memory Tools

**Tool 11: vector_db_search**
```
Purpose:    Semantic search across agent's long-term research memory
Source:     ChromaDB (dev) / Pinecone (production)
Rate limit: None (local) / generous (Pinecone)
Fallback:   Keyword search on episodic memory JSON log
Best for:   Retrieving prior research before making external API calls
```

**Tool 12: vector_db_store**
```
Purpose:    Store new research findings for future session retrieval
Source:     ChromaDB (dev) / Pinecone (production)
Rate limit: None meaningful
Fallback:   Append to episodic memory JSON log
Best for:   End of each research session — store all verified findings
```

### 5.3 Fallback Chain Summary

```
Primary               → Fallback 1            → Fallback 2          → Fallback 3
──────────────────────────────────────────────────────────────────────────────────
sec_filing_search     → vector_db_search      → web_search           → [DEGRADE]
financial_data_api    → sec_filing_search     → yfinance             → vector_db_search
earnings_transcript   → web_search            → vector_db_search     → [DEGRADE]
web_search            → news_sentiment        → vector_db_search     → [DEGRADE]
news_sentiment        → web_search            → vector_db_search     → [DEGRADE]
company_profile       → financial_data_api    → web_search           → vector_db_search
peer_comparison       → web_search            → financial_data_api   → [DEGRADE]
```

### 5.4 Tool Selection Anti-Patterns (What to Avoid)

These are explicitly detected and logged as warnings in the tool efficiency metric (AB-1):

- Using `web_search` for structured data → use `financial_data_api` or `sec_filing_search`
- Calling any external tool without first checking `vector_db_search`
- Making multiple calls to the same tool with identical parameters
- Using `financial_data_api` for a single ratio when a full statement call yields multiple needed metrics
- Accepting a single source's number without invoking `fact_checker` for material claims

---

## 6. Memory Architecture

### 6.1 Three-Layer Memory Design

Memory in this agent serves fundamentally different purposes at three timescales:

```
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 1: SHORT-TERM MEMORY (Working Memory)                     │
│  Scope:    Current research session only                         │
│  Medium:   LLM context window                                    │
│  Capacity: ~150K tokens (Claude Sonnet 4 = 200K, reserve 50K)   │
│  Contents: Current query, plan, all TAO traces, gathered data    │
│  Managed:  Sliding window summarization at 70% capacity          │
└──────────────────────────────────────────────────────────────────┘
           ↓ Persisted at session end
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 2: LONG-TERM MEMORY (Semantic Memory)                     │
│  Scope:    Persistent across all sessions                        │
│  Medium:   ChromaDB (dev) / Pinecone (prod)                      │
│  Capacity: Effectively unlimited                                 │
│  Contents: Verified research findings, chunked by source type    │
│  Schema:   See Section 6.3                                       │
│  Queried:  Before every external API call (Memory First rule)    │
└──────────────────────────────────────────────────────────────────┘
           ↓ Indexed separately
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 3: EPISODIC MEMORY (Strategy Memory)                      │
│  Scope:    Persistent, structured log of agent behaviour         │
│  Medium:   JSON file + SQLite index                              │
│  Contents: Tool success/failure rates, effective research        │
│            strategies per query type, error recovery patterns    │
│  Used by:  Planner — informs which tools to prioritize for       │
│            this type of query based on past session outcomes     │
└──────────────────────────────────────────────────────────────────┘
```

### 6.2 Short-Term Memory Management

Context window management is critical. The strategy:

**Budget allocation (200K token window):**
```
System prompt + tool registry:    ~8K tokens   (4%)
Research plan:                    ~2K tokens   (1%)
Active Thought-Action-Obs traces: ~40K tokens  (20%)
Gathered data (current step):     ~80K tokens  (40%)
Prior step summaries:             ~40K tokens  (20%)
Generation headroom:              ~30K tokens  (15%)
```

**Summarization trigger:** When total context reaches 70% of limit (140K tokens), the agent:
1. Identifies the earliest Thought-Action-Observation traces
2. Calls a summarization LLM pass: "Compress these traces into key findings only"
3. Replaces raw traces with compressed summary
4. Retains all raw traces in a separate session log file for the Trace Gallery

### 6.3 Vector Database Schema

```python
# Each record stored in Chroma/Pinecone:
{
    "id": "tsla-10k-2024-risk-supply-chain-001",
    "content": "Tesla faces significant supply chain risks...",  # Text to embed
    "embedding": [...],     # Generated by text-embedding-3-small (1536 dims)
    "metadata": {
        "ticker": "TSLA",
        "source_type": "10-K",          # 10-K / 10-Q / 8-K / earnings_call / news / analysis
        "date": "2024-10-15",           # Source document date
        "section": "Risk Factors",      # Document section
        "confidence": 0.95,             # Agent's confidence score
        "verified": True,               # Whether fact-checked against 2+ sources
        "session_id": "session-001",    # Research session that created this record
        "chunk_index": 3,               # Position within source document
        "word_count": 187               # For context budgeting
    }
}
```

### 6.4 Chunking Strategy by Document Type

| Source Type | Chunking Strategy | Rationale |
|-------------|------------------|-----------|
| SEC 10-K / 10-Q | By section (Risk Factors, MD&A, Financials); sub-chunk by individual risk factor | Preserves semantic coherence of regulatory language |
| Earnings Transcripts | By speaker turn; keep Q&A pairs together | Analyst question + management answer is one semantic unit |
| News Articles | By paragraph; always include headline + first paragraph in every chunk | Provides article context without full duplication |
| Financial Statements | Stored as structured JSON, NOT embedded | Numerical data retrieved via metadata filter, not semantic search |
| Research Reports | By section with 150-word overlap between chunks | Preserves cross-section narrative context |

### 6.5 Episodic Memory Schema

```json
{
  "session_id": "session-2024-01-15-001",
  "query_type": "risk_assessment",
  "companies": ["TSLA"],
  "tool_performance": {
    "sec_filing_search": {"calls": 2, "successes": 2, "avg_latency_ms": 1240},
    "financial_data_api": {"calls": 3, "successes": 2, "avg_latency_ms": 890},
    "web_search": {"calls": 4, "successes": 4, "avg_latency_ms": 620}
  },
  "strategy_notes": "For risk assessments, SEC 10-K should be retrieved first — it anchors all risk categories. News search is most valuable for risks not yet in filings.",
  "errors_encountered": ["financial_data_api timeout on call 3 — recovered via sec_filing_search fallback"],
  "final_quality_score": 0.87,
  "memory_hit_rate": 0.35
}
```

---

## 7. Multi-Source Synthesis Engine

### 7.1 Source Reliability Hierarchy (Corrected)

The synthesis engine weights all claims by the tier of their source:

```
Tier 1 (Trust weight: 1.00): SEC filings (10-K, 10-Q, 8-K)
        Legally mandated, audited, subject to criminal penalties
        → Use as ground truth for all numerical financial data

Tier 2 (Trust weight: 0.90): Financial data APIs (Bloomberg, FactSet, Refinitiv, FMP)
        Curated from primary sources with data quality controls
        → Use for structured metrics, ratios, historical series

Tier 3 (Trust weight: 0.75): Earnings call transcripts
        Direct management voice — valuable but subject to spin
        → Use for forward guidance, strategic priorities, management tone

Tier 4 (Trust weight: 0.60): Major news outlets (Reuters, Bloomberg News, FT, WSJ)
        Professional journalism with editorial oversight
        → Use for recent events, market context, analyst sentiment

Tier 5 (Trust weight: 0.20): Social media, forums, anonymous sources
        Unverified, potentially manipulative
        → Use only for sentiment signals, never for factual claims
```

Note: Tier 4 and Tier 5 are intentionally reversed from the project document's erroneous ordering.
See ERROR_LOG.md Error 001.

### 7.2 Conflict Resolution Protocol

When two sources report different values for the same metric:

```
Step 1: IDENTIFY
        Detect conflict: abs(value_A - value_B) / max(value_A, value_B) > 0.05 (5% threshold)

Step 2: ASSESS TIERS
        Compare source tiers. If Tier 1 vs Tier 3, prefer Tier 1 unconditionally.

Step 3: CHECK TEMPORAL ALIGNMENT
        Are both values for the same period? Financial APIs sometimes return TTM
        while filings report fiscal year — not a real conflict.

Step 4: CHECK FOR RESTATEMENTS
        Did the company restate financials? Newer sources supersede older ones.

Step 5: APPLY HIGHEST-TIER RULE
        If steps 3-4 don't resolve it, use the highest-tier source value.

Step 6: DOCUMENT
        Always note in the report: "Source A reported X; Source B reported Y.
        We used Source A (Tier N) as it is more authoritative. Discrepancy may
        reflect [restatement / different reporting periods / data lag]."
```

### 7.3 The Three Synthesis Techniques

**Narrative Threading:** Connects data points chronologically or thematically into a coherent arc.
A revenue growth number from the 10-K, management commentary from the earnings call, and a news
report of a new contract are woven together: "Revenue grew 15% YoY (10-K FY2024), driven by the
enterprise segment according to the CEO (Q3 2024 earnings call). This growth trajectory may be
sustained by the recently announced $2.1B AWS contract (Reuters, Nov 2024)."

**Quantitative Triangulation:** For any material number, collect the same metric from 3 sources:
- If 2/3 agree → report with high confidence
- If all 3 disagree → report the range and explain the discrepancy
- If only 1 source → report with explicit low-confidence flag

**Sentiment-Fact Alignment:** Compare qualitative tone (news, transcripts) against quantitative
reality (financials). Divergence is itself an analytical finding worth highlighting.

---

## 8. Error Handling and Graceful Degradation

### 8.1 Error Taxonomy

| Error Category | Examples | Recovery Strategy |
|---------------|----------|------------------|
| API Unavailability | EDGAR down, FMP 503 | Exponential backoff → fallback chain |
| Rate Limiting | 429 Too Many Requests | Wait with jitter → try alternative tool |
| Authentication | Expired API key | Fail fast → alert + use fallback |
| Malformed Response | Missing fields, wrong types | Parse what's available → flag gaps |
| Timeout | >30s response time | Retry once → fallback |
| Hallucination | LLM invents data | Chain-of-verification catches it |
| Circular Reasoning | Same tool, same params, repeated | Iteration limit + pattern detection |
| Context Overflow | >70% context window | Summarization trigger |

### 8.2 Retry with Exponential Backoff

```python
@retry(
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(5),
    before_sleep=log_retry_attempt,
    retry=retry_if_exception_type((APIError, TimeoutError, RateLimitError))
)
async def execute_tool_with_retry(tool_name: str, params: dict) -> ToolResult:
    return await tool_registry.execute(tool_name, params)
```

Jitter: Random 0–500ms added to each retry delay to prevent thundering herd.

### 8.3 Circuit Breaker Pattern

After 3 consecutive failures within 60 seconds, a tool's circuit breaker opens:
- Open state: All calls to this tool immediately route to fallback (no retry attempts)
- Half-open state: After 5 minutes, one probe request is allowed
- Closed state: Normal operation resumes after successful probe

This prevents a failing tool from consuming the entire 20-call budget on retries.

### 8.4 Graceful Degradation Protocol

When all fallbacks are exhausted for a required research section:

```
1. Do NOT fabricate data to fill the gap
2. Mark the section: "[DATA UNAVAILABLE: <tool_name> and all fallbacks failed.
   Specific error: <error_message>. This section could not be completed.]"
3. Complete all other sections normally
4. Add to Research Methodology Notes: "The following data could not be retrieved
   and may require manual research: [list]"
5. Downgrade the confidence score for the overall report
6. Log the failure pattern in episodic memory for future planning
```

---

## 9. RAG Pipeline Design

### 9.1 Six-Stage Pipeline

```
Stage 1: QUERY TRANSFORMATION
         Raw query → 3-5 specific retrieval queries via LLM decomposition
         Example: "Is Tesla a good investment?" →
           ["Tesla revenue growth operating margin 2023 2024",
            "Tesla competitive risks electric vehicle market share",
            "Tesla balance sheet debt cash flow",
            "Tesla management forward guidance Q4 2024",
            "Tesla analyst price target consensus"]

Stage 2: MULTI-SOURCE RETRIEVAL (parallel)
         ├── vector_db_search (long-term memory — always first)
         ├── sec_filing_search (EDGAR filings)
         ├── financial_data_api (structured data)
         ├── web_search (recent news and analysis)
         └── earnings_transcript (management commentary)

Stage 3: RELEVANCE FILTERING + RE-RANKING
         Cross-encoder re-ranking (Cohere rerank-english-v3)
         Filter: relevance score < 0.3 threshold → discard
         Domain-specific note: "EBITDA margin" is relevant to "operating margins" query

Stage 4: CONTEXT ASSEMBLY
         Token budget: 40% primary data / 30% supporting / 20% system / 10% generation
         Ordering: Most recent first; diverse source types enforced
         Tagging: Every chunk tagged with source, date, company, tier

Stage 5: GROUNDED GENERATION
         LLM generates draft using ONLY the assembled context
         Explicit instruction: "If the context does not support a claim, say so"
         Citation format enforced: every factual claim → [Source: X, Date: Y, Page: Z]

Stage 6: POST-GENERATION VERIFICATION
         Separate LLM call: "Identify all claims in the draft not supported by context"
         Unsupported claims → corrected from source data or flagged with disclaimer
         Numerical claims → routed to fact_checker tool for dual-source confirmation
```

### 9.2 HyDE for Complex Analytical Queries

For queries classified as "analytical" (not simple factual lookups), HyDE (Hypothetical Document
Embedding) is applied to improve retrieval:

```python
# Generate hypothetical ideal answer first
hypothetical_answer = llm.generate(
    f"Write a 200-word expert analysis answering: {query}"
)

# Use hypothetical answer embedding as retrieval query
# (More semantically similar to answer documents than the raw question)
retrieval_embedding = embed(hypothetical_answer)
results = vector_db.similarity_search(retrieval_embedding, top_k=10)
```

---

## 10. Evaluation Framework

### 10.1 20+ Quality Metrics

**Category 1: Factual Accuracy (5 metrics)**

| Metric | Description | Target | Implementation |
|--------|-------------|--------|---------------|
| FA-1 Numerical Accuracy | % numerical claims matching authoritative sources | >98% | Automated: extract numbers, cross-ref source data |
| FA-2 Citation Accuracy | % citations that are real and support the claim | 100% | Automated: verify each URL/source exists and contains cited claim |
| FA-3 Temporal Accuracy | All data correctly attributed to right time period | 100% | Automated: date extraction + period validation |
| FA-4 Entity Accuracy | Company names, tickers, executives are correct | 100% | Automated: NER + known entity list validation |
| FA-5 Hallucination Rate | Claims with no source in retrieved context | 0 | Automated: Chain-of-verification LLM pass |

**Category 2: Completeness (4 metrics)**

| Metric | Description | Target | Implementation |
|--------|-------------|--------|---------------|
| CO-1 Section Coverage | All required sections present | 100% | Automated: Template section checklist |
| CO-2 Source Diversity | Distinct source types used | ≥4 types | Automated: Count unique source_type values in tool_call_log |
| CO-3 Temporal Coverage | Historical trend coverage | ≥3 years | Automated: Date range of cited sources |
| CO-4 Risk Factor Coverage | % of 10-K risk factors addressed | ≥80% | Automated: NLP overlap vs. 10-K risk factor list |

**Category 3: Analytical Depth (4 metrics)**

| Metric | Description | Target | Implementation |
|--------|-------------|--------|---------------|
| AD-1 Insight Density | Non-obvious observations per page | ≥3/page | LLM-as-judge |
| AD-2 Cross-Source Synthesis | Info from 2+ sources connected into one insight | ≥5/report | LLM-as-judge + citation graph analysis |
| AD-3 Quantitative Reasoning | Original calculations derived from raw data | ≥10/report | Automated: detect calculation steps in report |
| AD-4 Forward-Looking Analysis | Reasoned projections present | ≥2 sections | LLM-as-judge |

**Category 4: Coherence and Structure (4 metrics)**

| Metric | Description | Target | Implementation |
|--------|-------------|--------|---------------|
| CS-1 Logical Flow | Sections follow logical progression | 0 logical gaps | LLM-as-judge with structured rubric |
| CS-2 Internal Consistency | No self-contradictions | 0 contradictions | Automated: Contradiction detection LLM pass |
| CS-3 Executive Summary Quality | Summary accurately reflects full report | High alignment | Automated: ROUGE/BERTScore vs. body |
| CS-4 Professional Formatting | Template compliance, appropriate tables | Pass/fail per criterion | Automated: Structure checklist |

**Category 5: Agent Behaviour (5 metrics)**

| Metric | Description | Target | Formula |
|--------|-------------|--------|---------|
| AB-1 Tool Efficiency | Useful calls / total calls | ≥70% | cited_tool_results / total_tool_calls |
| AB-2 Error Recovery | Errors recovered via fallback | ≥90% | recovered_errors / total_errors |
| AB-3 Planning Quality | Plan coverage and revision quality | Qualitative | LLM-as-judge |
| AB-4 Memory Utilization | Memory hits vs. external calls | ≥0.3 | **memory_hits / total_api_calls** (division, not multiplication) |
| AB-5 Latency | Query to delivery time | <5 min | Automated: wall clock measurement |

**Additional Metrics (4+)**

| Metric | Description | Target |
|--------|-------------|--------|
| AX-1 Confidence Calibration | Correlation between stated confidence and actual accuracy | >0.7 Pearson r |
| AX-2 Conflict Detection Rate | % of actual conflicts correctly identified | >95% |
| AX-3 Fallback Transparency | % of fallback uses disclosed in report | 100% |
| AX-4 Token Efficiency | Output quality per 1000 tokens consumed | Maximize |

### 10.2 LLM-as-Judge Implementation

For qualitative metrics (AD-1, AD-2, AD-4, CS-1, AB-3), a separate LLM evaluation call is made
with a structured rubric:

```python
JUDGE_PROMPT = """
You are an expert financial research quality evaluator.
Assess the following research report section on {metric_name}.

Scoring Rubric (1-5 scale):
5: Exceeds professional analyst standard
4: Meets professional analyst standard  
3: Adequate but missing key elements
2: Below standard, significant gaps
1: Fails to meet minimum requirements

Report section:
{section_text}

Return ONLY a JSON object: {"score": N, "reasoning": "...", "examples": [...]}
"""
```

---

## 11. Technology Stack and Rationale

### 11.1 Core Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Language | Python 3.11+ | Best LLM library ecosystem |
| Agent Framework | LangGraph 0.1+ | Stateful graph workflows, checkpointing, better than LangChain AgentExecutor for complex flows |
| Primary LLM | Claude Sonnet 4 (Anthropic) | 200K context window (critical for research tasks), excellent instruction following, strong tool use |
| Fallback LLM | GPT-4o-mini | Cost-efficient for summarization and verification sub-tasks |
| Vector DB (dev) | ChromaDB 0.4+ | Zero setup, local, free — ideal for development |
| Vector DB (prod) | Pinecone | Managed scaling, metadata filtering, 100K free vectors |
| Embeddings | text-embedding-3-small (OpenAI) | 1536 dimensions, $0.02/1M tokens — best cost/quality balance |
| Retry Logic | Tenacity 8.2+ | Declarative retry decorators, exponential backoff built-in |
| Testing | Pytest 8.0+ | Standard Python testing |

### 11.2 API Dependencies

| Service | Purpose | Free Tier | Priority |
|---------|---------|-----------|----------|
| SEC EDGAR (efts.sec.gov) | SEC filings | Free, no key | Required |
| Financial Modeling Prep | Financial data, transcripts | 250 req/day | Required |
| Tavily | Web search | 1000/month | Required |
| NewsAPI.org | News retrieval | 100 req/day | Required |
| Anthropic API | LLM reasoning | Free tier | Required |
| Alpha Vantage | Financial data fallback | 25 req/day | Fallback |
| Pinecone | Vector DB (production) | 100K vectors | Optional |

### 11.3 Why LangGraph over Plain LangChain

LangChain's `AgentExecutor` implements a simple while-loop over tool calls. LangGraph allows us to
represent the full agent workflow as a directed graph with explicit state transitions. This is critical
because:

1. **The Planner and Executor are separate nodes** — they can use different prompts and models
2. **Conditional edges** allow plan revision without code changes
3. **Checkpointing** lets the agent pause/resume (important for long research tasks)
4. **State is explicit and auditable** — every node reads/writes a typed state object
5. **Parallel execution** is natively supported for simultaneous tool calls

---

## 12. Data Flow Diagrams

### 12.1 Complete Request-Response Flow

```
User Query
    │
    ▼
Query Analyzer ──────────────────────────────────────────┐
    │ - Classify type (factual/analytical/comparative)    │
    │ - Extract company tickers                           │
    │ - Check if ambiguous                                │
    │                                                     │
    ├─ If ambiguous ──────────────────────────────────────┤
    │                                                     │
    │  Disambiguation Module                              │
    │  - Generate clarifying questions OR                 │
    │  - Make documented assumptions                      │
    │  - Log assumption in report methodology section     │
    │                                                     │
    ▼ (query is clear)                                   │
Memory Pre-Check (vector_db_search)──────────────────────┘
    │
    │ - Search for prior research on this company
    │ - If cache hit: include in plan, skip redundant steps
    │
    ▼
Planner LLM Call
    │ - Input: query + company list + memory hits + available tools
    │ - Output: structured JSON plan (steps + dependencies)
    │
    ▼
    ┌─── For each Plan Step ────────────────────────────────┐
    │                                                        │
    │  1. Check vector_db_search (Memory First rule)         │
    │     └─ Cache hit → use cached data, skip API call     │
    │     └─ Cache miss → proceed to tool execution         │
    │                                                        │
    │  2. Execute Tool (with circuit breaker check)          │
    │     └─ Circuit open → go to fallback immediately      │
    │     └─ Circuit closed → try primary tool               │
    │         └─ Success → observe + update state           │
    │         └─ Failure → exponential backoff + retry      │
    │             └─ Max retries → try Fallback 1           │
    │                 └─ Failure → try Fallback 2           │
    │                     └─ All fail → graceful degrade    │
    │                                                        │
    │  3. Conflict Detection                                 │
    │     └─ New data vs. existing data: conflict?          │
    │         └─ Yes → run Conflict Resolution Protocol     │
    │         └─ No → add to gathered_data                  │
    │                                                        │
    │  4. Decide: Plan revision needed?                      │
    │     └─ Unexpected finding → revise downstream steps   │
    │     └─ Context approaching limit → summarize          │
    │                                                        │
    └───────────────────────────────────────────────────────┘
    │
    ▼
Synthesis Engine
    │ - Weight all data by source tier
    │ - Resolve remaining conflicts
    │ - Generate narrative threads
    │ - Perform quantitative triangulation
    │
    ▼
Verification Pipeline
    │ - Extract all numerical claims
    │ - Cross-reference each via fact_checker
    │ - Flag unverifiable claims
    │ - Assign confidence scores per section
    │
    ▼
Report Generator
    │ - Format into required template sections
    │ - Attach source citations inline
    │ - Add Research Methodology Notes
    │ - Include confidence scores
    │
    ▼
Evaluation Module
    │ - Compute all 20+ metrics
    │ - Generate evaluation summary
    │
    ▼
vector_db_store (persist findings for future sessions)
    │
    ▼
Final Report Delivered
```

---

## 13. Security and Compliance Considerations

### 13.1 API Key Management

All API keys stored in `.env` file, never committed to git. The `.env.example` file shows required
variables without values. CI/CD pipeline checks for exposed secrets before any push.

```bash
# .env.example
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
FMP_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
NEWSAPI_KEY=your_key_here
PINECONE_API_KEY=your_key_here  # optional
```

### 13.2 No Investment Recommendations

The system prompt explicitly prohibits investment recommendations: "Do not make buy, sell, or hold
recommendations. Do not state that a company is a good or bad investment. Present analysis only."

### 13.3 Data Provenance

Every claim in the final report is traceable to a specific source document, retrieval date, and tool
call ID. This audit trail is stored in the session log and is never deleted.

---

## 14. Known Limitations and Future Extensions

### 14.1 Current Limitations

- **Rate limits constrain breadth:** Free API tiers limit the number of companies that can be
  researched in a single day. Challenge 4 (three-company comparison) may exhaust free tier quotas.

- **Context window limits depth:** Even at 200K tokens, very large SEC filings (some 10-Ks exceed
  300 pages) cannot be fully held in context. Chunking and selective retrieval are mitigations, not
  perfect solutions.

- **No real-time data:** The agent cannot access live market prices, intraday news, or real-time
  order flow. All financial data is at minimum hours old.

- **English only:** The current architecture does not support non-English regulatory documents.
  The India case study (Hindi/regional language filings) would require OCR + translation tooling.

### 14.2 Planned Extensions (Post-Project-1A)

- **Multi-agent architecture (Project AAI-3B):** Decompose into specialist agents (Financial
  Statement Agent, Sentiment Agent, Risk Agent, Synthesis Agent) for parallel execution.

- **India-specific tools:** MCA filing retrieval (AOC-4, MGT-7), SEBI circular search, BSE/NSE
  data feeds.

- **Real-time data integration:** Market data WebSocket feeds for live price/volume data.

- **Self-improving prompts:** Use episodic memory to automatically refine system prompts based on
  which prompt variants produced higher evaluation scores.

---

*Document: architecture_specification.md*  
*Project: Project1A-EyobKebede-AutonomousFinancialResearchAgent*  
*Version: 1.0 | Status: Day 1 Foundation Draft*
