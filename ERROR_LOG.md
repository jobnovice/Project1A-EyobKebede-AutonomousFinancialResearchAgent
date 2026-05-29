# ERROR_LOG.md
## Project1A-EyobKebede-AutonomousFinancialResearchAgent
## Deliberate Error Detection Log

**Candidate:** Eyob Kebede  
**Date:** 2026-05-28  
**Document Scanned:** Agentic AI - Autonomous Financial Research Agent with Multi-Source Synthesis (71 pages)  
**Errors Identified:** 7 / 7  

---

> **Assessment Note:** The project document states it contains exactly 7 deliberate factual or logical
> errors embedded throughout Parts A–E. Each error is documented below with: the location, the
> erroneous claim, the correct information, and the reasoning used to identify it.

---

## ERROR 001 — Source Reliability Hierarchy: Tiers 4 and 5 Are Inverted

**Location:** Part A, Section A6.2 — "Tiered Source Trustworthiness"  
**Page:** 21  
**Error Type:** Factual / Logical inaccuracy  
**Severity:** High — directly affects synthesis engine design

### Erroneous Claim
The document lists the source reliability hierarchy as:
- Tier 4: Social media posts and anonymous forum discussions
- Tier 5: Major news outlets (Reuters, Bloomberg News, Financial Times)

This places major professional news outlets **below** anonymous social media in trustworthiness.

### Why This Is Wrong
Professional journalism from Reuters, Bloomberg News, and the Financial Times involves:
- Named reporters with professional accountability
- Editorial oversight and fact-checking layers
- Legal liability for material misstatements
- Institutional reputation at stake

Anonymous social media and forum posts have none of these controls and are explicitly described in the
document itself as "unverified and subject to manipulation." Placing them above professional journalism
contradicts the document's own description of the tiers.

### Correct Hierarchy
- Tier 4 (corrected): **Major news outlets** (Reuters, Bloomberg News, Financial Times) — professional
  journalism with editorial oversight, but may contain errors or bias
- Tier 5 (corrected): **Social media posts and anonymous forum discussions** — crowd-sourced, unverified,
  subject to manipulation

### Impact on Implementation
The synthesis engine's conflict resolution logic must use the **corrected** hierarchy. When a Bloomberg
News article conflicts with a Reddit post, the agent must prefer Bloomberg — not the other way around.

---

## ERROR 002 — AB-4 Memory Utilization: Formula Uses Multiplication Instead of Division

**Location:** Part A, Section A5.2 — "Category 5: Agent Behaviour (5 Metrics)", metric AB-4  
**Page:** 19  
**Error Type:** Incorrect formula / logical error  
**Severity:** High — directly corrupts a core evaluation metric

### Erroneous Claim
> "Note: This metric is calculated as memory_hits **multiplied by** total_api_calls."

### Why This Is Wrong
The document's own description of AB-4 defines it as:

> "Measured as the **ratio** of memory hits to total external API calls (higher is better, target: >=0.3)"

A ratio is computed by **division**, not multiplication. If an agent has 30 memory hits out of 100 total
API calls:

- **Correct formula:** `30 / 100 = 0.30` → meets the ≥0.3 target ✓
- **Wrong formula:** `30 × 100 = 3000` → a meaningless number with no upper bound ✗

Multiplying hits by calls produces a value that cannot be interpreted as a ratio, cannot be compared
across sessions of different lengths, and has no sensible target threshold.

### Correct Formula
```
AB-4 Memory Utilization = memory_hits / total_api_calls
```

### Impact on Implementation
The `evaluation/metrics.py` module must implement this as division. Any evaluation dashboard displaying
AB-4 computed via multiplication would produce nonsensical results and fail to correctly identify whether
the ≥0.3 target has been met.

---

## ERROR 003 — SCAP Timeline: Wrong Year and Wrong Legislative Attribution

**Location:** Part A, Section A7.3 — "Query Disambiguation", last paragraph  
**Page:** 24  
**Error Type:** Factual historical inaccuracy (compound error — two facts wrong in one sentence)  
**Severity:** Medium — affects domain knowledge credibility

### Erroneous Claim
> "Note: The first US bank stress tests under SCAP were conducted in **2007** following the
> **Dodd-Frank Act**."

### Why This Is Wrong
This sentence contains two independent factual errors:

**Error A — The Year:**  
The Supervisory Capital Assessment Program (SCAP) was conducted in **2009**, not 2007. It was initiated
by the US Treasury and Federal Reserve in February 2009 and results were published in May 2009, as a
direct response to the 2008 financial crisis. In 2007, the crisis had not yet peaked and no such program
existed.

**Error B — The Legislative Attribution:**  
SCAP was **not** created by the Dodd-Frank Act. The Dodd-Frank Wall Street Reform and Consumer
Protection Act was signed into law on **July 21, 2010** — more than a year *after* SCAP was completed.
SCAP was an emergency executive action by the Treasury and Fed, not a Dodd-Frank requirement. The
annual stress tests *mandated by law* (DFAST — Dodd-Frank Act Stress Tests) came later, starting in
2013.

### Correct Statement
> "The first US bank stress tests under SCAP were conducted in **2009**, as an emergency response to the
> financial crisis, predating the Dodd-Frank Act (enacted 2010)."

### Impact on Implementation
The query disambiguation example in A7.3 is illustrative, so this error does not affect code. However,
if the agent's long-term memory is pre-populated with training context from this document, it must not
ingest this incorrect statement as fact.

---

## ERROR 004 — Indian Annual Filing Form: Incorrect Form Number and Filing Authority

**Location:** Part C, Section C4.2 — "Challenges Specific To Indian Market Research"  
**Page:** 42  
**Error Type:** Factual inaccuracy about regulatory filing requirements  
**Severity:** Medium — affects India-specific tool design

### Erroneous Claim
> "Filing format differences: Indian companies file annual returns using **Form 20-F** with the **MCA**
> (Ministry of Corporate Affairs), similar to the 10-K filing in the US system."

### Why This Is Wrong
This claim has two errors:

**Error A — Form 20-F is an SEC form, not an MCA form:**  
Form 20-F is filed with the **US Securities and Exchange Commission (SEC)** by foreign private issuers
that are listed on US exchanges. For example, Infosys, which is listed on the NYSE, files a Form 20-F
with the SEC. It has nothing to do with India's MCA.

**Error B — Indian domestic annual filings use different forms:**  
Indian companies filing with the MCA use:
- **Form AOC-4:** Filing of financial statements
- **Form MGT-7 / MGT-7A:** Annual return to the MCA

For disclosures to Indian stock exchanges (BSE/NSE), companies submit their **Annual Report** directly
to the exchanges under SEBI's Listing Obligations and Disclosure Requirements (LODR) Regulations.

### Correct Statement
> "Indian companies file financial statements with the MCA using Form AOC-4, and annual returns using
> Form MGT-7. Their Annual Reports are also submitted to BSE/NSE under SEBI's LODR Regulations.
> Indian companies cross-listed in the US additionally file Form 20-F with the SEC."

### Impact on Implementation
The India-specific tool for MCA filing retrieval must query the correct MCA21 portal using AOC-4 / MGT-7
identifiers, not look for Form 20-F filings on EDGAR.

---

## ERROR 005 — Industry Hallucination Rate Benchmark: Implausible and Unsupported Figure

**Location:** Part C, Case Study 3 — "Alpha Research Labs", Section C3.2  
**Page:** 40  
**Error Type:** Fabricated / implausible statistic  
**Severity:** Medium — tests whether students accept stated numbers uncritically

### Erroneous Claim
> "Note: Industry average hallucination rates for unverified financial agents are typically around
> **45-60%**."

### Why This Is Wrong
A hallucination rate of 45–60% would mean that nearly every other sentence in an agent's output is
fabricated. This would render the system entirely unusable for any purpose — no financial institution
would deploy or even test such a system. The claim is:

1. **Not cited** — no source, paper, or benchmark dataset is referenced
2. **Internally inconsistent** — the document's own target is <2% hallucination rate (metric FA-5),
   implying the baseline is much closer to achievable without heroic effort
3. **Contradicted by published research** — academic benchmarks on LLM hallucination in structured
   tasks typically show rates in the 3–25% range depending on domain and verification setup, not 45–60%
4. **Implausible by design** — it is embedded in the document specifically to see whether students
   accept authoritative-sounding numbers without scrutiny

The actual hallucination rate Alpha Research Labs achieved before redesign (23%) is a plausible
real-world figure. The 45–60% "industry average" is the planted error.

### Correct Understanding
There is no established industry-standard benchmark for financial agent hallucination rates as of 2024–2025.
Published figures vary widely by task type and model. A 23% rate (as described in the case study for
a naive ReAct loop) is plausible; 45–60% is not.

### Impact on Implementation
Do not use 45–60% as a baseline in evaluation reports or optimization logs. The FA-5 target of 0
hallucinations remains the goal; the pre-redesign baseline of ~23% is a reasonable starting comparison.

---

## ERROR 006 — OpenAI text-embedding-3-large: Incorrect Native Dimension Count

**Location:** Part E, Section E2.2 — "Recommended Embedding Models"  
**Page:** 62  
**Error Type:** Technical specification error  
**Severity:** Medium — affects vector database schema design and retrieval quality decisions

### Erroneous Claim
> "OpenAI text-embedding-3-large: **1024 dimensions**, $0.13 per million tokens."

### Why This Is Wrong
OpenAI's `text-embedding-3-large` model has a native embedding dimension of **3072**, not 1024.

The model does support **dimension reduction** — you can request output at 256, 1024, or 3072
dimensions using the `dimensions` parameter in the API call. However:

- The **native/default** dimension is 3072
- Stating "1024 dimensions" as the model's dimension count is incorrect
- The comparison to `text-embedding-3-small` (1536 dimensions native) also makes the 1024 figure
  suspicious — it would mean the "large" model has *fewer* dimensions than the "small" one by default,
  which makes no sense architecturally

For context:
| Model | Native Dimensions | Supported Reduction |
|-------|------------------|---------------------|
| text-embedding-3-small | 1536 | 512, 1536 |
| text-embedding-3-large | **3072** | 256, 1024, 3072 |

### Correct Statement
> "OpenAI text-embedding-3-large: **3072 dimensions** natively (reducible to 256 or 1024),
> $0.13 per million tokens."

### Impact on Implementation
The vector database schema in `memory/vector_store.py` must be initialized with the correct dimension
count. Creating a Chroma or Pinecone index with `dimensions=1024` when using text-embedding-3-large at
default settings will cause dimension mismatch errors at insertion time.

```python
# WRONG (based on document error):
collection = chroma_client.create_collection(
    name="ara1_memory",
    metadata={"hnsw:space": "cosine"},
    embedding_function=OpenAIEmbeddingFunction(
        model_name="text-embedding-3-large",
        dimensions=1024  # ERROR: This requests reduced dimensions
    )
)

# CORRECT (default native dimensions):
collection = chroma_client.create_collection(
    name="ara1_memory",
    metadata={"hnsw:space": "cosine"},
    embedding_function=OpenAIEmbeddingFunction(
        model_name="text-embedding-3-large",
        dimensions=3072  # Native dimension count
    )
)
```

---

## ERROR 007 — Scoring Matrix: Description Says "Five Dimensions" but Table Has Four

**Location:** Part B, Section B3.1 — "Scoring Matrix"  
**Page:** 33  
**Error Type:** Internal inconsistency / logical error  
**Severity:** Low — does not affect implementation, but tests reading comprehension

### Erroneous Claim
> "Each challenge is scored across **five dimensions**."

The table that immediately follows shows only **four** scoring columns:
- Accuracy
- Completeness
- Insight
- Architecture

### Why This Is Wrong
Counting the columns in the scoring table confirms there are four dimensions, not five. The row totals
are internally consistent with four columns (e.g., C1: 20+15+10+5 = 50 ✓), confirming the table
itself is correct and the prose description is wrong.

This is distinct from the Part F submission rubric (which genuinely has seven dimensions for the
overall project assessment). The B3.1 in-simulation scoring table has four dimensions.

### Correct Statement
> "Each challenge is scored across **four dimensions**: Accuracy, Completeness, Insight, and Architecture."

### Impact on Implementation
The evaluation dashboard (`evaluation/dashboard.py`) should display four simulation scoring dimensions,
not five. The total of 700 points is correctly computed from four dimensions.

---

## Summary Table

| # | Location | Error Type | Correct Version | Severity |
|---|----------|-----------|-----------------|----------|
| 001 | A6.2 Source Hierarchy | Tiers 4 & 5 swapped | News > Social Media | High |
| 002 | A5.2 Metric AB-4 | Multiply → should be divide | `hits / total_calls` | High |
| 003 | A7.3 SCAP | Wrong year (2007→2009) + wrong act | SCAP=2009, pre-Dodd-Frank | Medium |
| 004 | C4.2 Indian Filing | Form 20-F is an SEC form | MCA uses AOC-4 / MGT-7 | Medium |
| 005 | C3.2 Hallucination % | 45-60% industry average | Unsupported/implausible | Medium |
| 006 | E2.2 Embedding dims | text-embedding-3-large = 1024 | Native = 3072 | Medium |
| 007 | B3.1 Scoring dims | "five dimensions" | Four columns in table | Low |

---

## Detection Methodology

Errors were identified using the following first-principles approach:

1. **Internal consistency checks** — comparing the document's own definitions against its formulas and
   examples (caught Errors 002, 007)
2. **Domain knowledge verification** — cross-referencing financial and regulatory facts against known
   history (caught Errors 003, 004)
3. **Plausibility filtering** — flagging statistics that are implausibly large or small and lack citations
   (caught Error 005)
4. **Technical specification lookup** — verifying API/model specs against known documentation
   (caught Error 006)
5. **Logical hierarchy analysis** — checking that ordered lists are internally consistent with their own
   descriptions (caught Error 001)

---

*Document: ERROR_LOG.md | Project: Project1A-EyobKebede-AutonomousFinancialResearchAgent | Version: 1.0*
