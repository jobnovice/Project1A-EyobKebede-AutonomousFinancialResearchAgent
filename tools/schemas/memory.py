"""
tools/schemas/memory.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

JSON schemas for Group 3: Memory Tools (2 tools).

These tools implement the Memory First principle — vector_db_search
MUST be called before any external API call to check for cached findings.
vector_db_store MUST be called at session end to persist all verified findings.
"""

# ---------------------------------------------------------------------------
# Tool 11: vector_db_search
# ---------------------------------------------------------------------------
VECTOR_DB_SEARCH: dict = {
    "name": "vector_db_search",
    "description": (
        "Search the agent's long-term vector memory (ChromaDB / Pinecone) for previously "
        "researched information using semantic similarity. THIS IS THE FIRST TOOL TO CALL "
        "for every research sub-task before making any external API call — this is the "
        "Memory First principle. A cache hit avoids an external API call, reducing cost, "
        "latency, and rate-limit exposure. A hit contributes to metric AB-4 "
        "(Memory Utilization ≥ 0.3). Returns document chunks with similarity scores "
        "so you can assess relevance before using the content."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural language search query describing the information you're looking for. "
                    "Be specific: include company name, metric type, and time period. "
                    "Example: 'Tesla operating margin risk factors FY2024' or "
                    "'Apple revenue iPhone segment Q3 2024'."
                ),
            },
            "top_k": {
                "type": "integer",
                "description": (
                    "Number of most-similar document chunks to return (1-20). "
                    "Default is 5. Use higher values (10-15) for broad queries "
                    "where multiple relevant chunks may exist."
                ),
                "minimum": 1,
                "maximum": 20,
                "default": 5,
            },
            "filter": {
                "type": "object",
                "description": (
                    "Optional metadata filter to narrow the search scope. "
                    "Supported filter keys: "
                    "'ticker' (string): limit to a specific company. "
                    "'source_type' (string): one of '10-K', '10-Q', '8-K', "
                    "'earnings_call', 'news', 'analysis'. "
                    "'date_after' (string, ISO format): only chunks from after this date. "
                    "'date_before' (string, ISO format): only chunks from before this date. "
                    "'verified' (boolean): only return fact-checked chunks. "
                    "Example: {'ticker': 'TSLA', 'source_type': '10-K', 'verified': true}"
                ),
                "properties": {
                    "ticker": {"type": "string"},
                    "source_type": {
                        "type": "string",
                        "enum": ["10-K", "10-Q", "8-K", "earnings_call", "news", "analysis"],
                    },
                    "date_after": {"type": "string", "format": "date"},
                    "date_before": {"type": "string", "format": "date"},
                    "verified": {"type": "boolean"},
                    "session_id": {"type": "string"},
                },
                "additionalProperties": False,
            },
            "min_similarity": {
                "type": "number",
                "description": (
                    "Minimum cosine similarity threshold (0.0-1.0) for a result to be "
                    "returned. Default is 0.70. Lower values return more results but "
                    "with higher chance of irrelevance. Do not go below 0.50."
                ),
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.70,
            },
        },
        "required": ["query"],
    },
}

# ---------------------------------------------------------------------------
# Tool 12: vector_db_store
# ---------------------------------------------------------------------------
VECTOR_DB_STORE: dict = {
    "name": "vector_db_store",
    "description": (
        "Store verified research findings in the agent's long-term vector memory "
        "for retrieval in future sessions. Call this tool at the END of every "
        "successful research step after facts have been verified via fact_checker. "
        "Do NOT store unverified claims — only verified=True content enters long-term "
        "memory. Storing findings now enables Challenge 7 (sector analysis with memory) "
        "to retrieve cross-session knowledge and contributes to metric AB-4. "
        "Each call stores one semantically coherent chunk (one risk factor, one "
        "financial metric set, one management quote — not entire documents)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": (
                    "The text content to embed and store. Should be one semantically "
                    "coherent chunk: a single risk factor, a financial metric with context, "
                    "or a management quote with attribution. "
                    "Target length: 100-500 words. Do not store raw API response dumps."
                ),
            },
            "metadata": {
                "type": "object",
                "description": (
                    "Metadata tags for future filtering and retrieval. "
                    "All fields are used by vector_db_search filters."
                ),
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Company ticker symbol this content is about.",
                    },
                    "source_type": {
                        "type": "string",
                        "enum": ["10-K", "10-Q", "8-K", "earnings_call", "news", "analysis"],
                        "description": "The type of source document this content came from.",
                    },
                    "date": {
                        "type": "string",
                        "format": "date",
                        "description": "Date of the source document (ISO format: YYYY-MM-DD).",
                    },
                    "section": {
                        "type": "string",
                        "description": (
                            "Section of the source document (e.g., 'Risk Factors', "
                            "'MD&A', 'Financial Statements', 'Q&A Session')."
                        ),
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score from fact_checker (0.0-1.0).",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "verified": {
                        "type": "boolean",
                        "description": (
                            "Whether this content has been cross-referenced by fact_checker. "
                            "Only store verified=true content in long-term memory."
                        ),
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Research session ID that produced this content.",
                    },
                    "chunk_index": {
                        "type": "integer",
                        "description": "Position of this chunk within its source document.",
                    },
                },
                "required": ["ticker", "source_type", "date", "verified"],
                "additionalProperties": False,
            },
            "document_id": {
                "type": "string",
                "description": (
                    "Optional explicit document ID in the format "
                    "'[ticker]-[source_type]-[date]-[chunk_index]-[topic]'. "
                    "Example: 'tsla-10k-2024-003-supply-chain-risk'. "
                    "Auto-generated if not provided."
                ),
            },
        },
        "required": ["content", "metadata"],
    },
}

# ---------------------------------------------------------------------------
# Export: all Group 3 schemas as an ordered list
# ---------------------------------------------------------------------------
MEMORY_SCHEMAS: list[dict] = [
    VECTOR_DB_SEARCH,
    VECTOR_DB_STORE,
]
