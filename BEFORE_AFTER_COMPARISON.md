# Before vs After: RAG Consolidation

## Architecture Comparison

### BEFORE: Separate RAG Implementations

```
┌─────────────────────────────────────────────────────────────┐
│ Initial RAG (rag/retrieval.py)                              │
│ - Fixed weights: BM25=0.5, Semantic=0.5                     │
│ - No intelligence                                            │
│ - Used for all initial retrievals                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ User sees results                                            │
│ - May not be optimal for query type                         │
│ - User clicks "Retry" if unsatisfied                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ RAG Agent (agents/rag_agent.py)                             │
│ - Analyzes query type                                        │
│ - Recommends optimal weights                                │
│ - Only used for retry                                        │
└─────────────────────────────────────────────────────────────┘
```

**Problems:**
- ❌ Initial retrieval uses suboptimal fixed weights
- ❌ Requires user to manually retry for better results
- ❌ RAG agent intelligence only available after retry
- ❌ Two separate implementations doing similar things

---

### AFTER: Unified RAG with Intelligent Decomposition

```
┌─────────────────────────────────────────────────────────────┐
│ Decomposition LLM (ai/decomposition.py)                     │
│ - Analyzes query type for each sub-prompt                   │
│ - Recommends optimal BM25/semantic weights                  │
│ - Provides reasoning for weight selection                   │
│ - Single LLM call for decomposition + optimization          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Initial RAG (rag/retrieval.py)                              │
│ - Uses optimized weights from decomposition                 │
│ - MT599 Swift: BM25=0.8, Semantic=0.2                       │
│ - Legal: BM25=0.5, Semantic=0.5                             │
│ - Conceptual: BM25=0.3, Semantic=0.7                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ User sees optimized results                                  │
│ - Better results from the start                             │
│ - Retry still available if needed                           │
└─────────────────────────────────────────────────────────────┘
                            ↓ (optional)
┌─────────────────────────────────────────────────────────────┐
│ RAG Agent (agents/rag_agent.py)                             │
│ - Still available for retry                                 │
│ - Provides second opinion                                   │
│ - Can adjust based on actual results                        │
└─────────────────────────────────────────────────────────────┘
```

**Benefits:**
- ✅ Optimal retrieval from the start
- ✅ Single LLM call for decomposition + optimization
- ✅ Retry still available for edge cases
- ✅ Comprehensive logging of all decisions

---

## Code Comparison

### Decomposition Output

#### BEFORE
```json
{
  "decomposition": [
    {
      "title": "MT599 Swift Format",
      "sub_prompt": "What is the MT599 Swift message format?"
    }
  ]
}
```

#### AFTER
```json
{
  "decomposition": [
    {
      "title": "MT599 Swift Format",
      "sub_prompt": "What is the MT599 Swift message format?",
      "rag_params": {
        "bm25_weight": 0.8,
        "semantic_weight": 0.2,
        "reasoning": "MT599 Swift is highly specific terminology requiring exact keyword matching"
      }
    }
  ]
}
```

---

### Retrieval Call

#### BEFORE
```python
relevant_chunks = retrieve_relevant_chunks(
    prompt=sub_prompt,
    chunks=chunks,
    model=embedding_model,
    top_k=RAG_TOP_K,
    reranker_model=reranker_model
    # No weights specified - uses default 0.5/0.5
)
```

#### AFTER
```python
# Extract optimized weights from decomposition
rag_params = sub_prompt_data.get("rag_params", {})
bm25_weight = rag_params.get("bm25_weight", 0.5)
semantic_weight = rag_params.get("semantic_weight", 0.5)

# Log the weights and reasoning
logger.info(f"Using RAG weights - BM25: {bm25_weight:.2f}, Semantic: {semantic_weight:.2f}")
log_rag_parameters(sub_prompt_title, sub_prompt, bm25_weight, semantic_weight, reasoning, "decomposition")

relevant_chunks = retrieve_relevant_chunks(
    prompt=sub_prompt,
    chunks=chunks,
    model=embedding_model,
    top_k=RAG_TOP_K,
    reranker_model=reranker_model,
    bm25_weight=bm25_weight,        # Optimized weight
    semantic_weight=semantic_weight  # Optimized weight
)
```

---

## Logging Comparison

### BEFORE
```
INFO - Decomposing prompt: 'What is the MT599 Swift...'
INFO - Successfully decomposed prompt into 2 sub-prompts with titles.
INFO - Retrieving relevant chunks for sub-prompt 'MT599 Swift Format'
INFO - Retrieved 10 chunks for sub-prompt 'MT599 Swift Format'
```

### AFTER
```
INFO - Decomposing prompt with RAG optimization: 'What is the MT599 Swift...'
INFO - RAG params for 'MT599 Swift Format': BM25=0.80, Semantic=0.20, Reasoning: MT599 Swift is highly specific terminology requiring exact keyword matching
INFO - Successfully decomposed prompt into 2 sub-prompts with titles and RAG parameters.
INFO - Retrieving relevant chunks for sub-prompt 'MT599 Swift Format'
INFO - Using RAG weights - BM25: 0.80, Semantic: 0.20
INFO - RAG weight reasoning: MT599 Swift is highly specific terminology requiring exact keyword matching
INFO - RAG: Running hybrid search with weights - BM25: 0.80, Semantic: 0.20
INFO - Retrieved 10 chunks for sub-prompt 'MT599 Swift Format' using optimized RAG weights

# Plus structured JSON log:
{
  "timestamp": "2025-10-06T10:30:45.123456",
  "type": "rag_parameters",
  "source": "decomposition",
  "sub_prompt_title": "MT599 Swift Format",
  "sub_prompt": "What is the MT599 Swift message format?",
  "bm25_weight": 0.8,
  "semantic_weight": 0.2,
  "reasoning": "MT599 Swift is highly specific terminology requiring exact keyword matching"
}
```

---

## Performance Comparison

### BEFORE
- **LLM Calls per Query:** 1 (decomposition) + N (retry if needed)
- **Initial Retrieval Quality:** Suboptimal (fixed 0.5/0.5 weights)
- **User Experience:** May need to retry for better results
- **Optimization Available:** Only after manual retry

### AFTER
- **LLM Calls per Query:** 1 (decomposition with optimization) + N (retry if needed)
- **Initial Retrieval Quality:** Optimized (query-specific weights)
- **User Experience:** Better results from the start
- **Optimization Available:** Immediately, with retry as backup

---

## Query Type Examples

| Query Type | Before (Fixed) | After (Optimized) | Reasoning |
|------------|---------------|-------------------|-----------|
| MT599 Swift | BM25: 0.5, Sem: 0.5 | BM25: 0.8, Sem: 0.2 | Exact terminology needs keyword matching |
| Loan Amount | BM25: 0.5, Sem: 0.5 | BM25: 0.6, Sem: 0.4 | Numerical data benefits from keywords |
| Legal Concept | BM25: 0.5, Sem: 0.5 | BM25: 0.3, Sem: 0.7 | Conceptual queries need semantic understanding |
| Interest Rate | BM25: 0.5, Sem: 0.5 | BM25: 0.6, Sem: 0.4 | Financial terms benefit from keyword precision |
| General Question | BM25: 0.5, Sem: 0.5 | BM25: 0.5, Sem: 0.5 | Balanced approach for general queries |

---

## Summary

The consolidation successfully:
1. ✅ Unified RAG intelligence into decomposition step
2. ✅ Provides optimal weights from the start
3. ✅ Maintains retry functionality for edge cases
4. ✅ Adds comprehensive logging for all decisions
5. ✅ Reduces need for manual retries
6. ✅ Improves initial result quality
7. ✅ No UI changes required
8. ✅ Backward compatible with fallbacks

