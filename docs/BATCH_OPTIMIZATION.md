# LLM Reranker - Batch Optimization

## Overview

The LLM-based fallback reranker uses **batch optimization** to score ALL query-document pairs in a **single API call** per subprompt, resulting in 10-15x performance improvement.

## The Problem

### Naive Approach: Individual API Calls

```python
# BAD: Making individual API calls for each chunk
for chunk in chunks:  # 15 chunks
    score = llm.score(query, chunk)  # 15 API calls!
    scores.append(score)

# Result: 15 API calls, ~15-30 seconds total
```

## The Solution: Batch Scoring

### Optimized Approach: Single API Call

```python
# GOOD: Batch all pairs into one API call
scores = llm.score_batch(all_pairs)  # 1 API call!

# Result: 1 API call, ~2-3 seconds total
```

**Speedup: ~10-15x faster!** 🚀

## Performance Comparison

| Approach | API Calls | Time (15 chunks) | Speedup |
|----------|-----------|------------------|---------|
| Individual | 15 | ~15-30s | 1x |
| Batch | 1 | ~2-3s | **10-15x** |

## Real-World Example

### Scenario: 3 Subprompts, 15 Chunks Each

**Without Batch Optimization:**
```
Subprompt 1: 15 calls × 2s = 30s
Subprompt 2: 15 calls × 2s = 30s
Subprompt 3: 15 calls × 2s = 30s
Total: 45 calls, 90 seconds
```

**With Batch Optimization:**
```
Subprompt 1: 1 call × 3s = 3s
Subprompt 2: 1 call × 3s = 3s
Subprompt 3: 1 call × 3s = 3s
Total: 3 calls, 9 seconds
```

**Result: 10x faster! (90s → 9s)** 🎉

## Implementation

The batch optimization is automatic and transparent:

```python
# Just call predict() - batching happens automatically
reranker = LLMRerankerModel()
scores = reranker.predict(pairs)  # All pairs scored in 1 call
```

## Context Window Utilization

For 15 pairs with ~300 char documents:

| Component | Tokens | % of 8K |
|-----------|--------|---------|
| Prompts | ~250 | 3% |
| Documents | ~1,125 | 14% |
| Response | ~60 | 1% |
| **Total** | **~1,485** | **19%** |

**Conclusion**: We use only ~19% of context window, so 15-20 pairs fit easily!

## Comparison with API Reranker

| Reranker | API Calls | Time (15 pairs) |
|----------|-----------|-----------------|
| Databricks API | 1 | ~100ms |
| LLM Fallback | 1 | ~2-3s |

**Both use 1 API call per subprompt!** The difference is speed, not efficiency.

## Summary

- ✅ **10-15x speedup** vs individual calls
- ✅ **1 API call per subprompt** (not per chunk)
- ✅ **Automatic** - no code changes needed
- ✅ **Efficient** - uses ~19% of context window

**Key Takeaway**: The LLM fallback reranker makes the same number of API calls as the Databricks API reranker! 🚀

