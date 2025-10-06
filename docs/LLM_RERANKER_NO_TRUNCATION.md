# LLM Reranker - No Truncation Policy

## Overview

The LLM-based fallback reranker **does not truncate document text** by default, unlike the Databricks reranker API which has a hard 512-token limit.

## Why No Truncation?

### Context Window Comparison

| Reranker Type | Token Limit | Character Limit | Truncation |
|---------------|-------------|-----------------|------------|
| **Databricks API** | 512 tokens | ~2,000 chars | **Required** |
| **LLM Fallback** | ~8,000 tokens | ~32,000 chars | **Not needed** |

### Key Differences

1. **API Reranker (512 tokens)**
   - Uses specialized cross-encoder model
   - Hard limit of 512 tokens
   - Must truncate query + document to fit
   - Uses BERT tokenizer for accurate token counting
   - Truncation is critical for functionality

2. **LLM Reranker (~8K tokens)**
   - Uses general-purpose LLM (databricks-llama-4-maverick)
   - Large context window (~8,000 tokens)
   - Can handle full document text
   - No truncation needed for typical documents
   - Better understanding of full context

## Implementation Details

### Default Behavior (No Truncation)

```python
# Initialize without truncation (default)
reranker = LLMRerankerModel()  # max_length=None

# Documents are used in full
pairs = [
    ["query", "This is a very long document with thousands of characters..."]
]
scores = reranker.predict(pairs)  # Uses full document text
```

### Optional Truncation (If Needed)

```python
# Initialize with explicit truncation (not recommended)
reranker = LLMRerankerModel(max_length=5000)  # Truncate at 5000 chars

# Documents longer than 5000 chars will be truncated
pairs = [
    ["query", "Very long document..."]
]
scores = reranker.predict(pairs)  # Truncates if > 5000 chars
```

## Code Changes

### Before (Aggressive Truncation)

```python
def __init__(self, max_length: int = 2000):
    """Initialize with 2000 char limit."""
    self.max_length = max_length  # Always truncate at 2000 chars
```

### After (No Truncation by Default)

```python
def __init__(self, max_length: int = None):
    """Initialize without truncation (recommended)."""
    self.max_length = max_length  # None = no truncation
```

## Benefits of No Truncation

### 1. Better Accuracy
- LLM sees full document context
- No loss of important information
- Better relevance scoring

### 2. Simpler Logic
- No need to decide what to truncate
- No risk of cutting off critical content
- More predictable behavior

### 3. Consistency
- Same document text used for scoring
- No variation based on truncation strategy
- Reproducible results

### 4. Flexibility
- Works with documents of any size (up to ~32K chars)
- No need to tune truncation parameters
- Handles edge cases naturally

## When Truncation Might Be Needed

Truncation is only needed in rare cases:

### 1. Extremely Long Documents (>30K characters)
```python
# For documents that exceed LLM context window
reranker = LLMRerankerModel(max_length=30000)
```

### 2. Cost Optimization
```python
# To reduce token usage and costs
reranker = LLMRerankerModel(max_length=10000)
```

### 3. Performance Optimization
```python
# To speed up processing (shorter prompts)
reranker = LLMRerankerModel(max_length=5000)
```

## Typical Document Sizes

For reference, here are typical document sizes in the application:

| Document Type | Typical Size | Needs Truncation? |
|---------------|--------------|-------------------|
| Chunk from RAG | 500-2000 chars | ❌ No |
| Full paragraph | 1000-3000 chars | ❌ No |
| Multiple paragraphs | 3000-10000 chars | ❌ No |
| Full document section | 10000-30000 chars | ❌ No |
| Entire document | 30000+ chars | ⚠️ Maybe |

**Conclusion**: For 99% of use cases, no truncation is needed.

## Logging

### With No Truncation (Default)
```
INFO - LLM-based fallback reranker initialized (no truncation)
INFO - LLM fallback reranker scoring 10 pairs
INFO - LLM fallback reranker completed scoring with mean score: 0.7234
```

### With Truncation (If Configured)
```
INFO - LLM-based fallback reranker initialized with max_length=5000
WARNING - Truncating document from 8500 to 5000 characters
INFO - LLM fallback reranker scoring 10 pairs
INFO - LLM fallback reranker completed scoring with mean score: 0.6891
```

## Comparison with API Reranker

### API Reranker (Always Truncates)
```python
# API reranker MUST truncate
model = DatabricksRerankerModel()  # max_length=512 tokens

# Example: Long document
query = "What is machine learning?"
document = "Machine learning is..." * 1000  # Very long

# Truncation happens automatically
truncated_query, truncated_doc = model._truncate_text_pair(query, document)
# Result: Both query and document truncated to fit 512 tokens
```

### LLM Reranker (No Truncation)
```python
# LLM reranker uses full text
model = LLMRerankerModel()  # max_length=None

# Same example
query = "What is machine learning?"
document = "Machine learning is..." * 1000  # Very long

# No truncation - uses full text
score = model._score_single_pair(query, document)
# Result: Full document text sent to LLM
```

## Performance Impact

### Token Usage

| Scenario | API Reranker | LLM Reranker | Difference |
|----------|--------------|--------------|------------|
| Short doc (500 chars) | ~125 tokens | ~125 tokens | Same |
| Medium doc (2000 chars) | 512 tokens (truncated) | ~500 tokens | LLM uses more |
| Long doc (10000 chars) | 512 tokens (truncated) | ~2500 tokens | LLM uses 5x more |

### Cost Implications

- **API Reranker**: Fixed cost per call (regardless of document size)
- **LLM Reranker**: Variable cost based on token usage
- **Impact**: LLM reranker may cost more for long documents, but provides better accuracy

### Speed Impact

- **API Reranker**: Fast (~100ms for 10 pairs)
- **LLM Reranker**: Slower (~2-5s for 10 pairs)
- **Impact**: No truncation doesn't significantly affect speed (LLM is already slower)

## Best Practices

### ✅ Recommended

1. **Use default (no truncation)** for best accuracy
   ```python
   reranker = LLMRerankerModel()  # No max_length
   ```

2. **Let the LLM handle full context** - it's designed for it
   ```python
   # Trust the LLM's large context window
   scores = reranker.predict(pairs)  # Full documents
   ```

3. **Monitor token usage** if cost is a concern
   ```python
   # Track costs in production
   logger.info(f"Scored {len(pairs)} pairs with full documents")
   ```

### ❌ Not Recommended

1. **Don't truncate unnecessarily**
   ```python
   # Avoid this unless you have a specific reason
   reranker = LLMRerankerModel(max_length=2000)
   ```

2. **Don't use aggressive truncation**
   ```python
   # This defeats the purpose of using LLM
   reranker = LLMRerankerModel(max_length=500)
   ```

3. **Don't truncate to match API reranker**
   ```python
   # The LLM can handle more - use it!
   reranker = LLMRerankerModel(max_length=512)  # Too restrictive
   ```

## Migration Notes

If you were previously using the LLM reranker with truncation:

### Before
```python
# Old code with aggressive truncation
reranker = LLMRerankerModel(max_length=2000)
```

### After
```python
# New code without truncation (better accuracy)
reranker = LLMRerankerModel()  # max_length=None (default)
```

**Impact**: 
- ✅ Better accuracy (full context)
- ✅ Simpler code (no truncation logic)
- ⚠️ Slightly higher token usage
- ⚠️ Slightly higher cost (but better results)

## Summary

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Default Behavior** | No truncation | LLM has large context window |
| **Token Limit** | ~8K tokens | Sufficient for most documents |
| **Character Limit** | ~32K chars | Handles typical use cases |
| **Truncation Option** | Available | For edge cases only |
| **Recommendation** | Use default | Best accuracy and simplicity |

## Conclusion

The LLM-based fallback reranker **does not truncate document text by default**, leveraging the LLM's large context window (~8K tokens) to provide better accuracy and simpler implementation. Truncation is only needed for extremely long documents (>30K characters) or specific cost/performance optimization scenarios.

**Key Takeaway**: Trust the LLM's large context window and use full document text for best results! 🎯

