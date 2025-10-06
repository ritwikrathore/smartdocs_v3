# Reranker Fallback Mechanism

## Overview

The application now includes an automatic fallback mechanism for the reranker component. When the Databricks reranker API is unavailable (due to timeout, 403 errors, or other issues), the system automatically falls back to an LLM-based reranker that provides the same functionality.

## How It Works

### Startup Process

1. **Primary Attempt**: The system first attempts to load the Databricks reranker API
   - Timeout: 60 seconds (configurable)
   - Tests the API with a sample query-document pair
   - Validates the response format and scores

2. **Fallback Trigger**: If the primary reranker fails due to:
   - HTTP 403 (Unauthorized) errors
   - Timeout after 60 seconds
   - Network errors
   - Invalid API responses
   
3. **Automatic Fallback**: The system automatically loads the LLM-based reranker
   - Uses the same Databricks LLM endpoint
   - Maintains the same interface as the API reranker
   - Returns scores in the same format (numpy array, 0-1 range)

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Startup                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│         load_databricks_reranker_model()                     │
│         (with 60-second timeout)                             │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
            ┌───────────┐       ┌──────────────┐
            │  Success  │       │    Failure   │
            └───────────┘       └──────────────┘
                    │                   │
                    │                   ▼
                    │       ┌──────────────────────────┐
                    │       │ ENABLE_LLM_RERANKER_     │
                    │       │ FALLBACK = True?         │
                    │       └──────────────────────────┘
                    │                   │
                    │           ┌───────┴────────┐
                    │           │                │
                    │           ▼                ▼
                    │       ┌──────┐        ┌──────┐
                    │       │ Yes  │        │  No  │
                    │       └──────┘        └──────┘
                    │           │                │
                    │           ▼                ▼
                    │   ┌──────────────┐   ┌──────────┐
                    │   │ Load LLM     │   │  Return  │
                    │   │ Reranker     │   │  None    │
                    │   └──────────────┘   └──────────┘
                    │           │                │
                    └───────────┴────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Reranker Available   │
                    │  (API or LLM-based)   │
                    └───────────────────────┘
```

## Configuration

### Config Settings (`src/keyword_code/config.py`)

```python
# Enable/disable Databricks reranker
USE_DATABRICKS_RERANKER = True

# Timeout for reranker API at startup (seconds)
RERANKER_API_TIMEOUT = 60

# Enable automatic fallback to LLM-based reranker
ENABLE_LLM_RERANKER_FALLBACK = True

# Maximum tokens for reranker input
RERANKER_MAX_TOKENS = 512
```

### Customization

To adjust the timeout:
```python
# In config.py
RERANKER_API_TIMEOUT = 30  # Reduce to 30 seconds
```

To disable fallback:
```python
# In config.py
ENABLE_LLM_RERANKER_FALLBACK = False  # No fallback, return None on failure
```

## LLM-Based Reranker Details

### How It Works

The LLM-based reranker (`src/keyword_code/models/llm_reranker.py`) uses the Databricks LLM to score query-document pairs:

1. **Batch Scoring**: ALL pairs for a subprompt are scored in a SINGLE LLM call for maximum efficiency
   - Example: 15 chunks → 1 API call (not 15 calls!)
   - Typical speedup: 10-15x faster than individual calls

2. **Score Normalization**: All scores are clamped to the [0.0, 1.0] range

3. **No Truncation**: Unlike the API reranker (512 token limit), the LLM reranker uses full document text without truncation, leveraging the LLM's large context window (~8K tokens = ~32K characters)

4. **Fallback**: If batch scoring fails, falls back to individual scoring (rare)

### Prompt Template

```
You are a relevance scoring system. Given a query and a document, 
score how relevant the document is to answering the query.

Query: {query}

Document: {document}

Provide a relevance score between 0.0 and 1.0, where:
- 0.0 = completely irrelevant
- 0.5 = somewhat relevant
- 1.0 = highly relevant and directly answers the query

Respond with ONLY a single number between 0.0 and 1.0, nothing else.
```

### Performance Characteristics

| Aspect | Databricks Reranker API | LLM-Based Fallback |
|--------|------------------------|-------------------|
| Speed | Fast (~100ms for 10 pairs) | Moderate (~2-3s for 15 pairs) |
| Accuracy | High (specialized model) | Good (general LLM) |
| Cost | Lower (dedicated endpoint) | Higher (LLM tokens) |
| Availability | Depends on API | Depends on LLM API |
| Batch Size | Unlimited | All pairs in 1 call |
| API Calls | 1 per batch | 1 per subprompt |
| Token Limit | 512 tokens (truncated) | ~8K tokens (no truncation) |
| Context Window | Limited | Large (~32K characters) |

## Error Handling

### Startup Errors

The system handles various error scenarios:

1. **403 Unauthorized**
   ```
   ERROR - Databricks reranker API error: 403, {"error_code":"403","message":"Unauthorized access..."}
   WARNING - Attempting to load LLM-based fallback reranker...
   INFO - ✓ Successfully loaded LLM-based fallback reranker
   ```

2. **Timeout**
   ```
   ERROR - ✗ Databricks reranker API timeout after 60 seconds at startup
   WARNING - Attempting to load LLM-based fallback reranker...
   INFO - ✓ Successfully loaded LLM-based fallback reranker
   ```

3. **Network Error**
   ```
   ERROR - ✗ Databricks reranker API request error at startup: Connection refused
   WARNING - Attempting to load LLM-based fallback reranker...
   INFO - ✓ Successfully loaded LLM-based fallback reranker
   ```

### Runtime Behavior

- **Timeout Only at Startup**: The 60-second timeout is only applied during startup testing
- **Runtime Errors**: During normal operation, API errors return zero scores (existing behavior)
- **Seamless Integration**: The rest of the application doesn't need to know which reranker is being used

## Logging

### Startup Logs

When the reranker loads successfully:
```
============================================================
RERANKER INITIALIZATION
============================================================
INFO - Attempting to load Databricks reranker model...
INFO - Loading Databricks reranker model: cpm-marco-ms
INFO - Using max token length: 512
INFO - API timeout set to: 60 seconds
INFO - Testing Databricks reranker API connection...
INFO - ✓ Databricks reranker model loaded successfully with score: 0.8234
INFO - ✓ Databricks reranker API loaded successfully
============================================================
```

When fallback is triggered:
```
============================================================
RERANKER INITIALIZATION
============================================================
INFO - Attempting to load Databricks reranker model...
ERROR - ✗ Databricks reranker API timeout after 60 seconds at startup
WARNING - Attempting to load LLM-based fallback reranker...
INFO - Loading LLM-based fallback reranker model
INFO - LLM-based fallback reranker initialized successfully
INFO - LLM fallback reranker scoring 2 pairs
INFO - LLM fallback reranker completed scoring with mean score: 0.7500
INFO - ✓ LLM fallback reranker loaded successfully. Test scores: [0.85 0.65]
INFO - ✓ LLM-based fallback reranker loaded successfully
INFO - ✓ LLM-based fallback reranker loaded successfully
============================================================
```

## Testing

### Manual Testing

To test the fallback mechanism:

1. **Simulate API Failure**: Temporarily modify the API endpoint to an invalid URL
   ```python
   # In databricks_reranker.py
   DATABRICKS_RERANKER_ENDPOINT = "https://invalid-endpoint.com/test"
   ```

2. **Restart the Application**: The fallback should trigger automatically

3. **Check Logs**: Look for the fallback messages in the logs

### Automated Testing

```python
# Test the LLM reranker directly
from src.keyword_code.models.llm_reranker import load_llm_reranker_model

model = load_llm_reranker_model()
test_pairs = [
    ["What is Python?", "Python is a programming language."],
    ["What is Python?", "The sky is blue."]
]
scores = model.predict(test_pairs)
print(f"Scores: {scores}")  # Should show higher score for first pair
```

## Troubleshooting

### Issue: Both rerankers fail

**Symptoms**: Logs show both API and LLM reranker failures

**Solution**: 
1. Check `DATABRICKS_API_KEY` in `.env` file
2. Verify network connectivity
3. Check LLM endpoint availability

### Issue: Fallback is slow

**Symptoms**: Reranking takes significantly longer than expected

**Solution**:
1. This is expected behavior - LLM-based reranking is slower
2. Consider increasing `RERANKER_API_TIMEOUT` to give API more time
3. Check if API is intermittently available

### Issue: Fallback not triggering

**Symptoms**: Application fails without attempting fallback

**Solution**:
1. Verify `ENABLE_LLM_RERANKER_FALLBACK = True` in config
2. Check logs for specific error messages
3. Ensure LLM client is properly configured

## Best Practices

1. **Monitor Startup Logs**: Always check which reranker loaded successfully
2. **Set Appropriate Timeout**: Balance between waiting for API and quick fallback
3. **Test Fallback Regularly**: Ensure the LLM reranker works in your environment
4. **Consider Costs**: LLM-based reranking uses more tokens and may be more expensive
5. **Performance Testing**: Test with your typical workload to understand performance impact

## Future Enhancements

Potential improvements to consider:

1. **Caching**: Cache LLM reranker scores for repeated queries
2. **Hybrid Approach**: Use API when available, cache LLM scores as backup
3. **Adaptive Timeout**: Adjust timeout based on historical API response times
4. **Health Checks**: Periodic checks to switch back to API when it becomes available
5. **Metrics**: Track which reranker is being used and performance metrics

