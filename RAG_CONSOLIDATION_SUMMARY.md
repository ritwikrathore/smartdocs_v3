# RAG Consolidation Implementation Summary

## Overview
Successfully implemented **Option 1**: Enhanced the decomposition LLM to include optimal RAG parameters (BM25/semantic weights) for each sub-prompt. This consolidates the RAG agent intelligence into the initial decomposition step, making optimal retrieval available from the start while keeping the retry functionality for user-initiated re-retrieval.

## Changes Made

### 1. Enhanced Decomposition (`src/keyword_code/ai/decomposition.py`)

#### Updated System Prompt
- Added RAG parameter guidelines to the system prompt
- LLM now analyzes each sub-prompt and recommends optimal BM25/semantic weights
- Provides reasoning for weight selection based on query type

#### Query Type Guidelines Added
- **Keyword-based queries** (MT599 Swift, codes, exact terms): BM25 weight 0.7-0.8
- **General legal/financial queries**: Balanced weights (0.5/0.5)
- **Conceptual/interpretive queries**: Semantic weight 0.6-0.7
- **Technical terminology**: BM25 weight ~0.6

#### Updated Output Schema
Each decomposed sub-prompt now includes:
```json
{
  "title": "Sub-prompt Title",
  "sub_prompt": "The actual question",
  "rag_params": {
    "bm25_weight": 0.75,
    "semantic_weight": 0.25,
    "reasoning": "MT599 Swift requires precise keyword matching"
  }
}
```

#### Enhanced Validation
- Validates RAG parameters exist and are properly formatted
- Ensures weights are in valid range (0.0-1.0)
- Normalizes weights to sum to 1.0 if needed
- Provides default balanced weights (0.5/0.5) if parameters are missing or invalid
- Logs all validation steps and corrections

### 2. Updated Processing Flow (`src/keyword_code/app.py`)

#### Extract RAG Parameters
- Extracts `rag_params` from each decomposed sub-prompt
- Falls back to default 0.5/0.5 if not provided
- Logs the weights and reasoning for each sub-prompt

#### Pass Weights to Retrieval
- Updated both preprocessed and non-preprocessed retrieval paths
- Passes `bm25_weight` and `semantic_weight` to `retrieve_relevant_chunks()`
- Stores RAG params in `sub_prompts_with_contexts` for potential retry use

### 3. New Logging Function (`src/keyword_code/utils/interaction_logger.py`)

#### Added `log_rag_parameters()`
Logs RAG parameter selection with:
- Sub-prompt title and text
- BM25 and semantic weights
- Reasoning for weight selection
- Source (decomposition, retry_agent, or default)
- Timestamp in JSON format

### 4. Enhanced RAG Agent Logging (`src/keyword_code/agents/rag_agent.py`)

#### Added Detailed Logging
- Logs recommended weights from optimization agent
- Logs query type identification
- Logs reasoning for recommendations
- Calls `log_rag_parameters()` for retry operations

### 5. Enhanced Retrieval Logging (`src/keyword_code/rag/retrieval.py`)

#### Added Weight Logging
- Logs BM25 and semantic weights at the start of each retrieval
- Provides visibility into which weights are actually being used

## Benefits Achieved

✅ **Single LLM Call for Optimization** - Decomposition and RAG optimization in one call
✅ **Optimal Retrieval from Start** - Each sub-prompt gets optimized weights immediately
✅ **Comprehensive Logging** - Complete audit trail of all RAG parameter decisions
✅ **Backward Compatible** - Falls back to 0.5/0.5 if RAG params are missing
✅ **Retry Still Available** - RAG agent remains for user-initiated retries
✅ **No UI Changes** - As requested, no changes to the user interface

## Log Output Examples

### Decomposition Log
```
INFO - Decomposing prompt with RAG optimization: 'What is the MT599 Swift...'
INFO - RAG params for 'MT599 Swift Format': BM25=0.80, Semantic=0.20, Reasoning: MT599 Swift is highly specific terminology requiring exact keyword matching
INFO - Successfully decomposed prompt into 2 sub-prompts with titles and RAG parameters.
```

### Retrieval Log
```
INFO - Retrieving relevant chunks for sub-prompt 'MT599 Swift Format' for document.pdf
INFO - Using RAG weights - BM25: 0.80, Semantic: 0.20
INFO - RAG weight reasoning: MT599 Swift is highly specific terminology requiring exact keyword matching
INFO - RAG: Running hybrid search with weights - BM25: 0.80, Semantic: 0.20
INFO - Retrieved 10 chunks for sub-prompt 'MT599 Swift Format' in document.pdf using optimized RAG weights
```

### Interaction Log (JSON)
```json
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

## Files Modified

1. **src/keyword_code/ai/decomposition.py** - Enhanced decomposition with RAG parameters
2. **src/keyword_code/app.py** - Extract and use RAG parameters during processing
3. **src/keyword_code/utils/interaction_logger.py** - Added RAG parameter logging function
4. **src/keyword_code/agents/rag_agent.py** - Enhanced retry logging
5. **src/keyword_code/rag/retrieval.py** - Added weight logging at retrieval time

## Files Created

1. **test_decomposition_rag_params.py** - Test script for decomposition with RAG parameters
2. **RAG_CONSOLIDATION_SUMMARY.md** - This document

## Testing

### Test Script: `test_decomposition_rag_params.py`
Tests decomposition with various query types:
- MT599 Swift queries (should get high BM25 weight)
- Financial data queries (should get moderate BM25 weight)
- Conceptual legal queries (should get high semantic weight)
- Mixed queries (should get balanced or varied weights)

### To Run Tests
```bash
python test_decomposition_rag_params.py
```

## Rollback Plan

If issues arise, rollback is simple:
1. Revert `decomposition.py` to return only `title` and `sub_prompt`
2. Revert `app.py` to use default 0.5/0.5 weights
3. System will work exactly as before

All changes are additive and backward compatible.

## Conclusion

Successfully consolidated RAG optimization into the decomposition step. The system now intelligently selects RAG parameters based on query type during decomposition, while maintaining the ability to retry with agent-optimized parameters if needed.

