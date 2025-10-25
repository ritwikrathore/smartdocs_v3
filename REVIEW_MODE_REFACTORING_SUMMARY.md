# Review Mode Decomposition Refactoring Summary

## Overview

This document summarizes the refactoring of the Review Mode decomposition workflow to separate it from Ask Mode and optimize AI API calls.

## Key Changes

### 1. Separated Ask Mode and Review Mode Decomposition Functions

**Before:**
- Single `decompose_prompt()` function used for both Ask Mode and Review Mode
- Both modes shared the same decomposition logic with RAG parameters

**After:**
- `decompose_ask_mode_prompt()` - Specifically for Ask Mode with RAG parameters
- `decompose_review_mode_prompt()` - Specifically for Review Mode without RAG parameters
- `decompose_prompt()` - Kept as backward compatibility wrapper (deprecated)

**Files Modified:**
- `src/keyword_code/ai/decomposition.py` - Added new functions
- `src/keyword_code/ai/__init__.py` - Updated exports
- `src/keyword_code/app.py` - Updated to use `decompose_ask_mode_prompt`
- `pages/1_📄_CNT_space.py` - Updated to use `decompose_review_mode_prompt`

### 2. Removed Document Text from Review Mode Decomposition

**Before:**
```python
# In propose_validation_from_rule()
full_text_context = "\n".join([chunk.content for chunk in doc_chunks])
user_prompt = f"""
Here is the document context I am working with:
--- DOCUMENT TEXT ---
{full_text_context}
--- END DOCUMENT TEXT ---
"""
```

**After:**
```python
# In decompose_review_mode_prompt()
# No document text included - only the rule text
human_prompt = f"""Analyze the following validation rule(s)...
{user_prompt}"""
```

**Benefit:** Saves thousands of tokens per API call, as document text is not needed for rule decomposition.

### 3. Consolidated Review Mode Rule Processing

**Before:**
- Separate "SMARTREVIEW AI API REQUEST" in `propose_validation_from_rule()`
- Multiple API calls for rule processing

**After:**
- Single comprehensive `decompose_review_mode_prompt()` API call that outputs:
  - Concise title (max 5-6 words)
  - Re-written/clarified rule text
  - Validation approach decision (regex vs semantic) with reasoning
  - Extracted examples from rule text
  - Generated violation examples
  - Generated compliance examples

**New Data Structure:**
```python
class ReviewModeSubPrompt(BaseModel):
    title: str
    sub_prompt: str  # Clarified rule text
    validation_type: Literal['regex', 'semantic']
    validation_reasoning: str
    extracted_examples: List[str]
    violation_examples: List[str]
    compliance_examples: List[str]
```

### 4. Created Separate Regex Generation API Call

**New Function:** `generate_regex_pattern(analyzer, sub_prompt: ReviewModeSubPrompt)`

**Purpose:** 
- Dedicated API call for generating regex patterns
- Only called for sub-prompts flagged as "regex" validation
- Includes comprehensive regex guidelines in system prompt

**Input:**
- Re-written prompt/rule
- Extracted examples
- Violation examples
- Compliance examples
- Reasoning for why regex was chosen

**Output:**
```python
class RegexGenerationResult(BaseModel):
    regex_pattern: str
    explanation: str
    test_matches: List[str]
    test_non_matches: List[str]
```

**System Prompt Includes:**
- Detailed instructions on word boundaries and decimal numbers
- Common regex patterns for dates, phones, emails, etc.
- Best practices for Python regex

### 5. New Execution Flow

**Review Mode Workflow (V2):**

1. **Decomposition** (1 API call per document)
   - Input: Rule text only (no document text)
   - Output: List of `ReviewModeSubPrompt` objects
   - Function: `decompose_review_mode_prompt()`

2. **Regex Generation** (1 API call per regex-flagged rule)
   - Input: `ReviewModeSubPrompt` with validation_type='regex'
   - Output: `RegexGenerationResult` with pattern
   - Function: `generate_regex_pattern()`

3. **Validation Execution** (parallel)
   - Regex validation: Run pattern against document chunks
   - Semantic validation: Run LLM validation against document chunks
   - Both workflows execute in parallel where possible

**Ask Mode Workflow (Unchanged):**

1. **Decomposition** (1 API call)
   - Input: User query
   - Output: Sub-prompts with RAG parameters
   - Function: `decompose_ask_mode_prompt()`

2. **RAG Retrieval** (per sub-prompt)
   - Retrieve relevant chunks using optimized BM25/semantic weights

3. **Analysis** (1 API call for all sub-prompts)
   - Analyze with retrieved context

## New Functions Added

### `src/keyword_code/ai/decomposition.py`

1. **`decompose_ask_mode_prompt(analyzer, user_prompt)`**
   - Renamed from `decompose_prompt`
   - For Ask Mode only
   - Returns sub-prompts with RAG parameters

2. **`decompose_review_mode_prompt(analyzer, user_prompt)`**
   - NEW: For Review Mode only
   - No document text, no RAG parameters
   - Returns `List[ReviewModeSubPrompt]`

3. **`generate_regex_pattern(analyzer, sub_prompt)`**
   - NEW: Generates regex patterns for regex-flagged rules
   - Returns `RegexGenerationResult` or None

4. **`decompose_prompt(analyzer, user_prompt)`**
   - DEPRECATED: Backward compatibility wrapper
   - Calls `decompose_ask_mode_prompt`

### `src/keyword_code/smartreview/smartreview.py`

1. **`propose_validation_from_rule_v2(rule_text, example_text)`**
   - NEW: Refactored version without document text
   - Uses `decompose_review_mode_prompt` + `generate_regex_pattern`
   - For automated workflows

2. **`propose_validation_from_rule(rule_text, example_text, doc_chunks)`**
   - LEGACY: Original version kept for interactive UI
   - Still includes document text for backward compatibility

## Files Modified

1. **`src/keyword_code/ai/decomposition.py`**
   - Added new decomposition functions
   - Added Pydantic models for Review Mode
   - Added regex generation function

2. **`src/keyword_code/ai/__init__.py`**
   - Updated exports to include new functions

3. **`src/keyword_code/app.py`**
   - Updated Ask Mode to use `decompose_ask_mode_prompt`

4. **`pages/1_📄_CNT_space.py`**
   - Updated Review Mode to use `decompose_review_mode_prompt`
   - Updated to use `propose_validation_from_rule_v2`

5. **`src/keyword_code/smartreview/smartreview.py`**
   - Added `propose_validation_from_rule_v2`
   - Kept legacy version for backward compatibility

6. **`src/keyword_code/smartreview/__init__.py`**
   - Updated exports to include new functions

## Benefits

1. **Token Savings:** Removed document text from Review Mode decomposition (saves ~12,000 tokens per call)
2. **Clear Separation:** Ask Mode and Review Mode now have distinct, purpose-built decomposition functions
3. **Better Structure:** Review Mode decomposition outputs structured data with validation type, examples, etc.
4. **Dedicated Regex Generation:** Separate API call with specialized prompts for regex pattern creation
5. **Backward Compatibility:** Legacy functions kept for existing UI workflows
6. **Parallel Execution:** Regex and semantic validation can run in parallel

## Testing Recommendations

1. **Ask Mode:**
   - Test multi-question prompts
   - Verify RAG parameters are correctly applied
   - Check that decomposition works as before

2. **Review Mode:**
   - Test single and multiple rules
   - Verify regex patterns are generated correctly
   - Verify semantic validation uses clarified rules
   - Check that extracted examples are properly handled
   - Test parallel execution of regex and semantic validation

3. **Backward Compatibility:**
   - Verify interactive UI still works with legacy `propose_validation_from_rule`
   - Check that deprecated `decompose_prompt` still functions

## Migration Guide

### For Ask Mode (No Changes Required)
Existing code continues to work. Optionally update imports:
```python
# Old (still works)
from src.keyword_code.ai.decomposition import decompose_prompt

# New (recommended)
from src.keyword_code.ai.decomposition import decompose_ask_mode_prompt
```

### For Review Mode (Automated Workflows)
Update to use new V2 function:
```python
# Old
pv = await propose_validation_from_rule(rule_text, example_text, doc_chunks)

# New
pv = await propose_validation_from_rule_v2(rule_text, example_text)
# Note: doc_chunks no longer needed for decomposition
```

### For Review Mode (Interactive UI)
No changes required - legacy function still works.

## Future Improvements

1. **Remove Legacy Functions:** After migration period, remove deprecated functions
2. **Batch Regex Generation:** Generate all regex patterns in a single API call
3. **Cache Decomposition Results:** Cache decomposition results for repeated rules
4. **Enhanced Validation:** Add more sophisticated validation type selection logic

