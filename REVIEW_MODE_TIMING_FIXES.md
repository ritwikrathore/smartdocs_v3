# Review Mode Timing and Completeness Fixes

## 🚨 Critical Finding

**ALL semantic validation was failing** due to a bug in `review_tools.py` that tried to access a non-existent `SR.client` attribute. This caused an `AttributeError` on every semantic validation attempt, which was being **silently swallowed** by the exception handlers.

**Result**: Only regex validation results were appearing in the UI. Semantic validation was completely broken.

**Fix**: Changed `SR.client._default_model` to `SR.DATABRICKS_LLM_MODEL` (lines 82 and 200 in `review_tools.py`).

## Issue Summary

Two related problems were observed in Review mode:

1. **Incomplete Results**: Sometimes only regex-based validation results appeared (semantic/LLM-based results missing), or vice versa
2. **Premature Display**: The UI sometimes displayed partial results before complete analysis finished

## Root Cause Analysis

### Primary Issue: Silent Exception Handling

**Location**: `src/keyword_code/agents/review_tools.py`

The semantic validation functions were silently swallowing ALL exceptions without logging:

```python
# Lines 97-99 (per-chunk semantic validation)
except Exception:
    pass
return []

# Lines 223-225 (batch semantic validation)
except Exception:
    # Skip batch on any LLM error; continue with others
    continue
```

**Impact**: When semantic validation failed due to:
- LLM API timeouts
- Network errors
- JSON parsing failures
- Model errors

...the failures were **completely silent** - no logs, no warnings, no errors. The validation would return empty results, making it appear as if no violations were found.

### Secondary Issue: Inefficient Task Orchestration

**Location**: `src/keyword_code/agents/review_orchestrator.py`

The orchestration logic created unnecessary empty tasks:

```python
# Lines 30-36 (original code)
for rule in template.rules:
    for chunk in doc_chunks:
        per_tasks.append(asyncio.create_task(_run_for_rule_chunk(rule, chunk)))
    
    # Per-rule semantic batch pass (single call per rule)
    if getattr(rule, "validation_type", None) == "semantic":
        per_tasks.append(asyncio.create_task(tools.run_semantic_batch(rule, doc_chunks)))
```

**Problem**: For semantic rules, this created:
- `len(doc_chunks)` empty tasks (one per chunk via `_run_for_rule_chunk`) that did nothing
- 1 batch task that actually performed the validation

**Example**: With 2 rules (1 regex, 1 semantic) and 10 pages:
- Regex rule: 10 tasks ✅ (correct)
- Semantic rule: 10 empty tasks + 1 batch task = 11 tasks (wasteful)

While this didn't directly cause failures, it created confusion and potential timing issues.

### Premature Display Analysis

**Finding**: Not an actual issue. The `run_async()` helper properly awaits all async operations using `asyncio.run()` or `loop.run_until_complete()`, ensuring complete execution before returning.

The perceived "premature display" was likely caused by incomplete results (Issue #1) making it appear as if the UI updated before completion.

## Fixes Implemented

### 1. Fixed Model Access Bug in `review_tools.py`

**Issue**: Lines 82 and 200 were trying to access `SR.client._default_model`, but the `smartreview` module doesn't have a `client` attribute.

**Fix**: Changed to access `SR.DATABRICKS_LLM_MODEL` constant instead:
```python
# Before (BROKEN):
model=getattr(SR.client, "_default_model", "databricks-llama-4-maverick")

# After (FIXED):
model=getattr(SR, "DATABRICKS_LLM_MODEL", "databricks-llama-4-maverick")
```

This was causing **ALL semantic validation to fail** with `AttributeError`, which was being silently swallowed before the logging fixes.

### 2. Enhanced Error Logging in `review_tools.py`

**Per-chunk semantic validation** (lines 97-106):
```python
except Exception as e:
    # Log the error but don't fail the entire validation
    import logging
    logger = logging.getLogger(__name__)
    logger.error(
        f"Semantic validation failed for rule '{getattr(rule, 'description', 'unknown')}' "
        f"on page {getattr(chunk, 'page_num', '?')}: {e}",
        exc_info=True
    )
return []
```

**Batch semantic validation** (lines 210-250):
- Added warning when LLM returns non-list response
- Added error logging for individual item parsing failures
- Added comprehensive error logging for batch failures with full exception details

### 3. Optimized Task Orchestration in `review_orchestrator.py`

**New logic** (lines 22-71):
```python
for rule in template.rules:
    validation_type = getattr(rule, "validation_type", None)
    
    if validation_type == "regex":
        # Regex validation: run per chunk
        for chunk in doc_chunks:
            per_tasks.append(asyncio.create_task(_run_for_rule_chunk(rule, chunk)))
    elif validation_type == "semantic":
        # Semantic validation: run batch mode (single call per rule across all chunks)
        per_tasks.append(asyncio.create_task(tools.run_semantic_batch(rule, doc_chunks)))
    else:
        # Unknown validation type: log warning and skip
        logger.warning(f"Unknown validation_type '{validation_type}' for rule...")
```

**Benefits**:
- No more unnecessary empty tasks for semantic rules
- Clear separation of regex (per-chunk) vs semantic (batch) execution
- Warning logged for unknown validation types
- More efficient task creation and execution

### 4. Enhanced Logging in `review_evaluator.py`

**Added logging** (lines 104, 119, 121-126):
- Info log when AI evaluation starts
- Info log when AI evaluation completes with result count
- Error log (with full traceback) when AI evaluation fails and falls back to deterministic scoring

### 5. Enhanced Logging in `smartreview.py`

**Added logging** (lines 561-568, 573, 586, 598):
- Log rule type breakdown (e.g., "2 regex, 3 semantic")
- Log when orchestrated review completes with finding count
- Log when validation template execution completes successfully
- Log when legacy execution completes with result count

### 6. Enhanced Logging in `pages/1_📄_CNT_space.py`

**Added logging** (lines 179-187, 292-299):
- Log when validation starts for each file
- Log when validation completes with result count
- Log detailed error information when validation fails
- Log when results are stored in session state with breakdown per file

## Testing Recommendations

### 1. Test with Mixed Rule Types

Create a template with both regex and semantic rules:
```
• All monetary values must use decimal precision (e.g., "5.5 billion" not "5 billion")
• Check for the pattern: \b\d+\s+billion\b
```

**Expected**: Both regex and semantic results should appear in the UI.

### 2. Test with LLM Failures

Temporarily set an invalid `DATABRICKS_API_KEY` or `DATABRICKS_BASE_URL`:

**Expected**: 
- Error logs should appear in the console showing the LLM failure
- Regex results should still appear (if any regex rules exist)
- Semantic results should be empty with clear error messages

### 3. Test with Large Documents

Upload a 50+ page document with multiple rules:

**Expected**:
- Console logs should show task creation and execution progress
- All results should appear after complete execution
- No partial results should be displayed

### 4. Monitor Console Logs

Check the server console for the new log messages:
```
INFO: Rule breakdown: {'regex': 2, 'semantic': 3}
INFO: Running AI evaluation on 15 findings...
INFO: AI evaluation completed. Returned 12 ranked findings.
INFO: Validation completed for document.pdf. Found 12 results.
INFO: Storing 1 aggregated results in session state.
INFO:   Result 1: document.pdf - 12 validation results
INFO: Results stored successfully. UI will update on next rerun.
```

## Impact Assessment

### Before Fixes
- ❌ Silent failures when LLM calls failed
- ❌ Incomplete results with no indication of what went wrong
- ❌ Wasteful task creation (10x more tasks than needed for semantic rules)
- ❌ Difficult to debug issues

### After Fixes
- ✅ All failures are logged with full error details
- ✅ Clear visibility into validation execution flow
- ✅ Efficient task orchestration (only necessary tasks created)
- ✅ Easy to identify and debug issues from console logs
- ✅ Better error messages for users when things go wrong

## Files Modified

1. `src/keyword_code/agents/review_tools.py` - Enhanced error logging in semantic validation
2. `src/keyword_code/agents/review_orchestrator.py` - Optimized task orchestration
3. `src/keyword_code/agents/review_evaluator.py` - Enhanced logging in AI evaluation
4. `src/keyword_code/smartreview/smartreview.py` - Enhanced logging in validation execution
5. `pages/1_📄_CNT_space.py` - Enhanced logging in auto review update

## Next Steps

1. **Test the fixes** with various scenarios (mixed rules, LLM failures, large documents)
2. **Monitor logs** during testing to ensure all error paths are properly logged
3. **Verify completeness** - ensure both regex and semantic results appear consistently
4. **Performance check** - verify the optimized orchestration improves execution time
5. **User feedback** - confirm the issues are resolved in production use

