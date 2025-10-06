# PDF Highlighting Fix Summary

## Problem Description
The PDF highlighting function was frequently failing to highlight the correct chunk or highlighting it incompletely. This was causing user frustration as citations and supporting evidence were not being properly visualized in the PDF viewer.

## Root Cause Analysis

### 1. **Critical Issue: Missing `bboxes` in Reranked Results**
**Location:** `src/keyword_code/rag/retrieval.py` lines 224-235 and 349-372

**Problem:** 
- When chunks are created in `pdf_processor.py`, they include a `bboxes` field that stores precise bounding box coordinates for highlighting
- During the reranking process, the code was creating NEW dictionaries with only these fields:
  - `text`
  - `page_num`
  - `score`
  - `chunk_id`
  - `retrieval_method`
- **The `bboxes` field was NOT being copied over!**
- When the highlighting code tried to access `chunk.get('bboxes', [])`, it would get an empty list, causing it to:
  - Fall back to imprecise highlighting methods
  - Highlight the wrong area
  - Highlight incompletely or not at all

**Fix:**
Changed the reranking result formatting to preserve ALL original chunk data:
```python
# Before (WRONG):
results.append({
    "text": chunk.get("text", ""),
    "page_num": chunk.get("page_num", -1),
    "score": float(score),
    "chunk_id": chunk.get("chunk_id", f"unknown_{chunk_index}"),
    "retrieval_method": "hybrid"
})

# After (CORRECT):
result_chunk = chunk.copy()  # Preserve ALL fields including bboxes
result_chunk["score"] = float(score)
result_chunk["retrieval_method"] = "hybrid"
results.append(result_chunk)
```

### 2. **Text Normalization Bug**
**Location:** `src/keyword_code/processors/pdf_processor.py` line 99

**Problem:**
- The regex pattern was using `r'\\s+'` (escaped backslash) instead of `r'\s+'`
- This meant the pattern was looking for literal backslash-s characters instead of whitespace
- This caused text normalization to fail, leading to character offset misalignment

**Fix:**
```python
# Before (WRONG):
block_text_cleaned = re.sub(r'\\s+', ' ', block_text_original).strip()

# After (CORRECT):
block_text_cleaned = re.sub(r'\s+', ' ', block_text_original).strip()
```

### 3. **Fragile Sentence Matching**
**Location:** `src/keyword_code/processors/pdf_processor.py` lines 133-144

**Problem:**
- Used simple `startswith()` matching which is fragile
- Would fail if there were any text variations or whitespace differences
- No fallback mechanism if exact match failed

**Fix:**
Added fuzzy matching fallback:
```python
# Try exact match first
if chunk_text_normalized.startswith(page_sentences[i].text.strip()):
    # Found match
    
# Fallback: Use fuzzy matching on first 50 chars
if not chunk_sentences and page_sentences:
    from fuzzywuzzy import fuzz
    chunk_start = chunk_text_normalized[:50]
    # Find best matching sentence with 80% similarity threshold
    # ...
```

### 4. **Limited Bounding Box Coverage**
**Location:** `src/keyword_code/processors/pdf_processor.py` lines 168-195

**Problem:**
- Only stored top 3 overlapping blocks
- Used 10% overlap threshold which was too strict
- This could miss text areas, especially for multi-line chunks

**Fix:**
- Increased from top 3 to top 5 blocks
- Lowered overlap threshold from 10% to 5%
- Added better logging for debugging

```python
# Before:
if overlap_length > 0.1 * block_length:  # 10% threshold
    overlapping_blocks.append((block_bbox, overlap_length))
# ...
for block_bbox, _ in overlapping_blocks[:3]:  # Top 3 only

# After:
if overlap_length > 0.05 * block_length:  # 5% threshold
    overlapping_blocks.append((block_bbox, overlap_length))
# ...
max_blocks = min(5, len(overlapping_blocks))  # Up to 5 blocks
for block_bbox, _ in overlapping_blocks[:max_blocks]:
```

### 5. **Insufficient Logging**
**Problem:**
- Hard to debug highlighting failures
- No visibility into which chunks were missing bboxes

**Fix:**
Added comprehensive logging:
- Debug logs for fallback bbox matching
- Warning logs when no bboxes found for a chunk
- Debug logs for fuzzy sentence matching

## Impact

### Before Fixes:
- ❌ Highlighting often failed or was incomplete
- ❌ Wrong areas highlighted due to missing bbox data
- ❌ Reranked chunks (most relevant ones!) had no highlighting capability
- ❌ Text normalization failures caused offset misalignment
- ❌ Difficult to debug issues

### After Fixes:
- ✅ All chunk data preserved through reranking pipeline
- ✅ Correct bounding boxes available for highlighting
- ✅ More robust text matching with fuzzy fallback
- ✅ Better coverage of text areas (5 blocks vs 3)
- ✅ Proper text normalization
- ✅ Better logging for debugging

## Testing Recommendations

1. **Test with reranked results:** Upload a PDF and verify that the highest-scoring (reranked) chunks are properly highlighted
2. **Test with multi-line chunks:** Verify that chunks spanning multiple lines are fully highlighted
3. **Test with complex layouts:** Try PDFs with tables, columns, or unusual formatting
4. **Check logs:** Look for warnings about missing bboxes - there should be significantly fewer
5. **Cross-page chunks:** Verify that chunks spanning page boundaries are handled correctly

## Files Modified

1. `src/keyword_code/rag/retrieval.py`
   - Fixed `rerank_results()` function (lines 224-234)
   - Fixed hybrid search without reranking (lines 348-371)

2. `src/keyword_code/processors/pdf_processor.py`
   - Fixed text normalization regex (line 99)
   - Improved sentence matching with fuzzy fallback (lines 133-166)
   - Improved bbox overlap detection (lines 168-198)
   - Added better logging (lines 200-216)

## Technical Details

### Why This Was Hard to Spot
1. The code appeared to work for some cases (when exact search found the phrase)
2. The fallback highlighting methods masked the underlying issue
3. The bbox data loss happened silently during reranking
4. No error messages were generated - just incorrect behavior

### Why This Fix Is Correct
1. Preserves data integrity through the entire pipeline
2. Maintains backward compatibility (all existing fields still present)
3. Minimal performance impact (shallow copy is fast)
4. Follows Python best practices for data preservation
5. Adds defensive programming with better logging

## Future Improvements

Consider these additional enhancements:
1. Add unit tests for bbox preservation through reranking
2. Add validation to ensure bboxes are present before highlighting
3. Consider storing bbox data separately from chunk text for better separation of concerns
4. Add metrics to track highlighting success rate
5. Consider caching bbox lookups for performance

