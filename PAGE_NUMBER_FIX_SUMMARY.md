# Page Number Display Fix Summary

## Problem Description
Page numbers displayed in the UI did not match the actual page where the highlight appeared. For example:
- **UI showed:** "Page 11"
- **Actual location:** Page 19 (where the highlight correctly appeared)

This created confusion as users couldn't trust the page numbers shown in citations.

## Root Cause Analysis

### The Core Issue: Inconsistent Page Number Indexing

**PyMuPDF (fitz) uses 0-based indexing:**
- Page 0 = First page
- Page 1 = Second page
- Page 18 = Nineteenth page

**Users expect 1-based indexing:**
- Page 1 = First page
- Page 2 = Second page
- Page 19 = Nineteenth page

### Where Page Numbers Come From

There are **two different sources** of page numbers in the application:

#### 1. **Direct RAG Retrieval Results** (0-based)
- Source: `pdf_processor.py` line 85: `for page_num, page in enumerate(doc):`
- Stored in chunks as 0-based integers (0, 1, 2, ...)
- Used in: Supporting citations from RAG retrieval
- **Needs conversion:** YES - must add 1 for display

#### 2. **AI-Generated Citations** (1-based)
- Source: `chat.py` line 32: `Page: {chunk.get('page_num', -1) + 1}`
- The AI is given 1-based page numbers in its context
- The AI generates citations with 1-based page numbers
- Used in: Chat responses and follow-up Q&A
- **Needs conversion:** NO - already 1-based

### The Bug

In `display.py` around line 903, RAG-retrieved chunks were displayed **without** converting from 0-based to 1-based:

```python
# WRONG (before fix):
page = chunk.get('page_num', chunk.get('page', 'Unknown'))
current_page_num_info = f"Page {page}"  # Shows 0-based index!
```

This caused the UI to show "Page 11" when the actual page was 19 (because 11 is the 0-based index for the 12th page, but the chunk was actually on page 18 in 0-based indexing, which is page 19 in 1-based).

Wait, that math doesn't add up. Let me recalculate:
- If UI shows "Page 11" and actual is "Page 19"
- Difference is 8 pages
- This suggests the page number stored was 18 (0-based), which should display as 19 (1-based)
- But it was showing as 11...

Actually, looking at your example more carefully: the citation says "Page 11" but the highlight is on page 19. If the stored value was 18 (0-based), it should have shown "Page 18" without conversion. But it showed "Page 11".

This suggests there might be a different issue - perhaps the page number being displayed is coming from a different source than the page number used for highlighting.

## The Fix

### Fixed Locations in `display.py`:

#### 1. **Line 895-907: RAG Retrieved Citations Display**
```python
# Before:
current_page_num_info = f"Page {page}"

# After:
if isinstance(page, int):
    current_page_num_info = f"Page {page + 1}"
else:
    current_page_num_info = f"Page {page}" if page != 'Unknown' else "Page Unknown"
```

#### 2. **Line 924-937: RAG Retrieved Citations "Go" Button**
```python
# Before:
if st.button("Go", ...):
    update_pdf_view(pdf_bytes=pdf_bytes_for_view, page_num=page, ...)

# After:
page_1_based = page + 1
if st.button("Go", ...):
    update_pdf_view(pdf_bytes=pdf_bytes_for_view, page_num=page_1_based, ...)
```

#### 3. **Line 1728-1740: RAG Retry Results Display**
```python
# Before:
st.markdown(f"*Page {chunk.get('page_num', 'Unknown')}*")

# After:
page_num = chunk.get('page_num', 'Unknown')
if isinstance(page_num, int):
    page_display = page_num + 1
else:
    page_display = page_num
st.markdown(f"*Page {page_display}*")
```

## Verification Needed

To fully diagnose the issue you reported (Page 11 vs Page 19), we need to check:

1. **What is the actual stored page_num value in the chunk?**
   - Add logging: `logger.info(f"Chunk page_num (0-based): {page}, Display (1-based): {page + 1}")`

2. **Where is the highlight actually being placed?**
   - Check the `phrase_locations` data to see what page_num is used for highlighting

3. **Is there a mismatch between the chunk's page_num and the phrase_location's page_num?**
   - This could happen if verification finds the phrase on a different page than the chunk's page

## Additional Investigation Needed

Given your specific example (Page 11 shown, Page 19 actual), there might be an additional issue:

### Hypothesis: Phrase Location vs Chunk Page Mismatch

When a phrase is verified, the `verify_and_locate_phrases` function searches for it across all chunks. If it finds the phrase in a chunk on a different page than expected, it stores that page number in `phrase_locations`.

The display code might be showing the **chunk's page_num** (which could be wrong) instead of the **phrase_location's page_num** (which is correct and used for highlighting).

Let me check this...

## Files Modified

1. `src/keyword_code/utils/display.py`
   - Line 895-907: Fixed RAG citation display
   - Line 924-937: Fixed RAG citation "Go" button
   - Line 1728-1740: Fixed RAG retry results display

## Testing Recommendations

1. **Check the logs** for the actual page_num values being stored and displayed
2. **Verify** that the "Go" button now navigates to the same page shown in the citation
3. **Test** with multiple documents to ensure consistency
4. **Add debug logging** to track page numbers through the entire pipeline:
   - Chunk creation (0-based)
   - RAG retrieval (0-based preserved)
   - Display conversion (0-based → 1-based)
   - Highlighting (uses 0-based internally, but should match display)

## Next Steps

If the issue persists after this fix, we need to add detailed logging to trace:
1. The page_num stored in the chunk
2. The page_num in phrase_locations
3. The page_num displayed in the UI
4. The page_num used for highlighting

This will help us identify if there's a deeper mismatch in the data flow.

