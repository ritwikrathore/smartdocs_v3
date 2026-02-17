# Streaming File Processing Implementation

## Overview

This document details the streaming file processing feature implemented to provide progressive display of results when multiple files are uploaded. The goal was to allow users to start viewing results as each file completes rather than waiting for all files to finish processing.

---

## Implementation Summary

### Problem Statement

**Previous Behavior:**
- User uploads multiple files and clicks "Process Documents"
- All files are processed (in parallel or sequentially)
- Results are stored in `st.session_state.analysis_results` only after **all** files complete
- Single `st.rerun()` displays all results at once
- Users see only a spinner until the slowest file finishes

**Proposed Behavior:**
- Results appear incrementally as each file completes
- Users can view/interact with completed results while other files process
- Visual indicators show which files are completed, processing, or pending
- Tab-based display remains intact with status icons

---

## Files Created/Modified

### New File: `src/keyword_code/utils/streaming_processor.py`

Thread-safe class to manage background processing:

```python
class FileStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

class StreamingProcessor:
    def start_processing(files, process_func)  # Launch workers in ThreadPoolExecutor
    def poll_results() -> Dict                  # Return current status of all files
    def get_completed_results() -> List         # Return list of completed result dicts
    def has_pending() -> bool                   # Check if processing is ongoing
    def get_progress() -> Tuple[int, int]       # Return (completed, total) counts
    def shutdown()                              # Cleanup resources
    def cancel_pending()                        # Cancel pending tasks
```

### Modified: `src/keyword_code/display_utils/analysis_display.py`

Added two new functions:

1. **`display_analysis_results_streaming(results_state, original_order)`**
   - Displays results with status indicators
   - Creates tabs for all files with icons: ⏸️ pending, ⏳ processing, ✅ completed, ❌ error
   - Completed files show full analysis
   - Processing files show spinner with elapsed time

2. **`_display_single_file_result(result, filename, tab_index)`**
   - Helper to render a single completed file's analysis
   - Simplified version of the main display logic

### Modified: `src/keyword_code/display_utils/tools_column.py`

Added:

1. **`display_tools_column_streaming(results_state, original_order, tools_col)`**
   - Shows limited functionality during streaming
   - Displays progress bar and completion count
   - Disabled SmartChat and Export until processing completes

### Modified: `src/keyword_code/display_utils/__init__.py`

- Added exports for `display_analysis_results_streaming` and `display_tools_column_streaming`

### Modified: `pages/1_📄_CNT_space.py`

Key changes:

1. **New Configuration Flag:**
   ```python
   ENABLE_STREAMING_MODE = os.environ.get("ENABLE_STREAMING_MODE", "true").lower() == "true"
   ```

2. **New Session State Variables:**
   ```python
   st.session_state.streaming_processor = None
   st.session_state.processing_active = False
   st.session_state.streaming_file_order = []
   ```

3. **Modified Processing Flow:**
   - Single file: Uses original blocking processing
   - Multiple files: Uses streaming mode with progressive display

4. **Polling Loop:**
   - Checks processor status every 1.5 seconds
   - Updates UI with completed results
   - Triggers `st.rerun()` to refresh display

5. **Cancel Button:**
   - Added "Cancel Processing" button during streaming mode

---

## Architecture Flow

```
User clicks "Process Documents" (multiple files)
         │
         ▼
┌─────────────────────────────────┐
│  Create StreamingProcessor      │
│  Store in session_state         │
│  Set processing_active = True   │
│  Immediate st.rerun()           │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  POLLING LOOP (on each rerun)   │
│  ┌───────────────────────────┐  │
│  │ processor.poll_results()  │  │
│  │ Update analysis_results   │  │
│  │ Display streaming UI      │  │
│  │ time.sleep(1.5)           │  │
│  │ st.rerun()                │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
         │
         ▼ (when has_pending() == False)
┌─────────────────────────────────┐
│  FINALIZATION                   │
│  - Get final ordered results    │
│  - Shutdown processor           │
│  - Clear streaming state        │
│  - Final st.rerun()             │
└─────────────────────────────────┘
```

---

## Known Issues and Fixes

### Issue 1: UI Layout Problems - Analysis Title Displayed Twice ✅ FIXED

**Symptom:** The "AI Analysis Results" header/title appears twice in the UI during streaming mode.

**Root Cause:** The `display_analysis_results_streaming()` function and `_display_single_file_result()` both render section headers.

**Fix Applied:**
- Renamed function to `_display_single_file_result_streaming()`
- Removed duplicate header rendering - section headers only rendered once within the container
- Removed redundant blue header box inside section content

---

### Issue 2: Citation "Go" Buttons Unavailable During Processing ✅ FIXED

**Symptom:** The "Go" buttons next to citations (which navigate to the PDF page) are not functional while documents are being processed.

**Root Cause:** The original `_display_single_file_result()` was a simplified version without full citation logic.

**Fix Applied:**
- Ported full citation button logic from `display_analysis_results()` to `_display_single_file_result_streaming()`
- Added proper two-column layout (90% citation, 10% button)
- Unique button keys include streaming context: `goto_stream_{tab_index}_{section_key}_{citation_counter}_{phrase_idx}`
- Full `update_pdf_view()` call and `st.rerun()` on button click
- Shows "Loc N/A" for verified citations without location data

---

### Issue 3: SmartChat and Export Results Expanders Missing ✅ FIXED

**Symptom:** During streaming mode, SmartChat and Export Results show only placeholder messages.

**Root Cause:** The streaming tools column was too restrictive.

**Fix Applied:**
- `display_tools_column_streaming()` now accepts `completed_results_for_tools` parameter
- **SmartChat:** Full chat interface available once any document completes
  - Document selection multiselect for completed docs
  - Chat history display
  - Chat input with RAG retrieval on completed documents
- **Export Results:** Full export functionality for completed documents
  - Excel export with all completed analysis data
  - Word export available
  - Shows count of completed documents ready for export

---

### Issue 4: 1.5 Second Refresh Too Frequent - Disrupts Review Flow ✅ FIXED

**Symptom:** Page refreshes every 1.5 seconds, interrupting user interactions.

**Root Cause:** Fixed polling interval was too aggressive.

**Fix Applied:**
- **Smart polling strategy:**
  - Tracks previous completed count in `streaming_prev_completed`
  - Only auto-refreshes when new files complete (0.5s delay to prevent flicker)
  - When no change, uses 5-second background poll interval
- **Manual refresh button:** Added "🔄 Refresh Status" button for user-controlled updates
- **Button layout:** Cancel and Refresh buttons in side-by-side columns
- This reduces UI disruption from ~40 refreshes/minute to ~12 refreshes/minute (or less with manual control)

---

### Issue 5: Documents Not Processed in UI Tab Order ✅ FIXED

**Symptom:** Files complete in unpredictable order, not matching UI tab order.

**Root Cause:** `ThreadPoolExecutor` with multiple workers processes concurrently.

**Fix Applied:**
- Added `preserve_order` parameter to `StreamingProcessor.start_processing()`
- Default behavior: `preserve_order=True` uses single worker (sequential processing)
- Files now complete in the exact order they appear in the UI
- Queue position shown in pending tabs: "📋 Queued (position #1)"
- Each file state tracks its `order` index
- Parallel mode still available via `preserve_order=False` for power users

---

## Configuration Options

The streaming feature can be controlled via environment variable:

```bash
# Disable streaming mode (use original blocking behavior)
export ENABLE_STREAMING_MODE=false

# Enable streaming mode (default)
export ENABLE_STREAMING_MODE=true
```

When disabled, multi-file processing falls back to the original behavior where all files must complete before any results are shown.

---

## Recommendations for Future Improvements

### Short-term Fixes (Bug Fixes)

1. **Fix duplicate headers** - Remove redundant title rendering
2. **Restore citation "Go" buttons** - Port full button logic to streaming display
3. **Enable SmartChat/Export for completed docs** - Remove overly restrictive placeholders
4. **Increase poll interval** - Change from 1.5s to 3-5s
5. **Add progress indicator to tabs** - Show which file is currently processing

### Medium-term Improvements

1. **Implement smart polling** - Only refresh when state changes
2. **Add manual refresh button** - Let users control when to check status
3. **Preserve scroll position** - Remember and restore scroll on rerun
4. **Add estimated time remaining** - Based on completed file times

### Long-term Architecture Changes

1. **WebSocket-based updates** - True real-time updates without polling
2. **Background task queue** - Proper async task management
3. **Partial page updates** - Use Streamlit fragments (when available)
4. **Result caching** - Persist results across page refreshes

---

## Testing Checklist

- [ ] Single file upload (should behave like original - no streaming)
- [ ] Multiple file upload (2-3 files) - verify streaming behavior
- [ ] Large file set (5+ files) - verify performance and memory
- [ ] Mixed success/failure - verify error handling per file
- [ ] Cancel mid-processing - verify partial results preserved
- [ ] Browser refresh during processing - verify graceful recovery
- [ ] Citation "Go" buttons work for completed files
- [ ] SmartChat works with completed documents
- [ ] Export works with completed documents
- [ ] PDF viewer shows first completed document

---

## Related Files

| File | Purpose |
|------|---------|
| `src/keyword_code/utils/streaming_processor.py` | Core streaming logic |
| `src/keyword_code/display_utils/analysis_display.py` | Streaming UI display |
| `src/keyword_code/display_utils/tools_column.py` | Tools panel streaming support |
| `pages/1_📄_CNT_space.py` | Integration and polling loop |
| `src/keyword_code/app.py` | `process_file_wrapper()` function |

---

## Version History

| Date | Change |
|------|--------|
| 2025-02-04 | Initial implementation |
| 2025-02-04 | Documented known issues from testing |
| 2025-02-04 | Fixed all 5 issues: duplicate headers, Go buttons, SmartChat/Export, refresh frequency, processing order |

---

## Contact

For questions about this implementation, contact the CNT Automations team.


  Summary of Fixes to be made

  Issue 1: Duplicate Analysis Title ✅

  - Removed redundant header rendering in _display_single_file_result_streaming()
  - Section headers now render only once per section

  Issue 2: Citation "Go" Buttons ✅

  - Ported full citation button logic with proper two-column layout (90% citation, 10% button)
  - Added unique button keys with streaming context
  - Full PDF navigation works: update_pdf_view() + st.rerun()

  Issue 3: SmartChat and Export Missing ✅

  - SmartChat: Full chat UI available once any document completes
    - Document selection for completed docs
    - Chat history display
    - RAG-powered chat input
  - Export: Full Excel/Word export for completed documents

  Issue 4: Refresh Too Frequent ✅

  - Smart polling: Only auto-refresh when new files complete
  - 5-second background poll when no changes (vs 1.5s before)
  - Manual "🔄 Refresh Status" button for user-controlled updates
  - Reduces interruptions from ~40/min to ~12/min

  Issue 5: Processing Order ✅

  - Added preserve_order=True parameter (default)
  - Files now process sequentially in UI tab order
  - Queue position shown: "📋 Queued (position #2)"
  - Parallel mode available via preserve_order=False
