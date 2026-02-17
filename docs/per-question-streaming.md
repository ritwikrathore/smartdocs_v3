# Per-Question Streaming Architecture

This document describes the per-question streaming implementation for SmartDocs single-file analysis, which enables progressive display of results as each sub-question is analyzed.

> **Related Documentation:**
> - `STREAMING_IMPLEMENTATION.md` - Multi-file streaming (processes files progressively)
> - This document covers single-file, per-question streaming (streams LLM responses)

## Overview

The streaming architecture reduces time-to-first-token by processing questions sequentially (RAG + LLM per question) rather than batching all RAG retrievals before starting any LLM calls.

## Flow Diagram

```
User clicks "Process Documents"
         │
         ▼
┌─────────────────────────────┐
│  Set streaming_pending      │
│  Set placeholder result     │
│  st.rerun()                 │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  RESULTS VIEW               │
│  (prompt box hidden)        │
│  Disabled "New Analysis"    │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Decomposition              │
│  (break prompt into         │
│   sub-questions)            │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  For each question:         │
│  ┌───────────────────────┐  │
│  │ 1. RAG retrieval      │  │
│  │ 2. Stream LLM response│──┼──► UI updates progressively
│  │ 3. Verify citations   │  │
│  └───────────────────────┘  │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  PDF Highlighting           │
│  (after all questions)      │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  st.rerun()                 │
│  Final display with         │
│  Go buttons & PDF viewer    │
└─────────────────────────────┘
```

## Key Components

### 1. Entry Point (`pages/1_📄_CNT_space.py`)

When "Process Documents" is clicked for single-file Ask mode:

```python
# Store streaming parameters in session state
st.session_state.streaming_pending = {
    "file_data": file_data,
    "filename": filename,
    "user_prompt": user_prompt,
    "preprocessed_data": preprocessed_file_data,
}
# Set a placeholder result to trigger results view
st.session_state.analysis_results = [{"filename": filename, "streaming": True}]
st.rerun()  # Switch to results view
```

### 2. Results View Detection

The results view checks for pending streaming:

```python
streaming_pending = st.session_state.get("streaming_pending")
if streaming_pending:
    # Run streaming analysis in results view
    result = run_streaming_analysis(...)
```

### 3. Streaming Function (`run_streaming_analysis`)

Located in `pages/1_📄_CNT_space.py`, this function:

- Creates the UI structure (two columns, progress bar)
- Processes streaming events from `process_file_streaming()`
- Updates placeholders as content arrives
- Triggers `st.rerun()` when complete

### 4. Backend Streaming (`src/keyword_code/app.py`)

`process_file_streaming()` is an async generator that yields events:

| Event Type | Description |
|------------|-------------|
| `decomposition` | Sub-prompts extracted from user query |
| `analysis_chunk` | Streaming token from LLM |
| `analysis_complete` | Section finished with verification results |
| `analysis_error` | Error processing a section |
| `verification_complete` | All sections done, PDF annotated |
| `error` | Fatal error |

### 5. Per-Question RAG (`_retrieve_single_subprompt`)

Instead of retrieving all chunks upfront:

```python
# OLD: Wait for all RAG before any streaming
sub_prompts_with_contexts = await retrieve_chunks_for_all_subprompts(...)

# NEW: RAG + Stream per question
for sub_prompt_data in sub_prompts:
    sub_prompt_ctx = await _retrieve_single_subprompt(sub_prompt_data, ...)
    async for chunk in analyzer.analyze_single_subprompt_streaming(...):
        yield chunk
```

### 6. Per-Section Verification (`PDFProcessor.verify_single_phrase`)

Citations are verified immediately after each section completes:

```python
for phrase in supporting_quotes:
    verified, locations = processor.verify_single_phrase(phrase)
    section_verification[phrase] = verified
    section_phrase_locations[phrase] = locations
```

This enables showing verification badges and page numbers during streaming.

## UI Components During Streaming

### Progress Bar
- Located below "AI Analysis Results" header
- Updates after each `analysis_complete` event
- No text, just visual progress

### Section Placeholders
Each section uses `st.empty()` containers:
- `content`: Analysis text (streams in, then formatted on complete)
- `citations`: Expander with verification badges

### No Interactive Elements
During streaming, the UI avoids buttons that could trigger reruns:
- Retry buttons: Disabled
- Go buttons: Not rendered (appear after final rerun)
- PDF viewer: Placeholder only

## File Structure

```
src/keyword_code/
├── app.py
│   ├── process_file_streaming()      # Main streaming generator
│   └── _retrieve_single_subprompt()  # Per-question RAG
├── ai/
│   ├── analyzer.py
│   │   └── analyze_single_subprompt_streaming()  # LLM streaming
│   └── databricks_llm.py
│       └── get_completion_stream_async()  # Async token streaming
├── processors/
│   └── pdf_processor.py
│       └── verify_single_phrase()    # Lightweight verification
└── display_utils/
    └── streaming_display.py          # (Unused - kept for reference)

pages/
└── 1_📄_CNT_space.py
    └── run_streaming_analysis()      # UI streaming handler
```

## Performance Characteristics

### Time to First Token
- **Before**: Decomposition + ALL RAG + First LLM call
- **After**: Decomposition + ONE RAG + First LLM token

### Total Time
Similar to batch processing since:
- RAG calls are now sequential (was parallel)
- But LLM streaming starts earlier
- Verification is incremental

### Memory
- Streaming chunks are small
- Results accumulated in state dict
- Final result same size as batch

## Error Handling

1. **RAG Failure**: Section gets empty context, LLM attempts with limited info
2. **LLM Streaming Error**: `analysis_error` event, fallback result created
3. **Verification Error**: Section marked unverified, no page location
4. **Fatal Error**: `error` event, displayed after rerun

## Limitations

1. **Single File Only**: Multi-file still uses batch processing
2. **Ask Mode Only**: Review mode uses different pipeline
3. **No Parallel Questions**: Questions processed sequentially for streaming UX
4. **Highlighting Deferred**: PDF annotations only after all questions complete
