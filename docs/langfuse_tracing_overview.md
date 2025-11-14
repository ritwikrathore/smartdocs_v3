# Langfuse Tracing Integration

## Summary
This document describes how Langfuse tracing is wired into the SmartDocs application, how sessions are propagated, and how the implementation behaves when tracing is disabled. It is intended for maintainers who need to extend tracing to additional components or diagnose trace output in Langfuse.

## Environment and Configuration
- Tracing relies on the Langfuse Python SDK. The helper module gracefully handles the SDK being absent.
- Enable tracing by setting the following environment variables (for example in `.env`):
  - `LANGFUSE_TRACING_ENABLED` (default `true` when keys are present)
  - `LANGFUSE_PUBLIC_KEY`
  - `LANGFUSE_SECRET_KEY`
  - `LANGFUSE_BASE_URL` (if using a self-hosted Langfuse instance)
- When keys are missing or `LANGFUSE_TRACING_ENABLED` resolves to false, all helper functions become no-ops and the application runs without tracing.

## Helper Module (`src/keyword_code/utils/langfuse_tracing.py`)
The helper module centralises interactions with the Langfuse SDK:

- `is_tracing_enabled()` checks for SDK availability, toggle flags, and credentials.
- `get_langfuse_client_cached()` lazily instantiates a client and caches it for reuse.
- Context managers:
  - `start_trace()` creates a root span (and propagates attributes). It accepts metadata, tags, and optional `session_id` / `user_id` so every document run shares a consistent context.
  - `start_span()` wraps non-LLM operations (e.g., rerankers) when needed.
  - `start_generation()` is reserved for LLM calls and captures model-specific metadata.
- Update helpers (`update_current_trace`, `update_current_span`, `update_current_generation`) map directly to Langfuse client methods but guard against missing clients.
- Output helpers (`set_generation_output`, `set_span_output`) and error reporters (`record_generation_error`, `record_span_error`) streamline common logging patterns.
- `optional_context()` reduces boilerplate where trace instrumentation is conditional.

All helper functions degrade safely when tracing is unavailable, ensuring no user-visible regressions.

## Session Tracking Flow
`process_file_wrapper` now resolves the Streamlit session identifier via `get_session_id()` and passes it to `start_trace()`.

- Success Path: When the session lookup succeeds, the trace is tagged with the session, filename, prompt, mode, and memory usage metadata so related retries or follow-up actions can be correlated in Langfuse.
- Failure Path: If session resolution fails, a debug log records the failure and the code continues with a trace that has no session context. The rest of the instrumentation remains intact.

## Instrumentation in `process_file_wrapper`
`src/keyword_code/app.py` owns the heavy document-processing workflow. The following instrumentation was added around the existing logic:

1. **Trace bootstrap**
   - Build a metadata payload (filename, prompt, mode, preprocessing state, pre-run memory usage).
   - Initialize a trace context with `start_trace`. The code manually enters the context so it can be closed cleanly in the `finally` block, regardless of early returns.

2. **Review Mode Shortcut**
   - If the request is in review mode, the function skips RAG and immediately updates the trace to a `skipped` state before returning.

3. **Keyword-only Shortcut**
   - When decomposition triggers keyword-only handling, the trace records a success outcome with the number of keyword sections and total occurrences before returning the keyword payload.

4. **Full Ask Mode**
   - After aggregation, the trace captures the total analysis sections and sub-prompt count for the successful run.

5. **Exception Handling**
   - Any raised exception updates both the current span and the root trace with `status="error"` and the exception message.
   - Memory cleanup still runs, ensuring existing stability safeguards remain intact.

6. **Trace Closure**
   - The trace context is always exited in `finally`. Any exit failures are logged at debug level to avoid hiding the root issue.

## Behaviour When Tracing Is Disabled
- The helper module returns `None` for every context manager when tracing is off.
- The manual context-management code detects this and simply skips entering/exiting the trace.
- All update helpers become no-ops, so the runtime path matches the pre-instrumentation behaviour.

## Extending Tracing
To instrument additional components (for example `DocumentAnalyzer` or reranker calls):
1. Wrap the target operation in `start_span` or `start_generation` from `utils.langfuse_tracing`.
2. Pass the relevant input metadata (prompt text, model name, top-K settings, etc.).
3. Use `set_generation_output` or `set_span_output` to capture outputs and token usage when available.
4. Update the current trace or span status in exception handlers to surface failures in Langfuse dashboards.

Doing so will automatically share the session context initiated in `process_file_wrapper`, keeping all downstream activity grouped in a single trace.

## Troubleshooting Checklist
- Validate that `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are present and correct.
- Confirm the host machine can reach the Langfuse server (`LANGFUSE_BASE_URL`).
- Check application logs for messages prefixed with "Langfuse" or debug statements about client initialization.
- Verify that `LANGFUSE_TRACING_ENABLED` has not been explicitly set to `false`.
- If traces are still missing, run with log level `DEBUG` to observe whether context managers are being entered successfully.
