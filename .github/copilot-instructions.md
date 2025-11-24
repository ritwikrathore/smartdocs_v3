# SmartDocs Copilot Guide

## Architecture & Data Flow
- Streamlit page `pages/1_📄_CNT_space.py` owns the UI, toggles Ask/Review modes, and calls into `src/keyword_code/app.py` for heavy lifting.
- File uploads land in `process_file_wrapper`, which reuses `preprocess_file` results (pdf/docx -> `PDFProcessor`/`WordProcessor` -> chunks + embeddings) cached in `st.session_state`.
- Ask mode pipeline: prompt decomposition (`ai/decomposition.py`) → per-subprompt RAG (`rag/retrieval.py`) → single-call LLM analysis (`ai/analyzer.py`) → PDF verification/annotation (`processors/pdf_processor.py`).

## RAG & Retrieval Conventions
- Always obtain embeddings via `models.embedding.load_embedding_model()`; reranking goes through `load_reranker_model()` to keep Databricks vs fallback logic intact.
- Hybrid retrieval expects `bm25_weight + semantic_weight = 1.0`; log adjustments with `log_rag_parameters` so `logs/rag_interactions_*.log` stays consistent.
- Async helpers (`utils.async_utils.run_async` / `retrieve_relevant_chunks_async`) are already threadpooled—avoid raw `asyncio.run` inside Streamlit callbacks.

## LLM Integration
- All LLM calls are funneled through `DocumentAnalyzer` and the cached `DatabricksLLMClient`; do not instantiate OpenAI clients directly.
- Databricks credentials must live in `.env` as `DATABRICKS_API_KEY`; `config.py` loads it once and writes both console + `logs/app_*.log` outputs per run.
- Respect the JSON contract emitted by `analyze_document_with_all_contexts`; downstream displays expect `analysis_summary`, `supporting_quotes`, and `analysis_context` fields.

## Review Mode & SmartReview
- Review mode bypasses Ask pipelines—`run_auto_review_update` decomposes checklist text, builds `SmartReview` rules, executes validators, then pushes annotated PDFs via the same verification stack.
- Validation agents live in `smartreview/smartreview.py`; prefer `propose_validation_from_rule_v2` and `execute_validation_template` rather than rolling new prompts.

## Fact Extraction & Agents
- Structured fact export uses `FactExtractionService`; pass the raw analysis JSON from Ask mode to `extract_facts` to stay in sync with Pydantic agent schemas.
- RAG retry buttons rely on `agents/rag_agent.py`; if you change retry payloads, update both the agent schema and the Streamlit state keys (`rag_analysis_requests`, `section_facts_requests`).

## UI & Display Patterns
- Shared UI helpers live in `display_utils/` and `utils/ui_helpers.py`; keep new widgets there so branding CSS (`apply_ui_styling`) stays centralized.
- Markdown rendered to HTML should flow through `render_limited_markdown` to enforce the restricted formatting the PDF verifier expects.

## Session, Temp Files, Memory
- Session-scoped data is managed in `initialize_session_state`; use `clear_session_for_new_query` before resetting analysis state to prevent stray temp files.
- Temporary artifacts are tracked by `utils.file_manager`; register new files so automatic cleanup (and `tmp/` hygiene) keeps working.
- Monitor large objects with `memory_monitor.cleanup_memory` instead of manual `del`—the app auto-clears embeddings when RAM pressure is high.

## Local Development
- Install with `python -m pip install -r requirements.txt` (Python 3.11+ recommended); launch with `streamlit run Home.py` from the repo root.
- spaCy model `en_core_web_sm` is downloaded at runtime into `models/spacy/`; ensure write access when running locally or in CI.
- Tests folder is currently empty—manual verification relies on logs, annotated PDFs, and fact extraction outputs; add regression notebooks or Streamlit scripts near `tests/` if needed.

## Langfuse Tracing
- Tracing is centralized in `utils/langfuse_tracing.py`. Use `start_trace` for new document runs and `continue_trace` to append spans (e.g., retries) to an existing trace ID.
- `process_file_wrapper` captures the trace ID and returns it in the result payload; `pages/1_📄_CNT_space.py` stores this ID in `st.session_state.analysis_results` to link subsequent user actions.
- Ensure `ENABLE_LANGFUSE_TRACING` is set in config/env.
